package gift

import (
	"image"
	"image/draw"
	"math"
)

// GaborCombineMode selects how the per-orientation gabor responses are
// combined into the final result.
type GaborCombineMode int

const (
	// GaborMaxResponse picks, for each pixel, the response with the largest
	// absolute value among all configured orientations.
	GaborMaxResponse GaborCombineMode = iota
	// GaborFirstOrientation uses the response of the first orientation only,
	// ignoring the rest of the orientation list.
	GaborFirstOrientation
)

const (
	// maxGaborKernelRadius bounds the generated gabor kernels so that
	// extreme wavelength/bandwidth combinations cannot produce
	// unreasonably large (slow, memory hungry) kernels.
	maxGaborKernelRadius = 64
	// gaborAspectRatio is the spatial aspect ratio (gamma) of the gabor
	// envelope. 0.5 gives an ellipse elongated along the orientation.
	gaborAspectRatio = 0.5
)

func gaborIsFinite(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0)
}

// gaborKernel generates a finite, zero-mean, real gabor kernel for the given
// orientation (degrees), wavelength (pixels) and bandwidth (octaves).
// The zero mean guarantees that constant image regions produce a zero
// response, so the filter never brightens or darkens flat areas.
// Returns the kernel size (odd) and the kernel of size*size elements.
func gaborKernel(thetaDeg, wavelength, bandwidth float32) (int, []float32) {
	// Convert bandwidth in octaves to the gaussian envelope sigma:
	// sigma = lambda/pi * sqrt(ln2/2) * (2^b+1)/(2^b-1)
	pow2b := math.Pow(2, float64(bandwidth))
	sigma := float64(wavelength) / math.Pi * math.Sqrt(math.Ln2/2) * (pow2b + 1) / (pow2b - 1)

	radius := int(math.Ceil(3 * sigma))
	if radius < 1 {
		radius = 1
	}
	if radius > maxGaborKernelRadius {
		radius = maxGaborKernelRadius
	}
	size := 2*radius + 1

	sin, cos := math.Sincos(float64(thetaDeg) * math.Pi / 180)
	kernel := make([]float32, size*size)

	var sum, envSum float64
	for j := 0; j < size; j++ {
		for i := 0; i < size; i++ {
			x := float64(i - radius)
			y := float64(j - radius)
			xr := x*cos + y*sin
			yr := -x*sin + y*cos
			envelope := math.Exp(-(xr*xr + gaborAspectRatio*gaborAspectRatio*yr*yr) / (2 * sigma * sigma))
			w := envelope * math.Cos(2*math.Pi*xr/float64(wavelength))
			kernel[j*size+i] = float32(w)
			sum += w
			envSum += envelope
		}
	}

	// Subtract the mean so that the kernel has zero DC response, then
	// normalize by the envelope energy so that a matched sinusoid of
	// amplitude A produces a response of roughly A/2 and the amount
	// parameter stays meaningful regardless of wavelength and bandwidth.
	mean := float32(sum / float64(len(kernel)))
	scale := float32(1)
	if envSum > 0 {
		scale = float32(1 / envSum)
	}
	for i := range kernel {
		kernel[i] = (kernel[i] - mean) * scale
	}

	return size, kernel
}

type gaborFilter struct {
	orientations []float32
	wavelength   float32
	bandwidth    float32
	amount       float32
	combine      GaborCombineMode
}

func (p *gaborFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

// convolveLuminance applies the kernel weights to the luminance plane with
// edge clamping and stores the per-pixel response into resp.
func convolveLuminance(resp, lum []float32, w, h int, weights []uvweight, options *Options) {
	parallelize(options.Parallelization, 0, h, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			for x := 0; x < w; x++ {
				var acc float32
				for _, wt := range weights {
					sx := x + wt.u
					if sx < 0 {
						sx = 0
					} else if sx > w-1 {
						sx = w - 1
					}
					sy := y + wt.v
					if sy < 0 {
						sy = 0
					} else if sy > h-1 {
						sy = h - 1
					}
					acc += lum[sy*w+sx] * wt.weight
				}
				resp[y*w+x] = acc
			}
		}
	})
}

func (p *gaborFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	w, h := srcb.Dx(), srcb.Dy()

	if w <= 0 || h <= 0 {
		return
	}

	if p.amount == 0 || len(p.orientations) == 0 {
		copyimage(dst, src, options)
		return
	}

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// Build the luminance plane.
	n := w * h
	lum := make([]float32, n)
	parallelize(options.Parallelization, 0, h, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				lum[y*w+x] = 0.299*px.r + 0.587*px.g + 0.114*px.b
			}
		}
	})

	orientations := p.orientations
	if p.combine == GaborFirstOrientation && len(orientations) > 1 {
		orientations = orientations[:1]
	}

	resp := make([]float32, n)
	tmp := make([]float32, n)
	for i, theta := range orientations {
		_, kernel := gaborKernel(theta, p.wavelength, p.bandwidth)
		_, weights := prepareConvolutionWeights(kernel, false)

		if i == 0 {
			convolveLuminance(resp, lum, w, h, weights, options)
			continue
		}
		convolveLuminance(tmp, lum, w, h, weights, options)
		parallelize(options.Parallelization, 0, h, func(pmin, pmax int) {
			for y := pmin; y < pmax; y++ {
				for x := 0; x < w; x++ {
					k := y*w + x
					if absf32(tmp[k]) > absf32(resp[k]) {
						resp[k] = tmp[k]
					}
				}
			}
		})
	}

	// Apply the luminance response to the RGB channels, keeping the
	// original alpha untouched.
	parallelize(options.Parallelization, 0, h, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				d := p.amount * resp[y*w+x]
				pixSetter.setPixel(
					dstb.Min.X+x, dstb.Min.Y+y,
					pixel{px.r + d, px.g + d, px.b + d, px.a},
				)
			}
		}
	})
}

// Gabor creates a filter that enhances directional textures (such as wood
// grain) using a bank of gabor kernels, without blurring the texture away
// like an isotropic blur would.
//
// The orientations parameter lists the kernel orientations in degrees
// (0 means the response varies along the x axis, 90 along the y axis);
// non-finite entries are ignored and angles are normalized to [0, 180).
// The wavelength parameter is the stripe period in pixels the filter is
// tuned to; non-positive or non-finite values fall back to a safe default.
// The bandwidth parameter is the frequency bandwidth in octaves and
// controls the size of the gaussian envelope; it is clamped to a safe range.
// The amount parameter scales the response added to the image;
// a zero amount makes the filter a no-op.
// The combine parameter selects how the per-orientation responses are
// merged: GaborMaxResponse keeps the strongest absolute response per pixel,
// GaborFirstOrientation uses only the first orientation.
//
// The filter computes the response on the image luminance and adds it to
// the RGB channels; the alpha channel is preserved. The generated kernels
// have zero mean, so constant regions are left unchanged. The output bounds
// are identical to the input bounds.
//
// Example:
//
//	// Enhance wood grain running in two directions.
//	g := gift.New(
//		gift.Gabor([]float32{0, 90}, 8, 1, 0.5, gift.GaborMaxResponse),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func Gabor(orientations []float32, wavelength, bandwidth, amount float32, combine GaborCombineMode) Filter {
	if !gaborIsFinite(wavelength) || wavelength < 1 {
		wavelength = 4
	}
	if !gaborIsFinite(bandwidth) || bandwidth <= 0 {
		bandwidth = 1
	}
	if bandwidth < 0.5 {
		bandwidth = 0.5
	}
	if bandwidth > 10 {
		bandwidth = 10
	}
	if !gaborIsFinite(amount) {
		amount = 0
	}
	if combine != GaborMaxResponse && combine != GaborFirstOrientation {
		combine = GaborMaxResponse
	}

	ors := make([]float32, 0, len(orientations))
	for _, o := range orientations {
		if !gaborIsFinite(o) {
			continue
		}
		o = float32(math.Mod(float64(o), 180))
		if o < 0 {
			o += 180
		}
		ors = append(ors, o)
	}
	if len(ors) == 0 {
		ors = []float32{0}
	}

	return &gaborFilter{
		orientations: ors,
		wavelength:   wavelength,
		bandwidth:    bandwidth,
		amount:       amount,
		combine:      combine,
	}
}
