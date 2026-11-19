package gift

import (
	"image"
	"image/draw"
	"math"
)

// haarForward1D performs one level of the normalized Haar wavelet transform
// on data[:n] in place, using tmp as scratch space.
// After the call data[:ceil(n/2)] holds the approximation (low-pass)
// coefficients and data[ceil(n/2):n] holds the detail (high-pass) coefficients.
// For odd n the last sample is carried into the approximation band unchanged.
func haarForward1D(data, tmp []float32, n int) {
	m := n / 2
	l := n - m
	for i := 0; i < m; i++ {
		a := data[2*i]
		b := data[2*i+1]
		tmp[i] = (a + b) / 2
		tmp[l+i] = (a - b) / 2
	}
	if l > m {
		tmp[m] = data[n-1]
	}
	copy(data[:n], tmp[:n])
}

// haarInverse1D reverses haarForward1D: it takes the approximation and detail
// coefficients packed by haarForward1D and restores the original samples.
func haarInverse1D(data, tmp []float32, n int) {
	m := n / 2
	l := n - m
	for i := 0; i < m; i++ {
		tmp[2*i] = data[i] + data[l+i]
		tmp[2*i+1] = data[i] - data[l+i]
	}
	if l > m {
		tmp[n-1] = data[m]
	}
	copy(data[:n], tmp[:n])
}

// softThreshold shrinks the detail coefficient d towards zero by t.
func softThreshold(d, t float32) float32 {
	if d > t {
		return d - t
	}
	if d < -t {
		return d + t
	}
	return 0
}

type waveletDenoiseFilter struct {
	levels    int
	threshold float32
}

func (p *waveletDenoiseFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *waveletDenoiseFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	w, h := srcb.Dx(), srcb.Dy()
	if w < 1 || h < 1 {
		return
	}

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// Extract the luminance channel.
	// A single w*h float32 plane (plus a max(w, h) scratch buffer per worker)
	// is all the extra memory the filter needs.
	lum := make([]float32, w*h)
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				lum[y*w+x] = 0.299*px.r + 0.587*px.g + 0.114*px.b
			}
		}
	})

	// Forward Haar decomposition of the luminance plane.
	// Each level transforms the approximation submatrix in the top-left
	// corner; sizes records the approximation size after every level.
	type size struct{ w, h int }
	sizes := []size{{w, h}}
	aw, ah := w, h
	for level := 0; level < p.levels && (aw > 1 || ah > 1); level++ {
		if aw > 1 {
			parallelize(options.Parallelization, 0, ah, func(start, stop int) {
				tmp := make([]float32, aw)
				for y := start; y < stop; y++ {
					haarForward1D(lum[y*w:y*w+aw], tmp, aw)
				}
			})
		}
		if ah > 1 {
			parallelize(options.Parallelization, 0, aw, func(start, stop int) {
				col := make([]float32, ah)
				tmp := make([]float32, ah)
				for x := start; x < stop; x++ {
					for y := 0; y < ah; y++ {
						col[y] = lum[y*w+x]
					}
					haarForward1D(col, tmp, ah)
					for y := 0; y < ah; y++ {
						lum[y*w+x] = col[y]
					}
				}
			})
		}
		aw = (aw + 1) / 2
		ah = (ah + 1) / 2
		sizes = append(sizes, size{aw, ah})
	}

	// Soft-threshold the detail coefficients (everything outside the final
	// approximation submatrix).
	t := p.threshold
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := 0; x < w; x++ {
				if x < aw && y < ah {
					continue
				}
				i := y*w + x
				lum[i] = softThreshold(lum[i], t)
			}
		}
	})

	// Inverse Haar reconstruction, undoing each level in reverse order
	// (columns first, then rows).
	for k := len(sizes) - 1; k > 0; k-- {
		cw, ch := sizes[k-1].w, sizes[k-1].h
		if ch > 1 {
			parallelize(options.Parallelization, 0, cw, func(start, stop int) {
				col := make([]float32, ch)
				tmp := make([]float32, ch)
				for x := start; x < stop; x++ {
					for y := 0; y < ch; y++ {
						col[y] = lum[y*w+x]
					}
					haarInverse1D(col, tmp, ch)
					for y := 0; y < ch; y++ {
						lum[y*w+x] = col[y]
					}
				}
			})
		}
		if cw > 1 {
			parallelize(options.Parallelization, 0, ch, func(start, stop int) {
				tmp := make([]float32, cw)
				for y := start; y < stop; y++ {
					haarInverse1D(lum[y*w:y*w+cw], tmp, cw)
				}
			})
		}
	}

	// Reconstruct the image: scale the RGB channels of each pixel by the
	// luminance ratio so that hue and saturation are preserved. The alpha
	// channel is passed through unchanged.
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				yNew := minf32(maxf32(lum[y*w+x], 0), 1)
				yOld := 0.299*px.r + 0.587*px.g + 0.114*px.b
				var r, g, b float32
				if yOld > 1.0e-6 {
					k := yNew / yOld
					r = minf32(maxf32(px.r*k, 0), 1)
					g = minf32(maxf32(px.g*k, 0), 1)
					b = minf32(maxf32(px.b*k, 0), 1)
				} else {
					// The original pixel is black, the hue is undefined:
					// output the denoised luminance as gray.
					r, g, b = yNew, yNew, yNew
				}
				pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, pixel{r, g, b, px.a})
			}
		}
	})
}

// WaveletDenoise creates a filter that reduces noise (e.g. the fine grain of
// night shots) in the luminance channel of an image. It performs a Haar
// wavelet decomposition of the luminance, shrinks the detail coefficients
// with a soft threshold and reconstructs the image. Unlike a plain blur it
// keeps strong edges (building outlines, thin lights) intact.
//
// The levels parameter is the number of decomposition levels, it is clamped
// to the range (1, 3). The threshold parameter is the soft threshold for the
// detail coefficients in luminance units of range (0, 1); higher values remove
// more noise. If the threshold is zero, negative or NaN the filter returns
// the original image unchanged. The hue and the alpha channel of the image
// are preserved.
//
// Example:
//
//	g := gift.New(
//		gift.WaveletDenoise(2, 0.05),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func WaveletDenoise(levels int, threshold float32) Filter {
	levels = minint(maxint(levels, 1), 3)
	if threshold <= 0 || math.IsNaN(float64(threshold)) {
		return &copyimageFilter{}
	}
	return &waveletDenoiseFilter{
		levels:    levels,
		threshold: threshold,
	}
}
