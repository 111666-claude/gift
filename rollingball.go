package gift

import (
	"image"
	"image/draw"
	"math"
)

// rollingBallEps is the smallest background level that is used for the
// normalization. Darker (near black) backgrounds are left untouched to avoid
// dividing by values close to zero.
const rollingBallEps = 1e-6

// rollingBallFilter implements the rolling ball background correction.
type rollingBallFilter struct {
	radius float32
}

func (p *rollingBallFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *rollingBallFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	if srcb.Dx() <= 0 || srcb.Dy() <= 0 {
		return
	}

	// A non-positive or non-finite radius has an explicit no-op behavior.
	radius := p.radius
	if !(radius > 0) || math.IsInf(float64(radius), 0) {
		copyimage(dst, src, options)
		return
	}

	w, h := srcb.Dx(), srcb.Dy()

	// A ball that is larger than the image cannot follow any structure of
	// it: the separable passes can only see w or h samples at a time, so a
	// larger radius would not change the result in a meaningful way. Clamping
	// the radius makes the behavior of arbitrarily large values explicit and
	// identical to that of a ball spanning the image.
	if max := float32(maxint(w, h)); radius > max {
		radius = max
	}

	// The ball surface is approximated by a downward paraboloid
	//
	//	g(dx, dy) = -(dx*dx + dy*dy) / (2*radius),
	//
	// whose curvature matches a ball of the given radius near its lowest
	// point. The paraboloid makes the transform separable and independent of
	// the radius: both the working memory and the number of operations are
	// proportional to the image size only.
	c := 1 / (2 * float64(radius))
	// An extremely small positive radius would overflow the float32 buffers
	// in the envelope calculation. Capping the curvature keeps every
	// intermediate value finite; the result is the same as for any radius
	// that is far too small to bridge image structures.
	const maxCurvature = 1e12
	if c > maxCurvature {
		c = maxCurvature
	}

	pixGetter := newPixelGetter(src)

	// Extract the luminance plane. Fully transparent pixels are marked with
	// NaN so that they are ignored by the background estimation: the hidden
	// color of such a pixel cannot influence the background.
	luma := make([]float32, w*h)
	nan := float32(math.NaN())
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			off := y * w
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				if px.a <= 0 {
					luma[off+x] = nan
					continue
				}
				luma[off+x] = 0.299*px.r + 0.587*px.g + 0.114*px.b
			}
		}
	})

	// The background is the morphological closing of the luminance plane
	// with the paraboloid: a dilation (rolling the ball underneath the
	// surface) followed by an erosion (rolling it on top). Dark structures
	// narrower than the ball, such as text strokes, are bridged, while the
	// slowly varying paper brightness is followed. The transform is
	// extensive, so the background is never below the source luminance.
	tmp := make([]float32, w*h)
	// Dilation: roll the ball underneath the surface.
	rollingBallEnvelopeRows(tmp, luma, w, h, c, true, options)
	rollingBallEnvelopeCols(luma, tmp, w, h, c, true, options)
	// Erosion: roll the ball on top of the dilated surface.
	rollingBallEnvelopeRows(tmp, luma, w, h, c, false, options)
	rollingBallEnvelopeCols(luma, tmp, w, h, c, false, options)
	// Now luma holds the background.

	// The reference level is the brightest background value. Since the
	// closing is never above the source luminance, it equals the brightest
	// visible luminance of the image. Missing (fully transparent) areas are
	// skipped.
	ref := float32(0)
	for _, b := range luma {
		if b > ref && !math.IsInf(float64(b), 1) && !math.IsNaN(float64(b)) {
			ref = b
		}
	}
	if !(ref > 0) {
		// A fully transparent or fully black image has nothing to correct.
		copyimage(dst, src, options)
		return
	}

	// Remove the background by normalizing every pixel to the reference
	// level. The same factor is applied to the R, G and B channels, so their
	// ratios (and therefore the hue) are preserved. Alpha is not touched.
	pixSetter := newPixelSetter(dst)
	dstb := dst.Bounds()
	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			off := (y - srcb.Min.Y) * w
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				b := luma[off+x-srcb.Min.X]
				if px.a > 0 && b > rollingBallEps && b <= ref {
					scale := ref / b
					px.r *= scale
					px.g *= scale
					px.b *= scale
					// If a channel would leave the [0, 1] range, scale all
					// channels down by the same factor. Unlike per-channel
					// clamping this keeps the channel ratios, so the hue of a
					// very bright color is preserved too.
					if m := maxf32(px.r, maxf32(px.g, px.b)); m > 1 {
						q := 1 / m
						px.r *= q
						px.g *= q
						px.b *= q
					}
				}
				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// rollingBallLine computes the lower envelope of the parabolas
//
//	src[q] + (x-q)^2 / (2*radius)
//
// and writes it to dst. Non-finite source samples are treated as missing: a
// pixel of a fully transparent region does not contribute a parabola. When
// maximize is true the function computes the upper envelope
// src[q] - (x-q)^2/(2*radius) instead, i.e. the dilation with the downward
// paraboloid. The v, fv and z slices are scratch buffers of length len(src).
//
// This is the linear-time lower envelope algorithm for parabolas; the cost
// does not depend on the radius.
func rollingBallLine(dst, src []float32, c float64, maximize bool, v []int, fv, z []float64) {
	n := len(src)
	k := 0

	for q := 0; q < n; q++ {
		s := float64(src[q])
		if math.IsNaN(s) || math.IsInf(s, 0) {
			continue
		}
		if maximize {
			s = -s
		}

		xint := math.Inf(-1)
		for k > 0 {
			p := v[k-1]
			xint = (fv[k-1]-s)/(2*c*float64(p-q)) + float64(p+q)*0.5
			if xint <= z[k-1] {
				k--
			} else {
				break
			}
		}

		if k == 0 {
			v[0] = q
			fv[0] = s
			z[0] = math.Inf(-1)
		} else {
			v[k] = q
			fv[k] = s
			z[k] = xint
		}
		k++
	}

	if k == 0 {
		// No finite samples: return the neutral element of the operation
		// (negative infinity for the maximum, positive infinity for the
		// minimum). Both are ignored by the following pass.
		fill := float32(math.Inf(1))
		if maximize {
			fill = float32(math.Inf(-1))
		}
		for i := range dst {
			dst[i] = fill
		}
		return
	}

	e := 0
	for x := 0; x < n; x++ {
		xf := float64(x)
		for e+1 < k && z[e+1] < xf {
			e++
		}
		d := xf - float64(v[e])
		val := c*d*d + fv[e]
		if maximize {
			val = -val
		}
		dst[x] = float32(val)
	}
}

// rollingBallEnvelopeRows applies rollingBallLine along the rows of the image.
func rollingBallEnvelopeRows(dst, src []float32, w, h int, c float64, maximize bool, options *Options) {
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		v := make([]int, w)
		fv := make([]float64, w)
		z := make([]float64, w)
		for y := start; y < stop; y++ {
			off := y * w
			rollingBallLine(dst[off:off+w], src[off:off+w], c, maximize, v, fv, z)
		}
	})
}

// rollingBallEnvelopeCols applies rollingBallLine along the columns of the image.
func rollingBallEnvelopeCols(dst, src []float32, w, h int, c float64, maximize bool, options *Options) {
	parallelize(options.Parallelization, 0, w, func(start, stop int) {
		v := make([]int, h)
		fv := make([]float64, h)
		z := make([]float64, h)
		col := make([]float32, h)
		out := make([]float32, h)
		for x := start; x < stop; x++ {
			for y := 0; y < h; y++ {
				col[y] = src[y*w+x]
			}
			rollingBallLine(out, col, c, maximize, v, fv, z)
			for y := 0; y < h; y++ {
				dst[y*w+x] = out[y]
			}
		}
	})
}

// RollingBall creates a filter that corrects a slowly varying brightness
// background, such as the yellowish center and dark corners of a paper scan.
//
// The background is estimated by rolling a ball of the given radius (in
// pixels) underneath the luminance image: dark structures narrower than the
// ball, such as text strokes, are bridged, while the slow background
// variation is followed. The background is then removed by normalizing every
// pixel to the brightest background value. As a result the paper becomes
// flat, while darker foreground structures keep their contrast relative to
// it.
//
// For color images only the luminance is corrected. The same scale factor is
// applied to the R, G and B channels, so the hue is preserved; if a channel
// would exceed the valid range, all channels are scaled down together so the
// hue is still preserved. The alpha channel is never modified. Fully
// transparent pixels are passed through unchanged and do not contribute to
// the background estimation.
//
// The ball is approximated by a paraboloid of the same curvature. This makes
// the algorithm separable: the background is computed in linear time and
// with a fixed amount of memory per pixel, independent of the radius. A
// radius larger than the image is clamped to the largest image dimension;
// such a ball already cannot follow any structure of the image.
//
// A radius that is zero, negative, NaN or infinite copies the source image
// unchanged. Constant images, fully transparent images and fully black images
// are also returned unchanged. Background levels at or below 1e-6 are not
// used for the normalization, so a nearly black background is not amplified
// into noise. Empty source images produce empty destination images. The
// filter preserves the image bounds and can be combined with other filters in
// a filter chain.
//
// Example:
//
//	g := gift.New(
//		gift.RollingBall(50),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func RollingBall(radius float32) Filter {
	return &rollingBallFilter{
		radius: radius,
	}
}
