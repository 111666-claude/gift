package gift

import (
	"image"
	"image/draw"
	"math"
)

type bloomFilter struct {
	radius    int
	intensity float32
	threshold float32
}

func (f *bloomFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (f *bloomFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	if srcb.Dx() <= 0 || srcb.Dy() <= 0 {
		return
	}

	// A non-positive or non-finite intensity disables the effect.
	if f.intensity <= 0 || !isFiniteFloat32(f.intensity) {
		copyimage(dst, src, options)
		return
	}

	// A non-positive radius means no blur: the glow cannot spread, so the
	// filter is an identity operation.
	radius := f.radius
	if radius <= 0 {
		copyimage(dst, src, options)
		return
	}

	// Clamp an excessively large radius to the image dimensions so that the
	// gaussian kernel can never grow beyond the amount of data available.
	if maxDim := maxint(srcb.Dx(), srcb.Dy()) - 1; radius > maxDim {
		radius = maxDim
	}
	if radius <= 0 {
		copyimage(dst, src, options)
		return
	}

	// A non-finite threshold is treated as the highest one: only pixels with
	// full luminance can pass it. Otherwise clamp it to the [0, 1] range.
	threshold := f.threshold
	if !isFiniteFloat32(threshold) {
		threshold = 1
	} else if threshold < 0 {
		threshold = 0
	} else if threshold > 1 {
		threshold = 1
	}

	dstb := dst.Bounds()
	pixGetterSrc := newPixelGetter(src)
	highlight := createTempImage(srcb)
	pixSetterHighlight := newPixelSetter(highlight)

	// Extract the bright pixels into a separate highlight image. The color
	// components are scaled with a linear knee factor k so that the darkest
	// possible contribution is zero and the pixels below the threshold stay
	// fully transparent (including their alpha channel).
	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetterSrc.getPixel(x, y)
				lum := 0.299*px.r + 0.587*px.g + 0.114*px.b

				var k float32
				switch {
				case lum <= threshold:
					k = 0
				case threshold >= 1:
					// Guard against a zero denominator when threshold is 1.
					if lum >= 1 {
						k = 1
					}
				default:
					k = (lum - threshold) / (1 - threshold)
					if k > 1 {
						k = 1
					}
				}

				pixSetterHighlight.setPixel(x, y, pixel{
					r: px.r * k,
					g: px.g * k,
					b: px.b * k,
					a: px.a * k,
				})
			}
		}
	})

	// Build a normalized gaussian kernel with the given blur radius. The
	// kernel reaches roughly 1% of its peak value at its outermost taps.
	sigma := float32(radius) / 3
	size := 2*radius + 1
	center := radius
	kernel := make([]float32, size)
	kernel[center] = gaussianBlurKernel(0, sigma)
	sum := kernel[center]
	for i := 1; i <= radius; i++ {
		w := gaussianBlurKernel(float32(i), sigma)
		kernel[center-i] = w
		kernel[center+i] = w
		sum += 2 * w
	}
	for i := range kernel {
		kernel[i] /= sum
	}

	// Blur the highlight image with a separable two-pass gaussian blur. The
	// convolution works on premultiplied colors internally, keeping the
	// blurred edges of semi-transparent highlights free of color fringing.
	tmp := createTempImage(srcb)
	convolve1dh(tmp, highlight, kernel, options)
	blurred := createTempImage(srcb)
	convolve1dv(blurred, tmp, kernel, options)

	// Add the blurred glow to the original image in premultiplied space.
	// Colors are only ever added to, never divided or lifted globally, so
	// pixels without glow remain exactly unchanged. The result is converted
	// back to straight alpha before being written to the destination.
	pixGetterBlur := newPixelGetter(blurred)
	pixSetterDst := newPixelSetter(dst)

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				orig := pixGetterSrc.getPixel(x, y)
				glow := pixGetterBlur.getPixel(x, y)

				// Premultiplied glow, scaled by the intensity.
				ga := glow.a * f.intensity
				gr := glow.r * ga
				gg := glow.g * ga
				gb := glow.b * ga

				// Premultiplied original color.
				oa := orig.a
				or := orig.r * oa
				og := orig.g * oa
				ob := orig.b * oa

				// Additive ("plus") compositing, clamped to the unit range.
				ra := minf32(oa+ga, 1)
				rr := minf32(or+gr, 1)
				rg := minf32(og+gg, 1)
				rb := minf32(ob+gb, 1)

				// Convert back to straight alpha.
				if ra > 0 {
					rr /= ra
					rg /= ra
					rb /= ra
				} else {
					rr, rg, rb = 0, 0, 0
				}

				pixSetterDst.setPixel(
					dstb.Min.X+x-srcb.Min.X,
					dstb.Min.Y+y-srcb.Min.Y,
					pixel{rr, rg, rb, ra},
				)
			}
		}
	})
}

// isFiniteFloat32 reports whether v is neither NaN nor an infinity.
func isFiniteFloat32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

// Bloom creates a filter that adds a soft glow (bloom) to the bright areas
// of an image. It simulates the light bleeding around luminous objects that
// is typical of neon lights in night scenes.
//
// The filter first extracts the pixels whose luminance is above the
// threshold, blurs them with a gaussian kernel of the given radius and adds
// the resulting glow back to the original image, scaled by the intensity.
//
// The radius parameter is the gaussian blur radius in pixels and must be
// positive; radii larger than the image dimensions are clamped to them.
// The intensity parameter must be positive and finite; it controls the
// strength of the added glow. A non-positive, NaN or infinite intensity
// disables the filter.
// The threshold parameter is in the [0, 1] range and sets the minimum
// luminance a pixel must have to contribute to the glow. Values outside the
// range are clamped to it; a NaN or infinite threshold is replaced by 1.0.
//
// Transparent pixels are handled using premultiplied-alpha compositing, so
// the glow can spread over transparent areas without color fringing, and
// pixels that do not exceed the threshold are left completely unchanged.
//
// Example:
//
//	g := gift.New(
//		gift.Bloom(10, 0.8, 0.7),
//	)
//	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func Bloom(radius int, intensity, threshold float32) Filter {
	return &bloomFilter{
		radius:    radius,
		intensity: intensity,
		threshold: threshold,
	}
}
