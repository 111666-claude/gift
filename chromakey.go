package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

const (
	// chromaKeyMaxDistance is the largest possible distance between two
	// non-premultiplied RGB colors as measured by chromaKeyDistance. Each
	// channel difference is at most 1, so the average of the three squared
	// differences is at most 1.
	chromaKeyMaxDistance = 1

	// chromaKeyMaxParameter is the upper bound for the tolerance and feather
	// parameters. Both are measured in the same normalized distance units as
	// chromaKeyDistance; values outside [0, 1] are clamped.
	chromaKeyMaxParameter = 1
)

type chromaKeyFilter struct {
	keyColor  color.Color
	tolerance float32
	feather   float32
	despill   bool
}

func (p *chromaKeyFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

// chromaKeyDistance returns the normalized Euclidean distance between a pixel
// and the key color. All channels are in the range [0, 1], so the result is in
// [0, 1] as well.
func chromaKeyDistance(px, key pixel) float32 {
	dr := px.r - key.r
	dg := px.g - key.g
	db := px.b - key.b
	d := sqrtf32((dr*dr + dg*dg + db*db) / 3)
	return minf32(maxf32(d, 0), chromaKeyMaxDistance)
}

// chromaKeyAlpha maps a distance to a keying alpha in [0, 1].
// Distances up to the tolerance are fully transparent; the next feather units
// of distance form a linear ramp; everything farther away is fully opaque.
func chromaKeyAlpha(distance, tolerance, feather float32) float32 {
	if distance <= tolerance {
		return 0
	}
	if feather <= 0 {
		return 1
	}
	if distance >= tolerance+feather {
		return 1
	}
	return (distance - tolerance) / feather
}

// despillPixel removes a cast of the key color from a pixel by clamping the
// key color's dominant channel to the larger of the other two channels. It is
// used only on partially keyed pixels, so fully opaque foreground pixels keep
// their original colors.
func despillPixel(px pixel, key pixel) pixel {
	switch {
	case key.r > key.g && key.r > key.b:
		px.r = minf32(px.r, maxf32(px.g, px.b))
	case key.g > key.r && key.g > key.b:
		px.g = minf32(px.g, maxf32(px.r, px.b))
	case key.b > key.r && key.b > key.g:
		px.b = minf32(px.b, maxf32(px.r, px.g))
	}
	// A key color without a strictly dominant channel (gray, magenta, cyan,
	// yellow) has no single spill direction; such pixels are left unchanged.
	return px
}

func chromaKeyIsFinite(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0)
}

// chromaKeyPixel converts a key color to non-premultiplied RGB. The alpha of
// the color is ignored, including for fully transparent NRGBA/NRGBA64 values
// that color.NRGBAModel would otherwise reduce to black.
func chromaKeyPixel(c color.Color) pixel {
	switch c := c.(type) {
	case color.NRGBA:
		return pixel{
			float32(c.R) * qf8,
			float32(c.G) * qf8,
			float32(c.B) * qf8,
			1,
		}
	case color.NRGBA64:
		return pixel{
			float32(c.R) * qf16,
			float32(c.G) * qf16,
			float32(c.B) * qf16,
			1,
		}
	case color.RGBA:
		if c.A == 0 {
			return pixel{0, 0, 0, 1}
		}
		q := float32(1) / float32(c.A)
		return pixel{
			minf32(float32(c.R)*q, 1),
			minf32(float32(c.G)*q, 1),
			minf32(float32(c.B)*q, 1),
			1,
		}
	case color.RGBA64:
		if c.A == 0 {
			return pixel{0, 0, 0, 1}
		}
		q := float32(1) / float32(c.A)
		return pixel{
			minf32(float32(c.R)*q, 1),
			minf32(float32(c.G)*q, 1),
			minf32(float32(c.B)*q, 1),
			1,
		}
	}

	r16, g16, b16, a16 := c.RGBA()
	if a16 == 0 {
		return pixel{0, 0, 0, 1}
	}
	q := float32(1) / float32(a16)
	return pixel{
		minf32(float32(r16)*q, 1),
		minf32(float32(g16)*q, 1),
		minf32(float32(b16)*q, 1),
		1,
	}
}

func (p *chromaKeyFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	if srcb.Dx() <= 0 || srcb.Dy() <= 0 {
		return
	}

	// A nil key color or a non-finite parameter has an explicit no-op
	// behavior instead of producing an undefined mapping.
	if p.keyColor == nil || !chromaKeyIsFinite(p.tolerance) || !chromaKeyIsFinite(p.feather) {
		copyimage(dst, src, options)
		return
	}

	tolerance := minf32(maxf32(p.tolerance, 0), chromaKeyMaxParameter)
	feather := minf32(maxf32(p.feather, 0), chromaKeyMaxParameter)

	// The key is a color, not a transparency level, so its alpha is ignored.
	key := chromaKeyPixel(p.keyColor)

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				dstx := dstb.Min.X + x - srcb.Min.X
				dsty := dstb.Min.Y + y - srcb.Min.Y

				// Original transparent pixels are preserved as they are: their
				// color is not meaningful and must not be modified.
				if px.a <= 0 {
					pixSetter.setPixel(dstx, dsty, pixel{px.r, px.g, px.b, 0})
					continue
				}

				keyAlpha := chromaKeyAlpha(chromaKeyDistance(px, key), tolerance, feather)

				// Spill only exists where the key color and the foreground are
				// mixed, i.e. on pixels this filter keys partially. Fully opaque
				// pixels keep the original foreground color (including pixels
				// that were already semi-transparent in the source) and fully
				// keyed pixels are invisible, so neither is recolored.
				if p.despill && keyAlpha > 0 && keyAlpha < 1 {
					px = despillPixel(px, key)
				}

				px.a *= keyAlpha
				pixSetter.setPixel(dstx, dsty, px)
			}
		}
	})
}

// ChromaKey creates a filter that makes a key color transparent, as used for
// green-screen (or blue-screen) footage that is composited over another image.
//
// The keyColor parameter is the background color to remove; its alpha channel
// is ignored. The tolerance and feather parameters are normalized color
// distances in the range [0, 1] and are clamped to that range. Distance is the
// Euclidean distance between non-premultiplied RGB colors divided by sqrt(3),
// so an exact match has distance 0 and a color's complement has distance 1.
//
// Pixels whose distance from the key color is at most the tolerance become
// fully transparent. The next feather units of distance form a continuous
// linear alpha ramp, and pixels farther away keep their original alpha. A
// feather of zero produces a hard keyed/unkeyed edge. The distance does not
// depend on the alpha channel, and the original alpha of the source pixel is
// preserved wherever the pixel is not keyed.
//
// If despill is true, the key color's dominant channel (for example green) is
// clamped to the larger of the other two channels on partially keyed pixels.
// This removes the colored fringe that would otherwise appear when a
// semi-transparent edge is composited over a light or dark background. Fully
// opaque foreground pixels, fully keyed pixels and originally transparent
// pixels are not recolored.
//
// The source image is not modified and the output has the same size as the
// source, so the filter can be used anywhere in a filter chain. A nil key
// color and non-finite tolerance or feather values produce an unchanged copy
// of the source instead of an undefined mapping.
//
// Example:
//
//	g := gift.New(
//		gift.ChromaKey(color.NRGBA{0, 255, 0, 255}, 0.1, 0.2, true),
//	)
//	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func ChromaKey(keyColor color.Color, tolerance, feather float32, despill bool) Filter {
	return &chromaKeyFilter{
		keyColor:  keyColor,
		tolerance: tolerance,
		feather:   feather,
		despill:   despill,
	}
}
