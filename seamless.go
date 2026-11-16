package gift

import (
	"image"
	"image/draw"
	"math"
)

// seamlessTileFilter makes an image tile seamlessly by blending the areas
// adjacent to its opposite edges (see SeamlessTile).
type seamlessTileFilter struct {
	horizontalWidth int
	verticalWidth   int
	strength        float32
}

func (p *seamlessTileFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

// seamlessBandWidth returns the effective transition width for an image
// dimension of n pixels. The width is clamped to half of the dimension so
// that the transition bands of the opposite edges never overlap. Dimensions
// of zero or one pixel are left unchanged: they already repeat without a
// visible seam and have no room for a transition band.
func seamlessBandWidth(width, n int) int {
	if width <= 0 || n <= 1 {
		return 0
	}
	if maxWidth := n / 2; width > maxWidth {
		return maxWidth
	}
	return width
}

// seamlessAxis holds the mirror partner and the blend weight for every pixel
// along one image axis.
type seamlessAxis struct {
	partner []int
	weight  []float32
}

// newSeamlessAxis precomputes the source mirror partner and the blend weight
// for every index of an image axis that starts at min and has n pixels.
// Only the width pixels adjacent to each end of the axis have a non-zero
// weight. The weight decays quadratically from 0.5*strength at the edge to
// zero at the inner boundary of the band, so the transition fades out without
// introducing an internal edge.
func newSeamlessAxis(min, n, width int, strength float32) seamlessAxis {
	axis := seamlessAxis{
		partner: make([]int, n),
		weight:  make([]float32, n),
	}
	for i := 0; i < n; i++ {
		axis.partner[i] = min + i
	}
	if width <= 0 {
		return axis
	}
	for i := 0; i < n; i++ {
		var d int
		switch {
		case i < width:
			d = i
		case i >= n-width:
			d = n - 1 - i
		default:
			continue
		}
		// Blend the pixel with the one mirrored across the opposite edge.
		// The mirror pairing makes the first and the last pixel of the axis
		// converge to the same value at full strength, which is what removes
		// the seam when two copies of the image are placed next to each other.
		axis.partner[i] = min + n - 1 - i
		f := float32(width-d) / float32(width)
		axis.weight[i] = 0.5 * strength * f * f
	}
	return axis
}

func (p *seamlessTileFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	hw := seamlessBandWidth(p.horizontalWidth, srcb.Dx())
	vw := seamlessBandWidth(p.verticalWidth, srcb.Dy())
	if hw == 0 && vw == 0 {
		copyimage(dst, src, options)
		return
	}

	dstb := dst.Bounds()
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)
	ax := newSeamlessAxis(srcb.Min.X, srcb.Dx(), hw, p.strength)
	ay := newSeamlessAxis(srcb.Min.Y, srcb.Dy(), vw, p.strength)

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			yp := ay.partner[y-srcb.Min.Y]
			wy := ay.weight[y-srcb.Min.Y]

			// Order the two vertical samples by their coordinate and carry the
			// weights along. Using the same order no matter which side of the
			// image the pixel belongs to makes the blend of the opposite edges
			// bit-exact.
			y0, y1 := y, yp
			wy0, wy1 := 1-wy, wy
			if y1 < y0 {
				y0, y1 = y1, y0
				wy0, wy1 = wy1, wy0
			}

			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				xp := ax.partner[x-srcb.Min.X]
				wx := ax.weight[x-srcb.Min.X]

				x0, x1 := x, xp
				wx0, wx1 := 1-wx, wx
				if x1 < x0 {
					x0, x1 = x1, x0
					wx0, wx1 = wx1, wx0
				}

				var px pixel
				switch {
				case wx1 > 0 && wy1 > 0:
					var acc seamlessAccum
					acc.add(pixGetter.getPixel(x0, y0), wx0*wy0)
					acc.add(pixGetter.getPixel(x1, y0), wx1*wy0)
					acc.add(pixGetter.getPixel(x0, y1), wx0*wy1)
					acc.add(pixGetter.getPixel(x1, y1), wx1*wy1)
					px = acc.pixel()
				case wx1 > 0:
					var acc seamlessAccum
					acc.add(pixGetter.getPixel(x0, y), wx0)
					acc.add(pixGetter.getPixel(x1, y), wx1)
					px = acc.pixel()
				case wy1 > 0:
					var acc seamlessAccum
					acc.add(pixGetter.getPixel(x, y0), wy0)
					acc.add(pixGetter.getPixel(x, y1), wy1)
					px = acc.pixel()
				default:
					px = pixGetter.getPixel(x, y)
				}

				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// seamlessAccum accumulates pixels in premultiplied form. Blending the colors
// weighted by alpha and dividing by the resulting alpha afterwards prevents
// the hidden colors of transparent pixels from darkening the seam.
type seamlessAccum struct {
	r, g, b, a float32
}

func (p *seamlessAccum) add(px pixel, weight float32) {
	p.r += px.r * px.a * weight
	p.g += px.g * px.a * weight
	p.b += px.b * px.a * weight
	p.a += px.a * weight
}

func (p *seamlessAccum) pixel() pixel {
	if p.a <= 0 {
		return pixel{}
	}
	return pixel{p.r / p.a, p.g / p.a, p.b / p.a, p.a}
}

// seamlessStrength sanitizes the strength parameter. NaN and negative values
// are treated as zero (no blending), values above one (including +Inf) are
// clamped to one.
func seamlessStrength(strength float32) float32 {
	switch {
	case math.IsNaN(float64(strength)):
		return 0
	case strength < 0:
		return 0
	case strength > 1:
		return 1
	default:
		return strength
	}
}

// SeamlessTile creates a filter that makes an image tile seamlessly.
//
// The filter blends the horizontal and vertical transition bands with the
// pixels mirrored across the opposite edge, so that when two copies of the
// image are placed next to each other the colors on both sides of the joint
// match and no visible seam appears.
//
// horizontalWidth and verticalWidth specify the width in pixels of the
// transition band along the left and right edges and along the top and
// bottom edges respectively. A non-positive width disables the processing of
// that axis, so an image can be made tileable along a single direction. The
// widths are clamped to half of the corresponding image dimension; a
// dimension of zero or one pixel is left unchanged along that axis.
//
// strength specifies how strongly the edges are blended and is clamped to the
// range [0, 1]. strength = 1 removes the seam completely, while strength = 0
// returns an unchanged copy of the image. Partial values leave a
// proportionally weaker transition. NaN strength is treated as zero, -Inf is
// clamped to zero and +Inf is clamped to one.
//
// The colors are blended in premultiplied form, so the hidden colors of
// fully transparent pixels do not produce dark fringes. The filter does not
// modify the source image and can be used in a filter list with other
// filters.
//
// Example:
//
//	g := gift.New(
//		gift.SeamlessTile(64, 64, 1),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func SeamlessTile(horizontalWidth, verticalWidth int, strength float32) Filter {
	strength = seamlessStrength(strength)
	if strength == 0 || (horizontalWidth <= 0 && verticalWidth <= 0) {
		return &copyimageFilter{}
	}
	return &seamlessTileFilter{
		horizontalWidth: maxint(horizontalWidth, 0),
		verticalWidth:   maxint(verticalWidth, 0),
		strength:        strength,
	}
}
