/*
Package gift provides a set of useful image processing filters.

Basic usage:

	// 1. Create a new filter list and add some filters.
	g := gift.New(
	    gift.Resize(800, 0, gift.LanczosResampling),
	    gift.UnsharpMask(1, 1, 0),
	)

	// 2. Create a new image of the corresponding size.
	// dst is a new target image, src is the original image.
	dst := image.NewRGBA(g.Bounds(src.Bounds()))

	// 3. Use the Draw func to apply the filters to src and store the result in dst.
	g.Draw(dst, src)

*/
package gift

import (
	"image"
	"image/draw"
)

// Filter is an image processing filter.
type Filter interface {
	// Draw applies the filter to the src image and outputs the result to the dst image.
	Draw(dst draw.Image, src image.Image, options *Options)
	// Bounds calculates the appropriate bounds of an image after applying the filter.
	Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle)
}

// Options is the parameters passed to image processing filters.
type Options struct {
	Parallelization bool
}

var defaultOptions = Options{
	Parallelization: true,
}

// GIFT is a list of image processing filters.
type GIFT struct {
	Filters []Filter
	Options Options
}

// New creates a new filter list and initializes it with the given slice of filters.
func New(filters ...Filter) *GIFT {
	return &GIFT{
		Filters: filters,
		Options: defaultOptions,
	}
}

// SetParallelization enables or disables the image processing parallelization.
// Parallelization is enabled by default.
func (g *GIFT) SetParallelization(isEnabled bool) {
	g.Options.Parallelization = isEnabled
}

// Parallelization returns the current state of parallelization option.
func (g *GIFT) Parallelization() bool {
	return g.Options.Parallelization
}

// Add appends the given filters to the list of filters.
func (g *GIFT) Add(filters ...Filter) {
	g.Filters = append(g.Filters, filters...)
}

// Empty removes all the filters from the list.
func (g *GIFT) Empty() {
	g.Filters = []Filter{}
}

// Bounds calculates the appropriate bounds for the result image after applying all the added filters.
// Parameter srcBounds is the bounds of the source image.
//
// Example:
//
// 	src := image.NewRGBA(image.Rect(0, 0, 100, 200))
//	g := gift.New(gift.Rotate90())
//
// 	// calculate image bounds after applying rotation and create a new image of that size.
// 	dst := image.NewRGBA(g.Bounds(src.Bounds())) // dst bounds: (0, 0, 200, 100)
//
//
func (g *GIFT) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	b := srcBounds
	for _, f := range g.Filters {
		b = f.Bounds(b)
	}
	dstBounds = b
	return
}

// Draw applies all the added filters to the src image and outputs the result to the dst image.
func (g *GIFT) Draw(dst draw.Image, src image.Image) {
	if len(g.Filters) == 0 {
		copyimage(dst, src, &g.Options)
		return
	}

	first, last := 0, len(g.Filters)-1
	var tmpIn image.Image
	var tmpOut draw.Image

	for i, f := range g.Filters {
		if i == first {
			tmpIn = src
		} else {
			tmpIn = tmpOut
		}

		if i == last {
			tmpOut = dst
		} else {
			tmpOut = createTempImage(f.Bounds(tmpIn.Bounds()))
		}

		f.Draw(tmpOut, tmpIn, &g.Options)
	}
}

// Operator is an image composition operator.
type Operator int

// Composition operators.
const (
	CopyOperator Operator = iota
	OverOperator
)

// DrawAt applies all the added filters to the src image and outputs the result to the dst image
// at the specified position pt using the specified composition operator op.
func (g *GIFT) DrawAt(dst draw.Image, src image.Image, pt image.Point, op Operator) {
	switch op {
	case OverOperator:
		tb := g.Bounds(src.Bounds())
		tb = tb.Sub(tb.Min).Add(pt)
		tmp := createTempImage(tb)
		g.Draw(tmp, src)
		pixGetterDst := newPixelGetter(dst)
		pixGetterTmp := newPixelGetter(tmp)
		pixSetterDst := newPixelSetter(dst)
		ib := tb.Intersect(dst.Bounds())
		parallelize(g.Options.Parallelization, ib.Min.Y, ib.Max.Y, func(start, stop int) {
			for y := start; y < stop; y++ {
				for x := ib.Min.X; x < ib.Max.X; x++ {
					px0 := pixGetterDst.getPixel(x, y)
					px1 := pixGetterTmp.getPixel(x, y)
					c1 := px1.a
					c0 := (1 - c1) * px0.a
					cs := c0 + c1
					c0 /= cs
					c1 /= cs
					r := px0.r*c0 + px1.r*c1
					g := px0.g*c0 + px1.g*c1
					b := px0.b*c0 + px1.b*c1
					a := px0.a + px1.a*(1-px0.a)
					pixSetterDst.setPixel(x, y, pixel{r, g, b, a})
				}
			}
		})

	default:
		if pt.Eq(dst.Bounds().Min) {
			g.Draw(dst, src)
			return
		}
		if subimg, ok := getSubImage(dst, pt); ok {
			g.Draw(subimg, src)
			return
		}
		tb := g.Bounds(src.Bounds())
		tb = tb.Sub(tb.Min).Add(pt)
		tmp := createTempImage(tb)
		g.Draw(tmp, src)
		pixGetter := newPixelGetter(tmp)
		pixSetter := newPixelSetter(dst)
		ib := tb.Intersect(dst.Bounds())
		parallelize(g.Options.Parallelization, ib.Min.Y, ib.Max.Y, func(start, stop int) {
			for y := start; y < stop; y++ {
				for x := ib.Min.X; x < ib.Max.X; x++ {
					pixSetter.setPixel(x, y, pixGetter.getPixel(x, y))
				}
			}
		})
	}
}

// DrawAtMasked applies all the added filters to the src image and blends the result
// into the dst image at the position pt, modulating the strength of the blend
// per-pixel with the mask image. The op operator selects the composition semantics
// (CopyOperator or OverOperator).
//
// The mask is aligned with the output of the filter chain, not with the source
// image: its upper-left corner is placed at pt and it shares the destination
// coordinate space of the result. This keeps the mask aligned with pt even when
// the filters change the output bounds, for example when resizing or rotating.
// A mask pixel outside the mask bounds is treated as fully transparent. Pixels
// outside the dst bounds are clipped.
//
// The mask alpha controls the coverage of the filtered result (px1) over the
// existing destination pixel (px0):
//   - mask alpha 0.0 leaves the destination pixel untouched;
//   - mask alpha 1.0 fully replaces (CopyOperator) or fully composites
//     (OverOperator) the destination pixel with the filtered result;
//   - any intermediate value m blends the two proportionally.
//
// For CopyOperator the blend is the straight interpolation
// px = px0*(1-m) + px1*m. For OverOperator the source coverage is first scaled
// by the mask and then composited with the standard Porter-Duff source-over
// rule. With a fully opaque mask the result is identical to DrawAt.
//
// If mask is nil, the call is equivalent to DrawAt with the same operator.
// The src and mask images are only read, never modified.
func (g *GIFT) DrawAtMasked(dst draw.Image, src image.Image, mask image.Image, pt image.Point, op Operator) {
	if mask == nil {
		g.DrawAt(dst, src, pt, op)
		return
	}

	// tb is the output rectangle of the filter chain in the destination
	// coordinate space; mb is the mask rectangle placed with its Min at pt.
	tb := g.Bounds(src.Bounds())
	tb = tb.Sub(tb.Min).Add(pt)
	mb := mask.Bounds().Sub(mask.Bounds().Min).Add(pt)

	tmp := createTempImage(tb)
	g.Draw(tmp, src)

	pixGetterDst := newPixelGetter(dst)
	pixGetterTmp := newPixelGetter(tmp)
	pixSetterDst := newPixelSetter(dst)

	// Only pixels covered by all three images can be blended. Areas outside
	// the mask are treated as transparent and areas outside dst are clipped.
	ib := tb.Intersect(dst.Bounds()).Intersect(mb)
	if ib.Empty() {
		return
	}

	parallelize(g.Options.Parallelization, ib.Min.Y, ib.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := ib.Min.X; x < ib.Max.X; x++ {
				// Mask coordinates share the destination space: the mask Min
				// maps onto pt regardless of the mask's own origin.
				m := maskAlpha(mask, x-pt.X+mask.Bounds().Min.X, y-pt.Y+mask.Bounds().Min.Y)
				if m <= 0 {
					continue
				}

				px0 := pixGetterDst.getPixel(x, y)
				px1 := pixGetterTmp.getPixel(x, y)

				if op == OverOperator && m >= 1 {
					// Keep the exact OverOperator arithmetic of DrawAt for a
					// fully opaque mask so the two entry points agree bit for bit.
					c1 := px1.a
					c0 := (1 - c1) * px0.a
					cs := c0 + c1
					c0 /= cs
					c1 /= cs
					r := px0.r*c0 + px1.r*c1
					gg := px0.g*c0 + px1.g*c1
					b := px0.b*c0 + px1.b*c1
					a := px0.a + px1.a*(1-px0.a)
					pixSetterDst.setPixel(x, y, pixel{r, gg, b, a})
					continue
				}

				om := 1 - m
				var r, gg, b, a float32
				if op == OverOperator {
					// Masked source-over: scale the source coverage by the mask
					// and apply the standard Porter-Duff source-over rule.
					// m == 0 reproduces px0, m == 1 matches the DrawAt math.
					sa := px1.a * m
					a = px0.a + sa*(1-px0.a)
					if a > 0 {
						w1 := sa / a
						w0 := px0.a * (1 - sa) / a
						r = px1.r*w1 + px0.r*w0
						gg = px1.g*w1 + px0.g*w0
						b = px1.b*w1 + px0.b*w0
					}
				} else {
					// Straight interpolation of the straight-alpha pixels.
					r = px0.r*om + px1.r*m
					gg = px0.g*om + px1.g*m
					b = px0.b*om + px1.b*m
					a = px0.a*om + px1.a*m
				}
				pixSetterDst.setPixel(x, y, pixel{r, gg, b, a})
			}
		}
	})
}

// maskAlpha returns the normalized alpha of a mask pixel in the [0.0, 1.0]
// range. For alpha-enabled image types the alpha channel is used; for opaque
// grayscale images the gray value itself is the coverage.
func maskAlpha(mask image.Image, x, y int) float32 {
	switch mask := mask.(type) {
	case *image.Alpha:
		i := mask.PixOffset(x, y)
		return float32(mask.Pix[i]) * qf8
	case *image.Alpha16:
		i := mask.PixOffset(x, y)
		return float32(uint16(mask.Pix[i+0])<<8|uint16(mask.Pix[i+1])) * qf16
	case *image.Gray:
		i := mask.PixOffset(x, y)
		return float32(mask.Pix[i]) * qf8
	case *image.Gray16:
		i := mask.PixOffset(x, y)
		return float32(uint16(mask.Pix[i+0])<<8|uint16(mask.Pix[i+1])) * qf16
	case *image.NRGBA:
		i := mask.PixOffset(x, y)
		return float32(mask.Pix[i+3]) * qf8
	case *image.NRGBA64:
		i := mask.PixOffset(x, y)
		return float32(uint16(mask.Pix[i+6])<<8|uint16(mask.Pix[i+7])) * qf16
	case *image.RGBA:
		i := mask.PixOffset(x, y)
		return float32(mask.Pix[i+3]) * qf8
	case *image.RGBA64:
		i := mask.PixOffset(x, y)
		return float32(uint16(mask.Pix[i+6])<<8|uint16(mask.Pix[i+7])) * qf16
	default:
		_, _, _, a := mask.At(x, y).RGBA()
		return float32(a) * qf16
	}
}

func getSubImage(img draw.Image, pt image.Point) (draw.Image, bool) {
	if !pt.In(img.Bounds()) {
		return nil, false
	}
	switch img := img.(type) {
	case *image.Gray:
		return img.SubImage(image.Rectangle{pt, img.Bounds().Max}).(draw.Image), true
	case *image.Gray16:
		return img.SubImage(image.Rectangle{pt, img.Bounds().Max}).(draw.Image), true
	case *image.RGBA:
		return img.SubImage(image.Rectangle{pt, img.Bounds().Max}).(draw.Image), true
	case *image.RGBA64:
		return img.SubImage(image.Rectangle{pt, img.Bounds().Max}).(draw.Image), true
	case *image.NRGBA:
		return img.SubImage(image.Rectangle{pt, img.Bounds().Max}).(draw.Image), true
	case *image.NRGBA64:
		return img.SubImage(image.Rectangle{pt, img.Bounds().Max}).(draw.Image), true
	default:
		return nil, false
	}
}
