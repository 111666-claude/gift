package gift

import (
	"image"
	"image/draw"
)

// maxScale2xDimension is the largest source width or height whose doubled
// value still fits into an int.
const maxScale2xDimension = int(^uint(0)>>1) / 2

type scale2xFilter struct{}

// scale2xDoubleDimension returns the doubled size of a single axis. It reports
// false for empty axes and for axes whose doubled size does not fit into an
// int. The length is computed from the corner coordinates instead of using
// Rectangle.Dx or Rectangle.Dy, so a rectangle whose side length overflows an
// int is rejected instead of being mistaken for a small image.
func scale2xDoubleDimension(min, max int) (int, bool) {
	if max <= min {
		return 0, false
	}
	// Both corners are int values, so the true non-negative difference fits
	// into a uint and the subtraction is exact.
	size := uint64(max) - uint64(min)
	if size > uint64(maxScale2xDimension) {
		return 0, false
	}
	return int(2 * size), true
}

// scale2xOutputSize returns the size of the scaled image and reports whether
// the source can be scaled.
func scale2xOutputSize(srcBounds image.Rectangle) (w, h int, ok bool) {
	w, ok = scale2xDoubleDimension(srcBounds.Min.X, srcBounds.Max.X)
	if !ok {
		return 0, 0, false
	}
	h, ok = scale2xDoubleDimension(srcBounds.Min.Y, srcBounds.Max.Y)
	if !ok {
		return 0, 0, false
	}
	return w, h, true
}

func (p *scale2xFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	w, h, ok := scale2xOutputSize(srcBounds)
	if !ok {
		return image.Rectangle{}
	}
	return image.Rect(0, 0, w, h)
}

// scale2xPrevY returns the row above y clamped to the image bounds.
func scale2xPrevY(y, min int) int {
	if y <= min {
		return min
	}
	return y - 1
}

// scale2xNextY returns the row below y clamped to the image bounds.
func scale2xNextY(y, max int) int {
	if y >= max {
		return max
	}
	return y + 1
}

// scale2xColor normalizes a source pixel. The color of a fully transparent
// pixel is undefined (and is stored differently by different image types), so
// all fully transparent pixels are represented by the same zero value.
func scale2xColor(px pixel) pixel {
	if px.a == 0 {
		return pixel{}
	}
	return px
}

// scale2xPixels computes the 2x2 block of output pixels for the source pixel p
// and its orthogonal neighbors a (above), b (right), c (left) and d (below):
//
//	   a
//	c  p  b
//	   d
//
// The output block is:
//
//	e0 e1
//	e2 e3
func scale2xPixels(p, a, b, c, d pixel) (e0, e1, e2, e3 pixel) {
	e0, e1, e2, e3 = p, p, p, p
	if a != d && c != b {
		if c == a {
			e0 = c
		}
		if a == b {
			e1 = b
		}
		if c == d {
			e2 = c
		}
		if d == b {
			e3 = b
		}
	}
	return
}

// scale2xLoadRow reads and normalizes a row of the source image into buf.
func scale2xLoadRow(pixGetter *pixelGetter, y int, buf *[]pixel) {
	pixGetter.getPixelRow(y, buf)
	for i := range *buf {
		(*buf)[i] = scale2xColor((*buf)[i])
	}
}

func (p *scale2xFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}
	if src == nil || dst == nil {
		return
	}

	srcb := src.Bounds()
	w, h, ok := scale2xOutputSize(srcb)
	if !ok {
		return
	}
	if w <= 0 || h <= 0 {
		return
	}

	dstb := dst.Bounds()
	if dstb.Dx() <= 0 || dstb.Dy() <= 0 {
		return
	}

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)
	maxY := srcb.Max.Y - 1
	rowLen := int(uint64(srcb.Max.X) - uint64(srcb.Min.X))

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		above := make([]pixel, 0, rowLen)
		center := make([]pixel, 0, rowLen)
		below := make([]pixel, 0, rowLen)

		// Neighbors outside the image are clamped to the nearest edge pixel,
		// so border pixels follow the same rules as inner ones. Rows are
		// loaded once and rolled forward instead of being re-read.
		scale2xLoadRow(pixGetter, scale2xPrevY(start, srcb.Min.Y), &above)
		scale2xLoadRow(pixGetter, start, &center)

		for y := start; y < stop; y++ {
			scale2xLoadRow(pixGetter, scale2xNextY(y, maxY), &below)

			dsty := dstb.Min.Y + 2*(y-srcb.Min.Y)
			for i := range center {
				left := center[i]
				if i > 0 {
					left = center[i-1]
				}
				right := center[i]
				if i < len(center)-1 {
					right = center[i+1]
				}

				e0, e1, e2, e3 := scale2xPixels(center[i], above[i], right, left, below[i])

				dstx := dstb.Min.X + 2*i
				pixSetter.setPixel(dstx, dsty, e0)
				pixSetter.setPixel(dstx+1, dsty, e1)
				pixSetter.setPixel(dstx, dsty+1, e2)
				pixSetter.setPixel(dstx+1, dsty+1, e3)
			}

			above = above[:len(center)]
			copy(above, center)
			center = center[:len(below)]
			copy(center, below)
		}
	})
}

// Scale2x creates a filter that upscales an image to twice its width and
// height using the Scale2x (EPX) pixel-art algorithm. Each source pixel is
// replaced by a 2x2 block of pixels, so diagonal lines and outlines keep their
// hard edges instead of being smoothed by interpolation.
//
// For a source pixel P with the neighbors A (above), B (right), C (left) and
// D (below):
//
//	   A
//	C  P  B
//	   D
//
// the 2x2 output block is:
//
//	E0 E1
//	E2 E3
//
// where:
//
//	E0 = A if C == A and C != D and A != B, else P
//	E1 = B if A == B and A != C and B != D, else P
//	E2 = C if D == C and D != B and C != A, else P
//	E3 = D if B == D and B != A and D != C, else P
//
// Pixels outside the image are clamped to the nearest edge pixel, so borders,
// one pixel wide or high images and transparent areas are processed with the
// same rules as the interior. Two fully transparent pixels are always
// considered equal, regardless of their hidden color and of the source image
// type; when a transparent pixel is selected for an output subpixel, it is
// written as fully transparent black.
//
// The source is never modified and the filter can be used in a filter chain.
// Every image.Image implementation is accepted: premultiplied formats are
// converted to non-premultiplied colors and formats without a fast path are
// read through At, so each source pixel format has a defined result. If the
// source is empty, or doubling one of its dimensions would overflow an int,
// Bounds returns an empty rectangle and Draw leaves dst unchanged.
//
// Example:
//
//	g := gift.New(gift.Scale2x())
//	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func Scale2x() Filter {
	return &scale2xFilter{}
}
