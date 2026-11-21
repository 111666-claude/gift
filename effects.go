package gift

import (
	"image"
	"image/draw"
)

type pixelateFilter struct {
	size int
}

func (p *pixelateFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

// Support implements FilterSupport. It anchors the block grid to the
// coordinate origin. Region rendering uses supportRegion instead, which
// anchors to the original image origin as Draw does.
func (p *pixelateFilter) Support(dstRect image.Rectangle) (srcRect image.Rectangle) {
	return p.supportRegion(dstRect, image.Point{})
}

// supportRegion implements filterRegionSupport. Every pixel of a pixelation
// block takes the average color of the whole block, so the support of a
// rectangle is the union of all blocks it intersects. The block grid is
// anchored to the given image origin, matching the Draw behavior.
func (p *pixelateFilter) supportRegion(dstRect image.Rectangle, origin image.Point) (srcRect image.Rectangle) {
	blockSize := p.size
	if blockSize <= 1 {
		return dstRect
	}
	minX := dstRect.Min.X - origin.X
	minY := dstRect.Min.Y - origin.Y
	maxX := dstRect.Max.X - 1 - origin.X
	maxY := dstRect.Max.Y - 1 - origin.Y
	return image.Rect(
		floorDiv(minX, blockSize)*blockSize+origin.X,
		floorDiv(minY, blockSize)*blockSize+origin.Y,
		(floorDiv(maxX, blockSize)+1)*blockSize+origin.X,
		(floorDiv(maxY, blockSize)+1)*blockSize+origin.Y,
	)
}

// floorDiv returns the integer floor of a/b (b must be positive). Unlike Go's
// truncated division it rounds toward negative infinity, which is required for
// block alignment at negative coordinates.
func floorDiv(a, b int) int {
	if a >= 0 {
		return a / b
	}
	return -((-a + b - 1) / b)
}

func (p *pixelateFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	blockSize := p.size
	if blockSize <= 1 {
		copyimage(dst, src, options)
		return
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()

	// In region rendering the input may be a cropped sub-image, but the block
	// grid must stay anchored to the original image origin. For a standalone
	// Draw there is no region origin, so blocks anchor to the input origin
	// (the historical behavior).
	anchor := srcb.Min
	if options.regionOrigin != nil {
		anchor = *options.regionOrigin
	}
	relMinX := srcb.Min.X - anchor.X
	relMinY := srcb.Min.Y - anchor.Y
	firstBlockX := floorDiv(relMinX, blockSize)
	lastBlockX := floorDiv(relMinX+srcb.Dx()-1, blockSize)
	firstBlockY := floorDiv(relMinY, blockSize)
	lastBlockY := floorDiv(relMinY+srcb.Dy()-1, blockSize)
	numBlocksX := lastBlockX - firstBlockX + 1
	numBlocksY := lastBlockY - firstBlockY + 1

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	parallelize(options.Parallelization, 0, numBlocksY, func(bmin, bmax int) {
		for byi := bmin; byi < bmax; byi++ {
			by := firstBlockY + byi
			for bxi := 0; bxi < numBlocksX; bxi++ {
				bx := firstBlockX + bxi
				// block bounds in coordinates relative to the anchor
				bb := image.Rect(bx*blockSize, by*blockSize, (bx+1)*blockSize, (by+1)*blockSize)
				bbSrc := bb.Add(anchor).Intersect(srcb)
				bbDst := bbSrc.Sub(srcb.Min).Add(dstb.Min).Intersect(dstb)

				// calculate average color of the block
				var r, g, b, a float32
				var cnt float32
				for y := bbSrc.Min.Y; y < bbSrc.Max.Y; y++ {
					for x := bbSrc.Min.X; x < bbSrc.Max.X; x++ {
						px := pixGetter.getPixel(x, y)
						r += px.r
						g += px.g
						b += px.b
						a += px.a
						cnt++
					}
				}
				if cnt > 0 {
					r /= cnt
					g /= cnt
					b /= cnt
					a /= cnt
				}

				// set the calculated color for all pixels in the block
				for y := bbDst.Min.Y; y < bbDst.Max.Y; y++ {
					for x := bbDst.Min.X; x < bbDst.Max.X; x++ {
						pixSetter.setPixel(x, y, pixel{r, g, b, a})
					}
				}
			}
		}
	})
}

// Pixelate creates a filter that applies a pixelation effect to an image.
func Pixelate(size int) Filter {
	return &pixelateFilter{
		size: size,
	}
}
