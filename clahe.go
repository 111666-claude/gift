package gift

import (
	"image"
	"image/draw"
	"math"
)

// claheBins is the number of luminance histogram bins used by the CLAHE filter.
const claheBins = 256

type claheFilter struct {
	gridSize  int
	clipLimit float32
}

func (p *claheFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

// claheLuminance returns the BT.601 luma of a pixel in range (0, 1).
func claheLuminance(px pixel) float32 {
	return 0.299*px.r + 0.587*px.g + 0.114*px.b
}

// claheBin converts a luminance value in range (0, 1) to a histogram bin index.
func claheBin(lum float32) int {
	b := int(lum*(claheBins-1) + 0.5)
	if b < 0 {
		return 0
	}
	if b > claheBins-1 {
		return claheBins - 1
	}
	return b
}

// claheTileBounds splits a range of length n into count nearly-equal
// contiguous parts and returns the part boundaries (count+1 values).
func claheTileBounds(n, count int) []int {
	bounds := make([]int, count+1)
	div := n / count
	mod := n % count
	for i := 0; i <= count; i++ {
		bounds[i] = i*div + minint(i, mod)
	}
	return bounds
}

// claheTileLUT builds the mapping function for a single tile from its
// luminance histogram. The histogram is clipped at the limit derived from
// clipLimit and the excess counts are redistributed uniformly over all bins.
// The returned LUT maps luminance values in range (0, 1) to range (0, 1).
func claheTileLUT(hist *[claheBins]int, total int, clipLimit float32) *[claheBins]float32 {
	lut := &[claheBins]float32{}
	if total <= 0 {
		return lut
	}

	// A constant tile keeps the original values.
	nonzero := 0
	for i := 0; i < claheBins; i++ {
		if hist[i] > 0 {
			nonzero++
		}
	}
	if nonzero <= 1 {
		for i := 0; i < claheBins; i++ {
			lut[i] = float32(i) / (claheBins - 1)
		}
		return lut
	}

	// Clip the histogram and redistribute the excess counts uniformly.
	limit := int(clipLimit * float32(total) / claheBins)
	if limit < 1 {
		limit = 1
	}
	excess := 0
	for i := 0; i < claheBins; i++ {
		if hist[i] > limit {
			excess += hist[i] - limit
			hist[i] = limit
		}
	}
	inc := excess / claheBins
	rem := excess % claheBins
	for i := 0; i < claheBins; i++ {
		hist[i] += inc
	}
	for i := 0; i < rem; i++ {
		hist[i]++
	}

	// Build the cumulative distribution function.
	var cdf [claheBins]int
	sum := 0
	cdfMin := 0
	cdfMinFound := false
	for i := 0; i < claheBins; i++ {
		sum += hist[i]
		cdf[i] = sum
		if !cdfMinFound && sum > 0 {
			cdfMin = sum
			cdfMinFound = true
		}
	}

	if sum-cdfMin <= 0 {
		// The tile is constant: keep the original values.
		for i := 0; i < claheBins; i++ {
			lut[i] = float32(i) / (claheBins - 1)
		}
		return lut
	}

	for i := 0; i < claheBins; i++ {
		v := float32(cdf[i]-cdfMin) / float32(sum-cdfMin)
		lut[i] = minf32(maxf32(v, 0), 1)
	}
	return lut
}

// claheTileCoord locates the fractional tile coordinate of a pixel position
// for bilinear interpolation between the tile mapping functions. It returns
// the index of the tile to the left/above (guaranteed to have a valid
// neighbor at index+1 when the fraction is non-zero) and the fraction.
func claheTileCoord(pos, size, grid int) (int, float32) {
	if grid <= 1 {
		return 0, 0
	}
	// Pixel center position in tile units, shifted so that the tile
	// centers are at integer coordinates.
	t := (float32(pos)+0.5)*float32(grid)/float32(size) - 0.5
	i := int(math.Floor(float64(t)))
	f := t - float32(i)
	if i < 0 {
		return 0, 0
	}
	if i >= grid-1 {
		return grid - 2, 1
	}
	return i, f
}

func (p *claheFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	w, h := srcb.Dx(), srcb.Dy()
	if w == 0 || h == 0 {
		return
	}

	gridX := minint(maxint(p.gridSize, 1), w)
	gridY := minint(maxint(p.gridSize, 1), h)

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// Calculate the luminance of every pixel.
	lum := make([]float32, w*h)
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				lum[y*w+x] = claheLuminance(px)
			}
		}
	})

	// Build the clipped histogram mapping function for every tile.
	xb := claheTileBounds(w, gridX)
	yb := claheTileBounds(h, gridY)
	luts := make([]*[claheBins]float32, gridX*gridY)
	parallelize(options.Parallelization, 0, gridY, func(start, stop int) {
		for ty := start; ty < stop; ty++ {
			for tx := 0; tx < gridX; tx++ {
				var hist [claheBins]int
				for y := yb[ty]; y < yb[ty+1]; y++ {
					for x := xb[tx]; x < xb[tx+1]; x++ {
						hist[claheBin(lum[y*w+x])]++
					}
				}
				total := (yb[ty+1] - yb[ty]) * (xb[tx+1] - xb[tx])
				luts[ty*gridX+tx] = claheTileLUT(&hist, total, p.clipLimit)
			}
		}
	})

	// Map every pixel interpolating between the mapping functions of the
	// neighboring tiles and scale its RGB channels by the luminance ratio.
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			iy, fy := claheTileCoord(y, h, gridY)
			iy1 := minint(iy+1, gridY-1)
			for x := 0; x < w; x++ {
				ix, fx := claheTileCoord(x, w, gridX)
				ix1 := minint(ix+1, gridX-1)
				b := claheBin(lum[y*w+x])

				l00 := luts[iy*gridX+ix][b]
				l10 := luts[iy*gridX+ix1][b]
				l01 := luts[iy1*gridX+ix][b]
				l11 := luts[iy1*gridX+ix1][b]
				top := l00 + (l10-l00)*fx
				bot := l01 + (l11-l01)*fx
				m := top + (bot-top)*fy

				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				l := lum[y*w+x]
				if l > 1.0e-6 {
					s := m / l
					px.r *= s
					px.g *= s
					px.b *= s
				} else {
					px.r += m
					px.g += m
					px.b += m
				}
				pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, px)
			}
		}
	})
}

// CLAHE creates a filter that applies Contrast Limited Adaptive Histogram
// Equalization to the image. It improves the local contrast of images with
// uneven illumination (e.g. a scan that is dark on one side and bright on
// the other) where a global contrast adjustment would lose detail.
//
// The image is divided into a grid of gridSize x gridSize tiles. A histogram
// of the luminance is built for every tile, clipped at clipLimit times the
// average bin count with the excess redistributed uniformly over all bins,
// and turned into a mapping function. Every pixel is mapped interpolating
// between the functions of the neighboring tiles. The alpha channel is
// preserved and the RGB channels are scaled by the luminance ratio to avoid
// color shifts.
//
// The gridSize parameter must be positive, typical values are 4 to 16.
// The clipLimit parameter must be 1 or greater: 1 gives the strongest
// clipping, typical values are 2 to 40, a very large value disables clipping.
// Invalid parameters are clamped to the valid range.
//
// Example:
//
//	g := gift.New(
//		gift.CLAHE(8, 4),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func CLAHE(gridSize int, clipLimit float32) Filter {
	if gridSize < 1 {
		gridSize = 1
	}
	if !(clipLimit >= 1) {
		clipLimit = 1
	}
	return &claheFilter{
		gridSize:  gridSize,
		clipLimit: clipLimit,
	}
}
