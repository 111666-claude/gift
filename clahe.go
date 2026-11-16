package gift

import (
	"image"
	"image/draw"
	"math"
)

const (
	// claheHistSize is the number of luminance histogram bins.
	claheHistSize = 256
	// claheMaxTilesPerAxis is the upper bound for the tile grid dimensions.
	// It bounds the memory used by the per-tile mapping tables
	// (claheMaxTilesPerAxis^2 * claheHistSize * 4 bytes = 4 MiB).
	claheMaxTilesPerAxis = 64
)

type claheFilter struct {
	tilesX    int
	tilesY    int
	clipLimit float32
}

func (p *claheFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *claheFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	w, h := srcb.Dx(), srcb.Dy()
	if w <= 0 || h <= 0 {
		return
	}

	// Invalid parameters have an explicit no-op behavior: the source image is
	// copied unchanged instead of producing an undefined mapping.
	clipLimit := p.clipLimit
	if p.tilesX <= 0 || p.tilesY <= 0 || !(clipLimit > 0) || math.IsInf(float64(clipLimit), 0) {
		copyimage(dst, src, options)
		return
	}

	// A tile must contain at least one pixel. The upper bound keeps the memory
	// used by the mapping tables proportional to the clamped tile grid.
	tx := minint(minint(p.tilesX, w), claheMaxTilesPerAxis)
	ty := minint(minint(p.tilesY, h), claheMaxTilesPerAxis)
	if tx < 1 {
		tx = 1
	}
	if ty < 1 {
		ty = 1
	}

	pixGetter := newPixelGetter(src)

	// Tile boundaries are integer partitions of the image, so every tile owns
	// the exact pixels used to build its histogram.
	tileCentersX := claheTileCenters(w, tx)
	tileCentersY := claheTileCenters(h, ty)

	// Build a luminance histogram per tile. Fully transparent pixels are
	// skipped and partially transparent pixels are weighted by alpha so they
	// don't dominate the statistics.
	hists := make([]float32, tx*ty*claheHistSize)
	parallelize(options.Parallelization, 0, ty, func(start, stop int) {
		for tyi := start; tyi < stop; tyi++ {
			y0 := tyi * h / ty
			y1 := (tyi + 1) * h / ty
			for txi := 0; txi < tx; txi++ {
				x0 := txi * w / tx
				x1 := (txi + 1) * w / tx
				hist := hists[(tyi*tx+txi)*claheHistSize : (tyi*tx+txi+1)*claheHistSize]
				for y := y0; y < y1; y++ {
					for x := x0; x < x1; x++ {
						px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
						if px.a <= 0 {
							continue
						}
						lum := claheLuminance(px)
						bin := int(lum*float32(claheHistSize-1) + 0.5)
						if bin < 0 {
							bin = 0
						} else if bin > claheHistSize-1 {
							bin = claheHistSize - 1
						}
						hist[bin] += px.a
					}
				}
			}
		}
	})

	// Turn every histogram into a clipped cumulative distribution function.
	// identity marks tiles where the mapping must be skipped: empty and
	// single-level (uniform) tiles are returned unchanged, so flat areas are
	// not shifted or amplified.
	identity := make([]bool, tx*ty)
	parallelize(options.Parallelization, 0, tx*ty, func(start, stop int) {
		for t := start; t < stop; t++ {
			identity[t] = prepareCLAHELut(hists[t*claheHistSize:(t+1)*claheHistSize], clipLimit)
		}
	})

	// Interpolation weights along the horizontal axis are the same for every
	// row, so they are precomputed once.
	x0s := make([]int, w)
	x1s := make([]int, w)
	fxs := make([]float32, w)
	for i := 0; i < w; i++ {
		x0s[i], x1s[i], fxs[i] = claheInterp(tileCentersX, float32(i))
	}

	pixSetter := newPixelSetter(dst)

	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for iy := start; iy < stop; iy++ {
			ty0, ty1, fy := claheInterp(tileCentersY, float32(iy))
			row0 := ty0 * tx
			row1 := ty1 * tx

			for ix := 0; ix < w; ix++ {
				px := pixGetter.getPixel(srcb.Min.X+ix, srcb.Min.Y+iy)

				// Fully transparent pixels are passed through untouched,
				// including their color values.
				if px.a <= 0 {
					pixSetter.setPixel(dstb.Min.X+ix, dstb.Min.Y+iy, px)
					continue
				}

				lum := claheLuminance(px)

				x0, x1, fx := x0s[ix], x1s[ix], fxs[ix]
				m00 := claheMapValue(hists, identity, row0+x0, lum)
				m01 := claheMapValue(hists, identity, row0+x1, lum)
				m10 := claheMapValue(hists, identity, row1+x0, lum)
				m11 := claheMapValue(hists, identity, row1+x1, lum)

				m0 := m00 + (m01-m00)*fx
				m1 := m10 + (m11-m10)*fx
				mapped := m0 + (m1-m0)*fy

				// Apply the luminance change to all color channels. The alpha
				// channel is preserved and the result is clamped to stay in
				// range, which keeps every pixel finite.
				d := mapped - lum
				px.r = clamp01f32(px.r + d)
				px.g = clamp01f32(px.g + d)
				px.b = clamp01f32(px.b + d)

				pixSetter.setPixel(dstb.Min.X+ix, dstb.Min.Y+iy, px)
			}
		}
	})
}

// claheTileCenters returns the pixel-space centers of the integer tile
// partition of an axis of the given size split into the given number of tiles.
// The result contains strictly increasing values.
func claheTileCenters(size, tiles int) []float32 {
	centers := make([]float32, tiles)
	for t := 0; t < tiles; t++ {
		lo := t * size / tiles
		hi := (t + 1) * size / tiles
		centers[t] = float32(lo+hi-1) / 2
	}
	return centers
}

// claheInterp returns the two surrounding tile centers for the given pixel
// coordinate and the interpolation fraction between them. Coordinates outside
// the first and the last center are clamped to the nearest tile.
func claheInterp(centers []float32, pos float32) (i0, i1 int, f float32) {
	n := len(centers)
	if n <= 1 || pos <= centers[0] {
		return 0, 0, 0
	}
	if pos >= centers[n-1] {
		return n - 1, n - 1, 0
	}
	for i0+1 < n && centers[i0+1] <= pos {
		i0++
	}
	i1 = i0 + 1
	f = (pos - centers[i0]) / (centers[i1] - centers[i0])
	return
}

// prepareCLAHELut turns a luminance histogram into a clipped cumulative
// distribution function used as the tile's intensity mapping. It returns true
// when the mapping is the identity function.
func prepareCLAHELut(hist []float32, clipLimit float32) bool {
	var total float32
	minBin, maxBin := -1, -1
	for i, v := range hist {
		if v > 0 {
			if minBin < 0 {
				minBin = i
			}
			maxBin = i
		}
		total += v
	}

	// Empty tiles and tiles containing a single luminance level have no local
	// contrast to redistribute, so they are left unchanged.
	if total <= 0 || minBin < 0 || minBin == maxBin {
		return true
	}

	// clipLimit is expressed as a multiple of the average bin population.
	clipThresh := clipLimit * total / float32(claheHistSize)
	var excess float32
	for i, v := range hist {
		if v > clipThresh {
			excess += v - clipThresh
			hist[i] = clipThresh
		}
	}

	// Redistribute the clipped mass uniformly and build the cumulative
	// distribution in place. The clipped mass raises the mapping slope of the
	// occupied bins only up to the threshold, so a low clipLimit keeps the
	// output close to the input while a high one approaches plain histogram
	// equalization. No additional min/max normalization is applied: that
	// would undo the contrast limiting and amplify noise at every clipLimit.
	add := excess / float32(claheHistSize)
	var cum float32
	for i := range hist {
		hist[i] += add
		cum += hist[i]
		hist[i] = clamp01f32(cum / total)
	}

	return false
}

// claheMapValue returns the tile mapping for the luminance value y.
// Values between histogram bins are linearly interpolated to avoid banding.
func claheMapValue(hists []float32, identity []bool, tile int, y float32) float32 {
	if identity[tile] {
		return y
	}
	lut := hists[tile*claheHistSize : (tile+1)*claheHistSize]
	t := y * float32(claheHistSize-1)
	v := int(t)
	if v < 0 {
		return lut[0]
	}
	if v >= claheHistSize-1 {
		return lut[claheHistSize-1]
	}
	f := t - float32(v)
	return lut[v] + (lut[v+1]-lut[v])*f
}

// claheLuminance returns the luma of a pixel in the range (0, 1).
// Non-finite channel values are folded into the range so that they can never
// produce non-finite output pixels.
func claheLuminance(px pixel) float32 {
	return clamp01f32(0.299*px.r + 0.587*px.g + 0.114*px.b)
}

func clamp01f32(v float32) float32 {
	if !(v >= 0) { // also catches NaN
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// CLAHE creates a filter that applies Contrast Limited Adaptive Histogram
// Equalization (CLAHE) to the luminance of an image.
//
// The image is divided into tilesX by tilesY tiles. A luminance histogram is
// built for each tile and clipped to a limit of
//
//	clipLimit * (tile sample count) / 256
//
// where the tile sample count is alpha-weighted (transparent pixels are
// ignored entirely). The clipped mass is redistributed uniformly over all
// bins, which limits the contrast (and noise) amplification that plain
// histogram equalization produces: small clipLimit values keep the mapping
// close to the identity, large values approach plain histogram equalization.
//
// The per-tile mappings are bilinearly interpolated between neighboring tile
// centers, so no hard transitions appear at tile boundaries. Both the
// luminance values and the resulting mapping are interpolated linearly
// between histogram bins, which avoids banding. The luminance change is
// applied as the same offset to the R, G and B channels; the alpha channel
// is kept as is.
//
// Fully transparent pixels are passed through unchanged. Tiles that are empty
// or contain a single luminance level (flat color) produce the identity
// mapping, so solid areas are not shifted or amplified. Low-contrast tiles
// are amplified by a bounded factor controlled by clipLimit.
//
// Both tile counts are clamped to the corresponding image dimension and to
// 64, so at most 4096 tiles are used. That bounds the memory allocated for
// the per-tile 256-entry mapping tables to 4 MiB, independent of the image
// size. Besides that, the filter allocates a few slices proportional to the
// image width and height.
//
// If tilesX or tilesY is not positive, or clipLimit is not a finite positive
// number, the filter performs a plain copy without changing the image, which
// makes invalid parameters explicit rather than undefined. Empty source
// images produce empty destination images.
//
// Example:
//
//	g := gift.New(
//		gift.CLAHE(8, 8, 3),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func CLAHE(tilesX, tilesY int, clipLimit float32) Filter {
	return &claheFilter{
		tilesX:    tilesX,
		tilesY:    tilesY,
		clipLimit: clipLimit,
	}
}
