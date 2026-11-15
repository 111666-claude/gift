package gift

import (
	"image"
	"image/color"
	"image/draw"
)

type ditherFilter struct {
	palette []pixel
}

func (p *ditherFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

// nearestColor returns the palette color closest to the given color
// using squared Euclidean distance in RGB space.
// Ties are resolved in favor of the earliest palette entry.
func nearestColor(palette []pixel, r, g, b float32) pixel {
	best := palette[0]
	dr := r - best.r
	dg := g - best.g
	db := b - best.b
	bestDist := dr*dr + dg*dg + db*db
	for i := 1; i < len(palette); i++ {
		c := palette[i]
		dr = r - c.r
		dg = g - c.g
		db = b - c.b
		d := dr*dr + dg*dg + db*db
		if d < bestDist {
			bestDist = d
			best = c
		}
	}
	return best
}

func (p *ditherFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	srcb := src.Bounds()
	dstb := dst.Bounds()
	w := srcb.Dx()
	h := srcb.Dy()
	if w < 1 || h < 1 {
		return
	}

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// Error diffusion is inherently sequential: each pixel depends on the
	// quantization errors of its already processed neighbors. The filter
	// therefore runs in a single raster scan and produces exactly the same
	// result regardless of the Parallelization option.
	//
	// Floyd-Steinberg error distribution:
	//
	//            current    7/16
	//   3/16      5/16      1/16
	//
	// cur and nxt hold the accumulated errors for the current and the next
	// row. Two extra columns remove the need for horizontal bounds checks.
	cur := make([]pixel, w+2)
	nxt := make([]pixel, w+2)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)

			if px.a == 0 {
				// Fully transparent pixels are copied unchanged and do not
				// participate in error diffusion: the error accumulated for
				// them is discarded and they contribute no error themselves.
				pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, px)
				continue
			}

			e := cur[x+1]
			r := minf32(maxf32(px.r+e.r, 0), 1)
			g := minf32(maxf32(px.g+e.g, 0), 1)
			b := minf32(maxf32(px.b+e.b, 0), 1)

			c := nearestColor(p.palette, r, g, b)

			er := r - c.r
			eg := g - c.g
			eb := b - c.b

			cur[x+2].r += er * 7 / 16
			cur[x+2].g += eg * 7 / 16
			cur[x+2].b += eb * 7 / 16

			nxt[x+0].r += er * 3 / 16
			nxt[x+0].g += eg * 3 / 16
			nxt[x+0].b += eb * 3 / 16

			nxt[x+1].r += er * 5 / 16
			nxt[x+1].g += eg * 5 / 16
			nxt[x+1].b += eb * 5 / 16

			nxt[x+2].r += er * 1 / 16
			nxt[x+2].g += eg * 1 / 16
			nxt[x+2].b += eb * 1 / 16

			pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, pixel{c.r, c.g, c.b, px.a})
		}

		cur, nxt = nxt, cur
		for i := range nxt {
			nxt[i] = pixel{}
		}
	}
}

// Dither creates a filter that quantizes the image to the given palette using
// Floyd-Steinberg error diffusion dithering. It is useful for reducing an
// image to a small fixed set of colors (e.g. pixel art) without producing
// large color bands in gradients.
//
// Every opaque output pixel's RGB channels are taken exactly from the palette;
// the quantization error is propagated to the neighboring pixels so gradients
// keep their average brightness without introducing out-of-palette colors.
//
// The alpha channel of the source image is preserved. Fully transparent
// pixels are copied unchanged and neither receive nor propagate quantization
// error.
//
// Palette entries are converted to non-premultiplied RGB; their own alpha is
// ignored. Nil entries are skipped and duplicate colors are removed keeping
// the first occurrence, so the resulting palette is deterministic. If the
// palette is empty (or contains only nil entries), the filter copies the
// image unchanged.
//
// The filter is sequential by nature: the result does not depend on the
// Parallelization option.
//
// Example:
//
//	g := gift.New(
//		gift.Resize(64, 0, gift.NearestNeighborResampling),
//		gift.Dither([]color.Color{
//			color.NRGBA{0x1a, 0x1c, 0x2c, 0xff},
//			color.NRGBA{0x5d, 0x27, 0x5d, 0xff},
//			color.NRGBA{0xb1, 0x3e, 0x53, 0xff},
//			color.NRGBA{0xef, 0x7d, 0x57, 0xff},
//			color.NRGBA{0xff, 0xcd, 0x75, 0xff},
//		}),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func Dither(palette []color.Color) Filter {
	var colors []pixel
	seen := make(map[pixel]bool)
	for _, c := range palette {
		if c == nil {
			continue
		}
		n := color.NRGBAModel.Convert(c).(color.NRGBA)
		px := pixel{
			r: float32(n.R) / 0xff,
			g: float32(n.G) / 0xff,
			b: float32(n.B) / 0xff,
			a: 1,
		}
		if seen[px] {
			continue
		}
		seen[px] = true
		colors = append(colors, px)
	}
	if len(colors) == 0 {
		return &copyimageFilter{}
	}
	return &ditherFilter{palette: colors}
}
