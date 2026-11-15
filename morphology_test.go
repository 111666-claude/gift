package gift

import (
	"image"
	"image/draw"
	"testing"
)

// alphaGrid builds an NRGBA image from a visual alpha map:
// '.' = 0x00, '4' = 0x40, '8' = 0x80, 'F' = 0xFF.
// All pixels get the same RGB to verify that RGB is preserved.
func alphaGrid(r image.Rectangle, rgb [3]uint8, rows []string) *image.NRGBA {
	img := image.NewNRGBA(r)
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			var a uint8
			switch rows[y][x] {
			case '4':
				a = 0x40
			case '8':
				a = 0x80
			case 'F':
				a = 0xFF
			}
			i := img.PixOffset(r.Min.X+x, r.Min.Y+y)
			img.Pix[i+0] = rgb[0]
			img.Pix[i+1] = rgb[1]
			img.Pix[i+2] = rgb[2]
			img.Pix[i+3] = a
		}
	}
	return img
}

func TestDilateErode(t *testing.T) {
	rgb := [3]uint8{0x10, 0x20, 0x30}
	srcb := image.Rect(-1, -1, 4, 4) // non-zero Min, 5x5
	dstb := image.Rect(0, 0, 5, 5)

	testData := []struct {
		desc   string
		filter Filter
		srcPix []string
		dstPix []string
	}{
		{
			"dilate radius 0 is identity",
			Dilate(0),
			[]string{
				".....",
				"..8..",
				".8F8.",
				"..8..",
				".....",
			},
			[]string{
				".....",
				"..8..",
				".8F8.",
				"..8..",
				".....",
			},
		},
		{
			"erode negative radius is identity",
			Erode(-3),
			[]string{
				".....",
				"..8..",
				".8F8.",
				"..8..",
				".....",
			},
			[]string{
				".....",
				"..8..",
				".8F8.",
				"..8..",
				".....",
			},
		},
		{
			"dilate point radius 1 grows by one pixel",
			Dilate(1),
			[]string{
				".....",
				".....",
				"..F..",
				".....",
				".....",
			},
			[]string{
				".....",
				".FFF.",
				".FFF.",
				".FFF.",
				".....",
			},
		},
		{
			"dilate point radius 2 fills 5x5",
			Dilate(2),
			[]string{
				".....",
				".....",
				"..F..",
				".....",
				".....",
			},
			[]string{
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
			},
		},
		{
			"dilate huge radius is clamped and deterministic",
			Dilate(1 << 30),
			[]string{
				".....",
				".....",
				"..F..",
				".....",
				".....",
			},
			[]string{
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
			},
		},
		{
			"dilate fills one pixel hole",
			Dilate(1),
			[]string{
				"FFFFF",
				"FFFFF",
				"FF.FF",
				"FFFFF",
				"FFFFF",
			},
			[]string{
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
			},
		},
		{
			"dilate propagates semi-transparent edge",
			Dilate(1),
			[]string{
				".....",
				".....",
				"..8..",
				".....",
				".....",
			},
			[]string{
				".....",
				".888.",
				".888.",
				".888.",
				".....",
			},
		},
		{
			"dilate clipped by image edge",
			Dilate(1),
			[]string{
				"F....",
				".....",
				".....",
				".....",
				".....",
			},
			[]string{
				"FF...",
				"FF...",
				".....",
				".....",
				".....",
			},
		},
		{
			"erode opaque square recedes from edges",
			Erode(1),
			[]string{
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
			},
			[]string{
				".....",
				".FFF.",
				".FFF.",
				".FFF.",
				".....",
			},
		},
		{
			"erode 3x3 block leaves center",
			Erode(1),
			[]string{
				".....",
				".FFF.",
				".FFF.",
				".FFF.",
				".....",
			},
			[]string{
				".....",
				".....",
				"..F..",
				".....",
				".....",
			},
		},
		{
			"erode huge radius clears everything",
			Erode(1 << 30),
			[]string{
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
				"FFFFF",
			},
			[]string{
				".....",
				".....",
				".....",
				".....",
				".....",
			},
		},
		{
			"erode widens one pixel hole",
			Erode(1),
			[]string{
				"FFFFF",
				"FFFFF",
				"FF.FF",
				"FFFFF",
				"FFFFF",
			},
			[]string{
				".....",
				".....",
				".....",
				".....",
				".....",
			},
		},
		{
			"erode propagates semi-transparent minimum",
			Erode(1),
			[]string{
				"88888",
				"88888",
				"88488",
				"88888",
				"88888",
			},
			[]string{
				".....",
				".444.",
				".444.",
				".444.",
				".....",
			},
		},
	}

	for _, d := range testData {
		src := alphaGrid(srcb, rgb, d.srcPix)
		dst := image.NewNRGBA(d.filter.Bounds(src.Bounds()))
		d.filter.Draw(dst, src, nil)

		expected := alphaGrid(dstb, rgb, d.dstPix)
		if !checkBoundsAndPix(dst.Bounds(), expected.Bounds(), dst.Pix, expected.Pix) {
			t.Errorf("test [%s] failed: %#v, %#v", d.desc, dst.Bounds(), dst.Pix)
		}
	}

	// check no panics
	empty := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	Dilate(5).Draw(empty, empty, nil)
	Erode(5).Draw(empty, empty, nil)
	one := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	Dilate(5).Draw(one, one, nil)
	Erode(5).Draw(one, one, nil)
}

func TestDilateErodeFormats(t *testing.T) {
	// The same opaque shape filtered as NRGBA and as premultiplied RGBA
	// must produce the same pixels.
	rgb := [3]uint8{0x10, 0x20, 0x30}
	srcb := image.Rect(0, 0, 5, 5)
	shape := []string{
		".....",
		".....",
		"..F..",
		".....",
		".....",
	}

	for _, f := range []Filter{Dilate(1), Erode(1)} {
		srcN := alphaGrid(srcb, rgb, shape)
		dstN := image.NewNRGBA(f.Bounds(srcb))
		f.Draw(dstN, srcN, nil)

		srcR := image.NewRGBA(srcb)
		draw.Draw(srcR, srcb, srcN, srcb.Min, draw.Src)
		dstR := image.NewRGBA(f.Bounds(srcb))
		f.Draw(dstR, srcR, nil)

		// Alpha must match the NRGBA result exactly. RGB must be preserved
		// where it is representable: premultiplied storage keeps no color
		// for fully transparent pixels and rescales on alpha change.
		for i := 0; i < len(dstR.Pix); i += 4 {
			if dstR.Pix[i+3] != dstN.Pix[i+3] {
				t.Fatalf("rgba/nrgba alpha mismatch at byte %d: %#v",
					i, dstR.Pix[i:i+4])
			}
			if srcR.Pix[i+3] == 0xff && dstR.Pix[i+3] == 0xff {
				if dstR.Pix[i+0] != srcR.Pix[i+0] ||
					dstR.Pix[i+1] != srcR.Pix[i+1] ||
					dstR.Pix[i+2] != srcR.Pix[i+2] {
					t.Fatalf("rgba RGB changed at byte %d: %#v -> %#v",
						i, srcR.Pix[i:i+3], dstR.Pix[i:i+3])
				}
			}
		}
	}
}

func TestDilateErodeParallel(t *testing.T) {
	// Serial and parallel processing must produce identical pixels.
	w, h := 50, 41
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*37 + 11) % 251)
	}

	for _, f := range []Filter{Dilate(3), Erode(3), Dilate(17), Erode(17)} {
		dstSer := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dstSer, src, &Options{Parallelization: false})

		dstPar := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dstPar, src, &Options{Parallelization: true})

		if !checkBoundsAndPix(dstSer.Bounds(), dstPar.Bounds(), dstSer.Pix, dstPar.Pix) {
			t.Errorf("serial/parallel mismatch for %#v", f)
		}
	}
}

func TestDilateErodePreservesRGB(t *testing.T) {
	// Per-pixel distinct RGB must survive dilation/erosion of alpha.
	w, h := 9, 7
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(x*16 + y)
			src.Pix[i+1] = uint8(y*32 + x)
			src.Pix[i+2] = uint8(x ^ (y << 4))
			if (x+y)%3 == 0 {
				src.Pix[i+3] = 0xFF
			}
		}
	}

	for _, f := range []Filter{Dilate(2), Erode(2)} {
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				si := src.PixOffset(x, y)
				di := dst.PixOffset(x, y)
				if src.Pix[si] != dst.Pix[di] ||
					src.Pix[si+1] != dst.Pix[di+1] ||
					src.Pix[si+2] != dst.Pix[di+2] {
					t.Errorf("RGB changed at (%d, %d): %#v -> %#v",
						x, y, src.Pix[si:si+3], dst.Pix[di:di+3])
				}
			}
		}
	}
}
