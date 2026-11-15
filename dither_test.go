package gift

import (
	"image"
	"image/color"
	"testing"
)

func ditherTestPalette() []color.Color {
	return []color.Color{
		color.NRGBA{0x00, 0x00, 0x00, 0xff},
		color.NRGBA{0xff, 0xff, 0xff, 0xff},
	}
}

func colorInPalette(c color.NRGBA, palette []color.Color) bool {
	for _, p := range palette {
		if p == nil {
			continue
		}
		n := color.NRGBAModel.Convert(p).(color.NRGBA)
		if n.R == c.R && n.G == c.G && n.B == c.B {
			return true
		}
	}
	return false
}

func TestDitherGradient(t *testing.T) {
	// Horizontal grayscale gradient dithered to black & white:
	// every output pixel must be exactly one of the palette colors and
	// the local density of white pixels must track the source intensity
	// (no banding, no out-of-palette colors).
	palette := ditherTestPalette()
	w, h := 256, 32
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(x)
			src.SetNRGBA(x, y, color.NRGBA{v, v, v, 0xff})
		}
	}

	g := New(Dither(palette))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	whiteCount := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := dst.NRGBAAt(x, y)
			if c.A != 0xff {
				t.Errorf("Dither: alpha changed at (%d, %d): expected 255 got %d", x, y, c.A)
			}
			if !colorInPalette(c, palette) {
				t.Fatalf("Dither: out-of-palette color %v at (%d, %d)", c, x, y)
			}
			if c.R == 0xff {
				whiteCount++
			}
		}
	}

	// Average intensity of the gradient is ~0.5, so about half of the
	// pixels must become white.
	ratio := float64(whiteCount) / float64(w*h)
	if ratio < 0.4 || ratio > 0.6 {
		t.Errorf("Dither: bad white pixel ratio: expected ~0.5 got %v", ratio)
	}

	// Per-column white density must approximate the column intensity.
	for _, x := range []int{32, 128, 224} {
		cnt := 0
		for y := 0; y < h; y++ {
			if dst.NRGBAAt(x, y).R == 0xff {
				cnt++
			}
		}
		got := float64(cnt) / float64(h)
		want := float64(x) / 255
		if got < want-0.2 || got > want+0.2 {
			t.Errorf("Dither: column %d: white density %v, expected ~%v", x, got, want)
		}
	}
}

func TestDitherCheckerboard(t *testing.T) {
	// A mid-gray checkerboard must resolve to a mix of exactly the two
	// palette colors with sharp edges between cells.
	palette := ditherTestPalette()
	src := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			v := uint8(0x40)
			if (x+y)%2 == 0 {
				v = 0xc0
			}
			src.SetNRGBA(x, y, color.NRGBA{v, v, v, 0xff})
		}
	}

	dst := image.NewNRGBA(New(Dither(palette)).Bounds(src.Bounds()))
	New(Dither(palette)).Draw(dst, src)

	black, white := 0, 0
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			c := dst.NRGBAAt(x, y)
			if !colorInPalette(c, palette) {
				t.Fatalf("Dither: out-of-palette color %v at (%d, %d)", c, x, y)
			}
			if c.R == 0xff {
				white++
			} else {
				black++
			}
		}
	}
	if black == 0 || white == 0 {
		t.Errorf("Dither: checkerboard degenerated to a single color (black=%d, white=%d)", black, white)
	}
}

func TestDitherAlphaPreserved(t *testing.T) {
	palette := ditherTestPalette()
	src := image.NewNRGBA(image.Rect(0, 0, 4, 1))
	src.SetNRGBA(0, 0, color.NRGBA{0xc0, 0xc0, 0xc0, 0xff})
	src.SetNRGBA(1, 0, color.NRGBA{0x12, 0x34, 0x56, 0x00}) // fully transparent
	src.SetNRGBA(2, 0, color.NRGBA{0x40, 0x40, 0x40, 0x80}) // semi-transparent
	src.SetNRGBA(3, 0, color.NRGBA{0x40, 0x40, 0x40, 0xff})

	dst := image.NewNRGBA(image.Rect(0, 0, 4, 1))
	New(Dither(palette)).Draw(dst, src)

	// Alpha is preserved everywhere.
	for x, want := range []uint8{0xff, 0x00, 0x80, 0xff} {
		if got := dst.NRGBAAt(x, 0).A; got != want {
			t.Errorf("Dither: alpha at x=%d: expected %d got %d", x, want, got)
		}
	}

	// The transparent pixel is copied unchanged, including its RGB.
	if got := dst.NRGBAAt(1, 0); got != (color.NRGBA{0x12, 0x34, 0x56, 0x00}) {
		t.Errorf("Dither: transparent pixel modified: got %v", got)
	}

	// The quantization error of pixel 0 (0xc0 -> white, err < 0) must not
	// leak through the transparent pixel 1 into pixel 2. Pixel 2 at 0x40
	// must quantize to black; with leaked negative error it would still be
	// black, so also check pixel 3 which would receive propagated error
	// only from pixel 2 (0x40 -> black, err > 0 pushing 0x40 up but not
	// past 0.5 unless pixel 1's own error leaked in).
	if got := dst.NRGBAAt(2, 0); got.R != 0x00 || got.G != 0x00 || got.B != 0x00 {
		t.Errorf("Dither: pixel 2: expected black got %v", got)
	}
	if got := dst.NRGBAAt(3, 0); got.R != 0x00 || got.G != 0x00 || got.B != 0x00 {
		t.Errorf("Dither: pixel 3: expected black got %v", got)
	}
}

func TestDitherTransparentDoesNotPropagateError(t *testing.T) {
	// Two identical rows, one with a transparent gap. The pixels after the
	// gap must quantize exactly as if the error from before the gap had
	// been discarded.
	palette := ditherTestPalette()
	src := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	src.SetNRGBA(0, 0, color.NRGBA{0xff, 0x00, 0x00, 0xff}) // -> white, err = (0, +1, +1)
	src.SetNRGBA(1, 0, color.NRGBA{0x00, 0x00, 0x00, 0x00}) // transparent barrier
	src.SetNRGBA(2, 0, color.NRGBA{0x60, 0x60, 0x60, 0xff}) // 0.376 gray

	dst := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	New(Dither(palette)).Draw(dst, src)

	// 0.376 must quantize to black. If the error from pixel 0 leaked
	// through the transparent pixel, the green/blue channels would be
	// pushed up by 7/16 to ~0.81 and pixel 2 would become white-ish.
	if got := dst.NRGBAAt(2, 0); got.R != 0 || got.G != 0 || got.B != 0 {
		t.Errorf("Dither: error leaked through transparent pixel: got %v", got)
	}
}

func TestDitherEmptyPalette(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			src.SetNRGBA(x, y, color.NRGBA{uint8(x * 40), uint8(y * 40), 0x80, 0xff})
		}
	}

	for _, palette := range [][]color.Color{nil, {}, {nil, nil}} {
		dst := image.NewNRGBA(image.Rect(0, 0, 4, 4))
		New(Dither(palette)).Draw(dst, src)
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				if dst.NRGBAAt(x, y) != src.NRGBAAt(x, y) {
					t.Errorf("Dither: empty palette must copy the image, mismatch at (%d, %d)", x, y)
				}
			}
		}
	}
}

func TestDitherDuplicateAndNilColors(t *testing.T) {
	// Nil entries are skipped and duplicates collapsed: the result must be
	// identical to the one produced by the cleaned palette.
	src := image.NewNRGBA(image.Rect(0, 0, 32, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 32; x++ {
			v := uint8(x * 8)
			src.SetNRGBA(x, y, color.NRGBA{v, 0x80 - v/2, v, 0xff})
		}
	}

	dirty := []color.Color{
		nil,
		color.NRGBA{0x00, 0x00, 0x00, 0xff},
		color.NRGBA{0xff, 0xff, 0xff, 0xff},
		color.NRGBA{0x00, 0x00, 0x00, 0xff}, // duplicate
		nil,
		color.NRGBA{0xff, 0x00, 0x00, 0xff},
		color.NRGBA{0xff, 0xff, 0xff, 0xff}, // duplicate
	}
	clean := []color.Color{
		color.NRGBA{0x00, 0x00, 0x00, 0xff},
		color.NRGBA{0xff, 0xff, 0xff, 0xff},
		color.NRGBA{0xff, 0x00, 0x00, 0xff},
	}

	dst1 := image.NewNRGBA(image.Rect(0, 0, 32, 8))
	New(Dither(dirty)).Draw(dst1, src)
	dst2 := image.NewNRGBA(image.Rect(0, 0, 32, 8))
	New(Dither(clean)).Draw(dst2, src)

	for y := 0; y < 8; y++ {
		for x := 0; x < 32; x++ {
			if dst1.NRGBAAt(x, y) != dst2.NRGBAAt(x, y) {
				t.Fatalf("Dither: dirty and clean palettes differ at (%d, %d)", x, y)
			}
			if !colorInPalette(dst1.NRGBAAt(x, y), clean) {
				t.Fatalf("Dither: out-of-palette color %v at (%d, %d)", dst1.NRGBAAt(x, y), x, y)
			}
		}
	}
}

func TestDitherNonZeroBounds(t *testing.T) {
	palette := ditherTestPalette()
	src := image.NewNRGBA(image.Rect(5, 7, 21, 19))
	for y := 7; y < 19; y++ {
		for x := 5; x < 21; x++ {
			v := uint8((x - 5) * 16)
			src.SetNRGBA(x, y, color.NRGBA{v, v, v, 0xff})
		}
	}

	g := New(Dither(palette))
	dstb := g.Bounds(src.Bounds())
	if dstb != image.Rect(0, 0, 16, 12) {
		t.Fatalf("Dither: bad bounds: expected (0,0,16,12) got %v", dstb)
	}
	dst := image.NewNRGBA(dstb)
	g.Draw(dst, src)

	for y := 0; y < 12; y++ {
		for x := 0; x < 16; x++ {
			if c := dst.NRGBAAt(x, y); !colorInPalette(c, palette) {
				t.Fatalf("Dither: out-of-palette color %v at (%d, %d)", c, x, y)
			}
		}
	}
}

func TestDitherSmallImages(t *testing.T) {
	palette := ditherTestPalette()
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 1, 1),
		image.Rect(0, 0, 1, 10),
		image.Rect(0, 0, 10, 1),
		image.Rect(0, 0, 2, 2),
	} {
		src := image.NewNRGBA(r)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				src.SetNRGBA(x, y, color.NRGBA{0x80, 0x40, 0xc0, 0xff})
			}
		}
		g := New(Dither(palette))
		dst := image.NewNRGBA(g.Bounds(r))
		g.Draw(dst, src) // must not panic
		for y := 0; y < r.Dy(); y++ {
			for x := 0; x < r.Dx(); x++ {
				if c := dst.NRGBAAt(x, y); !colorInPalette(c, palette) {
					t.Fatalf("Dither: size %v: out-of-palette color %v at (%d, %d)", r.Size(), c, x, y)
				}
			}
		}
	}
}

func TestDitherParallelConsistency(t *testing.T) {
	// Error diffusion is sequential: enabling parallelization must not
	// change the result.
	palette := []color.Color{
		color.NRGBA{0x1a, 0x1c, 0x2c, 0xff},
		color.NRGBA{0xb1, 0x3e, 0x53, 0xff},
		color.NRGBA{0xff, 0xcd, 0x75, 0xff},
		color.NRGBA{0xf4, 0xf4, 0xf4, 0xff},
	}
	src := image.NewNRGBA(image.Rect(0, 0, 97, 61))
	for y := 0; y < 61; y++ {
		for x := 0; x < 97; x++ {
			src.SetNRGBA(x, y, color.NRGBA{
				uint8((x*3 + y*5) % 256),
				uint8((x*7 + y*2) % 256),
				uint8((x + y*11) % 256),
				0xff,
			})
		}
	}

	results := make([]*image.NRGBA, 2)
	for i, parallel := range []bool{false, true} {
		g := New(Dither(palette))
		g.SetParallelization(parallel)
		dst := image.NewNRGBA(g.Bounds(src.Bounds()))
		g.Draw(dst, src)
		results[i] = dst
	}

	for i := range results[0].Pix {
		if results[0].Pix[i] != results[1].Pix[i] {
			t.Fatalf("Dither: serial and parallel results differ at byte %d", i)
		}
	}
}

func TestDitherFilterChain(t *testing.T) {
	// The dither filter must compose with the existing filters in a chain.
	palette := ditherTestPalette()
	src := image.NewNRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			v := uint8(x*8 + y*4)
			src.SetNRGBA(x, y, color.NRGBA{v, 0xff - v, v, 0xff})
		}
	}

	g := New(
		Grayscale(),
		Contrast(20),
		Dither(palette),
	)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			if c := dst.NRGBAAt(x, y); !colorInPalette(c, palette) {
				t.Fatalf("Dither: filter chain produced out-of-palette color %v at (%d, %d)", c, x, y)
			}
		}
	}
}
