package gift

import (
	"image"
	"image/color"
	"testing"
)

// makeTransparentGreenStrip builds the 1x2048 fixture from the issue:
// even pixels are opaque red, odd pixels are fully transparent green.
// The hidden green color of the transparent pixels must never bleed into
// the visible red pixels when the image is filtered.
func makeTransparentGreenStrip() *image.NRGBA {
	src := image.NewNRGBA(image.Rect(0, 0, 2048, 1))
	for x := 0; x < 2048; x++ {
		i := x * 4
		if x%2 == 0 {
			src.Pix[i+0] = 255 // R
			src.Pix[i+1] = 0   // G
			src.Pix[i+2] = 0   // B
			src.Pix[i+3] = 255 // A
		} else {
			src.Pix[i+0] = 0   // R
			src.Pix[i+1] = 255 // G (hidden: alpha is 0)
			src.Pix[i+2] = 0   // B
			src.Pix[i+3] = 0   // A
		}
	}
	return src
}

// TestResizeTransparentColorNoBleed verifies requirement 2/3: downscaling the
// 1x2048 opaque-red / transparent-green strip to 1x1024 must produce pure red
// pixels with no green contamination. With box resampling every output pixel
// is exactly [255, 0, 0, 128].
func TestResizeTransparentColorNoBleed(t *testing.T) {
	src := makeTransparentGreenStrip()

	f := Resize(1024, 1, BoxResampling)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	if dst.Bounds().Dx() != 1024 || dst.Bounds().Dy() != 1 {
		t.Fatalf("expected 1024x1 output, got %v", dst.Bounds())
	}
	for x := 0; x < 1024; x++ {
		i := x * 4
		got := [4]uint8{dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]}
		want := [4]uint8{255, 0, 0, 128}
		if got != want {
			t.Fatalf("box resize pixel %d = %v, want %v", x, got, want)
		}
	}
}

// TestResizeTransparentColorNoBleedAllFilters checks that every interpolating
// resampler keeps the color pure red (interior alpha is 128; only the few
// edge-clamped pixels may deviate in alpha). Nearest neighbor copies source
// pixels verbatim and therefore never mixes colors at all.
func TestResizeTransparentColorNoBleedAllFilters(t *testing.T) {
	src := makeTransparentGreenStrip()

	for _, r := range []Resampling{LinearResampling, CubicResampling, LanczosResampling} {
		f := Resize(1024, 1, r)
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)

		for x := 0; x < 1024; x++ {
			i := x * 4
			if dst.Pix[i+1] != 0 || dst.Pix[i+2] != 0 || dst.Pix[i] < 250 {
				t.Errorf("%s pixel %d color polluted: %v", r, x,
					[4]uint8{dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]})
			}
			// Interior is unaffected by boundary clamping.
			if x >= 8 && x < 1016 && !nearU8(dst.Pix[i+3], 128) {
				t.Errorf("%s interior pixel %d alpha = %d, want 128", r, x, dst.Pix[i+3])
			}
		}
	}

	// Nearest neighbor: outputs must be exact source pixels (no bleed possible).
	f := Resize(1024, 1, NearestNeighborResampling)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	for x := 0; x < 1024; x++ {
		i := x * 4
		got := [4]uint8{dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]}
		red := got == [4]uint8{255, 0, 0, 255}
		transparentGreen := got == [4]uint8{0, 255, 0, 0}
		if !red && !transparentGreen {
			t.Fatalf("nearest resize pixel %d = %v is not an exact source pixel", x, got)
		}
	}
}

func nearU8(a, b uint8) bool {
	d := int(a) - int(b)
	if d < 0 {
		d = -d
	}
	return d <= 1
}

// TestResizeOpaqueUnchanged verifies requirement 4: fully opaque inputs keep
// their original behavior (the premultiplied/unpremultiplied round-trip is the
// identity when alpha is 1).
func TestResizeOpaqueUnchanged(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 2048, 1))
	for x := 0; x < 2048; x++ {
		i := x * 4
		src.Pix[i+0] = 255
		src.Pix[i+3] = 255
	}
	f := Resize(1024, 1, BoxResampling)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	for x := 0; x < 1024; x++ {
		i := x * 4
		got := [4]uint8{dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]}
		if got != [4]uint8{255, 0, 0, 255} {
			t.Fatalf("opaque resize pixel %d = %v, want [255 0 0 255]", x, got)
		}
	}
}

// TestRotateTransparentColorNoBleed verifies requirement 1/2 for rotation:
// rotating the strip must not produce any visible green/blue fringe, and the
// uncovered background (transparent) must not be painted opaque.
func TestRotateTransparentColorNoBleed(t *testing.T) {
	for _, angle := range []float32{30, 45, 90, 120} {
		for _, interp := range []Interpolation{NearestNeighborInterpolation, LinearInterpolation, CubicInterpolation} {
			src := makeTransparentGreenStrip()
			f := Rotate(angle, color.NRGBA{0, 0, 0, 0}, interp)
			dst := image.NewNRGBA(f.Bounds(src.Bounds()))
			f.Draw(dst, src, nil)

			visible := 0
			for i := 0; i < len(dst.Pix); i += 4 {
				r, g, b, a := dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]
				if a == 0 {
					// Fully transparent output remains fully transparent: the
					// background must never be painted opaque. Its hidden RGB is
					// not visible, so it must not affect any visible neighbor.
					continue
				}
				visible++
				if int(g) > int(r)+10 || int(b) > int(r)+10 {
					t.Errorf("rotate %.0f interp=%d visible pixel polluted: %v",
						angle, interp, [4]uint8{r, g, b, a})
				}
			}
			if visible == 0 {
				t.Errorf("rotate %.0f interp=%d produced no visible pixels", angle, interp)
			}
		}
	}
}

// TestBlurTransparentColorNoBleed verifies requirement 1/2 for gaussian blur:
// an opaque red block surrounded by fully transparent green must blur into
// pure red tones with no green contamination, and fully transparent corners
// must remain fully transparent (alpha must never be raised there).
func TestBlurTransparentColorNoBleed(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			i := src.PixOffset(x, y)
			if x >= 4 && x < 12 && y >= 4 && y < 12 {
				src.Pix[i+0] = 255
				src.Pix[i+3] = 255
			} else {
				src.Pix[i+1] = 255 // fully transparent green
			}
		}
	}

	f := GaussianBlur(2.0)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	for i := 0; i < len(dst.Pix); i += 4 {
		r, g, b, a := dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]
		if a > 0 && (int(g) > int(r)+10 || int(b) > int(r)+10) {
			t.Fatalf("blurred pixel polluted: %v", [4]uint8{r, g, b, a})
		}
	}

	for _, p := range [][2]int{{0, 0}, {15, 0}, {0, 15}, {15, 15}} {
		i := dst.PixOffset(p[0], p[1])
		// The corner stays fully transparent: blur must never raise alpha where
		// the source was transparent. (Hidden RGB at alpha=0 is not visible.)
		if dst.Pix[i+3] != 0 {
			t.Errorf("blur corner %v alpha = %d, want 0", p, dst.Pix[i+3])
		}
	}
}

// TestPremultiplyRoundTrip checks the premultiplied-alpha helpers used by all
// the linear filtering paths.
func TestPremultiplyRoundTrip(t *testing.T) {
	cases := []struct {
		in, want pixel
	}{
		{pixel{1, 0, 0, 1}, pixel{1, 0, 0, 1}},                 // opaque red
		{pixel{0, 1, 0, 0}, pixel{0, 0, 0, 0}},                 // fully transparent green -> hidden color dropped
		{pixel{1, 1, 1, 0.5}, pixel{1, 1, 1, 0.5}},             // 50% white round-trips
		{pixel{0.8, 0.4, 0.2, 0.25}, pixel{0.8, 0.4, 0.2, 0.25}},
	}
	for _, c := range cases {
		got := c.in.premultiply().unpremultiply()
		if !comparePixels(got, c.want, 1e-6) {
			t.Errorf("round-trip %v -> %v, want %v", c.in, got, c.want)
		}
	}

	// A premultiplied pixel with zero alpha unpremultiplies to zeroed color.
	if got := (pixel{0.3, 0.2, 0.1, 0}).unpremultiply(); got != (pixel{0, 0, 0, 0}) {
		t.Errorf("zero-alpha unpremultiply = %v, want zero pixel", got)
	}

	// Premultiplication actually scales color by alpha.
	pm := pixel{1, 1, 1, 0.5}.premultiply()
	if pm != (pixel{0.5, 0.5, 0.5, 0.5}) {
		t.Errorf("premultiply = %v, want {0.5 0.5 0.5 0.5}", pm)
	}
}
