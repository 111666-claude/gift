package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"
)

// chromaKeyTestImage builds an NRGBA image at the given bounds from a row-major
// slice of colors. The number of colors must match the image size.
func chromaKeyTestImage(r image.Rectangle, colors []color.NRGBA) *image.NRGBA {
	img := image.NewNRGBA(r)
	for i, c := range colors {
		x := r.Min.X + i%r.Dx()
		y := r.Min.Y + i/r.Dx()
		img.SetNRGBA(x, y, c)
	}
	return img
}

// chromaKeyAlphaAt returns the alpha channel of a pixel.
func chromaKeyAlphaAt(img image.Image, x, y int) uint8 {
	return color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA).A
}

// chromaKeyRGBAt returns the non-premultiplied RGB channels of a pixel.
func chromaKeyRGBAt(img image.Image, x, y int) (r, g, b uint8) {
	c := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
	return c.R, c.G, c.B
}

// chromaKeyExpectedAlpha mirrors the documented alpha ramp in float64 so that
// tests do not depend on the implementation's float32 rounding.
func chromaKeyExpectedAlpha(distance, tolerance, feather float64) float64 {
	if distance <= tolerance {
		return 0
	}
	if feather <= 0 {
		return 1
	}
	if distance >= tolerance+feather {
		return 1
	}
	return (distance - tolerance) / feather
}

func chromaKeyExpectedDistance(px [3]uint8, key [3]uint8) float64 {
	var sum float64
	for i := 0; i < 3; i++ {
		d := (float64(px[i]) - float64(key[i])) / 255
		sum += d * d
	}
	return math.Sqrt(sum / 3)
}

func TestChromaKeyBounds(t *testing.T) {
	srcb := image.Rect(-3, 5, 6, 9)
	src := image.NewNRGBA(srcb)
	f := ChromaKey(color.NRGBA{0, 255, 0, 255}, 0.1, 0.2, true)

	if b := f.Bounds(srcb); !b.Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
		t.Fatalf("unexpected filter bounds: %v", b)
	}

	// A destination with a different Min must receive the filtered image at
	// its own origin, and the returned GIFT bounds must match the source size.
	g := New(f)
	dstb := image.Rect(100, -50, 109, -46)
	dst := image.NewNRGBA(dstb)
	g.Draw(dst, src)

	if !dst.Bounds().Eq(dstb) {
		t.Fatalf("destination bounds changed: %v", dst.Bounds())
	}
	if b := g.Bounds(srcb); !b.Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
		t.Fatalf("unexpected gift bounds: %v", b)
	}

	// The top-left source pixel is the key color, so the top-left destination
	// pixel must be transparent; the offsets must line up on both axes.
	src.SetNRGBA(srcb.Min.X, srcb.Min.Y, color.NRGBA{0, 255, 0, 255})
	g.Draw(dst, src)
	if a := chromaKeyAlphaAt(dst, dstb.Min.X, dstb.Min.Y); a != 0 {
		t.Fatalf("expected transparent key pixel at destination origin, got alpha %d", a)
	}
}

func TestChromaKeyAlphaGradient(t *testing.T) {
	// A synthetic anti-aliased edge: a linear gradient from the green key
	// color to a skin tone. Distances are monotonic and the alpha ramp must be
	// continuous, so every intermediate step must stay between its neighbors.
	colors := []color.NRGBA{
		{0, 255, 0, 255},
		{16, 241, 2, 255},
		{32, 226, 4, 255},
		{48, 212, 6, 255},
		{64, 198, 8, 255},
		{80, 183, 10, 255},
		{96, 169, 12, 255},
		{112, 155, 14, 255},
		{128, 140, 16, 255},
		{144, 126, 18, 255},
		{160, 112, 20, 255},
		{176, 97, 22, 255},
		{192, 83, 24, 255},
		{208, 69, 26, 255},
		{224, 54, 28, 255},
		{240, 40, 30, 255},
	}
	expected := []uint8{
		0, 0, 0, 47, 97, 148, 197, 246,
		255, 255, 255, 255, 255, 255, 255, 255,
	}

	key := [3]uint8{0, 255, 0}
	srcb := image.Rect(0, 0, len(colors), 1)
	src := chromaKeyTestImage(srcb, colors)

	const tolerance, feather = 0.1, 0.25
	g := New(ChromaKey(color.NRGBA{0, 255, 0, 255}, tolerance, feather, false))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for i, c := range colors {
		gotA := chromaKeyAlphaAt(dst, i, 0)
		wantA := float64(expected[i])
		if math.Abs(float64(gotA)-wantA) > 1 {
			t.Errorf("pixel %d %v: expected alpha ~%.0f, got %d", i, c, wantA, gotA)
		}

		// The original alpha was fully opaque, so every alpha value must be
		// the expected ramp and never overflow.
		wantF := chromaKeyExpectedAlpha(chromaKeyExpectedDistance([3]uint8{c.R, c.G, c.B}, key), tolerance, feather)
		if math.Abs(float64(gotA)-wantF*255) > 1 {
			t.Errorf("pixel %d %v: expected alpha %.2f, got %d", i, c, wantF, gotA)
		}
	}

	// Continuity: a partial alpha must actually occur between the endpoints.
	partial := 0
	for i := 0; i < len(colors); i++ {
		a := chromaKeyAlphaAt(dst, i, 0)
		if a > 0 && a < 255 {
			partial++
		}
		if i > 0 && chromaKeyAlphaAt(dst, i-1, 0) > a {
			t.Errorf("alpha ramp is not monotonic at %d (%d > %d)",
				i, chromaKeyAlphaAt(dst, i-1, 0), a)
		}
	}
	if partial < 3 {
		t.Errorf("expected several partially transparent edge pixels, got %d", partial)
	}
}

func TestChromaKeyHardEdge(t *testing.T) {
	colors := []color.NRGBA{
		{0, 255, 0, 255},
		{16, 241, 2, 255},
		{32, 226, 4, 255},
		{48, 212, 6, 255},
		{64, 198, 8, 255},
		{80, 183, 10, 255},
		{240, 40, 30, 255},
	}
	// Distances from green: 0, 0.048, 0.098, 0.147, 0.195, 0.245, 0.733.
	// With tolerance 0.2 and no feather the first five are keyed and the last
	// two keep their full alpha.
	expected := []uint8{0, 0, 0, 0, 0, 255, 255}

	src := chromaKeyTestImage(image.Rect(0, 0, len(colors), 1), colors)
	g := New(ChromaKey(color.NRGBA{0, 255, 0, 255}, 0.2, 0, true))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for i := range colors {
		if got := chromaKeyAlphaAt(dst, i, 0); got != expected[i] {
			t.Errorf("pixel %d: expected alpha %d, got %d", i, expected[i], got)
		}
	}

	// Every alpha must be one of the two hard-edge states.
	for i := 0; i < len(colors); i++ {
		if a := chromaKeyAlphaAt(dst, i, 0); a != 0 && a != 255 {
			t.Errorf("feather 0 produced intermediate alpha %d at pixel %d", a, i)
		}
	}
}

func TestChromaKeyPreservesForegroundAndTransparency(t *testing.T) {
	// Real color swatches used as foreground, key background and edge samples.
	foreground := []color.NRGBA{
		{224, 172, 105, 255}, // skin
		{240, 40, 30, 255},   // red shirt
		{40, 60, 160, 255},   // blue trousers
		{90, 60, 40, 255},    // brown hair
		{250, 250, 245, 255}, // white highlight
		{220, 150, 80, 128},  // semi-transparent skin
	}
	key := color.NRGBA{0, 255, 0, 255}

	srcb := image.Rect(-2, -3, 4, 3) // 6x6 with non-zero Min
	colors := make([]color.NRGBA, srcb.Dx()*srcb.Dy())
	for i := range colors {
		colors[i] = key
	}
	for i, c := range foreground {
		colors[i] = c
	}
	// A pixel that is already fully transparent, with an arbitrary color that
	// must not be interpreted as a color to key.
	transparentIndex := 2*srcb.Dx() + 3
	colors[transparentIndex] = color.NRGBA{200, 255, 200, 0}

	src := chromaKeyTestImage(srcb, colors)
	orig := append([]uint8(nil), src.Pix...)

	g := New(ChromaKey(key, 0.1, 0.2, true))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	// The source image must not be modified.
	for i := range orig {
		if src.Pix[i] != orig[i] {
			t.Fatalf("source image was modified at byte %d: %d != %d", i, src.Pix[i], orig[i])
		}
	}

	for i, c := range foreground {
		x := i
		got := color.NRGBAModel.Convert(dst.At(x, 0)).(color.NRGBA)
		if got.A != c.A {
			t.Errorf("foreground %v: expected alpha %d, got %d", c, c.A, got.A)
		}
		if got.R != c.R || got.G != c.G || got.B != c.B {
			t.Errorf("foreground %v was recolored to %v", c, got)
		}
	}

	// Originally transparent pixels stay transparent and keep their color.
	tx := transparentIndex % srcb.Dx()
	ty := transparentIndex / srcb.Dx()
	got := color.NRGBAModel.Convert(dst.At(tx, ty)).(color.NRGBA)
	if got != colors[transparentIndex] {
		t.Errorf("transparent pixel changed: expected %v, got %v", colors[transparentIndex], got)
	}

	// Every remaining pixel is the key color and must be fully transparent.
	for y := 0; y < srcb.Dy(); y++ {
		for x := 0; x < srcb.Dx(); x++ {
			if x < len(foreground) && y == 0 || y*srcb.Dx()+x == transparentIndex {
				continue
			}
			if a := chromaKeyAlphaAt(dst, x, y); a != 0 {
				t.Errorf("key pixel (%d,%d) kept alpha %d", x, y, a)
			}
		}
	}
}

func TestChromaKeyDespill(t *testing.T) {
	// A typical anti-aliased fringe: green stays the dominant channel while
	// the pixel is only partially covered by the foreground.
	fringe := color.NRGBA{140, 190, 110, 255}
	key := color.NRGBA{0, 255, 0, 255}

	src := chromaKeyTestImage(image.Rect(0, 0, 1, 1), []color.NRGBA{fringe})

	const tolerance, feather = 0.2, 0.5
	distance := chromaKeyExpectedDistance([3]uint8{fringe.R, fringe.G, fringe.B}, [3]uint8{key.R, key.G, key.B})
	wantA := chromaKeyExpectedAlpha(distance, tolerance, feather)
	if wantA <= 0 || wantA >= 1 {
		t.Fatalf("test setup: fringe pixel must be partially keyed, got alpha %v", wantA)
	}

	without := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	New(ChromaKey(key, tolerance, feather, false)).Draw(without, src)

	with := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	New(ChromaKey(key, tolerance, feather, true)).Draw(with, src)

	// Despill must not change the generated alpha.
	if a0, a1 := chromaKeyAlphaAt(without, 0, 0), chromaKeyAlphaAt(with, 0, 0); a0 != a1 {
		t.Errorf("despill changed alpha: %d -> %d", a0, a1)
	}

	// The green channel is clamped to the larger of red and blue.
	r, g, b := chromaKeyRGBAt(with, 0, 0)
	if g > chromaKeyMaxUint8(r, b) {
		t.Errorf("despill left green dominant: %v", [3]uint8{r, g, b})
	}
	if g >= fringe.G {
		t.Errorf("despill did not reduce green: %d >= %d", g, fringe.G)
	}

	// Without despill the fringe remains green-dominant.
	r0, g0, b0 := chromaKeyRGBAt(without, 0, 0)
	if g0 <= chromaKeyMaxUint8(r0, b0) {
		t.Errorf("test setup: expected spill without despill, got %v", [3]uint8{r0, g0, b0})
	}

	// Composite over light and dark backgrounds. After despill the green
	// channel must not dominate the composite on either background.
	for _, bg := range []color.NRGBA{
		{255, 255, 255, 255},
		{0, 0, 0, 255},
		{64, 64, 64, 255},
		{200, 200, 200, 255},
	} {
		out := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		draw.Draw(out, out.Bounds(), &image.Uniform{bg}, image.Point{}, draw.Src)
		New(ChromaKey(key, tolerance, feather, true)).
			DrawAt(out, src, image.Point{}, OverOperator)

		cr, cg, cb := chromaKeyRGBAt(out, 0, 0)
		if excess := int(cg) - int(chromaKeyMaxUint8(cr, cb)); excess > 1 {
			t.Errorf("background %v: green fringe remains after despill (%d, %d, %d)", bg, cr, cg, cb)
		}

		// Sanity check that the same composite without despill does show it,
		// which proves the check above is meaningful.
		raw := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		draw.Draw(raw, raw.Bounds(), &image.Uniform{bg}, image.Point{}, draw.Src)
		New(ChromaKey(key, tolerance, feather, false)).
			DrawAt(raw, src, image.Point{}, OverOperator)
		rr, rg, rb := chromaKeyRGBAt(raw, 0, 0)
		if int(rg)-int(chromaKeyMaxUint8(rr, rb)) <= 1 {
			t.Errorf("test setup: background %v hides the spill even without despill", bg)
		}
	}
}

func TestChromaKeyCompositeEdge(t *testing.T) {
	// Composite the full anti-aliased edge (same gradient as the alpha test)
	// over both light and dark backgrounds using the Over operator, and check
	// that despill removes the green cast while the alpha ramp is unchanged.
	colors := []color.NRGBA{
		{0, 255, 0, 255},
		{16, 241, 2, 255},
		{32, 226, 4, 255},
		{48, 212, 6, 255},
		{64, 198, 8, 255},
		{80, 183, 10, 255},
		{96, 169, 12, 255},
		{112, 155, 14, 255},
		{128, 140, 16, 255},
	}
	key := color.NRGBA{0, 255, 0, 255}
	const tolerance, feather = 0.1, 0.25

	src := chromaKeyTestImage(image.Rect(0, 0, len(colors), 1), colors)

	keyed := image.NewNRGBA(image.Rect(0, 0, len(colors), 1))
	New(ChromaKey(key, tolerance, feather, true)).Draw(keyed, src)

	backgrounds := []color.NRGBA{
		{255, 255, 255, 255}, // light
		{0, 0, 0, 255},       // dark
		{128, 128, 128, 255}, // mid gray
		{240, 220, 180, 255}, // warm light
	}

	partialSeen := false
	for x := 0; x < len(colors); x++ {
		a := chromaKeyAlphaAt(keyed, x, 0)
		if a == 0 || a == 255 {
			continue
		}
		partialSeen = true

		for _, bg := range backgrounds {
			// Manual source-over of the despilled edge pixel on the background.
			c := color.NRGBAModel.Convert(keyed.At(x, 0)).(color.NRGBA)
			af := float64(c.A) / 255
			composite := func(bgv uint8, fgv uint8) uint8 {
				return uint8(math.Round(float64(fgv)*af + float64(bgv)*(1-af)))
			}
			r := composite(bg.R, c.R)
			g := composite(bg.G, c.G)
			b := composite(bg.B, c.B)

			// A visible green fringe is a composite whose green channel exceeds
			// both other channels by more than a rounding step.
			if int(g)-int(chromaKeyMaxUint8(r, b)) > 1 {
				t.Errorf("pixel %d over %v: green fringe remains (%d, %d, %d)", x, bg, r, g, b)
			}
		}
	}
	if !partialSeen {
		t.Fatal("test setup: expected partially transparent edge pixels")
	}

	// The same edge without despill must still show the fringe on at least one
	// background, confirming the assertions above are meaningful.
	raw := image.NewNRGBA(image.Rect(0, 0, len(colors), 1))
	New(ChromaKey(key, tolerance, feather, false)).Draw(raw, src)
	fringeSeen := false
	for x := 0; x < len(colors); x++ {
		a := chromaKeyAlphaAt(raw, x, 0)
		if a == 0 || a == 255 {
			continue
		}
		c := color.NRGBAModel.Convert(raw.At(x, 0)).(color.NRGBA)
		for _, bg := range backgrounds {
			af := float64(c.A) / 255
			g := float64(c.G)*af + float64(bg.G)*(1-af)
			r := float64(c.R)*af + float64(bg.R)*(1-af)
			b := float64(c.B)*af + float64(bg.B)*(1-af)
			if g-maxf64(r, b) > 1 {
				fringeSeen = true
			}
		}
	}
	if !fringeSeen {
		t.Fatal("test setup: the raw edge should show a green fringe before despill")
	}
}

func maxf64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func TestChromaKeyDespillSkipsOpaqueAndTransparentPixels(t *testing.T) {
	key := color.NRGBA{0, 255, 0, 255}
	colors := []color.NRGBA{
		{240, 40, 30, 255},   // fully opaque foreground
		{0, 255, 0, 255},     // fully keyed
		{200, 190, 170, 255}, // partial but green is not dominant
	}
	src := chromaKeyTestImage(image.Rect(0, 0, len(colors), 1), colors)

	g := New(ChromaKey(key, 0.2, 0.5, true))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	// Fully opaque foreground keeps its exact color.
	if r, gg, b := chromaKeyRGBAt(dst, 0, 0); r != 240 || gg != 40 || b != 30 {
		t.Errorf("opaque foreground was recolored: %d %d %d", r, gg, b)
	}
	// Fully keyed pixels are invisible; despill does not matter.
	if a := chromaKeyAlphaAt(dst, 1, 0); a != 0 {
		t.Errorf("key pixel kept alpha %d", a)
	}
	// The partial pixel must not have its dominant red/blue channels clamped
	// just because the key is green.
	if r, gg, b := chromaKeyRGBAt(dst, 2, 0); r != 200 || gg != 190 || b != 170 {
		t.Errorf("partial non-green pixel was recolored: %d %d %d", r, gg, b)
	}
}

func TestChromaKeyDespillChannelSelection(t *testing.T) {
	// For a key with a single dominant channel, that channel is the spill
	// channel and is clamped to the larger of the other two. Magenta has no
	// single dominant channel, so despill leaves the pixel alone.
	tests := []struct {
		desc string
		key  color.NRGBA
		px   color.NRGBA
		tol  float32
		f    float32
		want color.NRGBA // expected RGB, alpha is checked separately
	}{
		{"green spill", color.NRGBA{0, 255, 0, 255}, color.NRGBA{140, 190, 110, 255}, 0.2, 0.5, color.NRGBA{140, 140, 110, 255}},
		{"blue spill", color.NRGBA{0, 0, 255, 255}, color.NRGBA{120, 90, 200, 255}, 0.2, 0.5, color.NRGBA{120, 90, 120, 255}},
		{"red spill", color.NRGBA{255, 0, 0, 255}, color.NRGBA{200, 110, 90, 255}, 0.2, 0.5, color.NRGBA{110, 110, 90, 255}},
		{"magenta key has no spill channel", color.NRGBA{255, 0, 255, 255}, color.NRGBA{190, 80, 170, 255}, 0.2, 0.6, color.NRGBA{190, 80, 170, 255}},
	}

	for _, d := range tests {
		src := chromaKeyTestImage(image.Rect(0, 0, 1, 1), []color.NRGBA{d.px})
		dst := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		ChromaKey(d.key, d.tol, d.f, true).Draw(dst, src, nil)

		got := color.NRGBAModel.Convert(dst.At(0, 0)).(color.NRGBA)
		if got.A == 0 || got.A == 255 {
			t.Fatalf("%s: test setup: expected a partially keyed pixel, got alpha %d", d.desc, got.A)
		}
		if got.R != d.want.R || got.G != d.want.G || got.B != d.want.B {
			t.Errorf("%s: expected %v, got %v", d.desc, d.want, got)
		}
	}
}

func TestChromaKeyKeyColorAlphaIgnored(t *testing.T) {
	// The key color's alpha must not matter: NRGBA with alpha 0 still carries
	// the RGB meaning of the key and must key the same pixels as its opaque
	// equivalent. (color.NRGBAModel would turn such a color into black.)
	srcColors := []color.NRGBA{
		{0, 255, 0, 255},
		{240, 40, 30, 255},
		{0, 200, 0, 255},
	}
	src := chromaKeyTestImage(image.Rect(0, 0, 3, 1), srcColors)

	keys := []color.Color{
		color.NRGBA{0, 255, 0, 255},
		color.NRGBA{0, 255, 0, 0},
		color.NRGBA{0, 255, 0, 1},
		color.RGBA{0, 255, 0, 255},
		color.NRGBA64{0, 0xffff, 0, 0},
	}

	var expected []uint8
	for _, key := range keys {
		dst := image.NewNRGBA(image.Rect(0, 0, 3, 1))
		ChromaKey(key, 0.2, 0, false).Draw(dst, src, nil)
		var alphas []uint8
		for x := 0; x < 3; x++ {
			alphas = append(alphas, chromaKeyAlphaAt(dst, x, 0))
		}
		if expected == nil {
			expected = alphas
			continue
		}
		for x := range alphas {
			if alphas[x] != expected[x] {
				t.Errorf("key %v: alpha %d is %d, expected %d", key, x, alphas[x], expected[x])
			}
		}
	}
	if expected[0] != 0 || expected[1] != 255 || expected[2] != 0 {
		t.Errorf("unexpected keying result for green key: %v", expected)
	}
}

func TestChromaKeyInvalidParameters(t *testing.T) {
	key := color.NRGBA{0, 255, 0, 255}
	src := chromaKeyTestImage(image.Rect(0, 0, 2, 2), []color.NRGBA{
		{0, 255, 0, 255},
		{10, 250, 10, 255},
		{200, 30, 20, 255},
		{0, 0, 0, 0},
	})
	orig := append([]uint8(nil), src.Pix...)

	tests := []struct {
		desc string
		key  color.Color
		tol  float32
		f    float32
		desp bool
	}{
		{"nil key", nil, 0.1, 0.2, true},
		{"NaN tolerance", key, float32(math.NaN()), 0.2, true},
		{"+Inf tolerance", key, float32(math.Inf(1)), 0.2, true},
		{"-Inf tolerance", key, float32(math.Inf(-1)), 0.2, true},
		{"NaN feather", key, 0.1, float32(math.NaN()), true},
		{"+Inf feather", key, 0.1, float32(math.Inf(1)), true},
		{"-Inf feather", key, 0.1, float32(math.Inf(-1)), true},
	}

	for _, d := range tests {
		f := ChromaKey(d.key, d.tol, d.f, d.desp)
		if b := f.Bounds(src.Bounds()); !b.Eq(src.Bounds().Sub(src.Bounds().Min)) {
			t.Errorf("%s: unexpected bounds %v", d.desc, b)
		}
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		New(f).Draw(dst, src)
		for i := range orig {
			if dst.Pix[i] != orig[i] {
				t.Errorf("%s: expected an unchanged copy at byte %d: %d != %d",
					d.desc, i, dst.Pix[i], orig[i])
				break
			}
		}
	}

	// A negative tolerance and feather are clamped to zero: only the exact key
	// color is removed, with a hard edge.
	src = chromaKeyTestImage(image.Rect(0, 0, 3, 1), []color.NRGBA{
		{0, 255, 0, 255},
		{1, 255, 0, 255},
		{200, 30, 20, 255},
	})
	dst := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	New(ChromaKey(key, -1, -1, true)).Draw(dst, src)
	if a := chromaKeyAlphaAt(dst, 0, 0); a != 0 {
		t.Errorf("negative tolerance: expected exact key removed, got alpha %d", a)
	}
	if a := chromaKeyAlphaAt(dst, 1, 0); a != 255 {
		t.Errorf("negative tolerance: near key color must stay opaque, got alpha %d", a)
	}
	if a := chromaKeyAlphaAt(dst, 2, 0); a != 255 {
		t.Errorf("negative tolerance: foreground must stay opaque, got alpha %d", a)
	}

	// Parameters above the maximum distance are clamped to 1: every pixel is
	// within tolerance and the result is fully transparent, but still stable.
	dst = image.NewNRGBA(image.Rect(0, 0, 3, 1))
	New(ChromaKey(key, 100, 100, true)).Draw(dst, src)
	for x := 0; x < 3; x++ {
		if a := chromaKeyAlphaAt(dst, x, 0); a != 0 {
			t.Errorf("clamped tolerance: expected alpha 0 at %d, got %d", x, a)
		}
	}
}

func TestChromaKeyDegenerateImages(t *testing.T) {
	key := color.NRGBA{0, 255, 0, 255}

	// Fully transparent image: alpha stays zero, no color or alpha overflow.
	transparent := image.NewNRGBA(image.Rect(-1, -1, 3, 2))
	for i := range transparent.Pix {
		transparent.Pix[i] = 0xff
	}
	for y := -1; y < 2; y++ {
		for x := -1; x < 3; x++ {
			transparent.SetNRGBA(x, y, color.NRGBA{200, 255, 180, 0})
		}
	}
	f := ChromaKey(key, 0.1, 0.2, true)
	dst := image.NewNRGBA(f.Bounds(transparent.Bounds()))
	f.Draw(dst, transparent, nil)
	for y := 0; y < 3; y++ {
		for x := 0; x < 4; x++ {
			c := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
			if c.A != 0 {
				t.Fatalf("transparent image produced alpha %d at (%d,%d)", c.A, x, y)
			}
			// The color of an originally transparent pixel is preserved
			// exactly; in particular it is never turned into a visible color.
			if c.R != 200 || c.G != 255 || c.B != 180 || c.A != 0 {
				t.Fatalf("transparent image recolored pixel to %v", c)
			}
		}
	}

	// Image entirely filled with the key color and with a near-key color.
	for _, c := range []color.NRGBA{
		{0, 255, 0, 255},
		{0, 254, 2, 255},
	} {
		src := image.NewNRGBA(image.Rect(0, 0, 5, 4))
		draw.Draw(src, src.Bounds(), &image.Uniform{c}, image.Point{}, draw.Src)
		dst := image.NewNRGBA(image.Rect(0, 0, 5, 4))
		New(ChromaKey(key, 0.1, 0.3, true)).Draw(dst, src)
		for y := 0; y < 4; y++ {
			for x := 0; x < 5; x++ {
				if a := chromaKeyAlphaAt(dst, x, y); a != 0 {
					t.Fatalf("uniform key image %v kept alpha %d at (%d,%d)", c, a, x, y)
				}
			}
		}
	}

	// Zero-sized images must be handled without panicking.
	empty := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	dst = image.NewNRGBA(image.Rect(0, 0, 0, 0))
	ChromaKey(key, 0.1, 0.2, true).Draw(dst, empty, nil)
}

func TestChromaKeyImageTypes(t *testing.T) {
	key := color.NRGBA{0, 255, 0, 255}
	colors := []color.NRGBA{
		{0, 255, 0, 255},
		{32, 226, 4, 255},
		{112, 155, 14, 255},
		{240, 40, 30, 255},
	}
	const tolerance, feather = 0.1, 0.25
	f := ChromaKey(key, tolerance, feather, false)

	// The same pixels in different 8-bit image types must key identically.
	type variant struct {
		name string
		img  draw.Image
	}
	var variants []variant
	nrgba := chromaKeyTestImage(image.Rect(0, 0, len(colors), 1), colors)
	variants = append(variants, variant{"NRGBA", nrgba})

	rgba := image.NewRGBA(nrgba.Bounds())
	draw.Draw(rgba, rgba.Bounds(), nrgba, image.Point{}, draw.Src)
	variants = append(variants, variant{"RGBA", rgba})

	var expected []uint8
	for _, v := range variants {
		dst := image.NewNRGBA(f.Bounds(v.img.Bounds()))
		f.Draw(dst, v.img, nil)
		var alphas []uint8
		for x := 0; x < len(colors); x++ {
			alphas = append(alphas, chromaKeyAlphaAt(dst, x, 0))
		}
		if expected == nil {
			expected = alphas
		} else if len(alphas) != len(expected) {
			t.Fatalf("%s: unexpected alpha count %d", v.name, len(alphas))
		} else {
			for i := range alphas {
				if alphas[i] != expected[i] {
					t.Errorf("%s: alpha %d is %d, NRGBA produced %d", v.name, i, alphas[i], expected[i])
				}
			}
		}
	}

	// 16-bit NRGBA64 must produce the same ramp at higher precision.
	nrgba64 := image.NewNRGBA64(image.Rect(0, 0, len(colors), 1))
	for x := 0; x < len(colors); x++ {
		c := nrgba.NRGBAAt(x, 0)
		nrgba64.SetNRGBA64(x, 0, color.NRGBA64{
			uint16(c.R) * 0x101, uint16(c.G) * 0x101, uint16(c.B) * 0x101, 0xffff,
		})
	}
	dst64 := image.NewNRGBA64(f.Bounds(nrgba64.Bounds()))
	f.Draw(dst64, nrgba64, nil)
	for x := 0; x < len(colors); x++ {
		got := dst64.NRGBA64At(x, 0).A
		want := uint16(expected[x]) * 0x101
		if diff := int(got) - int(want); diff < -0x200 || diff > 0x200 {
			t.Errorf("NRGBA64 alpha %d is %d, expected ~%d", x, got, want)
		}
	}

	// Gray images: a gray key removes matching gray levels and scales the
	// luminance of partially keyed pixels by their alpha.
	grayKey := color.NRGBA{128, 128, 128, 255}
	gray := image.NewGray(image.Rect(0, 0, 3, 1))
	gray.Pix[0] = 128
	gray.Pix[1] = 0
	gray.Pix[2] = 255
	grayDst := image.NewGray(image.Rect(0, 0, 3, 1))
	ChromaKey(grayKey, 0.2, 0, false).Draw(grayDst, gray, nil)
	if grayDst.Pix[0] != 0 {
		t.Errorf("gray key pixel should be removed, got %d", grayDst.Pix[0])
	}
	if grayDst.Pix[1] != 0 || grayDst.Pix[2] != 255 {
		t.Errorf("gray foreground levels changed: %v", grayDst.Pix)
	}

	// Paletted output must not panic and must remain a valid palette image.
	palImg := image.NewPaletted(image.Rect(0, 0, 1, 1), []color.Color{
		color.NRGBA{0, 255, 0, 255},
		color.NRGBA{0, 0, 0, 0},
	})
	palDst := image.NewPaletted(image.Rect(0, 0, 1, 1), palImg.Palette)
	ChromaKey(key, 0.1, 0.2, true).Draw(palDst, palImg, nil)
	if palDst.Pix[0] >= uint8(len(palDst.Palette)) {
		t.Errorf("paletted output index out of range: %d", palDst.Pix[0])
	}
}

func TestChromaKeyParallelAndChain(t *testing.T) {
	key := color.NRGBA{0, 255, 0, 255}

	// A larger synthetic image with a gradient background and colored blocks.
	b := image.Rect(0, 0, 64, 48)
	src := image.NewNRGBA(b)
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			switch {
			case x < 20:
				// Green screen with a mild luminance gradient.
				src.SetNRGBA(x, y, color.NRGBA{0, uint8(230 + (x+y)%25), uint8((x + y) % 20), 255})
			case x < 40:
				// Foreground blocks.
				src.SetNRGBA(x, y, color.NRGBA{uint8(100 + x), uint8(40 + y), 60, 255})
			default:
				// Anti-aliased transition.
				t := float64(x-40) / 23
				src.SetNRGBA(x, y, color.NRGBA{
					uint8(255 * t), uint8(255 * (1 - 0.4*t)), uint8(50 * t), 255,
				})
			}
		}
	}

	run := func(parallel bool, filters ...Filter) *image.NRGBA {
		g := New(filters...)
		g.SetParallelization(parallel)
		dst := image.NewNRGBA(g.Bounds(src.Bounds()))
		g.Draw(dst, src)
		return dst
	}

	serial := run(false, ChromaKey(key, 0.1, 0.2, true))
	parallel := run(true, ChromaKey(key, 0.1, 0.2, true))
	if !chromaKeyEqualBytes(serial.Pix, parallel.Pix) {
		t.Error("parallel and serial results differ")
	}

	// The filter is composable: the output size is unchanged when followed by
	// another size-preserving filter, and alpha survives the chain.
	chain := run(true, ChromaKey(key, 0.1, 0.2, true), Grayscale())
	if !chain.Bounds().Eq(src.Bounds()) {
		t.Fatalf("filter chain changed bounds: %v", chain.Bounds())
	}
	if a := chromaKeyAlphaAt(chain, 0, 0); a != 0 {
		t.Errorf("keyed pixel lost its alpha in a chain: %d", a)
	}
	if a := chromaKeyAlphaAt(chain, 30, 20); a != 255 {
		t.Errorf("foreground pixel lost its alpha in a chain: %d", a)
	}

	// ChromaKey in front of a resizing filter must still work and produce the
	// resized bounds.
	g := New(ChromaKey(key, 0.1, 0.2, true), Resize(10, 10, NearestNeighborResampling))
	if b := g.Bounds(src.Bounds()); !b.Eq(image.Rect(0, 0, 10, 10)) {
		t.Fatalf("unexpected chained bounds: %v", b)
	}
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)
	if a := chromaKeyAlphaAt(dst, 0, 0); a != 0 {
		t.Errorf("resized chain lost keying: alpha %d", a)
	}
}

func chromaKeyEqualBytes(a, b []uint8) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func chromaKeyMaxUint8(a, b uint8) uint8 {
	if a > b {
		return a
	}
	return b
}
