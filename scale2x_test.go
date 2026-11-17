package gift

import (
	"image"
	"image/color"
	"math/rand"
	"testing"
)

// scale2xLegend maps the characters used by the fixtures to NRGBA colors.
// Characters with zero alpha have different hidden RGB values; the filter must
// treat all of them as the same transparent color.
var scale2xLegend = map[rune][4]uint8{
	'.': {0x00, 0x00, 0x00, 0x00},
	'R': {0xFF, 0x00, 0x00, 0xFF},
	'G': {0x00, 0xFF, 0x00, 0xFF},
	'B': {0x00, 0x00, 0xFF, 0xFF},
	'a': {0xC8, 0x64, 0x32, 0x00},
	'c': {0x05, 0x06, 0x07, 0x00},
	't': {0x0B, 0x16, 0x21, 0x00},
}

// newScale2xFixture builds an NRGBA image with the given bounds from rows of
// legend characters.
func newScale2xFixture(b image.Rectangle, rows []string) *image.NRGBA {
	img := image.NewNRGBA(b)
	for i := range img.Pix {
		img.Pix[i] = 0xEE // catch any pixel the fixture fails to fill
	}
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			px, ok := scale2xLegend[rune(rows[y][x])]
			if !ok {
				panic("unknown fixture character")
			}
			i := img.PixOffset(b.Min.X+x, b.Min.Y+y)
			img.Pix[i+0] = px[0]
			img.Pix[i+1] = px[1]
			img.Pix[i+2] = px[2]
			img.Pix[i+3] = px[3]
		}
	}
	return img
}

func TestScale2xFixtures(t *testing.T) {
	testData := []struct {
		desc       string
		srcb, dstb image.Rectangle
		srcRows    []string
		dstRows    []string
	}{
		{
			// Source pixel (2,2) is the inner corner where the transparent
			// background meets the L. Its left neighbor is an L arm and its
			// upper neighbor is transparent, so only subpixel E2 takes the
			// left neighbor; E0, E1 and E3 stay transparent.
			"L shape, diagonal corners",
			image.Rect(0, 0, 5, 5),
			image.Rect(0, 0, 10, 10),
			[]string{
				"RR...",
				"RR...",
				"RR...",
				"RRRRR",
				"RRRRR",
			},
			[]string{
				"RRRR......",
				"RRRR......",
				"RRRR......",
				"RRRR......",
				"RRRR......",
				"RRRRR.....",
				"RRRRRRRRRR",
				"RRRRRRRRRR",
				"RRRRRRRRRR",
				"RRRRRRRRRR",
			},
		},
		{
			// Transparent pixels with hidden colors. Interior (1,1) and (1,2)
			// plus the right edge treatment must ignore the hidden RGB of the
			// transparent neighbors, and every transparent output pixel must
			// be transparent black (zeroed by the fixture helper).
			"transparent pixels with hidden colors",
			image.Rect(-2, 1, 1, 4),
			image.Rect(0, 0, 6, 6),
			[]string{
				"GaG",
				"ctR",
				"GRG",
			},
			[]string{
				"GG..GG",
				"G...GG",
				"....RR",
				"...RRR",
				"GGRRRG",
				"GGRRGG",
			},
		},
		{
			// The 45 degree diagonal must stay a hard one pixel wide
			// staircase instead of becoming a blurred line.
			"diagonal line",
			image.Rect(9, -4, 14, 1),
			image.Rect(0, 0, 10, 10),
			[]string{
				"R....",
				".R...",
				"..R..",
				"...R.",
				"....R",
			},
			[]string{
				"RR........",
				"R.R.......",
				".RRR......",
				"..RRR.....",
				"...RRR....",
				"....RRR...",
				".....RRR..",
				"......RRR.",
				".......R.R",
				"........RR",
			},
		},
		{
			"L shape, non-zero destination bounds",
			image.Rect(-2, 3, 3, 8),
			image.Rect(-4, -4, 6, 6),
			[]string{
				"RR...",
				"RR...",
				"RR...",
				"RRRRR",
				"RRRRR",
			},
			[]string{
				"RRRR......",
				"RRRR......",
				"RRRR......",
				"RRRR......",
				"RRRR......",
				"RRRRR.....",
				"RRRRRRRRRR",
				"RRRRRRRRRR",
				"RRRRRRRRRR",
				"RRRRRRRRRR",
			},
		},
		{
			"single pixel",
			image.Rect(7, -3, 8, -2),
			image.Rect(0, 0, 2, 2),
			[]string{"B"},
			[]string{
				"BB",
				"BB",
			},
		},
		{
			"single column",
			image.Rect(0, 0, 1, 3),
			image.Rect(0, 0, 2, 6),
			[]string{"R", "G", "B"},
			[]string{
				"RR",
				"RR",
				"GG",
				"GG",
				"BB",
				"BB",
			},
		},
		{
			"single row",
			image.Rect(0, 0, 3, 1),
			image.Rect(0, 0, 6, 2),
			[]string{"RGB"},
			[]string{
				"RRGGBB",
				"RRGGBB",
			},
		},
	}

	for _, d := range testData {
		src := newScale2xFixture(d.srcb, d.srcRows)
		srcPix := append([]uint8(nil), src.Pix...)

		f := Scale2x()
		b := f.Bounds(src.Bounds())
		if !b.Eq(image.Rect(0, 0, 2*d.srcb.Dx(), 2*d.srcb.Dy())) {
			t.Errorf("test [%s] failed: unexpected bounds %v", d.desc, b)
			continue
		}

		dst := image.NewNRGBA(d.dstb)
		f.Draw(dst, src, nil)

		want := newScale2xFixture(d.dstb, d.dstRows)
		if !checkBoundsAndPix(dst.Bounds(), d.dstb, dst.Pix, want.Pix) {
			t.Errorf("test [%s] failed:\ngot  %v\nwant %v", d.desc, dst.Pix, want.Pix)
		}

		if !checkBoundsAndPix(src.Bounds(), d.srcb, src.Pix, srcPix) {
			t.Errorf("test [%s] failed: source image was modified", d.desc)
		}
	}
}

// scale2xReference is an independent, straightforward implementation of the
// Scale2x rules. It works directly on the source bytes instead of the pixel
// helpers used by the filter.
func scale2xReference(src *image.NRGBA) *image.NRGBA {
	b := src.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, 2*b.Dx(), 2*b.Dy()))
	if b.Empty() {
		return dst
	}

	type cpx struct{ r, g, b, a uint8 }

	at := func(x, y int) cpx {
		if x < b.Min.X {
			x = b.Min.X
		}
		if x > b.Max.X-1 {
			x = b.Max.X - 1
		}
		if y < b.Min.Y {
			y = b.Min.Y
		}
		if y > b.Max.Y-1 {
			y = b.Max.Y - 1
		}
		i := src.PixOffset(x, y)
		px := cpx{src.Pix[i], src.Pix[i+1], src.Pix[i+2], src.Pix[i+3]}
		if px.a == 0 {
			return cpx{}
		}
		return px
	}

	set := func(x, y int, px cpx) {
		i := dst.PixOffset(x, y)
		dst.Pix[i+0] = px.r
		dst.Pix[i+1] = px.g
		dst.Pix[i+2] = px.b
		dst.Pix[i+3] = px.a
	}

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			p, a, bb, cc, d := at(x, y), at(x, y-1), at(x+1, y), at(x-1, y), at(x, y+1)

			e0, e1, e2, e3 := p, p, p, p
			if a != d && cc != bb {
				if cc == a {
					e0 = a
				}
				if a == bb {
					e1 = bb
				}
				if cc == d {
					e2 = cc
				}
				if d == bb {
					e3 = bb
				}
			}

			dx := 2 * (x - b.Min.X)
			dy := 2 * (y - b.Min.Y)
			set(dx, dy, e0)
			set(dx+1, dy, e1)
			set(dx, dy+1, e2)
			set(dx+1, dy+1, e3)
		}
	}

	return dst
}

func TestScale2xReference(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	alphas := []uint8{0x00, 0x00, 0x40, 0x80, 0xFF}

	for w := 1; w <= 6; w++ {
		for h := 1; h <= 6; h++ {
			for iter := 0; iter < 10; iter++ {
				srcb := image.Rect(rnd.Intn(7)-3, rnd.Intn(7)-3, 0, 0)
				srcb.Max = srcb.Min.Add(image.Pt(w, h))
				src := image.NewNRGBA(srcb)
				for i := 0; i < len(src.Pix); i += 4 {
					// Fully transparent pixels get arbitrary hidden colors.
					for j := 0; j < 3; j++ {
						src.Pix[i+j] = uint8(rnd.Intn(4) * 0x40)
					}
					src.Pix[i+3] = alphas[rnd.Intn(len(alphas))]
				}
				if w == 1 && h == 1 && iter == 0 {
					// Make sure at least one fixture has an interesting alpha.
					src.Pix[3] = 0xFF
				}

				want := scale2xReference(src)
				for _, parallel := range []bool{false, true} {
					f := Scale2x()
					dst := image.NewNRGBA(f.Bounds(src.Bounds()))
					f.Draw(dst, src, &Options{Parallelization: parallel})
					if !checkBoundsAndPix(dst.Bounds(), want.Bounds(), dst.Pix, want.Pix) {
						t.Fatalf("reference mismatch (w=%d h=%d parallel=%v)", w, h, parallel)
					}
				}
			}
		}
	}
}

func TestScale2xBounds(t *testing.T) {
	f := Scale2x()
	maxInt := int(^uint(0) >> 1)

	testData := []struct {
		desc string
		srcb image.Rectangle
		want image.Rectangle
	}{
		{"zero", image.Rectangle{}, image.Rect(0, 0, 0, 0)},
		{"empty with position", image.Rect(5, 5, 5, 5), image.Rect(0, 0, 0, 0)},
		// image.Rect canonicalizes inverted corners, so the negative sizes are
		// built directly. A non-canonical rectangle that starts after it ends
		// has a negative width or height.
		{"negative size", image.Rectangle{Min: image.Pt(4, 4), Max: image.Pt(1, 1)}, image.Rect(0, 0, 0, 0)},
		{"negative width", image.Rectangle{Min: image.Pt(4, 0), Max: image.Pt(1, 3)}, image.Rect(0, 0, 0, 0)},
		{"negative height", image.Rectangle{Min: image.Pt(0, 4), Max: image.Pt(3, 1)}, image.Rect(0, 0, 0, 0)},
		{"non-zero min", image.Rect(-5, -5, 0, 0), image.Rect(0, 0, 10, 10)},
		{"max width", image.Rect(0, 0, maxScale2xDimension, 1), image.Rect(0, 0, 2*maxScale2xDimension, 2)},
		{"width overflow", image.Rect(0, 0, maxScale2xDimension+1, 1), image.Rect(0, 0, 0, 0)},
		{"height overflow", image.Rect(0, 0, 1, maxScale2xDimension+1), image.Rect(0, 0, 0, 0)},
		{"max int", image.Rect(0, 0, maxInt, maxInt), image.Rect(0, 0, 0, 0)},
		// A length that itself overflows an int must not be mistaken for a
		// small or negative size.
		{"length overflows int", image.Rectangle{Min: image.Pt(-1, 0), Max: image.Pt(maxInt, 1)}, image.Rect(0, 0, 0, 0)},
		{"height length overflows int", image.Rectangle{Min: image.Pt(0, -1), Max: image.Pt(1, maxInt)}, image.Rect(0, 0, 0, 0)},
	}

	for _, d := range testData {
		got := f.Bounds(d.srcb)
		if !got.Eq(d.want) {
			t.Errorf("test [%s] failed: got %v, want %v", d.desc, got, d.want)
		}
	}
}

func TestScale2xEmptyDraw(t *testing.T) {
	testData := []image.Rectangle{
		{},
		image.Rect(-3, 2, -3, 2),
		{Min: image.Pt(2, 2), Max: image.Pt(0, 0)},
	}

	for _, srcb := range testData {
		// Use a small 1x1 image for the inverted rectangle because
		// image.NewNRGBA refuses to allocate negative dimensions.
		var src *image.NRGBA
		if srcb.Dx() < 0 || srcb.Dy() < 0 {
			src = image.NewNRGBA(image.Rect(0, 0, 1, 1))
			src.Rect = srcb
		} else {
			src = image.NewNRGBA(srcb)
		}
		dst := image.NewNRGBA(image.Rect(-1, -1, 2, 2))
		for i := range dst.Pix {
			dst.Pix[i] = 0xAA
		}
		before := append([]uint8(nil), dst.Pix...)

		Scale2x().Draw(dst, src, nil)

		if !checkBoundsAndPix(dst.Bounds(), image.Rect(-1, -1, 2, 2), dst.Pix, before) {
			t.Errorf("test [%v] failed: dst was modified", srcb)
		}
	}
}

// scale2xAtImage exposes only the generic image.Image interface, so it is read
// through At instead of any of the optimized pixel getter paths.
type scale2xAtImage struct {
	r      image.Rectangle
	colors []color.NRGBA
}

func (m *scale2xAtImage) ColorModel() color.Model { return color.NRGBAModel }

func (m *scale2xAtImage) Bounds() image.Rectangle { return m.r }

func (m *scale2xAtImage) At(x, y int) color.Color {
	if !image.Pt(x, y).In(m.r) {
		return color.NRGBA{}
	}
	return m.colors[(y-m.r.Min.Y)*m.r.Dx()+(x-m.r.Min.X)]
}

func TestScale2xGenericFormat(t *testing.T) {
	srcb := image.Rect(-1, 2, 4, 7)
	srcRows := []string{
		"RR...",
		"RR...",
		"RR...",
		"RRRRR",
		"RRRRR",
	}
	nrgba := newScale2xFixture(srcb, srcRows)

	// Only fully opaque and fully transparent colors are used, so the
	// generic At path and the NRGBA path are expected to agree exactly.
	colors := make([]color.NRGBA, 0, srcb.Dx()*srcb.Dy())
	for y := 0; y < srcb.Dy(); y++ {
		for x := 0; x < srcb.Dx(); x++ {
			i := nrgba.PixOffset(srcb.Min.X+x, srcb.Min.Y+y)
			colors = append(colors, color.NRGBA{
				R: nrgba.Pix[i+0],
				G: nrgba.Pix[i+1],
				B: nrgba.Pix[i+2],
				A: nrgba.Pix[i+3],
			})
		}
	}
	generic := &scale2xAtImage{r: srcb, colors: colors}

	want := image.NewNRGBA(Scale2x().Bounds(srcb))
	Scale2x().Draw(want, nrgba, nil)

	dst := image.NewNRGBA(Scale2x().Bounds(srcb))
	Scale2x().Draw(dst, generic, nil)

	if !checkBoundsAndPix(dst.Bounds(), want.Bounds(), dst.Pix, want.Pix) {
		t.Errorf("generic source format mismatch:\ngot  %v\nwant %v", dst.Pix, want.Pix)
	}

	// image.Alpha is not one of the formats with a fast path either.
	alpha := image.NewAlpha(image.Rect(0, 0, 3, 2))
	alpha.Pix = []uint8{0x00, 0x40, 0xFF, 0x80, 0x20, 0x00}
	dst1 := image.NewNRGBA(Scale2x().Bounds(alpha.Bounds()))
	Scale2x().Draw(dst1, alpha, nil)
	dst2 := image.NewNRGBA(Scale2x().Bounds(alpha.Bounds()))
	Scale2x().Draw(dst2, alpha, nil)
	if !checkBoundsAndPix(dst1.Bounds(), dst2.Bounds(), dst1.Pix, dst2.Pix) {
		t.Errorf("generic source format is not deterministic")
	}
}

func TestScale2xChain(t *testing.T) {
	srcb := image.Rect(1, -2, 4, 1)
	srcRows := []string{
		"R..",
		"RR.",
		"RRR",
	}
	src := newScale2xFixture(srcb, srcRows)

	// Apply Scale2x twice through a filter chain.
	g := New(Scale2x(), Scale2x())
	b := g.Bounds(srcb)
	if !b.Eq(image.Rect(0, 0, 4*srcb.Dx(), 4*srcb.Dy())) {
		t.Fatalf("unexpected chained bounds %v", b)
	}
	dst := image.NewNRGBA(b)
	g.Draw(dst, src)

	tmp := image.NewNRGBA(Scale2x().Bounds(srcb))
	Scale2x().Draw(tmp, src, nil)
	want := image.NewNRGBA(Scale2x().Bounds(tmp.Bounds()))
	Scale2x().Draw(want, tmp, nil)

	if !checkBoundsAndPix(dst.Bounds(), want.Bounds(), dst.Pix, want.Pix) {
		t.Errorf("chained Scale2x mismatch")
	}

	// Scale2x followed by an existing filter.
	g = New(Scale2x(), Invert())
	dst = image.NewNRGBA(g.Bounds(srcb))
	g.Draw(dst, src)

	want = image.NewNRGBA(Scale2x().Bounds(srcb))
	Scale2x().Draw(want, src, nil)
	inv := image.NewNRGBA(image.Rect(0, 0, want.Bounds().Dx(), want.Bounds().Dy()))
	Invert().Draw(inv, want, nil)

	if !checkBoundsAndPix(dst.Bounds(), inv.Bounds(), dst.Pix, inv.Pix) {
		t.Errorf("Scale2x -> Invert chain mismatch")
	}
}

func BenchmarkScale2x(b *testing.B) {
	src := image.NewNRGBA(image.Rect(0, 0, 256, 256))
	for i := range src.Pix {
		src.Pix[i] = uint8(i)
	}
	g := New(Scale2x())
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Draw(dst, src)
	}
}
