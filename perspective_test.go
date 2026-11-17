package gift

import (
	"bytes"
	"image"
	"image/color"
	"math"
	"testing"
)

func TestPerspectiveErrors(t *testing.T) {
	testData := []struct {
		desc           string
		tl, tr, br, bl Point
		err            error
	}{
		{
			"nan coordinate",
			Point{math.NaN(), 0}, Point{10, 0}, Point{10, 10}, Point{0, 10},
			ErrPerspectiveNonFinitePoint,
		},
		{
			"positive infinity coordinate",
			Point{0, 0}, Point{math.Inf(1), 0}, Point{10, 10}, Point{0, 10},
			ErrPerspectiveNonFinitePoint,
		},
		{
			"negative infinity coordinate",
			Point{0, 0}, Point{10, 0}, Point{10, math.Inf(-1)}, Point{0, 10},
			ErrPerspectiveNonFinitePoint,
		},
		{
			"collinear points",
			Point{0, 0}, Point{1, 0}, Point{2, 0}, Point{3, 0},
			ErrPerspectiveDegenerateQuad,
		},
		{
			"three collinear points",
			Point{0, 0}, Point{5, 5}, Point{10, 10}, Point{0, 10},
			ErrPerspectiveDegenerateQuad,
		},
		{
			"coincident points",
			Point{0, 0}, Point{0, 0}, Point{10, 10}, Point{0, 10},
			ErrPerspectiveDegenerateQuad,
		},
		{
			"all points identical",
			Point{5, 5}, Point{5, 5}, Point{5, 5}, Point{5, 5},
			ErrPerspectiveDegenerateQuad,
		},
		{
			"self-intersecting bowtie",
			Point{0, 0}, Point{10, 10}, Point{10, 0}, Point{0, 10},
			ErrPerspectiveSelfIntersectingQuad,
		},
		{
			"self-intersecting swapped",
			Point{0, 0}, Point{10, 0}, Point{0, 10}, Point{10, 10},
			ErrPerspectiveSelfIntersectingQuad,
		},
		{
			"huge output bounds",
			Point{0, 0}, Point{1e7, 0}, Point{1e7, 1e7}, Point{0, 1e7},
			ErrPerspectiveOutputTooLarge,
		},
		{
			"valid quad",
			Point{0, 0}, Point{10, 0}, Point{10, 10}, Point{0, 10},
			nil,
		},
		{
			"valid concave quad",
			Point{0, 0}, Point{10, 0}, Point{5, 3}, Point{0, 10},
			nil,
		},
		{
			"valid tiny quad",
			Point{0, 0}, Point{1, 0}, Point{1, 1}, Point{0, 1},
			nil,
		},
	}

	for _, d := range testData {
		f, err := Perspective(d.tl, d.tr, d.br, d.bl, NearestNeighborInterpolation)
		if err != d.err {
			t.Errorf("test [%s]: expected error %v got %v", d.desc, d.err, err)
		}
		if d.err != nil && f != nil {
			t.Errorf("test [%s]: expected nil filter on error", d.desc)
		}
	}
}

func TestPerspectiveBounds(t *testing.T) {
	testData := []struct {
		desc           string
		tl, tr, br, bl Point
		srcb           image.Rectangle
		dstb           image.Rectangle
	}{
		{
			"axis-aligned quad",
			Point{2, 3}, Point{10, 3}, Point{10, 9}, Point{2, 9},
			image.Rect(0, 0, 4, 4),
			image.Rect(2, 3, 10, 9),
		},
		{
			"fractional corners",
			Point{0.5, 0.5}, Point{4.5, 0.5}, Point{4.5, 4.5}, Point{0.5, 4.5},
			image.Rect(0, 0, 4, 4),
			image.Rect(0, 0, 5, 5),
		},
		{
			"negative coordinates",
			Point{-4, -4}, Point{2, -4}, Point{2, 2}, Point{-4, 2},
			image.Rect(0, 0, 4, 4),
			image.Rect(-4, -4, 2, 2),
		},
		{
			"empty source",
			Point{0, 0}, Point{4, 0}, Point{4, 4}, Point{0, 4},
			image.Rect(0, 0, 0, 0),
			image.Rect(0, 0, 0, 0),
		},
	}

	for _, d := range testData {
		f, err := Perspective(d.tl, d.tr, d.br, d.bl, NearestNeighborInterpolation)
		if err != nil {
			t.Errorf("test [%s]: unexpected error %v", d.desc, err)
			continue
		}
		if b := f.Bounds(d.srcb); !b.Eq(d.dstb) {
			t.Errorf("test [%s]: expected bounds %v got %v", d.desc, d.dstb, b)
		}
	}
}

func TestPerspectiveIdentity(t *testing.T) {
	src := image.NewGray(image.Rect(-1, -1, 3, 3))
	src.Pix = []uint8{
		0x10, 0x20, 0x30, 0x40,
		0x50, 0x60, 0x70, 0x80,
		0x90, 0xa0, 0xb0, 0xc0,
		0xd0, 0xe0, 0xf0, 0x0f,
	}
	srcPixBackup := append([]uint8(nil), src.Pix...)

	interpolations := []struct {
		name string
		i    Interpolation
	}{
		{"nearest", NearestNeighborInterpolation},
		{"linear", LinearInterpolation},
		{"cubic", CubicInterpolation},
	}

	for _, interp := range interpolations {
		// The source rectangle corners map to themselves: identity transform.
		f, err := Perspective(Point{0, 0}, Point{4, 0}, Point{4, 4}, Point{0, 4}, interp.i)
		if err != nil {
			t.Fatalf("identity %s: unexpected error %v", interp.name, err)
		}
		dst := image.NewGray(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)

		if !checkBoundsAndPix(dst.Bounds(), image.Rect(0, 0, 4, 4), dst.Pix, srcPixBackup) {
			t.Errorf("identity %s failed: %#v, %#v", interp.name, dst.Bounds(), dst.Pix)
		}
	}

	if !bytes.Equal(src.Pix, srcPixBackup) {
		t.Error("identity: source image was modified")
	}
}

func TestPerspectiveCheckerboard(t *testing.T) {
	// 4x4 checkerboard of 2x2 blocks, magnified 2x by an affine perspective.
	src := image.NewGray(image.Rect(0, 0, 4, 4))
	src.Pix = []uint8{
		0x10, 0x10, 0x20, 0x20,
		0x10, 0x10, 0x20, 0x20,
		0x30, 0x30, 0x40, 0x40,
		0x30, 0x30, 0x40, 0x40,
	}
	expected := image.NewGray(image.Rect(0, 0, 8, 8))
	expected.Pix = []uint8{
		0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20,
		0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20,
		0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20,
		0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20,
		0x30, 0x30, 0x30, 0x30, 0x40, 0x40, 0x40, 0x40,
		0x30, 0x30, 0x30, 0x30, 0x40, 0x40, 0x40, 0x40,
		0x30, 0x30, 0x30, 0x30, 0x40, 0x40, 0x40, 0x40,
		0x30, 0x30, 0x30, 0x30, 0x40, 0x40, 0x40, 0x40,
	}

	f, err := Perspective(Point{0, 0}, Point{8, 0}, Point{8, 8}, Point{0, 8}, NearestNeighborInterpolation)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	dst := image.NewGray(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	if !checkBoundsAndPix(dst.Bounds(), expected.Bounds(), dst.Pix, expected.Pix) {
		t.Errorf("checkerboard failed: %#v, %#v", dst.Bounds(), dst.Pix)
	}
}

func TestPerspectiveTrapezoid(t *testing.T) {
	// 8x8 source with distinct quadrants.
	src := image.NewGray(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			var v uint8
			switch {
			case x < 4 && y < 4:
				v = 0x10
			case x >= 4 && y < 4:
				v = 0x20
			case x >= 4 && y >= 4:
				v = 0x40
			default:
				v = 0x30
			}
			src.Pix[y*8+x] = v
		}
	}

	// A true perspective (non-affine) trapezoid: billboard tilted away.
	f, err := Perspective(Point{2, 0}, Point{6, 0}, Point{8, 8}, Point{0, 8}, NearestNeighborInterpolation)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	dst := image.NewGray(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	if !dst.Bounds().Eq(image.Rect(0, 0, 8, 8)) {
		t.Fatalf("expected bounds (0,0,8,8) got %v", dst.Bounds())
	}

	at := func(x, y int) uint8 { return dst.Pix[y*8+x] }

	// Pixels outside the destination quad are transparent (zero).
	for _, pt := range []image.Point{{0, 0}, {7, 0}, {0, 3}} {
		if at(pt.X, pt.Y) != 0x00 {
			t.Errorf("expected transparent pixel at %v got %#02x", pt, at(pt.X, pt.Y))
		}
	}

	// Hand-computed interior samples (inverse homography, nearest neighbor).
	expected := []struct {
		x, y int
		v    uint8
	}{
		{2, 0, 0x10}, // near top-left corner  -> src (1, 0)
		{5, 0, 0x20}, // near top-right corner -> src (6, 0)
		{7, 7, 0x40}, // near bottom-right     -> src (7, 7)
		{0, 7, 0x30}, // near bottom-left      -> src (0, 7)
		{4, 2, 0x20}, // interior              -> src (4, 3)
		{3, 5, 0x30}, // interior              -> src (3, 6)
		{4, 5, 0x40}, // interior              -> src (4, 6)
		{1, 3, 0x30}, // interior              -> src (0, 4)
	}
	for _, e := range expected {
		if at(e.x, e.y) != e.v {
			t.Errorf("expected %#02x at (%d, %d) got %#02x", e.v, e.x, e.y, at(e.x, e.y))
		}
	}
}

func TestPerspectiveAlphaInterpolation(t *testing.T) {
	// 2x2 semi-transparent source stretched 2x horizontally with linear
	// interpolation to verify premultiplied alpha/color blending.
	src := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	src.SetNRGBA(0, 0, color.NRGBA{200, 0, 0, 128})
	src.SetNRGBA(1, 0, color.NRGBA{0, 0, 100, 255})
	src.SetNRGBA(0, 1, color.NRGBA{200, 0, 0, 128})
	src.SetNRGBA(1, 1, color.NRGBA{0, 0, 100, 255})
	srcPixBackup := append([]uint8(nil), src.Pix...)

	f, err := Perspective(Point{0, 0}, Point{4, 0}, Point{4, 2}, Point{0, 2}, LinearInterpolation)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	if !dst.Bounds().Eq(image.Rect(0, 0, 4, 2)) {
		t.Fatalf("expected bounds (0,0,4,2) got %v", dst.Bounds())
	}

	type px struct{ r, g, b, a uint8 }
	expected := []struct {
		x, y int
		c    px
	}{
		{0, 0, px{200, 0, 0, 96}},   // 0.75 * src(0,0) + 0.25 * transparent
		{1, 0, px{120, 0, 40, 160}}, // 0.75 * src(0,0) + 0.25 * src(1,0)
		{2, 0, px{29, 0, 86, 223}},  // 0.25 * src(0,0) + 0.75 * src(1,0)
		{3, 0, px{0, 0, 100, 191}},  // 0.75 * src(1,0) + 0.25 * transparent
		{0, 1, px{200, 0, 0, 96}},
		{1, 1, px{120, 0, 40, 160}},
		{2, 1, px{29, 0, 86, 223}},
		{3, 1, px{0, 0, 100, 191}},
	}
	approx := func(a, b uint8) bool {
		d := int(a) - int(b)
		return d >= -1 && d <= 1
	}
	for _, e := range expected {
		c := dst.NRGBAAt(e.x, e.y)
		if !approx(c.R, e.c.r) || !approx(c.G, e.c.g) || !approx(c.B, e.c.b) || !approx(c.A, e.c.a) {
			t.Errorf("at (%d, %d): expected %v got %v", e.x, e.y, e.c, c)
		}
	}

	if !bytes.Equal(src.Pix, srcPixBackup) {
		t.Error("source image was modified")
	}
}

func TestPerspectiveChain(t *testing.T) {
	src := image.NewGray(image.Rect(0, 0, 4, 4))
	src.Pix = []uint8{
		0x10, 0x20, 0x30, 0x40,
		0x50, 0x60, 0x70, 0x80,
		0x90, 0xa0, 0xb0, 0xc0,
		0xd0, 0xe0, 0xf0, 0x0f,
	}
	srcPixBackup := append([]uint8(nil), src.Pix...)

	pf, err := Perspective(Point{0, 0}, Point{4, 0}, Point{4, 4}, Point{0, 4}, NearestNeighborInterpolation)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	// Perspective chained with other filters: identity perspective followed by
	// a horizontal flip must equal a plain horizontal flip.
	g := New(pf, FlipHorizontal())
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	expected := []uint8{
		0x40, 0x30, 0x20, 0x10,
		0x80, 0x70, 0x60, 0x50,
		0xc0, 0xb0, 0xa0, 0x90,
		0x0f, 0xf0, 0xe0, 0xd0,
	}
	if !checkBoundsAndPix(dst.Bounds(), image.Rect(0, 0, 4, 4), dst.Pix, expected) {
		t.Errorf("chain failed: %#v, %#v", dst.Bounds(), dst.Pix)
	}

	if !bytes.Equal(src.Pix, srcPixBackup) {
		t.Error("source image was modified")
	}
}
