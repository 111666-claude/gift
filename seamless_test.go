package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"
)

// seamlessTestImage builds an NRGBA image at the given bounds from a row-major
// slice of colors. The number of colors must match the image size.
func seamlessTestImage(r image.Rectangle, colors []color.NRGBA) *image.NRGBA {
	img := image.NewNRGBA(r)
	if len(colors) != r.Dx()*r.Dy() {
		panic("seamlessTestImage: color count does not match image size")
	}
	for i, c := range colors {
		img.SetNRGBA(r.Min.X+i%r.Dx(), r.Min.Y+i/r.Dx(), c)
	}
	return img
}

func seamlessNRGBAAt(img image.Image, x, y int) color.NRGBA {
	return color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
}

func seamlessAbsDiff(a, b uint8) int {
	if a > b {
		return int(a - b)
	}
	return int(b - a)
}

// seamlessMaxChannelDiff returns the largest per-channel difference of two
// colors, including alpha.
func seamlessMaxChannelDiff(a, b color.NRGBA) int {
	d := seamlessAbsDiff(a.R, b.R)
	d = maxint(d, seamlessAbsDiff(a.G, b.G))
	d = maxint(d, seamlessAbsDiff(a.B, b.B))
	d = maxint(d, seamlessAbsDiff(a.A, b.A))
	return d
}

// fourEdgeImage returns an image whose four edges have distinct colors:
// the left edge is red, the right edge blue, the top edge green and the
// bottom edge yellow. The interior is a neutral gray.
func fourEdgeImage(r image.Rectangle) *image.NRGBA {
	colors := make([]color.NRGBA, 0, r.Dx()*r.Dy())
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			var c color.NRGBA
			switch {
			case x == 0 && y == 0:
				c = color.NRGBA{255, 255, 255, 255} // top-left corner
			case x == r.Dx()-1 && y == 0:
				c = color.NRGBA{255, 255, 0, 255} // top-right corner
			case x == 0 && y == r.Dy()-1:
				c = color.NRGBA{0, 255, 255, 255} // bottom-left corner
			case x == r.Dx()-1 && y == r.Dy()-1:
				c = color.NRGBA{0, 0, 0, 255} // bottom-right corner
			case x == 0:
				c = color.NRGBA{255, 0, 0, 255}
			case x == r.Dx()-1:
				c = color.NRGBA{0, 0, 255, 255}
			case y == 0:
				c = color.NRGBA{0, 255, 0, 255}
			case y == r.Dy()-1:
				c = color.NRGBA{255, 255, 0, 255}
			default:
				c = color.NRGBA{128, 128, 128, 255}
			}
			colors = append(colors, c)
		}
	}
	return seamlessTestImage(r, colors)
}

func TestSeamlessTileBounds(t *testing.T) {
	srcb := image.Rect(-3, 5, 9, 17)
	src := fourEdgeImage(srcb)
	f := SeamlessTile(3, 4, 1)

	if b := f.Bounds(srcb); !b.Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
		t.Fatalf("unexpected filter bounds: %v", b)
	}

	g := New(f)
	if b := g.Bounds(srcb); !b.Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
		t.Fatalf("unexpected gift bounds: %v", b)
	}

	// The destination must keep its own bounds and receive the result at its
	// own origin regardless of the source origin.
	dstb := image.Rect(100, -50, 100+srcb.Dx(), -50+srcb.Dy())
	dst := image.NewNRGBA(dstb)
	g.Draw(dst, src)
	if !dst.Bounds().Eq(dstb) {
		t.Fatalf("destination bounds changed: %v", dst.Bounds())
	}

	// The source must not be modified.
	want := fourEdgeImage(srcb)
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			if got, want := src.NRGBAAt(x, y), want.NRGBAAt(x, y); got != want {
				t.Fatalf("source modified at (%d,%d): got %v want %v", x, y, got, want)
			}
		}
	}
}

func TestSeamlessTileSeams(t *testing.T) {
	// With four distinct edge colors, the opposite edges must converge to the
	// same color so that adjacent copies join without a jump.
	srcb := image.Rect(0, 0, 16, 12)
	src := fourEdgeImage(srcb)
	g := New(SeamlessTile(4, 4, 1))
	dst := image.NewNRGBA(g.Bounds(srcb))
	g.Draw(dst, src)

	for y := 0; y < srcb.Dy(); y++ {
		left := seamlessNRGBAAt(dst, 0, y)
		right := seamlessNRGBAAt(dst, srcb.Dx()-1, y)
		if d := seamlessMaxChannelDiff(left, right); d != 0 {
			t.Errorf("horizontal seam row %d: left %v right %v differ by %d", y, left, right, d)
		}
	}
	for x := 0; x < srcb.Dx(); x++ {
		top := seamlessNRGBAAt(dst, x, 0)
		bottom := seamlessNRGBAAt(dst, x, srcb.Dy()-1)
		if d := seamlessMaxChannelDiff(top, bottom); d != 0 {
			t.Errorf("vertical seam column %d: top %v bottom %v differ by %d", x, top, bottom, d)
		}
	}

	// All four corners must converge to the same color at full strength.
	corners := []color.NRGBA{
		seamlessNRGBAAt(dst, 0, 0),
		seamlessNRGBAAt(dst, srcb.Dx()-1, 0),
		seamlessNRGBAAt(dst, 0, srcb.Dy()-1),
		seamlessNRGBAAt(dst, srcb.Dx()-1, srcb.Dy()-1),
	}
	for i := 1; i < len(corners); i++ {
		if d := seamlessMaxChannelDiff(corners[0], corners[i]); d != 0 {
			t.Errorf("corner %d %v differs from corner 0 %v by %d", i, corners[i], corners[0], d)
		}
	}

	// Tile two copies horizontally and vertically; the junction must be
	// continuous, i.e. no pair across the tile boundary may jump more than the
	// pairs found inside the tile next to it.
	tiled := image.NewNRGBA(image.Rect(0, 0, srcb.Dx()*2, srcb.Dy()*2))
	for ty := 0; ty < 2; ty++ {
		for tx := 0; tx < 2; tx++ {
			pt := image.Pt(tx*srcb.Dx(), ty*srcb.Dy())
			draw.Draw(tiled, image.Rect(pt.X, pt.Y, pt.X+srcb.Dx(), pt.Y+srcb.Dy()), dst, image.Point{}, draw.Src)
		}
	}
	for y := 0; y < tiled.Bounds().Dy(); y++ {
		if c0, c1 := seamlessNRGBAAt(tiled, srcb.Dx()-1, y), seamlessNRGBAAt(tiled, srcb.Dx(), y); seamlessMaxChannelDiff(c0, c1) != 0 {
			t.Errorf("tiled horizontal junction row %d: %v | %v", y, c0, c1)
		}
	}
	for x := 0; x < tiled.Bounds().Dx(); x++ {
		if c0, c1 := seamlessNRGBAAt(tiled, x, srcb.Dy()-1), seamlessNRGBAAt(tiled, x, srcb.Dy()); seamlessMaxChannelDiff(c0, c1) != 0 {
			t.Errorf("tiled vertical junction column %d: %v | %v", x, c0, c1)
		}
	}
}

func TestSeamlessTileCenterFidelity(t *testing.T) {
	srcb := image.Rect(0, 0, 20, 20)
	colors := make([]color.NRGBA, 0, srcb.Dx()*srcb.Dy())
	for y := 0; y < srcb.Dy(); y++ {
		for x := 0; x < srcb.Dx(); x++ {
			colors = append(colors, color.NRGBA{uint8(x * 12), uint8(y * 12), uint8((x + y) * 6), 255})
		}
	}
	src := seamlessTestImage(srcb, colors)

	// The transition bands are 4 pixels wide, so the central 12x12 area must
	// stay untouched.
	g := New(SeamlessTile(4, 4, 1))
	dst := image.NewNRGBA(g.Bounds(srcb))
	g.Draw(dst, src)

	for y := 4; y < srcb.Dy()-4; y++ {
		for x := 4; x < srcb.Dx()-4; x++ {
			if got, want := seamlessNRGBAAt(dst, x, y), src.NRGBAAt(x, y); got != want {
				t.Fatalf("center pixel (%d,%d) changed: got %v want %v", x, y, got, want)
			}
		}
	}
}

func TestSeamlessTileAlpha(t *testing.T) {
	// The two opposite edges are fully transparent but carry hidden colors
	// (green on the left, red on the right). The neighbouring opaque pixels
	// are blue. Premultiplied blending must not pull the hidden colors into
	// the visible result and must not create a dark fringe.
	r := image.Rect(0, 0, 8, 4)
	colors := make([]color.NRGBA, 0, r.Dx()*r.Dy())
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			switch {
			case x == 0:
				colors = append(colors, color.NRGBA{0, 255, 0, 0}) // hidden green
			case x == r.Dx()-1:
				colors = append(colors, color.NRGBA{255, 0, 0, 0}) // hidden red
			default:
				colors = append(colors, color.NRGBA{0, 0, 255, 255})
			}
		}
	}
	src := seamlessTestImage(r, colors)
	g := New(SeamlessTile(2, 0, 1))
	dst := image.NewNRGBA(g.Bounds(r))
	g.Draw(dst, src)

	for y := 0; y < r.Dy(); y++ {
		// Both edge pixels are a blend of two fully transparent pixels.
		for _, x := range []int{0, r.Dx() - 1} {
			if got := seamlessNRGBAAt(dst, x, y); got.A != 0 {
				t.Errorf("edge pixel (%d,%d) must stay transparent, got %v", x, y, got)
			}
		}
		// The pixels next to the edges are opaque blue and must stay blue.
		for _, x := range []int{1, r.Dx() - 2} {
			c := seamlessNRGBAAt(dst, x, y)
			if c.A != 255 || c.R != 0 || c.G != 0 || c.B != 255 {
				t.Errorf("pixel (%d,%d) changed color: got %v want opaque blue", x, y, c)
			}
		}
	}

	// A half transparent edge blended with the opposite edge must keep a
	// consistent alpha and a color derived only from the visible contributions.
	r2 := image.Rect(0, 0, 6, 2)
	colors2 := make([]color.NRGBA, 0, r2.Dx()*r2.Dy())
	for y := 0; y < r2.Dy(); y++ {
		for x := 0; x < r2.Dx(); x++ {
			switch x {
			case 0:
				colors2 = append(colors2, color.NRGBA{255, 0, 0, 128})
			case r2.Dx() - 1:
				colors2 = append(colors2, color.NRGBA{0, 0, 255, 128})
			default:
				colors2 = append(colors2, color.NRGBA{0, 0, 0, 0})
			}
		}
	}
	src2 := seamlessTestImage(r2, colors2)
	g2 := New(SeamlessTile(1, 0, 1))
	dst2 := image.NewNRGBA(g2.Bounds(r2))
	g2.Draw(dst2, src2)

	for y := 0; y < r2.Dy(); y++ {
		left := seamlessNRGBAAt(dst2, 0, y)
		right := seamlessNRGBAAt(dst2, r2.Dx()-1, y)
		if left != right {
			t.Errorf("half transparent seam row %d: left %v right %v", y, left, right)
		}
		// Both edges must be a 50/50 mix of the two visible edge colors.
		if left.A != 128 {
			t.Errorf("half transparent seam alpha: got %d want 128", left.A)
		}
		if left.R != 128 || left.B != 128 {
			t.Errorf("half transparent seam color: got %v want purple", left)
		}
	}
}

func TestSeamlessTileSingleAxis(t *testing.T) {
	srcb := image.Rect(0, 0, 10, 8)
	src := fourEdgeImage(srcb)
	midX := srcb.Dx() / 2
	midY := srcb.Dy() / 2

	// Horizontal only: the pixels outside the left/right transition bands must
	// be copied for every row, and the vertical variation must stay intact.
	g := New(SeamlessTile(3, 0, 1))
	dst := image.NewNRGBA(g.Bounds(srcb))
	g.Draw(dst, src)
	for y := 0; y < srcb.Dy(); y++ {
		for x := 3; x < srcb.Dx()-3; x++ {
			if got, want := seamlessNRGBAAt(dst, x, y), src.NRGBAAt(x, y); got != want {
				t.Errorf("horizontal only: interior pixel (%d,%d) changed: got %v want %v", x, y, got, want)
			}
		}
	}
	for x := 3; x < srcb.Dx()-3; x++ {
		if got, want := seamlessNRGBAAt(dst, x, 0), src.NRGBAAt(x, 0); got != want {
			t.Errorf("horizontal only: top row pixel %d changed: got %v want %v", x, got, want)
		}
		if got, want := seamlessNRGBAAt(dst, x, srcb.Dy()-1), src.NRGBAAt(x, srcb.Dy()-1); got != want {
			t.Errorf("horizontal only: bottom row pixel %d changed: got %v want %v", x, got, want)
		}
	}
	for y := 0; y < srcb.Dy(); y++ {
		left := seamlessNRGBAAt(dst, 0, y)
		right := seamlessNRGBAAt(dst, srcb.Dx()-1, y)
		if d := seamlessMaxChannelDiff(left, right); d != 0 {
			t.Errorf("horizontal only: seam row %d differs by %d", y, d)
		}
	}
	// The bottom row must still differ from the top row, otherwise the filter
	// would have processed both axes.
	if d := seamlessMaxChannelDiff(seamlessNRGBAAt(dst, midX, 0), seamlessNRGBAAt(dst, midX, srcb.Dy()-1)); d == 0 {
		t.Errorf("horizontal only: vertical variation was removed")
	}

	// Vertical only: the pixels outside the top/bottom transition bands must be
	// copied for every column, and the horizontal variation must stay intact.
	g = New(SeamlessTile(0, 3, 1))
	dst = image.NewNRGBA(g.Bounds(srcb))
	g.Draw(dst, src)
	for y := 3; y < srcb.Dy()-3; y++ {
		for x := 0; x < srcb.Dx(); x++ {
			if got, want := seamlessNRGBAAt(dst, x, y), src.NRGBAAt(x, y); got != want {
				t.Errorf("vertical only: interior pixel (%d,%d) changed: got %v want %v", x, y, got, want)
			}
		}
	}
	for y := 3; y < srcb.Dy()-3; y++ {
		if got, want := seamlessNRGBAAt(dst, 0, y), src.NRGBAAt(0, y); got != want {
			t.Errorf("vertical only: left column pixel %d changed: got %v want %v", y, got, want)
		}
		if got, want := seamlessNRGBAAt(dst, srcb.Dx()-1, y), src.NRGBAAt(srcb.Dx()-1, y); got != want {
			t.Errorf("vertical only: right column pixel %d changed: got %v want %v", y, got, want)
		}
	}
	for x := 0; x < srcb.Dx(); x++ {
		top := seamlessNRGBAAt(dst, x, 0)
		bottom := seamlessNRGBAAt(dst, x, srcb.Dy()-1)
		if d := seamlessMaxChannelDiff(top, bottom); d != 0 {
			t.Errorf("vertical only: seam column %d differs by %d", x, d)
		}
	}
	if d := seamlessMaxChannelDiff(seamlessNRGBAAt(dst, 0, midY), seamlessNRGBAAt(dst, srcb.Dx()-1, midY)); d == 0 {
		t.Errorf("vertical only: horizontal variation was removed")
	}
}

func TestSeamlessTileStrength(t *testing.T) {
	// A single pixel wide left/right edge with values 0 and 255. Strength 1
	// must converge the edges, strength 0 must copy the image, and a partial
	// strength must reduce the seam proportionally.
	newSrc := func() *image.NRGBA {
		r := image.Rect(0, 0, 4, 1)
		return seamlessTestImage(r, []color.NRGBA{
			{0, 0, 0, 255},
			{10, 10, 10, 255},
			{20, 20, 20, 255},
			{255, 255, 255, 255},
		})
	}

	g := New(SeamlessTile(1, 0, 1))
	dst := image.NewNRGBA(g.Bounds(image.Rect(0, 0, 4, 1)))
	g.Draw(dst, newSrc())
	if d := seamlessMaxChannelDiff(seamlessNRGBAAt(dst, 0, 0), seamlessNRGBAAt(dst, 3, 0)); d != 0 {
		t.Errorf("strength 1 seam diff: got %d want 0", d)
	}

	g = New(SeamlessTile(1, 0, 0.5))
	dst = image.NewNRGBA(g.Bounds(image.Rect(0, 0, 4, 1)))
	g.Draw(dst, newSrc())
	// Each edge moves a quarter of the way towards the opposite edge, so the
	// remaining jump is half of the original 255.
	d := seamlessAbsDiff(seamlessNRGBAAt(dst, 0, 0).R, seamlessNRGBAAt(dst, 3, 0).R)
	if d != 127 && d != 128 {
		t.Errorf("strength 0.5 seam diff: got %d want 127 or 128", d)
	}

	// strength 0 returns an unchanged copy.
	g = New(SeamlessTile(1, 0, 0))
	dst = image.NewNRGBA(g.Bounds(image.Rect(0, 0, 4, 1)))
	g.Draw(dst, newSrc())
	for x := 0; x < 4; x++ {
		if got, want := seamlessNRGBAAt(dst, x, 0), newSrc().NRGBAAt(x, 0); got != want {
			t.Errorf("strength 0: pixel %d changed: got %v want %v", x, got, want)
		}
	}
}

func TestSeamlessTileInvalidParameters(t *testing.T) {
	srcb := image.Rect(0, 0, 8, 6)
	src := fourEdgeImage(srcb)

	// Non-finite strengths must yield deterministic results and never
	// produce non-finite pixels.
	for _, strength := range []float32{
		0, -1, 1, 2,
		float32(math.NaN()),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
	} {
		f := SeamlessTile(3, 3, strength)
		if b := f.Bounds(srcb); !b.Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
			t.Errorf("strength %v: unexpected bounds %v", strength, b)
		}
		g := New(f)
		dst := image.NewNRGBA(g.Bounds(srcb))
		g.Draw(dst, src)
		for y := 0; y < srcb.Dy(); y++ {
			for x := 0; x < srcb.Dx(); x++ {
				c := seamlessNRGBAAt(dst, x, y)
				if c.A > 0 {
					// Opaque input must stay opaque.
					if c.A != 255 {
						t.Errorf("strength %v: unexpected alpha %d at (%d,%d)", strength, c.A, x, y)
					}
				}
			}
		}
	}

	// NaN strength must behave like zero.
	nan := SeamlessTile(2, 2, float32(math.NaN()))
	zero := SeamlessTile(2, 2, 0)
	if _, ok := nan.(*copyimageFilter); !ok {
		t.Errorf("NaN strength must produce a copy filter, got %T", nan)
	}
	if _, ok := zero.(*copyimageFilter); !ok {
		t.Errorf("zero strength must produce a copy filter, got %T", zero)
	}

	// Negative widths disable the corresponding axis.
	f := SeamlessTile(-5, -5, 1)
	if _, ok := f.(*copyimageFilter); !ok {
		t.Errorf("negative widths must produce a copy filter, got %T", f)
	}
	f = SeamlessTile(-5, 2, 1)
	sf, ok := f.(*seamlessTileFilter)
	if !ok {
		t.Fatalf("expected seamlessTileFilter, got %T", f)
	}
	if sf.horizontalWidth != 0 || sf.verticalWidth != 2 {
		t.Errorf("unexpected filter widths: %d, %d", sf.horizontalWidth, sf.verticalWidth)
	}

	// Widths larger than half of the image are clamped, not rejected.
	g := New(SeamlessTile(1000, 1000, 1))
	dst := image.NewNRGBA(g.Bounds(srcb))
	g.Draw(dst, src)
	if !dst.Bounds().Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
		t.Errorf("oversized widths: unexpected bounds %v", dst.Bounds())
	}
	for y := 0; y < srcb.Dy(); y++ {
		left := seamlessNRGBAAt(dst, 0, y)
		right := seamlessNRGBAAt(dst, srcb.Dx()-1, y)
		if d := seamlessMaxChannelDiff(left, right); d != 0 {
			t.Errorf("oversized widths: seam row %d differs by %d", y, d)
		}
	}
}

func TestSeamlessTileSmallImages(t *testing.T) {
	for _, r := range []image.Rectangle{
		{},
		image.Rect(0, 0, 0, 5),
		image.Rect(0, 0, 5, 0),
		image.Rect(0, 0, 1, 1),
		image.Rect(0, 0, 1, 7),
		image.Rect(0, 0, 7, 1),
		image.Rect(0, 0, 2, 2),
		image.Rect(0, 0, 3, 3),
	} {
		src := image.NewNRGBA(r)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				src.SetNRGBA(x, y, color.NRGBA{uint8((x * 37) % 256), uint8((y * 53) % 256), 77, 255})
			}
		}
		f := SeamlessTile(4, 4, 1)
		dst := image.NewNRGBA(f.Bounds(r))
		f.Draw(dst, src, nil)
		if !dst.Bounds().Eq(image.Rect(0, 0, r.Dx(), r.Dy())) {
			t.Errorf("bounds %v: got %v", r, dst.Bounds())
		}
		// An image with at most one pixel on both axes has nothing to blend and
		// must be copied.
		if r.Dx() <= 1 && r.Dy() <= 1 {
			for y := 0; y < r.Dy(); y++ {
				for x := 0; x < r.Dx(); x++ {
					if got, want := seamlessNRGBAAt(dst, x, y), src.NRGBAAt(x, y); got != want {
						t.Errorf("bounds %v: pixel (%d,%d) changed: got %v want %v", r, x, y, got, want)
					}
				}
			}
		}
	}
}

func TestSeamlessTileFilterChain(t *testing.T) {
	srcb := image.Rect(0, 0, 12, 8)
	src := fourEdgeImage(srcb)

	// SeamlessTile must be usable before and after other filters.
	g := New(
		Grayscale(),
		SeamlessTile(2, 2, 1),
		Invert(),
	)
	dst := image.NewNRGBA(g.Bounds(srcb))
	g.Draw(dst, src)
	if !dst.Bounds().Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
		t.Fatalf("unexpected bounds: %v", dst.Bounds())
	}

	// The seam must still be continuous after the following filter.
	for y := 0; y < srcb.Dy(); y++ {
		left := seamlessNRGBAAt(dst, 0, y)
		right := seamlessNRGBAAt(dst, srcb.Dx()-1, y)
		if d := seamlessMaxChannelDiff(left, right); d != 0 {
			t.Errorf("filter chain seam row %d differs by %d", y, d)
		}
	}
}

func TestSeamlessTileParallel(t *testing.T) {
	r := image.Rect(-2, -3, 30, 22)
	src := fourEdgeImage(r)
	var results [2]*image.NRGBA
	for i, parallel := range []bool{false, true} {
		g := New(SeamlessTile(5, 4, 1))
		g.SetParallelization(parallel)
		dst := image.NewNRGBA(g.Bounds(r))
		g.Draw(dst, src)
		results[i] = dst
	}
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			a := results[0].NRGBAAt(r.Min.X+x, r.Min.Y+y)
			b := results[1].NRGBAAt(r.Min.X+x, r.Min.Y+y)
			if a != b {
				t.Fatalf("parallel mismatch at (%d,%d): %v vs %v", x, y, a, b)
			}
		}
	}
}

func TestSeamlessTilePixelFormats(t *testing.T) {
	r := image.Rect(-1, 2, 7, 8)
	srcNRGBA := fourEdgeImage(r)
	g := New(SeamlessTile(2, 2, 1))
	want := image.NewNRGBA(g.Bounds(r))
	g.Draw(want, srcNRGBA)

	// The same pixels converted to other formats must produce the same result
	// within the quantization error of each format.
	toRGBA := image.NewRGBA(r)
	draw.Draw(toRGBA, r, srcNRGBA, r.Min, draw.Src)
	toNRGBA64 := image.NewNRGBA64(r)
	draw.Draw(toNRGBA64, r, srcNRGBA, r.Min, draw.Src)

	type variant struct {
		name string
		src  image.Image
		dst  draw.Image
		tol  int
	}
	variants := []variant{
		{"RGBA", toRGBA, image.NewRGBA(g.Bounds(r)), 1},
		{"NRGBA64", toNRGBA64, image.NewNRGBA64(g.Bounds(r)), 1},
	}
	for _, v := range variants {
		g.Draw(v.dst, v.src)
		for y := 0; y < r.Dy(); y++ {
			for x := 0; x < r.Dx(); x++ {
				got := seamlessNRGBAAt(v.dst, x, y)
				wantpx := want.NRGBAAt(x, y)
				if d := seamlessMaxChannelDiff(got, wantpx); d > v.tol {
					t.Errorf("%s: pixel (%d,%d) differs by %d: got %v want %v", v.name, x, y, d, got, wantpx)
				}
			}
		}
	}
}
