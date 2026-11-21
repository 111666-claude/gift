package gift

import (
	"image"
	"image/color"
	"math"
	"reflect"
	"testing"
)

// newComponentsImage creates an NRGBA image with the given bounds. The alpha
// values are given as a string grid where '.' is fully transparent and '0'-'f'
// is a hexadecimal alpha value (0-15 scaled to 0-255).
func newComponentsImage(b image.Rectangle, rows []string) *image.NRGBA {
	img := image.NewNRGBA(b)
	for y, row := range rows {
		if len(row) != b.Dx() {
			panic("bad row length")
		}
		for x, ch := range row {
			var a uint8
			switch ch {
			case '.':
				a = 0
			case 'f':
				a = 255
			default:
				if ch < '0' || ch > 'e' {
					panic("bad alpha char")
				}
				a = uint8(ch-'0') * 17
			}
			i := img.PixOffset(b.Min.X+x, b.Min.Y+y)
			img.Pix[i+0] = 200
			img.Pix[i+1] = 100
			img.Pix[i+2] = 50
			img.Pix[i+3] = a
		}
	}
	return img
}

func TestGetAlpha16(t *testing.T) {
	testData := []struct {
		desc string
		img  image.Image
		want [4]uint16
	}{
		{
			"nrgba",
			&image.NRGBA{Pix: []uint8{0, 0, 0, 0x00, 0, 0, 0, 0x80, 0, 0, 0, 0xff},
				Stride: 4, Rect: image.Rect(0, 0, 3, 1)},
			[4]uint16{0x0000, 0x8080, 0xffff, 0},
		},
		{
			"rgba",
			&image.RGBA{Pix: []uint8{0, 0, 0, 0x00, 0, 0, 0, 0x80, 0, 0, 0, 0xff},
				Stride: 4, Rect: image.Rect(0, 0, 3, 1)},
			[4]uint16{0x0000, 0x8080, 0xffff, 0},
		},
		{
			"nrgba64",
			&image.NRGBA64{Pix: []uint8{
				0, 0, 0, 0, 0, 0, 0x00, 0x00,
				0, 0, 0, 0, 0, 0, 0x12, 0x34,
				0, 0, 0, 0, 0, 0, 0xab, 0xcd,
			}, Stride: 8, Rect: image.Rect(0, 0, 3, 1)},
			[4]uint16{0x0000, 0x1234, 0xabcd, 0},
		},
		{
			"rgba64",
			&image.RGBA64{Pix: []uint8{
				0, 0, 0, 0, 0, 0, 0x00, 0x00,
				0, 0, 0, 0, 0, 0, 0x12, 0x34,
				0, 0, 0, 0, 0, 0, 0xab, 0xcd,
			}, Stride: 8, Rect: image.Rect(0, 0, 3, 1)},
			[4]uint16{0x0000, 0x1234, 0xabcd, 0},
		},
		{
			"gray",
			image.NewGray(image.Rect(0, 0, 3, 1)),
			[4]uint16{0xffff, 0xffff, 0xffff, 0},
		},
		{
			"gray16",
			image.NewGray16(image.Rect(0, 0, 3, 1)),
			[4]uint16{0xffff, 0xffff, 0xffff, 0},
		},
		{
			"ycbcr",
			image.NewYCbCr(image.Rect(0, 0, 3, 1), image.YCbCrSubsampleRatio444),
			[4]uint16{0xffff, 0xffff, 0xffff, 0},
		},
		{
			"paletted",
			palettedAlphaImage(),
			[4]uint16{0x0000, 0x8080, 0xffff, 0},
		},
		{
			"generic",
			image.NewUniform(color.NRGBA{0, 0, 0, 0x80}),
			[4]uint16{0x8080, 0x8080, 0x8080, 0x8080},
		},
	}

	for _, d := range testData {
		pg := newPixelGetter(d.img)
		b := d.img.Bounds()
		for i, x := range []int{b.Min.X, b.Min.X + 1, b.Min.X + 2, b.Max.X} {
			if x >= b.Max.X {
				continue
			}
			want := d.want[i]
			if got := pg.getAlpha16(x, b.Min.Y); got != want {
				t.Errorf("test [%s] failed: x=%d expected %#x got %#x", d.desc, x, want, got)
			}
		}
	}

	// getAlpha16 must not allocate for specialized pixel types.
	pg := newPixelGetter(image.NewNRGBA(image.Rect(0, 0, 1, 1)))
	if n := testing.AllocsPerRun(100, func() { _ = pg.getAlpha16(0, 0) }); n != 0 {
		t.Errorf("getAlpha16 allocates for NRGBA: %v", n)
	}
}

func palettedAlphaImage() *image.Paletted {
	img := image.NewPaletted(image.Rect(0, 0, 3, 1), []color.Color{
		color.NRGBA{0, 0, 0, 0},
		color.NRGBA{0, 0, 0, 0x80},
		color.NRGBA{0, 0, 0, 0xff},
	})
	img.Pix[0], img.Pix[1], img.Pix[2] = 0, 1, 2
	return img
}

func TestComponents(t *testing.T) {
	testData := []struct {
		desc         string
		bounds       image.Rectangle
		rows         []string
		threshold    int
		connectivity Connectivity
		want         []Component
	}{
		{
			"empty image",
			image.Rect(0, 0, 0, 0),
			nil,
			0,
			Connectivity4,
			nil,
		},
		{
			"fully transparent",
			image.Rect(0, 0, 4, 3),
			[]string{"....", "....", "...."},
			0,
			Connectivity8,
			nil,
		},
		{
			"single pixel",
			image.Rect(0, 0, 3, 2),
			[]string{"...", ".f."},
			0,
			Connectivity4,
			[]Component{{1, image.Rect(1, 1, 2, 2), 1, 1, 1}},
		},
		{
			"two separate parts",
			image.Rect(0, 0, 5, 3),
			[]string{"ff..f", "ff..f", "....."},
			0,
			Connectivity4,
			[]Component{
				{1, image.Rect(0, 0, 2, 2), 4, 0.5, 0.5},
				{2, image.Rect(4, 0, 5, 2), 2, 4, 0.5},
			},
		},
		{
			"scan order numbering",
			image.Rect(0, 0, 5, 3),
			[]string{"...f.", "f....", "..f.."},
			0,
			Connectivity8,
			[]Component{
				{1, image.Rect(3, 0, 4, 1), 1, 3, 0},
				{2, image.Rect(0, 1, 1, 2), 1, 0, 1},
				{3, image.Rect(2, 2, 3, 3), 1, 2, 2},
			},
		},
		{
			"diagonal contact 4-way",
			image.Rect(0, 0, 2, 2),
			[]string{"f.", ".f"},
			0,
			Connectivity4,
			[]Component{
				{1, image.Rect(0, 0, 1, 1), 1, 0, 0},
				{2, image.Rect(1, 1, 2, 2), 1, 1, 1},
			},
		},
		{
			"diagonal contact 8-way",
			image.Rect(0, 0, 2, 2),
			[]string{"f.", ".f"},
			0,
			Connectivity8,
			[]Component{
				{1, image.Rect(0, 0, 2, 2), 2, 0.5, 0.5},
			},
		},
		{
			"single pixel hole",
			image.Rect(0, 0, 3, 3),
			[]string{"fff", "f.f", "fff"},
			0,
			Connectivity4,
			[]Component{
				{1, image.Rect(0, 0, 3, 3), 8, 1, 1},
			},
		},
		{
			"ring at 8-way",
			image.Rect(0, 0, 3, 3),
			[]string{"fff", "f.f", "fff"},
			0,
			Connectivity8,
			[]Component{
				{1, image.Rect(0, 0, 3, 3), 8, 1, 1},
			},
		},
		{
			"non-zero bounds",
			image.Rect(-2, 5, 2, 8),
			[]string{"f...", ".f..", "..ff"},
			0,
			Connectivity4,
			[]Component{
				{1, image.Rect(-2, 5, -1, 6), 1, -2, 5},
				{2, image.Rect(-1, 6, 0, 7), 1, -1, 6},
				{3, image.Rect(0, 7, 2, 8), 2, 0.5, 7},
			},
		},
		{
			"threshold includes partial alpha",
			image.Rect(0, 0, 3, 1),
			[]string{"17f"},
			0x80,
			Connectivity4,
			[]Component{
				{1, image.Rect(2, 0, 3, 1), 1, 2, 0},
			},
		},
		{
			"threshold is inclusive",
			image.Rect(0, 0, 3, 1),
			[]string{"78f"},
			0x88,
			Connectivity4,
			[]Component{
				{1, image.Rect(1, 0, 3, 1), 2, 1.5, 0},
			},
		},
		{
			"zero alpha character is transparent",
			image.Rect(0, 0, 4, 1),
			[]string{"0f0f"},
			0,
			Connectivity4,
			[]Component{
				{1, image.Rect(1, 0, 2, 1), 1, 1, 0},
				{2, image.Rect(3, 0, 4, 1), 1, 3, 0},
			},
		},
		{
			"threshold 255",
			image.Rect(0, 0, 3, 1),
			[]string{"7fe"},
			255,
			Connectivity4,
			[]Component{
				{1, image.Rect(1, 0, 2, 1), 1, 1, 0},
			},
		},
		{
			"snake shape",
			image.Rect(0, 0, 4, 4),
			[]string{
				"f...",
				"fff.",
				"..f.",
				"..ff",
			},
			0,
			Connectivity4,
			[]Component{
				{1, image.Rect(0, 0, 4, 4), 7, 10.0 / 7.0, 11.0 / 7.0},
			},
		},
		{
			"centroid is pixel average",
			image.Rect(0, 0, 5, 2),
			[]string{
				"f...f",
				"..f..",
			},
			0,
			Connectivity8,
			[]Component{
				{1, image.Rect(0, 0, 1, 1), 1, 0, 0},
				{2, image.Rect(4, 0, 5, 1), 1, 4, 0},
				{3, image.Rect(2, 1, 3, 2), 1, 2, 1},
			},
		},
	}

	for _, d := range testData {
		src := newComponentsImage(d.bounds, d.rows)
		got, err := Components(src, d.threshold, d.connectivity)
		if err != nil {
			t.Errorf("test [%s] failed: unexpected error: %v", d.desc, err)
			continue
		}
		if len(got) != len(d.want) {
			t.Errorf("test [%s] failed: expected %d components, got %d: %#v", d.desc, len(d.want), len(got), got)
			continue
		}
		for i := range d.want {
			w := d.want[i]
			g := got[i]
			if g.Label != w.Label || !g.Bounds.Eq(w.Bounds) || g.NumPixels != w.NumPixels ||
				math.Abs(g.CentroidX-w.CentroidX) > 1e-9 || math.Abs(g.CentroidY-w.CentroidY) > 1e-9 {
				t.Errorf("test [%s] failed: component %d: expected %#v, got %#v", d.desc, i, w, g)
			}
		}
	}
}

func TestComponentsErrors(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))

	if _, err := Components(nil, 0, Connectivity4); err != ErrNilImage {
		t.Errorf("expected ErrNilImage, got %v", err)
	}
	if _, _, err := LabelComponents(nil, 0, Connectivity4); err != ErrNilImage {
		t.Errorf("expected ErrNilImage, got %v", err)
	}

	for _, threshold := range []int{-1, 256, 1000} {
		if _, err := Components(img, threshold, Connectivity4); err != ErrInvalidAlphaThreshold {
			t.Errorf("threshold %d: expected ErrInvalidAlphaThreshold, got %v", threshold, err)
		}
		if _, _, err := LabelComponents(img, threshold, Connectivity4); err != ErrInvalidAlphaThreshold {
			t.Errorf("label threshold %d: expected ErrInvalidAlphaThreshold, got %v", threshold, err)
		}
	}

	for _, c := range []Connectivity{0, 1, 3, 5, 6, 7, 9, -4} {
		if _, err := Components(img, 0, c); err != ErrInvalidConnectivity {
			t.Errorf("connectivity %d: expected ErrInvalidConnectivity, got %v", c, err)
		}
		if _, _, err := LabelComponents(img, 0, c); err != ErrInvalidConnectivity {
			t.Errorf("label connectivity %d: expected ErrInvalidConnectivity, got %v", c, err)
		}
	}

	// Validation order is documented: the source image is checked first.
	if _, err := Components(nil, -1, Connectivity4); err != ErrNilImage {
		t.Errorf("expected ErrNilImage with nil image and bad threshold, got %v", err)
	}
}

func TestLabelComponents(t *testing.T) {
	// A hand-drawn character sheet with several detached parts: a head, a torso
	// with a single-pixel hole and an attached leg, a detached arm, and a pair of
	// diagonally touching accessory pixels. Bounds start away from the origin.
	src := newComponentsImage(image.Rect(-1, 2, 7, 10), []string{
		"..fff...",
		"..fff...",
		"........",
		".fff.f..",
		".f.f....",
		".fff....",
		"..f....f",
		"..ff..f.",
	})
	wantLabels := []string{
		"..111...",
		"..111...",
		"........",
		".222.3..",
		".2.2....",
		".222....",
		"..2....4",
		"..22..4.",
	}

	dst, comps, err := LabelComponents(src, 0, Connectivity8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !dst.Bounds().Eq(src.Bounds()) {
		t.Fatalf("unexpected label bounds: %v", dst.Bounds())
	}

	wantComps := []Component{
		{1, image.Rect(1, 2, 4, 4), 6, 2, 2.5},
		{2, image.Rect(0, 5, 3, 10), 11, 12.0 / 11.0, 74.0 / 11.0},
		{3, image.Rect(4, 5, 5, 6), 1, 4, 5},
		{4, image.Rect(5, 8, 7, 10), 2, 5.5, 8.5},
	}
	if len(comps) != len(wantComps) {
		t.Fatalf("expected %d components, got %d: %#v", len(wantComps), len(comps), comps)
	}
	for i, w := range wantComps {
		g := comps[i]
		if g.Label != w.Label || !g.Bounds.Eq(w.Bounds) || g.NumPixels != w.NumPixels ||
			math.Abs(g.CentroidX-w.CentroidX) > 1e-9 || math.Abs(g.CentroidY-w.CentroidY) > 1e-9 {
			t.Errorf("component %d: expected %#v, got %#v", i, w, g)
		}
	}

	// Check the label image pixel by pixel and make sure labels, bounds and
	// statistics agree with each other.
	for y := 0; y < src.Bounds().Dy(); y++ {
		for x := 0; x < src.Bounds().Dx(); x++ {
			var want uint16
			if ch := wantLabels[y][x]; ch != '.' {
				want = uint16(ch - '0')
			}
			p := image.Pt(src.Bounds().Min.X+x, src.Bounds().Min.Y+y)
			got := dst.Gray16At(p.X, p.Y).Y
			if got != want {
				t.Errorf("label mismatch at %v: expected %d, got %d", p, want, got)
			}

			// Every labeled pixel must be inside its component's bounding box and
			// the source pixel must be visible.
			if want == 0 {
				continue
			}
			if !p.In(comps[want-1].Bounds) {
				t.Errorf("pixel %v labeled %d is outside the component bounds %v", p, want, comps[want-1].Bounds)
			}
			if src.NRGBAAt(p.X, p.Y).A == 0 {
				t.Errorf("pixel %v labeled %d is transparent in the source", p, want)
			}
		}
	}

	// Recount each label from the label image and compare against the reported
	// statistics: this catches a label written for the wrong component.
	for li, c := range comps {
		label := uint16(li + 1)
		count := 0
		var bb image.Rectangle
		for y := dst.Bounds().Min.Y; y < dst.Bounds().Max.Y; y++ {
			for x := dst.Bounds().Min.X; x < dst.Bounds().Max.X; x++ {
				if dst.Gray16At(x, y).Y != label {
					continue
				}
				count++
				p := image.Rect(x, y, x+1, y+1)
				if bb.Empty() {
					bb = p
				} else {
					bb = bb.Union(p)
				}
			}
		}
		if count != c.NumPixels {
			t.Errorf("component %d: label image has %d pixels, NumPixels is %d", li+1, count, c.NumPixels)
		}
		if !bb.Eq(c.Bounds) {
			t.Errorf("component %d: label image bounds %v, reported %v", li+1, bb, c.Bounds)
		}
	}

	// The same input must produce the same output (stable numbering).
	dst2, comps2, err := LabelComponents(src, 0, Connectivity8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(dst.Pix, dst2.Pix) || !reflect.DeepEqual(comps, comps2) {
		t.Error("component labels are not stable between runs")
	}
}

func TestLabelComponentsSameAsComponents(t *testing.T) {
	src := newComponentsImage(image.Rect(0, 0, 8, 8), []string{
		"f...f...",
		".f..f...",
		"..f.....",
		"........",
		"fff..f.f",
		"...f....",
		"...f....",
		"........",
	})

	for _, connectivity := range []Connectivity{Connectivity4, Connectivity8} {
		comps, err := Components(src, 10, connectivity)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		dst, comps2, err := LabelComponents(src, 10, connectivity)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(comps, comps2) {
			t.Errorf("connectivity %d: Components and LabelComponents disagree", connectivity)
		}

		// Re-derive the statistics from the label image alone.
		counts := make([]int, len(comps)+1)
		xsums := make([]int, len(comps)+1)
		ysums := make([]int, len(comps)+1)
		for y := dst.Bounds().Min.Y; y < dst.Bounds().Max.Y; y++ {
			for x := dst.Bounds().Min.X; x < dst.Bounds().Max.X; x++ {
				l := int(dst.Gray16At(x, y).Y)
				if l == 0 {
					continue
				}
				counts[l]++
				xsums[l] += x
				ysums[l] += y
			}
		}
		for i, c := range comps {
			l := i + 1
			if counts[l] != c.NumPixels {
				t.Errorf("connectivity %d: component %d count mismatch", connectivity, l)
			}
			if math.Abs(float64(xsums[l])/float64(counts[l])-c.CentroidX) > 1e-9 ||
				math.Abs(float64(ysums[l])/float64(counts[l])-c.CentroidY) > 1e-9 {
				t.Errorf("connectivity %d: component %d centroid mismatch", connectivity, l)
			}
		}
	}
}

func TestLabelComponentsTransparent(t *testing.T) {
	src := image.NewNRGBA(image.Rect(2, 3, 6, 5))
	dst, comps, err := LabelComponents(src, 0, Connectivity8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != 0 {
		t.Errorf("expected no components, got %#v", comps)
	}
	if !dst.Bounds().Eq(src.Bounds()) || len(dst.Pix) != 4*2*2 {
		t.Fatalf("unexpected label image: bounds %v, pix %d", dst.Bounds(), len(dst.Pix))
	}
	for i, v := range dst.Pix {
		if v != 0 {
			t.Errorf("unexpected non-zero label at byte %d: %d", i, v)
		}
	}
}

func TestComponentsAlpha16Threshold(t *testing.T) {
	// 16-bit alpha values are compared using their most significant byte, so a
	// threshold of 0x12 accepts 0x12ff but rejects 0x11ff.
	img := image.NewNRGBA64(image.Rect(0, 0, 4, 1))
	alphas := []uint16{0x0000, 0x11ff, 0x1200, 0x12ff}
	for x, a := range alphas {
		i := img.PixOffset(x, 0)
		img.Pix[i+6] = uint8(a >> 8)
		img.Pix[i+7] = uint8(a)
	}

	comps, err := Components(img, 0x12, Connectivity4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != 1 || !comps[0].Bounds.Eq(image.Rect(2, 0, 4, 1)) || comps[0].NumPixels != 2 {
		t.Errorf("unexpected components: %#v", comps)
	}

	// A threshold of zero still ignores fully transparent pixels.
	comps, err = Components(img, 0, Connectivity4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != 1 || !comps[0].Bounds.Eq(image.Rect(1, 0, 4, 1)) || comps[0].NumPixels != 3 {
		t.Errorf("unexpected components at zero threshold: %#v", comps)
	}
}

func TestComponentsOpaqueAndGenericImages(t *testing.T) {
	// Opaque images without an alpha channel: every pixel is a component pixel.
	gray := image.NewGray(image.Rect(0, 0, 4, 2))
	gray.Pix = []uint8{0, 0, 0, 0, 0, 0, 0, 0}
	comps, err := Components(gray, 0, Connectivity4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != 1 || comps[0].NumPixels != 8 || !comps[0].Bounds.Eq(image.Rect(0, 0, 4, 2)) {
		t.Errorf("unexpected Gray components: %#v", comps)
	}
	// Gray pixels with value 0 are opaque; alpha threshold 255 must select them.
	if comps, err = Components(gray, 255, Connectivity4); err != nil || len(comps) != 1 {
		t.Errorf("unexpected Gray threshold components: %#v, %v", comps, err)
	}

	// A generic image.Image implementation goes through the At() path.
	transparent := boundedImage{image.NewUniform(color.NRGBA{10, 20, 30, 0}), image.Rect(0, 0, 3, 2)}
	if comps, err = Components(transparent, 0, Connectivity8); err != nil || len(comps) != 0 {
		t.Errorf("unexpected transparent generic components: %#v, %v", comps, err)
	}
	opaque := boundedImage{image.NewUniform(color.NRGBA{10, 20, 30, 40}), image.Rect(-2, -1, 1, 1)}
	comps, err = Components(opaque, 0, Connectivity8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != 1 || comps[0].NumPixels != 6 || !comps[0].Bounds.Eq(image.Rect(-2, -1, 1, 1)) {
		t.Errorf("unexpected generic components: %#v", comps)
	}
	if a := newPixelGetter(opaque).getAlpha16(0, 0); a != 0x2828 {
		t.Errorf("unexpected Uniform alpha: %#x", a)
	}

	// Sub-images with a non-zero origin behave like the full image.
	full := newComponentsImage(image.Rect(0, 0, 4, 4), []string{"f...", ".f..", "..f.", "...f"})
	sub := full.SubImage(image.Rect(1, 1, 4, 4))
	comps, err = Components(sub, 0, Connectivity8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != 1 {
		t.Fatalf("expected 1 component in sub-image, got %#v", comps)
	}
	if !comps[0].Bounds.Eq(image.Rect(1, 1, 4, 4)) || comps[0].NumPixels != 3 {
		t.Errorf("unexpected sub-image component: %#v", comps[0])
	}
}

func TestComponentsManyRegions(t *testing.T) {
	// Tiny isolated 2x2 blocks arranged in a grid. This exercises the run-based
	// representation on a large sparse image.
	const side = 500
	img := image.NewNRGBA(image.Rect(0, 0, side, side))
	blocks := 0
	for y := 0; y+1 < side; y += 4 {
		for x := 0; x+1 < side; x += 4 {
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					i := img.PixOffset(x+dx, y+dy)
					img.Pix[i+3] = 0xff
				}
			}
			blocks++
		}
	}
	comps, err := Components(img, 0, Connectivity8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != blocks {
		t.Fatalf("expected %d components, got %d", blocks, len(comps))
	}
	for i, c := range comps {
		if c.NumPixels != 4 || c.Label != i+1 {
			t.Fatalf("unexpected component %d: %#v", i, c)
		}
	}

	// A single full-width line is one component with a one-pixel-tall box.
	line := image.NewNRGBA(image.Rect(0, 0, side, side))
	for x := 0; x < side; x++ {
		line.Pix[line.PixOffset(x, 250)+3] = 0xff
	}
	comps, err = Components(line, 0, Connectivity4)
	if err != nil || len(comps) != 1 || comps[0].NumPixels != side || !comps[0].Bounds.Eq(image.Rect(0, 250, side, 251)) {
		t.Errorf("unexpected line component: %#v, %v", comps, err)
	}
}

func TestLabelComponentsTooMany(t *testing.T) {
	// 65536 isolated single-pixel components cannot fit in a Gray16 label image.
	const side = 512
	img := image.NewNRGBA(image.Rect(0, 0, side, side))
	for y := 0; y < side; y += 2 {
		for x := 0; x < side; x += 2 {
			img.Pix[img.PixOffset(x, y)+3] = 0xff
		}
	}
	// side/2 * side/2 = 65536 components.
	if comps, err := Components(img, 0, Connectivity4); err != nil || len(comps) != 65536 {
		t.Fatalf("expected 65536 components, got %d, %v", len(comps), err)
	}
	dst, comps, err := LabelComponents(img, 0, Connectivity4)
	if err != ErrTooManyComponents {
		t.Fatalf("expected ErrTooManyComponents, got %v", err)
	}
	if dst != nil || comps != nil {
		t.Error("expected nil results along with ErrTooManyComponents")
	}

	// 65535 components is still representable: remove one pixel from the grid,
	// which removes a single isolated component.
	img.Pix[img.PixOffset(0, side-2)+3] = 0
	dst, comps, err = LabelComponents(img, 0, Connectivity4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(comps) != 65535 {
		t.Fatalf("expected 65535 components, got %d", len(comps))
	}
	maxLabel := uint16(0)
	for i := 0; i < len(dst.Pix); i += 2 {
		l := uint16(dst.Pix[i])<<8 | uint16(dst.Pix[i+1])
		if l > maxLabel {
			maxLabel = l
		}
	}
	if maxLabel != 65535 {
		t.Errorf("expected max label 65535, got %d", maxLabel)
	}
}

type boundedImage struct {
	image.Image
	bounds image.Rectangle
}

func (p boundedImage) Bounds() image.Rectangle { return p.bounds }

func TestComponentsMalformedBounds(t *testing.T) {
	// A malformed source rectangle must not panic; LabelComponents returns an
	// empty label image at the Min point.
	src := &image.NRGBA{Pix: []uint8{}, Stride: 0, Rect: image.Rectangle{image.Pt(3, 4), image.Pt(1, 2)}}
	comps, err := Components(src, 0, Connectivity4)
	if err != nil || len(comps) != 0 {
		t.Errorf("unexpected malformed bounds components: %#v, %v", comps, err)
	}
	dst, comps, err := LabelComponents(src, 0, Connectivity4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dst == nil || !dst.Bounds().Empty() {
		t.Errorf("unexpected malformed label bounds: %v", dst)
	}
	if len(comps) != 0 {
		t.Errorf("unexpected malformed label components: %#v", comps)
	}
}

func TestComponentsSparseAllocations(t *testing.T) {
	// A 3000x3000 image with only a handful of runs. The analysis must allocate a
	// number of objects proportional to the number of runs, not the number of
	// pixels.
	img := image.NewNRGBA(image.Rect(0, 0, 3000, 3000))
	for _, p := range []image.Point{{0, 0}, {2999, 0}, {1500, 1500}, {0, 2999}, {2999, 2999}} {
		i := img.PixOffset(p.X, p.Y)
		img.Pix[i+3] = 0xff
	}
	// Two-row run to ensure the union-find path is exercised too.
	img.Pix[img.PixOffset(10, 10)+3] = 0xff
	img.Pix[img.PixOffset(10, 11)+3] = 0xff

	if _, err := Components(img, 0, Connectivity4); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	const maxAllocs = 20
	if n := testing.AllocsPerRun(10, func() { _, _ = Components(img, 0, Connectivity4) }); n > maxAllocs {
		t.Errorf("sparse image analysis allocates %v objects, want <= %d", n, maxAllocs)
	}
}

func BenchmarkComponents(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+3] = 0xff
	}
	for x := 0; x < 1024; x += 16 {
		for y := 0; y < 1024; y += 16 {
			img.Pix[img.PixOffset(x, y)+3] = 0
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Components(img, 0, Connectivity4)
	}
}
