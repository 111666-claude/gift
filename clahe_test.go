package gift

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestCLAHE(t *testing.T) {
	testData := []struct {
		desc           string
		tilesX, tilesY int
		clipLimit      float32
		srcb, dstb     image.Rectangle
		srcPix, dstPix []uint8
	}{
		{
			"clahe (2x2 tiles, clip 2)",
			2, 2, 2,
			image.Rect(0, 0, 4, 4),
			image.Rect(0, 0, 4, 4),
			[]uint8{
				10, 40, 90, 230,
				20, 30, 210, 220,
				0, 5, 250, 255,
				100, 110, 120, 130,
			},
			[]uint8{
				13, 46, 91, 231,
				24, 34, 209, 219,
				2, 8, 249, 255,
				103, 113, 120, 130,
			},
		},
		{
			"clahe (2x2 tiles, clip 4)",
			2, 2, 4,
			image.Rect(0, 0, 4, 4),
			image.Rect(0, 0, 4, 4),
			[]uint8{
				0, 16, 32, 48,
				64, 80, 96, 112,
				128, 144, 160, 176,
				192, 208, 224, 240,
			},
			[]uint8{
				5, 22, 36, 54,
				70, 86, 100, 117,
				127, 144, 158, 175,
				192, 209, 223, 241,
			},
		},
		{
			"clahe (flat image)",
			2, 2, 8,
			image.Rect(0, 0, 4, 4),
			image.Rect(0, 0, 4, 4),
			[]uint8{
				77, 77, 77, 77,
				77, 77, 77, 77,
				77, 77, 77, 77,
				77, 77, 77, 77,
			},
			[]uint8{
				77, 77, 77, 77,
				77, 77, 77, 77,
				77, 77, 77, 77,
				77, 77, 77, 77,
			},
		},
		{
			"clahe (1x1 image)",
			8, 8, 2,
			image.Rect(0, 0, 1, 1),
			image.Rect(0, 0, 1, 1),
			[]uint8{123},
			[]uint8{123},
		},
		{
			"clahe (1x6 image)",
			4, 4, 4,
			image.Rect(0, 0, 1, 6),
			image.Rect(0, 0, 1, 6),
			[]uint8{0, 40, 90, 150, 210, 255},
			[]uint8{0, 42, 94, 150, 208, 255},
		},
		{
			"clahe (6x1 image)",
			4, 4, 4,
			image.Rect(0, 0, 6, 1),
			image.Rect(0, 0, 6, 1),
			[]uint8{0, 40, 90, 150, 210, 255},
			[]uint8{0, 42, 94, 150, 208, 255},
		},
		{
			"clahe (zero tiles)",
			0, 2, 3,
			image.Rect(-2, -2, 2, 2),
			image.Rect(0, 0, 4, 4),
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			"clahe (negative tiles)",
			-3, 2, 3,
			image.Rect(-2, -2, 2, 2),
			image.Rect(0, 0, 4, 4),
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			"clahe (zero clip limit)",
			2, 2, 0,
			image.Rect(-2, -2, 2, 2),
			image.Rect(0, 0, 4, 4),
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			"clahe (negative clip limit)",
			2, 2, -1,
			image.Rect(-2, -2, 2, 2),
			image.Rect(0, 0, 4, 4),
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			"clahe (infinite clip limit)",
			2, 2, float32(math.Inf(1)),
			image.Rect(-2, -2, 2, 2),
			image.Rect(0, 0, 4, 4),
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			"clahe (0x0 image)",
			3, 3, 3,
			image.Rect(-1, -1, -1, -1),
			image.Rect(0, 0, 0, 0),
			[]uint8{},
			[]uint8{},
		},
	}

	for _, d := range testData {
		src := image.NewGray(d.srcb)
		src.Pix = d.srcPix

		f := CLAHE(d.tilesX, d.tilesY, d.clipLimit)
		if b := f.Bounds(src.Bounds()); !b.Eq(image.Rect(0, 0, d.srcb.Dx(), d.srcb.Dy())) {
			t.Errorf("test [%s] failed: unexpected filter bounds: %v", d.desc, b)
		}
		dst := image.NewGray(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)

		if !checkBoundsAndPix(dst.Bounds(), d.dstb, dst.Pix, d.dstPix) {
			t.Errorf("test [%s] failed: %#v, %#v", d.desc, dst.Bounds(), dst.Pix)
		}
	}
}

// claheScene builds a deterministic "night scene" test image: the left 60%
// is dark with a narrow tonal range and fine detail, the right 40% is bright
// with a wide tonal range and comparable detail.
func claheScene() *image.Gray {
	const w, h = 160, 96
	img := image.NewGray(image.Rect(0, 0, w, h))
	noise := func(x, y, seed int) float64 {
		v := x*374761393 + y*668265263 + seed*1442695040
		v = (v ^ (v >> 13)) * 1274126177
		v = v ^ (v >> 16)
		if v < 0 {
			v = -v
		}
		return float64(v%1000)/1000 - 0.5
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var v float64
			if x < 96 {
				v = 30 + float64(x)*14/95 + noise(x, y, 1)*6
			} else {
				v = 160 + float64(x-96)*85/63 + noise(x, y, 2)*6
			}
			img.SetGray(x, y, color.Gray{claheTestU8(v)})
		}
	}
	return img
}

func claheTestU8(v float64) uint8 {
	if !(v >= 0) {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}

func TestCLAHELocalContrast(t *testing.T) {
	src := claheScene()
	const w, h = 160, 96

	std := func(img *image.Gray, x0, x1 int) float64 {
		var n, sum, sum2 float64
		for y := 0; y < h; y++ {
			for x := x0; x < x1; x++ {
				v := float64(img.GrayAt(x, y).Y)
				n++
				sum += v
				sum2 += v * v
			}
		}
		mean := sum / n
		variance := sum2/n - mean*mean
		if variance < 0 {
			return 0
		}
		return math.Sqrt(variance)
	}

	srcDark := std(src, 0, 96)
	srcBright := std(src, 96, w)

	// The dark region must gain much more local contrast than the already
	// bright region, and the gain must grow with clipLimit while staying
	// bounded (no extreme noise amplification).
	dst := image.NewGray(src.Bounds())
	CLAHE(8, 6, 4).Draw(dst, src, nil)
	dstDark := std(dst, 0, 96)
	dstBright := std(dst, 96, w)

	if dstDark <= srcDark {
		t.Errorf("dark region contrast not increased: %.2f -> %.2f", srcDark, dstDark)
	}
	if dstDark/srcDark > 3.5 {
		t.Errorf("dark region contrast amplified too much: %.2f -> %.2f", srcDark, dstDark)
	}
	if dstBright/srcBright > 1.5 {
		t.Errorf("bright region contrast amplified too much: %.2f -> %.2f", srcBright, dstBright)
	}

	// clipLimit controls the amplification: increasing it must never reduce
	// the dark-region standard deviation substantially.
	prev := 0.0
	for _, clipLimit := range []float32{1, 2, 4, 8} {
		CLAHE(8, 6, clipLimit).Draw(dst, src, nil)
		got := std(dst, 0, 96)
		if got < prev-0.5 {
			t.Errorf("clip limit %v reduced dark contrast: %.2f -> %.2f", clipLimit, prev, got)
		}
		prev = got
	}
}

func TestCLAHEClipLimit(t *testing.T) {
	// A noisy flat patch: the output range must be bounded and must grow
	// monotonically with the clip limit, staying finite.
	const w, h = 128, 128
	src := image.NewGray(image.Rect(0, 0, w, h))
	noise := func(x, y int) int {
		v := x*374761393 + y*668265263 + 1442695040
		v = (v ^ (v >> 13)) * 1274126177
		v = v ^ (v >> 16)
		if v < 0 {
			v = -v
		}
		return v%21 - 10
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetGray(x, y, color.Gray{uint8(128 + noise(x, y))})
		}
	}

	rangeOf := func(img *image.Gray) int {
		lo, hi := 255, 0
		for _, v := range img.Pix {
			if int(v) < lo {
				lo = int(v)
			}
			if int(v) > hi {
				hi = int(v)
			}
		}
		return hi - lo
	}

	prev := -1
	for _, clipLimit := range []float32{1, 2, 4, 8, 16, 64} {
		dst := image.NewGray(src.Bounds())
		CLAHE(8, 8, clipLimit).Draw(dst, src, nil)
		r := rangeOf(dst)
		if prev >= 0 && r < prev {
			t.Errorf("clip limit %v reduced the output range: %d -> %d", clipLimit, prev, r)
		}
		prev = r
	}

	// A moderate clip limit must amplify the noise far less than an
	// effectively unlimited one (plain histogram equalization).
	limited := image.NewGray(src.Bounds())
	unlimited := image.NewGray(src.Bounds())
	CLAHE(8, 8, 2).Draw(limited, src, nil)
	CLAHE(8, 8, 1.0e9).Draw(unlimited, src, nil)
	if rangeOf(limited)*2 > rangeOf(unlimited) {
		t.Errorf("clip limit does not limit the amplification: limited range %d, unlimited range %d",
			rangeOf(limited), rangeOf(unlimited))
	}

	// A very small clip limit must be close to the identity.
	dst := image.NewGray(src.Bounds())
	CLAHE(8, 8, 1.0e-6).Draw(dst, src, nil)
	maxDiff := 0
	for i := range src.Pix {
		d := int(dst.Pix[i]) - int(src.Pix[i])
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1 {
		t.Errorf("tiny clip limit is not close to identity: max diff %d", maxDiff)
	}
}

func TestCLAHEFlatAndTransparent(t *testing.T) {
	// Pure solid color images of different levels must not change at all.
	for _, level := range []uint8{0, 1, 128, 254, 255} {
		src := image.NewGray(image.Rect(0, 0, 32, 24))
		for i := range src.Pix {
			src.Pix[i] = level
		}
		dst := image.NewGray(src.Bounds())
		CLAHE(4, 4, 8).Draw(dst, src, nil)
		for i := range src.Pix {
			if dst.Pix[i] != level {
				t.Fatalf("flat level %d changed at pixel %d: %d", level, i, dst.Pix[i])
			}
		}
	}

	// Transparent pixels keep their color and alpha; alpha is never modified.
	const w, h = 16, 16
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a := uint8(255)
			if x < 8 {
				a = 0
			}
			src.SetNRGBA(x, y, color.NRGBA{uint8(20 + x*10), uint8(30 + y*8), 100, a})
		}
	}
	dst := image.NewNRGBA(src.Bounds())
	CLAHE(4, 4, 3).Draw(dst, src, nil)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sp := src.NRGBAAt(x, y)
			dp := dst.NRGBAAt(x, y)
			if dp.A != sp.A {
				t.Fatalf("alpha changed at (%d, %d): %d -> %d", x, y, sp.A, dp.A)
			}
			if sp.A == 0 && dp != sp {
				t.Fatalf("transparent pixel changed at (%d, %d): %v -> %v", x, y, sp, dp)
			}
		}
	}
}

func TestCLAHESeams(t *testing.T) {
	// A smooth ramp must stay smooth: adjacent-pixel jumps at tile borders
	// must not exceed those inside tiles, and second differences must stay
	// small (no visible tile seams).
	const w, h = 256, 4
	src := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetGray(x, y, color.Gray{uint8(x * 255 / (w - 1))})
		}
	}

	for _, tiles := range []int{2, 4, 8, 16} {
		dst := image.NewGray(src.Bounds())
		CLAHE(tiles, 2, 4).Draw(dst, src, nil)
		step := w / tiles
		maxJumpAll, maxJumpInterior := 0, 0
		maxSecondAll, maxSecondBoundary := 0, 0
		for x := 1; x < w; x++ {
			d := int(dst.GrayAt(x, 0).Y) - int(dst.GrayAt(x-1, 0).Y)
			if d < 0 {
				d = -d
			}
			if d > maxJumpAll {
				maxJumpAll = d
			}
			if x%step != 0 && d > maxJumpInterior {
				maxJumpInterior = d
			}
		}
		for x := 1; x < w-1; x++ {
			d := int(dst.GrayAt(x+1, 0).Y) - 2*int(dst.GrayAt(x, 0).Y) + int(dst.GrayAt(x-1, 0).Y)
			if d < 0 {
				d = -d
			}
			if d > maxSecondAll {
				maxSecondAll = d
			}
			if (x%step == 0 || (x+1)%step == 0) && d > maxSecondBoundary {
				maxSecondBoundary = d
			}
		}
		if maxJumpAll > maxJumpInterior+1 {
			t.Errorf("%d tiles: visible jump at tile boundary: all %d, interior %d",
				tiles, maxJumpAll, maxJumpInterior)
		}
		if maxSecondAll > 3 {
			t.Errorf("%d tiles: second difference too large: %d", tiles, maxSecondAll)
		}
		// Eight-bit rounding can produce second differences of ±2 anywhere
		// along a smooth ramp; a seam would show up as a larger, isolated
		// jump at the tile boundary.
		if maxSecondBoundary > 2 {
			t.Errorf("%d tiles: seam at a tile boundary: %d", tiles, maxSecondBoundary)
		}
	}

	// The same ramp in color must not show per-channel seams either: the
	// channel differences must stay small everywhere and the same offset must
	// be applied to all channels.
	const cw, chh = 192, 4
	csrc := image.NewNRGBA(image.Rect(0, 0, cw, chh))
	for y := 0; y < chh; y++ {
		for x := 0; x < cw; x++ {
			v := uint8(x * 255 / (cw - 1))
			csrc.SetNRGBA(x, y, color.NRGBA{v, uint8(40 + int(v)*3/4), uint8(255 - int(v)), 255})
		}
	}
	cdst := image.NewNRGBA(csrc.Bounds())
	CLAHE(6, 2, 4).Draw(cdst, csrc, nil)
	for y := 0; y < chh; y++ {
		for x := 1; x < cw; x++ {
			p0 := cdst.NRGBAAt(x-1, y)
			p1 := cdst.NRGBAAt(x, y)
			for _, d := range []int{
				int(p1.R) - int(p0.R),
				int(p1.G) - int(p0.G),
				int(p1.B) - int(p0.B),
			} {
				if d < 0 {
					d = -d
				}
				if d > 8 {
					t.Fatalf("color seam at (%d, %d): jump %d", x, y, d)
				}
			}
		}
	}

	// A black/white step must not get a seam halo: the flat areas on either
	// side stay flat away from the edge.
	const sw, sh = 64, 16
	step := image.NewGray(image.Rect(0, 0, sw, sh))
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			v := uint8(40)
			if x >= 37 {
				v = 200
			}
			step.SetGray(x, y, color.Gray{v})
		}
	}
	out := image.NewGray(step.Bounds())
	CLAHE(8, 2, 4).Draw(out, step, nil)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			v := int(out.GrayAt(x, y).Y)
			if x < 35 && v > 45 {
				t.Errorf("halo in the dark area at (%d, %d): %d", x, y, v)
			}
			if x > 42 && v < 195 {
				t.Errorf("halo in the bright area at (%d, %d): %d", x, y, v)
			}
		}
	}
}

func TestCLAHEParallel(t *testing.T) {
	src := claheScene()
	rgba := image.NewRGBA(src.Bounds())
	for y := 0; y < rgba.Rect.Dy(); y++ {
		for x := 0; x < rgba.Rect.Dx(); x++ {
			v := src.GrayAt(x, y).Y
			rgba.SetRGBA(x, y, color.RGBA{v, uint8(255 - int(v)), uint8((int(v) * 3) & 0xff), 255})
		}
	}

	for _, tiles := range [][2]int{{1, 1}, {4, 3}, {8, 8}, {17, 5}} {
		dst1 := image.NewRGBA(rgba.Bounds())
		dst2 := image.NewRGBA(rgba.Bounds())
		f := CLAHE(tiles[0], tiles[1], 3)
		f.Draw(dst1, rgba, &Options{Parallelization: true})
		f.Draw(dst2, rgba, &Options{Parallelization: false})
		if string(dst1.Pix) != string(dst2.Pix) {
			t.Errorf("parallel and serial results differ for %dx%d tiles", tiles[0], tiles[1])
		}
	}
}

func TestCLAHETileBounds(t *testing.T) {
	// Tile counts are clamped to the image size and to 64 per axis, so an
	// oversized grid must behave exactly like the clamped one.
	src := image.NewGray(image.Rect(0, 0, 120, 90))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*31 + i/7*13) & 0xff)
	}
	for _, c := range []struct {
		tx1, ty1, tx2, ty2 int
	}{
		{64, 64, 200, 200},
		{120, 90, 1000, 1000},
		{64, 64, 64 + 5000, 64},
	} {
		dst1 := image.NewGray(src.Bounds())
		dst2 := image.NewGray(src.Bounds())
		CLAHE(c.tx1, c.ty1, 3).Draw(dst1, src, nil)
		CLAHE(c.tx2, c.ty2, 3).Draw(dst2, src, nil)
		if string(dst1.Pix) != string(dst2.Pix) {
			t.Errorf("tile grid not clamped: %dx%d differs from %dx%d", c.tx1, c.ty1, c.tx2, c.ty2)
		}
	}
}

func TestCLAHENoPanic(t *testing.T) {
	// Degenerate inputs must neither panic nor produce non-finite pixels.
	testData := []struct {
		desc      string
		src       image.Image
		tilesX    int
		tilesY    int
		clipLimit float32
	}{
		{"empty", image.NewGray(image.Rect(0, 0, 0, 0)), 4, 4, 3},
		{"empty with invalid params", image.NewGray(image.Rect(0, 0, 0, 0)), -1, 0, -1},
		{"single pixel", image.NewGray(image.Rect(0, 0, 1, 1)), 1, 1, 1},
		{"one row", image.NewGray(image.Rect(0, 0, 7, 1)), 5, 5, 2},
		{"one column", image.NewGray(image.Rect(0, 0, 1, 7)), 5, 5, 2},
		{"huge tiles", image.NewGray(image.Rect(0, 0, 5, 5)), 1 << 20, 1 << 20, 2},
		{"huge clip", claheScene(), 3, 3, float32(math.MaxFloat32)},
		{"tiny clip", claheScene(), 3, 3, float32(math.SmallestNonzeroFloat32)},
		{"NaN clip", claheScene(), 3, 3, float32(math.NaN())},
		{"subimage bounds", image.NewGray(image.Rect(-20, 10, -4, 26)), 4, 4, 2},
	}

	for _, d := range testData {
		dst := image.NewNRGBA(image.Rect(0, 0, d.src.Bounds().Dx(), d.src.Bounds().Dy()))
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("test [%s] panicked: %v", d.desc, r)
				}
			}()
			CLAHE(d.tilesX, d.tilesY, d.clipLimit).Draw(dst, d.src, nil)
		}()
		for i, v := range dst.Pix {
			if v > 255 {
				t.Errorf("test [%s] produced out of range byte at %d: %d", d.desc, i, v)
			}
		}
	}
}

func TestCLAHEComposition(t *testing.T) {
	// A filter list containing CLAHE must give the same result as applying
	// the filters one by one, and must not affect filters around it.
	src := claheScene()

	g := New(
		CLAHE(8, 6, 2),
		Contrast(10),
		Grayscale(),
	)
	if b := g.Bounds(src.Bounds()); !b.Eq(image.Rect(0, 0, src.Bounds().Dx(), src.Bounds().Dy())) {
		t.Fatalf("unexpected bounds after composition: %v", b)
	}
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	// The GIFT pipeline uses NRGBA64 temporary images between filters, so
	// the sequential reference does the same.
	step := createTempImage(src.Bounds())
	CLAHE(8, 6, 2).Draw(step, src, &defaultOptions)
	step2 := createTempImage(step.Bounds())
	Contrast(10).Draw(step2, step, &defaultOptions)
	step3 := image.NewNRGBA(step.Bounds())
	Grayscale().Draw(step3, step2, &defaultOptions)

	if string(dst.Pix) != string(step3.Pix) {
		t.Error("composed result differs from sequential filter application")
	}

	// DrawAt must still work and must leave pixels outside the filtered
	// rectangle untouched; only the filtered area may change.
	bg := image.NewNRGBA(image.Rect(0, 0, 40, 40))
	for i := range bg.Pix {
		bg.Pix[i] = 0xAA
	}
	g2 := New(CLAHE(3, 3, 2))
	g2.DrawAt(bg, src, image.Pt(4, 6), CopyOperator)
	// Everything above the pasted region and to its left is untouched.
	for y := 0; y < 40; y++ {
		for x := 0; x < 40; x++ {
			outside := x < 4 || x >= 4+src.Bounds().Dx() || y < 6 || y >= 6+src.Bounds().Dy()
			if outside && bg.NRGBAAt(x, y) != (color.NRGBA{0xAA, 0xAA, 0xAA, 0xAA}) {
				t.Fatalf("DrawAt modified a pixel outside the destination area at (%d, %d)", x, y)
			}
		}
	}
}
