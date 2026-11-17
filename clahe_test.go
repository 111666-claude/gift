package gift

import (
	"image"
	"testing"
)

func makeGrayGradient(w, h int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Pix[y*w+x] = uint8(x * 0xff / (w - 1))
		}
	}
	return img
}

func TestCLAHEConstantImage(t *testing.T) {
	// A constant image must survive CLAHE unchanged: every tile is constant,
	// so every tile mapping is the identity.
	src := image.NewGray(image.Rect(0, 0, 16, 16))
	for i := range src.Pix {
		src.Pix[i] = 100
	}
	g := New(CLAHE(4, 2))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if !dst.Bounds().Size().Eq(src.Bounds().Size()) {
		t.Fatalf("CLAHE: bad dst bounds: %v", dst.Bounds())
	}
	for i, v := range dst.Pix {
		d := int(v) - 100
		if d < -1 || d > 1 {
			t.Errorf("CLAHE constant image: index %d: expected 100 got %d", i, v)
		}
	}
}

func TestCLAHESingleTileNoClip(t *testing.T) {
	// A single tile covering the whole image, no clipping: plain histogram
	// equalization. Three pixels at 10 and one at 255: cdfMin normalization
	// maps 10 -> 0 and 255 -> 255.
	src := image.NewGray(image.Rect(0, 0, 2, 2))
	src.Pix[0] = 10
	src.Pix[1] = 10
	src.Pix[2] = 10
	src.Pix[3] = 255

	dst := image.NewGray(New(CLAHE(1, 1000)).Bounds(src.Bounds()))
	New(CLAHE(1, 1000)).Draw(dst, src)

	expected := []uint8{0, 0, 0, 255}
	for i, v := range dst.Pix {
		if v != expected[i] {
			t.Errorf("CLAHE single tile: index %d: expected %d got %d", i, expected[i], v)
		}
	}
}

func TestCLAHESingleTileClip(t *testing.T) {
	// Same image with clipLimit = 1: the histogram is clipped at 1 and the
	// excess of 2 is redistributed one count each to bins 0 and 1.
	// The resulting cdf is 1, 2, ..., 3 (bin 10), ..., 4 (bin 255), so
	// bin 10 maps to (3-1)/(4-1) = 2/3 -> 170 and bin 255 to 255.
	src := image.NewGray(image.Rect(0, 0, 2, 2))
	src.Pix[0] = 10
	src.Pix[1] = 10
	src.Pix[2] = 10
	src.Pix[3] = 255

	g := New(CLAHE(1, 1))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	expected := []uint8{170, 170, 170, 255}
	for i, v := range dst.Pix {
		if v != expected[i] {
			t.Errorf("CLAHE clip: index %d: expected %d got %d", i, expected[i], v)
		}
	}
}

func TestCLAHEGradient(t *testing.T) {
	// A fixed horizontal gray gradient: the global range must be preserved
	// and the local contrast must increase.
	src := makeGrayGradient(256, 16)
	g := New(CLAHE(8, 1000))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if !dst.Bounds().Size().Eq(src.Bounds().Size()) {
		t.Fatalf("CLAHE gradient: bad dst bounds: %v", dst.Bounds())
	}
	if v := dst.Pix[0]; v != 0 {
		t.Errorf("CLAHE gradient: left edge: expected 0 got %d", v)
	}
	if v := dst.Pix[255]; v != 255 {
		t.Errorf("CLAHE gradient: right edge: expected 255 got %d", v)
	}
	// Inside one tile (tile 3 covers x in [96, 128)) the equalized slope is
	// much steeper than the original slope of 1.
	if d := int(dst.Pix[115]) - int(dst.Pix[110]); d < 10 {
		t.Errorf("CLAHE gradient: local contrast too low: dst[115]-dst[110] = %d", d)
	}
}

func TestCLAHEColorAndAlpha(t *testing.T) {
	// Dark and bright pure red pixels with a non-opaque alpha channel.
	// The alpha channel must be preserved and the zero green/blue channels
	// must stay zero (no color shift introduced by the luminance scaling).
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			i := src.PixOffset(x, y)
			if x < 2 {
				src.Pix[i+0] = 26
			} else {
				src.Pix[i+0] = 230
			}
			src.Pix[i+1] = 0
			src.Pix[i+2] = 0
			src.Pix[i+3] = 128
		}
	}
	g := New(CLAHE(1, 1000))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			i := dst.PixOffset(x, y)
			r, gg, b, a := dst.Pix[i+0], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]
			if a != 128 {
				t.Errorf("CLAHE color: (%d, %d): alpha: expected 128 got %d", x, y, a)
			}
			if gg != 0 || b != 0 {
				t.Errorf("CLAHE color: (%d, %d): expected zero g/b got %d/%d", x, y, gg, b)
			}
			if x < 2 && r != 0 {
				t.Errorf("CLAHE color: (%d, %d): expected r 0 got %d", x, y, r)
			}
			if x >= 2 && r != 255 {
				t.Errorf("CLAHE color: (%d, %d): expected r 255 got %d", x, y, r)
			}
		}
	}
}

func TestCLAHETinyTiles(t *testing.T) {
	// A grid finer than the image is clamped to one pixel per tile; every
	// 1-pixel tile is constant, so the result is the original image.
	src := image.NewGray(image.Rect(0, 0, 3, 3))
	for i := range src.Pix {
		src.Pix[i] = uint8(10 + i*20)
	}
	g := New(CLAHE(8, 2))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for i, v := range dst.Pix {
		d := int(v) - int(src.Pix[i])
		if d < -1 || d > 1 {
			t.Errorf("CLAHE tiny tiles: index %d: expected %d got %d", i, src.Pix[i], v)
		}
	}
}

func TestCLAHEEdgeTiles(t *testing.T) {
	// An 8x8 mid-gray image with one bright pixel in the corner tile and a
	// 4x4 grid (2x2 pixel tiles). The corner tile equalizes to 0/255, all
	// other tiles are constant and keep their values; pixels between tiles
	// are blended by the interpolation.
	src := image.NewGray(image.Rect(0, 0, 8, 8))
	for i := range src.Pix {
		src.Pix[i] = 128
	}
	src.Pix[src.PixOffset(1, 1)] = 255

	g := New(CLAHE(4, 1000))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if v := dst.Pix[dst.PixOffset(0, 0)]; v != 0 {
		t.Errorf("CLAHE edge tiles: (0, 0): expected 0 got %d", v)
	}
	if v := dst.Pix[dst.PixOffset(1, 1)]; v != 255 {
		t.Errorf("CLAHE edge tiles: (1, 1): expected 255 got %d", v)
	}
	if v := dst.Pix[dst.PixOffset(7, 7)]; v != 128 {
		t.Errorf("CLAHE edge tiles: (7, 7): expected 128 got %d", v)
	}
	// (2, 2) sits between the corner tile and the constant tiles: the
	// interpolation must produce a value strictly between 0 and 128.
	if v := dst.Pix[dst.PixOffset(2, 2)]; v == 0 || v >= 128 {
		t.Errorf("CLAHE edge tiles: (2, 2): expected interpolated value in (0, 128) got %d", v)
	}
}

func TestCLAHENonDivisibleSize(t *testing.T) {
	// The image size is not divisible by the grid size: tiles get different
	// sizes and the result must still be stable.
	src := makeGrayGradient(13, 7)
	g := New(CLAHE(4, 2))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if !dst.Bounds().Size().Eq(src.Bounds().Size()) {
		t.Fatalf("CLAHE non-divisible: bad dst bounds: %v", dst.Bounds())
	}
	if v := dst.Pix[0]; v > 1 {
		t.Errorf("CLAHE non-divisible: left edge: expected ~0 got %d", v)
	}
	if v := dst.Pix[12]; v < 254 {
		t.Errorf("CLAHE non-divisible: right edge: expected ~255 got %d", v)
	}
}

func TestCLAHEInvalidParams(t *testing.T) {
	// Invalid parameters are clamped and must produce a stable result.
	src := makeGrayGradient(5, 5)
	for _, f := range []Filter{
		CLAHE(0, 2),
		CLAHE(-3, 2),
		CLAHE(4, 0),
		CLAHE(4, -1),
		CLAHE(0, 0),
	} {
		g := New(f)
		dst := image.NewGray(g.Bounds(src.Bounds()))
		g.Draw(dst, src)
		if !dst.Bounds().Size().Eq(src.Bounds().Size()) {
			t.Errorf("CLAHE invalid params: bad dst bounds: %v", dst.Bounds())
		}
	}

	// Degenerate images must not panic.
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 1, 1),
		image.Rect(0, 0, 1, 10),
		image.Rect(0, 0, 10, 1),
	} {
		src := image.NewGray(r)
		g := New(CLAHE(8, 2))
		dst := image.NewGray(g.Bounds(src.Bounds()))
		g.Draw(dst, src)
	}
}

func TestCLAHEHalfDarkHalfBright(t *testing.T) {
	// The motivating use case: the left part is dark, the right part is
	// bright, and the boundary does not align with the tile grid. Interior
	// tiles are constant and keep their values while the tiles around the
	// boundary equalize locally, producing intermediate values that a
	// global contrast adjustment cannot create.
	src := image.NewGray(image.Rect(0, 0, 32, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 32; x++ {
			if x < 14 {
				src.Pix[y*32+x] = 40
			} else {
				src.Pix[y*32+x] = 200
			}
		}
	}
	g := New(CLAHE(4, 1000))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if v := dst.Pix[dst.PixOffset(0, 8)]; v != 40 {
		t.Errorf("CLAHE halves: far left: expected 40 got %d", v)
	}
	if v := dst.Pix[dst.PixOffset(31, 8)]; v != 200 {
		t.Errorf("CLAHE halves: far right: expected 200 got %d", v)
	}
	seen := map[uint8]bool{}
	for _, v := range dst.Pix {
		seen[v] = true
	}
	if len(seen) <= 2 {
		t.Errorf("CLAHE halves: expected intermediate values near the boundary, got %d distinct values", len(seen))
	}
}
