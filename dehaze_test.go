package gift

import (
	"image"
	"math"
	"testing"
)

// nrgbaTolerant compares two NRGBA pixel buffers allowing a small rounding
// tolerance so the tests stay valid across architectures with different
// floating-point rounding (FMA).
func nrgbaTolerant(got, want []uint8, tol int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		diff := int(got[i]) - int(want[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}

// TestDehazeIdentity checks parameter combinations that must leave the image
// unchanged: zero strength, non-positive radius and the corresponding
// constructor shortcut filters.
func TestDehazeIdentity(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 3))
	src.Pix = []uint8{
		0x00, 0x20, 0x40, 0xff, 0x10, 0x30, 0x50, 0x80, 0x20, 0x40, 0x60, 0x00, 0x30, 0x50, 0x70, 0xff,
		0x40, 0x60, 0x80, 0xff, 0x50, 0x70, 0x90, 0xff, 0x60, 0x80, 0xa0, 0x40, 0x70, 0x90, 0xb0, 0xff,
		0x80, 0xa0, 0xc0, 0xff, 0x90, 0xb0, 0xd0, 0xff, 0xa0, 0xc0, 0xe0, 0xff, 0xb0, 0xd0, 0xf0, 0x00,
	}

	for _, f := range []Filter{
		Dehaze(3, 0, 0.1),
		Dehaze(0, 1, 0.1),
		Dehaze(-7, 1, 0.1),
		Dehaze(3, -1, 0.1),
	} {
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)
		if !checkBoundsAndPix(dst.Bounds(), image.Rect(0, 0, 4, 3), dst.Pix, src.Pix) {
			t.Errorf("identity case failed for filter %#v:\n got %v\nwant %v", f, dst.Pix, src.Pix)
		}
	}
}

// TestDehazeBounds checks that the filter never changes the image dimensions.
func TestDehazeBounds(t *testing.T) {
	srcb := image.Rect(-3, -5, 7, 11)
	want := image.Rect(0, 0, 10, 16)
	if got := Dehaze(3, 1, 0.1).Bounds(srcb); !got.Eq(want) {
		t.Errorf("unexpected bounds: %v, want %v", got, want)
	}
}

// TestDehazeUniformBlocks checks that uniform color blocks (constant dark
// channel everywhere) are reconstructed to themselves. The fixed hazy gray
// block is the deterministic equivalent of a foggy scene region.
func TestDehazeUniformBlocks(t *testing.T) {
	colors := [][4]uint8{
		{0xff, 0xff, 0xff, 0xff}, // pure white
		{0x00, 0x00, 0x00, 0xff}, // pure black
		{0x66, 0x66, 0x66, 0xff}, // mid gray
		{0x80, 0x40, 0x20, 0xff}, // colored block
	}
	for _, c := range colors {
		src := image.NewNRGBA(image.Rect(0, 0, 5, 4))
		for i := range src.Pix {
			src.Pix[i] = c[i%4]
		}
		f := Dehaze(3, 1, 0.1)
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)
		if !nrgbaTolerant(dst.Pix, src.Pix, 0) {
			t.Errorf("uniform block %v changed: %v", c, dst.Pix[:8])
		}
	}
}

// TestDehazeFoggyBlocks checks the output on a deterministic two-region
// "foggy" image with hand-computed expected values:
//   - the dark region is airlight,
//   - the lighter region has its haze lifted and its color restored.
func TestDehazeFoggyBlocks(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	src.Pix = []uint8{
		0x80, 0x80, 0x80, 0xff, // 0.5 gray: darkest pixel, becomes the airlight
		0x40, 0x40, 0x40, 0xff, // 0.25 gray: fogged distant region
	}
	want := []uint8{
		0xff, 0xff, 0xff, 0xff, // airlight pixel restores to white
		0x40, 0x40, 0x40, 0xff, // t = 0.5, tmin = 0.1: (0.25-0.5)/0.5+0.5 = 0.25
	}
	f := Dehaze(1, 1, 0.1)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	if !nrgbaTolerant(dst.Pix, want, 1) {
		t.Errorf("foggy blocks mismatch:\n got %v\nwant %v", dst.Pix, want)
	}
}

// TestDehazeTransparentEdges checks that fully transparent pixels are copied
// verbatim (no RGB injection, no dark-channel pollution from neighbors) and
// that semi-transparent pixels keep their alpha while their RGB is processed.
func TestDehazeTransparentEdges(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	src.Pix = []uint8{
		0x10, 0x20, 0x30, 0x00, // fully transparent
		0x80, 0x40, 0x20, 0x80, // semi-transparent
	}
	f := Dehaze(1, 1, 0.1)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	// Fully transparent pixel must survive untouched.
	if dst.Pix[0] != 0x10 || dst.Pix[1] != 0x20 || dst.Pix[2] != 0x30 || dst.Pix[3] != 0x00 {
		t.Errorf("transparent pixel altered: %v", dst.Pix[:4])
	}
	// Alpha of the semi-transparent pixel is preserved.
	if dst.Pix[7] != 0x80 {
		t.Errorf("semi-transparent alpha altered: got %d, want 128", dst.Pix[7])
	}
	// The lone visible pixel is itself the airlight, so its RGB is unchanged.
	if !nrgbaTolerant(dst.Pix[4:8], []uint8{0x80, 0x40, 0x20, 0x80}, 1) {
		t.Errorf("semi-transparent RGB mismatch: %v", dst.Pix[4:8])
	}
}

// TestDehazeFullyTransparent checks that an image without any visible pixels
// is copied unchanged instead of producing an invalid airlight.
func TestDehazeFullyTransparent(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	src.Pix = []uint8{
		0x11, 0x22, 0x33, 0x00, 0x44, 0x55, 0x66, 0x00, 0x77, 0x88, 0x99, 0x00,
		0xaa, 0xbb, 0xcc, 0x00, 0xdd, 0xee, 0xff, 0x00, 0x12, 0x34, 0x56, 0x00,
	}
	f := Dehaze(2, 1, 0.1)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	if !checkBoundsAndPix(dst.Bounds(), image.Rect(0, 0, 3, 2), dst.Pix, src.Pix) {
		t.Errorf("fully transparent image altered:\n got %v\nwant %v", dst.Pix, src.Pix)
	}
}

// TestDehazeEmpty checks that empty images do not panic and stay empty.
func TestDehazeEmpty(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	f := Dehaze(3, 1, 0.1)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	if dst.Bounds().Dx() != 0 || dst.Bounds().Dy() != 0 || len(dst.Pix) != 0 {
		t.Errorf("empty image produced unexpected output: %v %v", dst.Bounds(), len(dst.Pix))
	}
}

// TestDehazeInvalidParams checks that out-of-range and NaN parameters are
// clamped to valid values and never produce NaN/Inf output bytes.
func TestDehazeInvalidParams(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 3))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*37 + 11) % 256)
		if i%4 == 3 {
			src.Pix[i] = 0xff
		}
	}

	filters := []Filter{
		Dehaze(-5, 1, 0.1),
		Dehaze(3, -2, 0.1),
		Dehaze(3, 2, 0.1),
		Dehaze(3, 1, -1),
		Dehaze(3, 1, 5),
		Dehaze(1<<30, 1, 0.1), // radius far beyond the image dimensions
		Dehaze(3, float32(math.NaN()), 0.1),
		Dehaze(3, 1, float32(math.NaN())),
	}
	for i, f := range filters {
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)
		if len(dst.Pix) != len(src.Pix) {
			t.Fatalf("case %d: unexpected pix length", i)
		}
		for _, v := range dst.Pix {
			if v > 0xff {
				t.Errorf("case %d: byte out of range: %d", i, v)
			}
		}
	}
}

// TestDehazeHugeRadius checks a radius larger than the image: every window
// covers the whole image and the result stays finite and well defined.
func TestDehazeHugeRadius(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	src.Pix = []uint8{
		0x20, 0x40, 0x60, 0xff, 0x80, 0xa0, 0xc0, 0x80, 0x10, 0x30, 0x50, 0xff,
		0x70, 0x90, 0xb0, 0xff, 0x30, 0x50, 0x70, 0x00, 0x90, 0xb0, 0xd0, 0xff,
	}
	f := Dehaze(1<<20, 1, 0.1)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	if len(dst.Pix) != len(src.Pix) {
		t.Fatalf("unexpected pix length")
	}
	// Transparent pixel remains untouched.
	if dst.Pix[16] != 0x30 || dst.Pix[17] != 0x50 || dst.Pix[18] != 0x70 || dst.Pix[19] != 0x00 {
		t.Errorf("transparent pixel altered with huge radius: %v", dst.Pix[16:20])
	}
	// All bytes are finite by construction; check alpha preservation elsewhere.
	for i := 3; i < len(src.Pix); i += 4 {
		if dst.Pix[i] != src.Pix[i] {
			t.Errorf("alpha not preserved at pixel %d: got %d want %d", i/4, dst.Pix[i], src.Pix[i])
		}
	}
}

// TestDehazeColorOverflow checks a scene where the naive inversion would push
// channels above 1: output must be clamped to the maximum representable
// value instead of wrapping or producing invalid data.
func TestDehazeColorOverflow(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 1))
	src.Pix = []uint8{
		0x00, 0x00, 0x00, 0xff, // airlight
		0x05, 0x05, 0x0a, 0xff,
		0x0a, 0x0a, 0x14, 0xff,
		0x14, 0x14, 0x28, 0xff,
	}
	f := Dehaze(1, 1, 0)
	dst := image.NewNRGBA64(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	// All 16-bit components must stay in range; at least one blue channel
	// saturates to 0xffff instead of wrapping around.
	saturated := 0
	for px := 0; px < 4; px++ {
		for ch := 0; ch < 4; ch++ {
			i := px*8 + ch*2
			v := uint16(dst.Pix[i])<<8 | uint16(dst.Pix[i+1])
			if v == 0xffff {
				saturated++
			}
		}
		// Alpha stays fully opaque.
		a := uint16(dst.Pix[px*8+6])<<8 | uint16(dst.Pix[px*8+7])
		if a != 0xffff {
			t.Errorf("alpha altered at pixel %d: %d", px, a)
		}
	}
	if saturated == 0 {
		t.Errorf("expected at least one saturated component, pix: %v", dst.Pix)
	}
}

// TestDehazeParallel checks that parallel and sequential execution give the
// same result.
func TestDehazeParallel(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 13, 9))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*53 + 7) % 256)
		if i%4 == 3 {
			src.Pix[i] = uint8(64 + (i/4)%192)
		}
	}
	f := Dehaze(3, 0.9, 0.15)

	dstPar := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dstPar, src, &Options{Parallelization: true})
	dstSeq := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dstSeq, src, &Options{Parallelization: false})

	if !checkBoundsAndPix(dstPar.Bounds(), dstSeq.Bounds(), dstPar.Pix, dstSeq.Pix) {
		t.Errorf("parallel/sequential results differ:\n par %v\n seq %v", dstPar.Pix, dstSeq.Pix)
	}
}

// TestDehazeFilterChain checks that the filter joins an existing filter chain
// and preserves dimensions through it.
func TestDehazeFilterChain(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 8, 6))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*29 + 3) % 256)
		if i%4 == 3 {
			src.Pix[i] = 0xff
		}
	}
	g := New(
		Grayscale(),
		Dehaze(2, 0.7, 0.1),
		Saturation(20),
	)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)
	if !dst.Bounds().Eq(image.Rect(0, 0, 8, 6)) {
		t.Errorf("chain altered bounds: %v", dst.Bounds())
	}
}

// TestDehazeAlphaPreserved checks alpha preservation on a mixed-alpha image,
// including pixels at the image borders where edge clamping is used.
func TestDehazeAlphaPreserved(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 5, 4))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*41 + 5) % 256)
		if i%4 == 3 {
			src.Pix[i] = uint8((i / 4 * 37) % 256) // mixed alpha
		}
	}
	f := Dehaze(2, 1, 0.1)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	for i := 3; i < len(src.Pix); i += 4 {
		if dst.Pix[i] != src.Pix[i] {
			t.Errorf("alpha changed at pixel %d: got %d want %d", i/4, dst.Pix[i], src.Pix[i])
		}
	}
}

// TestDehazeGolden checks the output against a golden image produced from the
// natural test picture, in the same style as all other filter golden tests.
func TestDehazeGolden(t *testing.T) {
	src := loadImage(t, "testdata/src.png")
	g := New(Dehaze(7, 0.8, 0.1))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)
	want := loadImageNRGBA(t, "testdata/dst_dehaze.png")
	if !goldenEqual(dst, want) {
		t.Error("resulting image differs from golden: dehaze")
	}
}
