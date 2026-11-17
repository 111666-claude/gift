package gift

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// nrgbaFromPix builds an NRGBA image with the given bounds and pixels.
func nrgbaFromPix(b image.Rectangle, pix []uint8) *image.NRGBA {
	img := image.NewNRGBA(b)
	img.Pix = pix
	return img
}

// bloomPixEqual compares two NRGBA pixel slices allowing for a small
// rounding tolerance introduced by the internal 16-bit temporary images.
func bloomPixEqual(pix1, pix2 []uint8, tolerance uint8) bool {
	if len(pix1) != len(pix2) {
		return false
	}
	for i := range pix1 {
		d := int(pix1[i]) - int(pix2[i])
		if d < 0 {
			d = -d
		}
		if d > int(tolerance) {
			return false
		}
	}
	return true
}

func getNRGBAPix(t *testing.T, f Filter, src *image.NRGBA, parallel bool) *image.NRGBA {
	t.Helper()
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, &Options{Parallelization: parallel})
	return dst
}

// TestBloomBounds makes sure the output always keeps the source dimensions.
func TestBloomBounds(t *testing.T) {
	f := Bloom(5, 1, 0.5)
	for _, b := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 1, 1),
		image.Rect(-3, -7, 4, 9),
		image.Rect(10, 20, 31, 42),
	} {
		got := f.Bounds(b)
		want := image.Rect(0, 0, b.Dx(), b.Dy())
		if !got.Eq(want) {
			t.Errorf("bounds mismatch for %v: got %v, want %v", b, got, want)
		}
	}
}

// TestBloomEmpty checks that empty images do not panic and produce empty
// output of the expected size.
func TestBloomEmpty(t *testing.T) {
	for _, parallel := range []bool{true, false} {
		src := image.NewNRGBA(image.Rect(0, 0, 0, 0))
		dst := image.NewNRGBA(Bloom(2, 1, 0).Bounds(src.Bounds()))
		Bloom(2, 1, 0).Draw(dst, src, &Options{Parallelization: parallel})
		if dst.Bounds() != src.Bounds() {
			t.Errorf("empty image bounds changed: %v", dst.Bounds())
		}
	}
}

// TestBloomIdentity covers every parameter combination that must leave the
// source untouched.
func TestBloomIdentity(t *testing.T) {
	srcPix := []uint8{
		0x10, 0x20, 0x30, 0x40, 0x80, 0x90, 0xA0, 0xB0,
		0xC0, 0xD0, 0xE0, 0xFF, 0x00, 0x00, 0x00, 0x00,
	}
	srcb := image.Rect(0, 0, 2, 2)

	cases := []struct {
		desc              string
		radius            int
		intensity, thresh float32
	}{
		{"zero radius", 0, 1, 0},
		{"negative radius", -5, 1, 0},
		{"zero intensity", 3, 0, 0},
		{"negative intensity", 3, -1, 0},
		{"NaN intensity", 3, float32(math.NaN()), 0},
		{"+Inf intensity", 3, float32(math.Inf(1)), 0},
		{"-Inf intensity", 3, float32(math.Inf(-1)), 0},
	}
	for _, d := range cases {
		for _, parallel := range []bool{true, false} {
			src := nrgbaFromPix(srcb, append([]uint8(nil), srcPix...))
			dst := getNRGBAPix(t, Bloom(d.radius, d.intensity, d.thresh), src, parallel)
			if !bloomPixEqual(dst.Pix, srcPix, 0) {
				t.Errorf("case %q (parallel=%v): expected identity, got %v", d.desc, parallel, dst.Pix)
			}
		}
	}
}

// TestBloomDarkUntouched verifies that pixels below the threshold, including
// fully transparent ones, are not modified at all.
func TestBloomDarkUntouched(t *testing.T) {
	// One bright pixel surrounded by dark and transparent pixels.
	srcb := image.Rect(0, 0, 5, 1)
	src := image.NewNRGBA(srcb)
	src.SetNRGBA(2, 0, colorOf(255, 255, 255, 255))
	// Everything else stays the zero value: fully transparent black.

	f := Bloom(1, 1, 0.9)
	for _, parallel := range []bool{true, false} {
		dst := getNRGBAPix(t, f, src, parallel)

		for x := 0; x < 5; x++ {
			if x == 2 {
				continue
			}
			i := x * 4
			// The glow may reach neighbors only through their alpha channel
			// and premultiplied color; a fully dark pixel far enough away
			// (outside the kernel reach) must remain exactly transparent.
			if x == 0 || x == 4 {
				if dst.Pix[i] != 0 || dst.Pix[i+1] != 0 || dst.Pix[i+2] != 0 || dst.Pix[i+3] != 0 {
					t.Errorf("parallel=%v: dark pixel at x=%d was modified: %v", parallel, x, dst.Pix[i:i+4])
				}
			}
		}
	}
}

// TestBloomOpaqueDarkUntouched checks that an opaque dark region below the
// threshold is never globally brightened: only pixels within the blur
// radius of a bright source can change, and the lift there is the additive
// glow rather than a uniform offset.
func TestBloomOpaqueDarkUntouched(t *testing.T) {
	srcb := image.Rect(0, 0, 5, 1)
	src := image.NewNRGBA(srcb)
	for x := 0; x < 4; x++ {
		src.SetNRGBA(x, 0, colorOf(10, 10, 10, 255))
	}
	src.SetNRGBA(4, 0, colorOf(255, 255, 255, 255))

	f := Bloom(1, 1, 0.5)
	dst := getNRGBAPix(t, f, src, true)

	// The leftmost pixels are out of blur reach of the bright pixel: they
	// must remain exactly unchanged.
	for _, x := range []int{0, 1, 2} {
		want := []uint8{10, 10, 10, 255}
		if !bloomPixEqual(dst.Pix[x*4:x*4+4], want, 0) {
			t.Errorf("dark opaque pixel x=%d changed: got %v, want %v", x, dst.Pix[x*4:x*4+4], want)
		}
	}
}

// TestBloomHaloDecay uses a bright point on a black background and verifies
// that the glow is strongest at the source and decays monotonically outward.
func TestBloomHaloDecay(t *testing.T) {
	const radius = 3
	const n = 11
	srcb := image.Rect(0, 0, n, 1)
	src := image.NewNRGBA(srcb)
	center := n / 2
	src.SetNRGBA(center, 0, colorOf(255, 255, 255, 255))

	f := Bloom(radius, 1, 0)
	dst := getNRGBAPix(t, f, src, true)

	alphaAt := func(x int) uint8 { return dst.Pix[x*4+3] }

	if alphaAt(center) != 255 {
		t.Errorf("center alpha: got %d, want 255", alphaAt(center))
	}

	// Alpha of the glow must never increase while moving away from the
	// center. Within the kernel reach it decays strictly; past the reach it
	// stays at zero.
	for x := center + 1; x < n; x++ {
		if alphaAt(x) > alphaAt(x-1) {
			t.Errorf("glow alpha increased with distance: x=%d (%d) > x=%d (%d)",
				x, alphaAt(x), x-1, alphaAt(x-1))
		}
	}
	for x := center - 1; x >= 0; x-- {
		if alphaAt(x) > alphaAt(x+1) {
			t.Errorf("glow alpha increased with distance: x=%d (%d) > x=%d (%d)",
				x, alphaAt(x), x+1, alphaAt(x+1))
		}
	}

	// Symmetry of the halo around the bright point.
	for dx := 1; dx <= center; dx++ {
		l := dst.Pix[(center-dx)*4 : (center-dx+1)*4]
		r := dst.Pix[(center+dx)*4 : (center+dx+1)*4]
		if !bloomPixEqual(l, r, 1) {
			t.Errorf("asymmetric glow at dx=%d: %v vs %v", dx, l, r)
		}
	}

	// Pixels beyond the kernel reach stay exactly transparent.
	if alphaAt(0) != 0 || alphaAt(n-1) != 0 {
		t.Errorf("glow leaked beyond kernel reach: %d, %d", alphaAt(0), alphaAt(n-1))
	}

	// The glow color must stay gray (white light), not pick up fringe colors.
	for x := 0; x < n; x++ {
		i := x * 4
		r, g, b := dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2]
		if maxu8(r, g, b)-minu8(r, g, b) > 2 {
			t.Errorf("glow at x=%d is not neutral gray: %d %d %d", x, r, g, b)
		}
	}
}

// TestBloomExactValues checks the actual composited bytes for a small image
// against hand-computed premultiplied-alpha results.
func TestBloomExactValues(t *testing.T) {
	// 1x3 image: transparent black, opaque white, transparent black.
	srcb := image.Rect(0, 0, 3, 1)
	src := image.NewNRGBA(srcb)
	src.SetNRGBA(1, 0, colorOf(255, 255, 255, 255))

	// radius 1 -> sigma = 1/3; normalized kernel {w1, w0, w1} with
	// w0 = exp(0), w1 = exp(-4.5), sum = w0 + 2*w1.
	w0 := 1.0
	w1 := math.Exp(-4.5)
	sum := w0 + 2*w1
	k1 := float32(w1 / sum) // outer tap, roughly 0.01087

	// At a neighbor pixel the blurred glow alpha equals k1 while the straight
	// glow color stays white (1,1,1). Additive compositing over a transparent
	// black pixel therefore yields straight white (stored as 255,255,255) with
	// an 8-bit alpha of round(k1*255) = 3.
	neighborAlpha := f32u8(k1 * 255)
	if neighborAlpha != 3 {
		t.Fatalf("hand-computed neighbor alpha changed: got %d", neighborAlpha)
	}

	f := Bloom(1, 1, 0)
	dst := getNRGBAPix(t, f, src, false)

	want := []uint8{
		255, 255, 255, 3,
		255, 255, 255, 255,
		255, 255, 255, 3,
	}
	if !bloomPixEqual(dst.Pix, want, 1) {
		t.Errorf("exact bloom output mismatch:\n got %v\nwant %v", dst.Pix, want)
	}
}

// TestBloomThresholdGating verifies the threshold selects which pixels feed
// the glow and that a threshold of 1 on a non-HDR image produces no glow.
func TestBloomThresholdGating(t *testing.T) {
	srcb := image.Rect(0, 0, 5, 1)

	build := func() *image.NRGBA {
		src := image.NewNRGBA(srcb)
		src.SetNRGBA(0, 0, colorOf(40, 40, 40, 255))    // lum ~= 0.157
		src.SetNRGBA(2, 0, colorOf(200, 200, 200, 255)) // lum ~= 0.784
		src.SetNRGBA(4, 0, colorOf(10, 10, 10, 255))    // dark, beyond radius-1 reach
		return src
	}

	// High threshold: only the bright middle pixel glows; the dark pixel on
	// the far right is outside the radius-1 kernel reach and stays unchanged.
	fHigh := Bloom(1, 1, 0.5)
	dstHigh := getNRGBAPix(t, fHigh, build(), true)
	right := dstHigh.Pix[4*4 : 4*4+4]
	if !bloomPixEqual(right, []uint8{10, 10, 10, 255}, 0) {
		t.Errorf("dark pixel affected by high-threshold bloom: %v", right)
	}

	// Threshold of 1: nothing in a normal [0,1] image exceeds it, so the
	// whole image must be copied unchanged.
	fMax := Bloom(2, 2, 1)
	src := build()
	dstMax := getNRGBAPix(t, fMax, src, true)
	if !bloomPixEqual(dstMax.Pix, src.Pix, 0) {
		t.Errorf("threshold=1 changed the image: %v", dstMax.Pix)
	}

	// Threshold above 1 is clamped to 1 and behaves the same way.
	fOver := Bloom(2, 2, 5)
	dstOver := getNRGBAPix(t, fOver, build(), true)
	if !bloomPixEqual(dstOver.Pix, src.Pix, 0) {
		t.Errorf("threshold>1 was not clamped: %v", dstOver.Pix)
	}

	// NaN threshold is replaced by 1: again an identity result.
	fNaN := Bloom(2, 2, float32(math.NaN()))
	dstNaN := getNRGBAPix(t, fNaN, build(), true)
	if !bloomPixEqual(dstNaN.Pix, src.Pix, 0) {
		t.Errorf("NaN threshold was not handled: %v", dstNaN.Pix)
	}
}

// TestBloomSemiTransparent checks the alpha handling: the glow of a
// semi-transparent bright pixel spreads into the neighboring transparent
// pixels with correctly premultiplied colors.
func TestBloomSemiTransparent(t *testing.T) {
	srcb := image.Rect(0, 0, 3, 1)
	src := image.NewNRGBA(srcb)
	// 50% opaque white in the middle.
	src.SetNRGBA(1, 0, colorOf(255, 255, 255, 128))

	f := Bloom(1, 1, 0)
	dst := getNRGBAPix(t, f, src, true)

	w0 := 1.0
	w1 := math.Exp(-4.5)
	k1 := float32(w1 / (w0 + 2*w1))

	// Center: blurred alpha = 0.5*w0/sum. Additive union with original alpha.
	centerBlurredA := float32(0.5) * float32(w0/(w0+2*w1))
	wantCenterA := f32u8(minf32(0.5+centerBlurredA, 1) * 255)

	center := dst.Pix[1*4 : 1*4+4]
	if absInt(int(center[3]), int(wantCenterA)) > 1 {
		t.Errorf("semi-transparent center alpha: got %d, want %d", center[3], wantCenterA)
	}
	// Straight color of a white glow must remain white.
	if center[0] != 255 || center[1] != 255 || center[2] != 255 {
		t.Errorf("semi-transparent center lost its white color: %v", center)
	}

	// Neighbor: originally transparent, only the glow contributes. Colors are
	// stored straight, so it is white with a small alpha.
	wantNeighborA := f32u8(0.5 * k1 * 255)
	for _, x := range []int{0, 2} {
		px := dst.Pix[x*4 : x*4+4]
		if absInt(int(px[3]), int(wantNeighborA)) > 1 {
			t.Errorf("neighbor x=%d alpha: got %d, want %d", x, px[3], wantNeighborA)
		}
		// A white glow keeps straight RGB at full value regardless of alpha.
		if px[0] != 255 || px[1] != 255 || px[2] != 255 {
			t.Errorf("neighbor x=%d lost its white glow color: %v", x, px)
		}
	}
}

// TestBloomExtremePixels ensures values never overflow and no NaN is
// produced for the brightest possible pixels.
func TestBloomExtremePixels(t *testing.T) {
	srcb := image.Rect(0, 0, 3, 3)
	src := image.NewNRGBA(srcb)
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			src.SetNRGBA(x, y, colorOf(255, 255, 255, 255))
		}
	}

	dst := getNRGBAPix(t, Bloom(2, 100, 0), src, true)
	for i := 0; i < len(dst.Pix); i++ {
		if dst.Pix[i] != 255 {
			t.Fatalf("fully white image must stay fully white, pix[%d]=%d", i, dst.Pix[i])
		}
	}
}

// TestBloomHugeRadius checks that an oversized radius is clamped instead of
// allocating a gigantic kernel, and still produces a valid result.
func TestBloomHugeRadius(t *testing.T) {
	srcb := image.Rect(0, 0, 4, 3)
	src := image.NewNRGBA(srcb)
	src.SetNRGBA(0, 0, colorOf(255, 255, 255, 255))

	dst := getNRGBAPix(t, Bloom(math.MaxInt32, 1, 0), src, true)
	if !dst.Bounds().Eq(image.Rect(0, 0, 4, 3)) {
		t.Fatalf("bounds changed with huge radius: %v", dst.Bounds())
	}
	for i := 0; i < len(dst.Pix); i++ {
		if dst.Pix[i] > 255 {
			t.Fatalf("invalid output byte with huge radius: %v", dst.Pix)
		}
	}

	// The bright pixel must remain at full opacity even with the clamp.
	if dst.Pix[3] != 255 {
		t.Errorf("bright pixel alpha with huge radius: got %d, want 255", dst.Pix[3])
	}
}

// TestBloomIntensity verifies that increasing the intensity strengthens the
// glow while leaving pixels without glow untouched.
func TestBloomIntensity(t *testing.T) {
	build := func() *image.NRGBA {
		src := image.NewNRGBA(image.Rect(0, 0, 7, 1))
		src.SetNRGBA(3, 0, colorOf(255, 255, 255, 255))
		return src
	}

	weak := getNRGBAPix(t, Bloom(2, 0.25, 0), build(), true)
	strong := getNRGBAPix(t, Bloom(2, 1, 0), build(), true)

	for x := 0; x < 7; x++ {
		i := x * 4
		if x == 3 {
			continue
		}
		if strong.Pix[i+3] < weak.Pix[i+3] {
			t.Errorf("x=%d: stronger intensity produced weaker glow alpha: %d < %d",
				x, strong.Pix[i+3], weak.Pix[i+3])
		}
	}

	// Far edges are outside the kernel and remain transparent at any intensity.
	if weak.Pix[3] != 0 || strong.Pix[3] != 0 || weak.Pix[6*4+3] != 0 || strong.Pix[6*4+3] != 0 {
		t.Errorf("edge pixel should stay transparent: weak=%v strong=%v", weak.Pix, strong.Pix)
	}
}

// TestBloomFilterChain makes sure the filter plugs into the regular filter
// chain, keeps the bounds intact and cooperates with other filters.
func TestBloomFilterChain(t *testing.T) {
	src := image.NewNRGBA(image.Rect(-2, -3, 3, 2))
	for y := src.Bounds().Min.Y; y < src.Bounds().Max.Y; y++ {
		for x := src.Bounds().Min.X; x < src.Bounds().Max.X; x++ {
			if x == 0 && y == 0 {
				src.SetNRGBA(x, y, colorOf(255, 255, 255, 255))
			} else {
				src.SetNRGBA(x, y, colorOf(20, 20, 30, 255))
			}
		}
	}

	g := New(
		Grayscale(),
		Bloom(2, 0.7, 0.2),
		GaussianBlur(0.5),
	)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	wantBounds := image.Rect(0, 0, 5, 5)
	if !dst.Bounds().Eq(wantBounds) {
		t.Errorf("chain bounds: got %v, want %v", dst.Bounds(), wantBounds)
	}
	if len(dst.Pix) != 5*5*4 {
		t.Errorf("chain output size mismatch: %d bytes", len(dst.Pix))
	}

	// After Grayscale all channels must be equal (GaussianBlur and Bloom of
	// neutral-gray light keep them equal).
	for i := 0; i < len(dst.Pix); i += 4 {
		r, gg, b := dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2]
		if maxu8(r, gg, b)-minu8(r, gg, b) > 1 {
			t.Errorf("chained bloom introduced a color cast at pixel %d: %d %d %d", i/4, r, gg, b)
		}
	}
}

// TestBloomParallelism checks that the parallel and serial code paths yield
// the same pixels.
func TestBloomParallelism(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 13, 7))
	for y := 0; y < 7; y++ {
		for x := 0; x < 13; x++ {
			if (x+y)%3 == 0 {
				src.SetNRGBA(x, y, colorOf(220, 180, 120, 200))
			} else if (x+y)%2 == 0 {
				src.SetNRGBA(x, y, colorOf(30, 20, 40, 255))
			}
		}
	}

	f := Bloom(3, 0.9, 0.15)
	serial := getNRGBAPix(t, f, src, false)
	parallel := getNRGBAPix(t, f, src, true)
	if !bloomPixEqual(serial.Pix, parallel.Pix, 0) {
		t.Errorf("parallel and serial results differ")
	}
}

// TestBloomDifferentImageTypes exercises the filter through the generic pixel
// accessor with an RGBA (premultiplied) and a Gray source.
func TestBloomDifferentImageTypes(t *testing.T) {
	rgba := image.NewRGBA(image.Rect(0, 0, 3, 1))
	rgba.Set(1, 0, colorOf(255, 255, 255, 255))
	dstRGBA := image.NewNRGBA(Bloom(1, 1, 0).Bounds(rgba.Bounds()))
	Bloom(1, 1, 0).Draw(dstRGBA, rgba, nil)
	if dstRGBA.Pix[1*4+3] != 255 {
		t.Errorf("RGBA source: bright pixel lost opacity: %v", dstRGBA.Pix)
	}

	gray := image.NewGray(image.Rect(0, 0, 3, 1))
	gray.Set(1, 0, colorOf(255, 255, 255, 255))
	dstGray := image.NewNRGBA(Bloom(1, 1, 0.5).Bounds(gray.Bounds()))
	Bloom(1, 1, 0.5).Draw(dstGray, gray, nil)
	if dstGray.Bounds() != image.Rect(0, 0, 3, 1) {
		t.Errorf("gray source bounds changed: %v", dstGray.Bounds())
	}
	if dstGray.Pix[1*4+3] != 255 {
		t.Errorf("Gray source: bright pixel lost opacity: %v", dstGray.Pix)
	}
}

func colorOf(r, g, b, a uint8) color.NRGBA {
	return color.NRGBA{R: r, G: g, B: b, A: a}
}

func maxu8(vals ...uint8) uint8 {
	m := vals[0]
	for _, v := range vals[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func minu8(vals ...uint8) uint8 {
	m := vals[0]
	for _, v := range vals[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func absInt(a, b int) int {
	if a > b {
		return a - b
	}
	return b - a
}
