package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"
)

// makeCurveGrayRamp returns an 8-bit gray image with a horizontal ramp from
// 0 to 255.
func makeCurveGrayRamp(w, h int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Pix[y*w+x] = uint8(x * 0xff / (w - 1))
		}
	}
	return img
}

// drawCurve applies the filter to src and returns the result.
func drawCurve(src image.Image, f Filter) draw.Image {
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	return dst
}

func TestCurveIdentity(t *testing.T) {
	// A curve without control points, a single point on the diagonal and
	// unordered duplicate points on the diagonal are all the identity and
	// must return the image unchanged. The image includes a fully transparent
	// pixel with non-zero color channels.
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(x * 60)
			src.Pix[i+1] = uint8(y * 80)
			src.Pix[i+2] = uint8(255 - x*30)
			src.Pix[i+3] = uint8(255 - y*40)
		}
	}
	i := src.PixOffset(2, 2)
	src.Pix[i+0], src.Pix[i+1], src.Pix[i+2], src.Pix[i+3] = 10, 20, 30, 0

	for _, f := range []Filter{
		Curve(),
		Curve(CurvePoint{0.5, 0.5}),
		Curve(CurvePoint{0, 0}, CurvePoint{1, 1}),
		Curve(
			CurvePoint{1, 1},
			CurvePoint{0.25, 0.25},
			CurvePoint{0.75, 0.75},
			CurvePoint{0, 0},
			CurvePoint{0.25, 0.25},
		),
	} {
		dst := drawCurve(src, f).(*image.NRGBA)
		if !dst.Rect.Eq(src.Rect) {
			t.Fatalf("Curve identity: bad bounds: expected %v got %v", src.Rect, dst.Rect)
		}
		for i := range src.Pix {
			if dst.Pix[i] != src.Pix[i] {
				t.Fatalf("Curve identity: index %d: expected %d got %d", i, src.Pix[i], dst.Pix[i])
			}
		}
	}
}

func TestCurveGrayLinear(t *testing.T) {
	// Two control points define a straight line through them: a curve from
	// (0, 0.2) to (1, 0.8) must map the gray value v to 0.2 + 0.6*v/255.
	src := makeCurveGrayRamp(256, 1)
	f := Curve(CurvePoint{0, 0.2}, CurvePoint{1, 0.8})
	dst := drawCurve(src, f).(*image.NRGBA)

	for x := 0; x < 256; x++ {
		want := 0.2 + 0.6*float64(x)/255
		got := float64(dst.Pix[dst.PixOffset(x, 0)])
		if math.Abs(got-want*255) > 1.0 {
			t.Errorf("Curve linear: x %d: expected %.2f got %.0f", x, want*255, got)
		}
	}
}

func TestCurveInterpolatesControlPoints(t *testing.T) {
	// The monotone cubic interpolation must pass through every control point
	// and stay within the control value range.
	points := []CurvePoint{
		{0, 0},
		{0.1, 0.02},
		{0.3, 0.35},
		{0.5, 0.4},
		{0.7, 0.85},
		{1, 1},
	}
	pts := prepareCurvePoints(points)
	m := curveTangents(pts)

	for _, p := range points {
		got := curveEval(pts, m, float64(p.X))
		if math.Abs(got-float64(p.Y)) > 1.0e-6 {
			t.Errorf("Curve interpolation: x %v: expected %v got %v", p.X, p.Y, got)
		}
	}

	// Dense sampling: finite, within [0, 1], and non-decreasing.
	prev := -1.0
	for i := 0; i <= 10000; i++ {
		x := float64(i) / 10000
		v := curveEval(pts, m, x)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Curve interpolation: x %v: non-finite value %v", x, v)
		}
		if v < 0 || v > 1 {
			t.Fatalf("Curve interpolation: x %v: value %v outside [0, 1]", x, v)
		}
		if v < prev {
			t.Fatalf("Curve interpolation: x %v: value %v below previous %v", x, v, prev)
		}
		prev = v
	}
}

func TestCurveMonotoneNoOvershoot(t *testing.T) {
	// Five control points with a flat middle section. A non-shape-preserving
	// interpolant (e.g. plain Catmull-Rom) would overshoot above 0.8 between
	// the control points. PCHIP must keep the section flat.
	f := Curve(
		CurvePoint{0, 0},
		CurvePoint{0.25, 0.8},
		CurvePoint{0.5, 0.8},
		CurvePoint{0.75, 0.8},
		CurvePoint{1, 1},
	)
	src := makeCurveGrayRamp(256, 1)
	dst := drawCurve(src, f).(*image.NRGBA)

	prev := -1
	for x := 0; x < 256; x++ {
		v := int(dst.Pix[dst.PixOffset(x, 0)])
		if v < prev {
			t.Errorf("Curve no overshoot: x %d: value %d below previous %d", x, v, prev)
		}
		prev = v
	}
	for x := 64; x <= 191; x++ {
		want := 0.8 * 255
		got := float64(dst.Pix[dst.PixOffset(x, 0)])
		if math.Abs(got-want) > 1.0 {
			t.Errorf("Curve no overshoot: flat section x %d: expected %.2f got %.0f", x, want, got)
		}
	}
	if v := dst.Pix[dst.PixOffset(0, 0)]; v != 0 {
		t.Errorf("Curve no overshoot: x 0: expected 0 got %d", v)
	}
	if v := dst.Pix[dst.PixOffset(255, 0)]; v != 255 {
		t.Errorf("Curve no overshoot: x 255: expected 255 got %d", v)
	}
}

func TestCurveColorAndAlpha(t *testing.T) {
	// Colors must keep their channel ratios when the luminance is scaled, and
	// the alpha channel must be preserved.
	src := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	colors := []color.NRGBA{
		{200, 100, 50, 255},
		{26, 0, 0, 128},
		{120, 140, 180, 64},
	}
	for x, c := range colors {
		i := src.PixOffset(x, 0)
		src.Pix[i+0] = c.R
		src.Pix[i+1] = c.G
		src.Pix[i+2] = c.B
		src.Pix[i+3] = c.A
	}

	f := Curve(
		CurvePoint{0, 0},
		CurvePoint{0.5, 0.7},
		CurvePoint{1, 1},
	)
	dst := drawCurve(src, f).(*image.NRGBA)

	for x := 0; x < 3; x++ {
		i := dst.PixOffset(x, 0)
		r, g, b, a := dst.Pix[i+0], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3]
		srcPix := src.Pix[src.PixOffset(x, 0):]
		if a != srcPix[3] {
			t.Errorf("Curve color: pixel %d: alpha: expected %d got %d", x, srcPix[3], a)
		}

		srcPixF := [3]float32{
			float32(srcPix[0]) / 255,
			float32(srcPix[1]) / 255,
			float32(srcPix[2]) / 255,
		}
		y := 0.299*srcPixF[0] + 0.587*srcPixF[1] + 0.114*srcPixF[2]
		y2 := float32(0.0)
		if y > 1.0e-6 {
			// y2 = curve(y): only the endpoint/segment is needed for the
			// expected ratio check below, so evaluate the filter's LUT.
			y2 = f.(*curveFilter).lut[curveLUTIndex(y)]
		}

		got := [3]float32{float32(r) / 255, float32(g) / 255, float32(b) / 255}
		for c := 0; c < 3; c++ {
			want := srcPixF[c] * (y2 / y)
			// Saturated channels are clamped, so compare only unsaturated
			// ones against the expected scaled value.
			if want <= 1 {
				if math.Abs(float64(got[c]-want)) > 2.0/255 {
					t.Errorf("Curve color: pixel %d channel %d: expected %.4f got %.4f", x, c, want, got[c])
				}
			}
		}
	}
}

func TestCurveBlackAndWhitePoints(t *testing.T) {
	// Overriding the black point lifts pure black to gray: with a zero
	// luminance the curve value is added uniformly to all channels.
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	src.Pix = []uint8{
		0, 0, 0, 255,
		255, 255, 255, 255,
	}
	f := Curve(CurvePoint{0, 0.1}, CurvePoint{1, 0.9})
	dst := drawCurve(src, f).(*image.NRGBA)

	for x := 0; x < 2; x++ {
		i := dst.PixOffset(x, 0)
		r, g, b := dst.Pix[i+0], dst.Pix[i+1], dst.Pix[i+2]
		if r != g || g != b {
			t.Errorf("Curve black/white: pixel %d: expected neutral color got (%d, %d, %d)", x, r, g, b)
		}
	}
	// Black -> 0.1 (26), white -> 0.9 (229).
	if got := dst.Pix[dst.PixOffset(0, 0)]; math.Abs(float64(got)-26) > 1 {
		t.Errorf("Curve black/white: black point: expected 26 got %d", got)
	}
	if got := dst.Pix[dst.PixOffset(1, 0)]; math.Abs(float64(got)-229) > 1 {
		t.Errorf("Curve black/white: white point: expected 229 got %d", got)
	}
}

func TestCurvePointNormalization(t *testing.T) {
	// Duplicate X values keep the last point, unsorted points are accepted,
	// non-finite points are dropped, and missing endpoints are added.
	src := makeCurveGrayRamp(256, 1)
	reference := drawCurve(src, Curve(CurvePoint{0.5, 0.6}))

	equivalent := []struct {
		name string
		f    Filter
	}{
		{"duplicate", Curve(CurvePoint{0.5, 0.2}, CurvePoint{0.5, 0.6})},
		{"unsorted", Curve(CurvePoint{1, 1}, CurvePoint{0.5, 0.6}, CurvePoint{0, 0})},
		{
			"non-finite",
			Curve(
				CurvePoint{float32(math.NaN()), 0.5},
				CurvePoint{0.5, float32(math.NaN())},
				CurvePoint{0.5, 0.6},
				CurvePoint{float32(math.Inf(1)), float32(math.Inf(-1))},
			),
		},
		{"auto endpoints", Curve(CurvePoint{0, 0}, CurvePoint{0.5, 0.6}, CurvePoint{1, 1})},
	}
	for _, d := range equivalent {
		dst := drawCurve(src, d.f).(*image.NRGBA)
		ref := reference.(*image.NRGBA)
		for i := range ref.Pix {
			if dst.Pix[i] != ref.Pix[i] {
				t.Fatalf("Curve normalization (%s): index %d: expected %d got %d", d.name, i, ref.Pix[i], dst.Pix[i])
			}
		}
	}

	// The curve passes through (0.5, 0.6): the mid-gray pixel maps to 153.
	got := reference.(*image.NRGBA).Pix[reference.(*image.NRGBA).PixOffset(128, 0)]
	if math.Abs(float64(got)-153) > 2 {
		t.Errorf("Curve normalization: mid-gray: expected ~153 got %d", got)
	}

	// Out-of-range coordinates are clamped: (-1, 2) becomes (0, 1).
	clamped := drawCurve(src, Curve(CurvePoint{-1, 2}, CurvePoint{1, 1})).(*image.NRGBA)
	for x := 0; x < 256; x++ {
		if v := clamped.Pix[clamped.PixOffset(x, 0)]; v != 255 {
			t.Fatalf("Curve normalization: clamped x %d: expected 255 got %d", x, v)
		}
	}

	// All points invalid: the curve is the identity.
	invalid := drawCurve(src, Curve(
		CurvePoint{float32(math.NaN()), float32(math.NaN())},
		CurvePoint{float32(math.Inf(1)), 0.5},
	)).(*image.NRGBA)
	for x := 0; x < 256; x++ {
		want := uint8(x)
		if v := invalid.Pix[invalid.PixOffset(x, 0)]; v != want {
			t.Fatalf("Curve normalization: invalid points: x %d: expected %d got %d", x, want, v)
		}
	}
}

func TestCurveExtreme(t *testing.T) {
	src := makeCurveGrayRamp(256, 1)

	// A strictly decreasing curve inverts the tones monotonically.
	inverted := drawCurve(src, Curve(CurvePoint{0, 1}, CurvePoint{1, 0})).(*image.NRGBA)
	prev := 256
	for x := 0; x < 256; x++ {
		v := int(inverted.Pix[inverted.PixOffset(x, 0)])
		if v > prev {
			t.Errorf("Curve extreme: inverted x %d: value %d above previous %d", x, v, prev)
		}
		prev = v
	}
	if v := inverted.Pix[inverted.PixOffset(0, 0)]; v != 255 {
		t.Errorf("Curve extreme: inverted black: expected 255 got %d", v)
	}
	if v := inverted.Pix[inverted.PixOffset(255, 0)]; v != 0 {
		t.Errorf("Curve extreme: inverted white: expected 0 got %d", v)
	}

	// Solid curves pin the whole image to one tone.
	solidBlack := drawCurve(src, Curve(CurvePoint{0, 0}, CurvePoint{1, 0})).(*image.NRGBA)
	solidWhite := drawCurve(src, Curve(CurvePoint{0, 1}, CurvePoint{1, 1})).(*image.NRGBA)
	for x := 0; x < 256; x++ {
		if v := solidBlack.Pix[solidBlack.PixOffset(x, 0)]; v != 0 {
			t.Errorf("Curve extreme: solid black x %d: expected 0 got %d", x, v)
		}
		if v := solidWhite.Pix[solidWhite.PixOffset(x, 0)]; v != 255 {
			t.Errorf("Curve extreme: solid white x %d: expected 255 got %d", x, v)
		}
	}

	// A nearly vertical section must stay finite and in range.
	steep := make([]CurvePoint, 0, 1002)
	for i := 0; i <= 1000; i++ {
		steep = append(steep, CurvePoint{X: float32(i) / 1000, Y: float32(i) / 1000})
	}
	steep = append(steep, CurvePoint{0.5, 0}, CurvePoint{0.5001, 1})
	dst := drawCurve(src, Curve(steep...)).(*image.NRGBA)
	for x := 0; x < 256; x++ {
		if v := dst.Pix[dst.PixOffset(x, 0)]; v > 255 {
			t.Fatalf("Curve extreme: steep x %d: value %d out of range", x, v)
		}
	}
}

func TestCurveEmptyImage(t *testing.T) {
	// Degenerate images must not panic.
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 0, 10),
		image.Rect(0, 0, 10, 0),
		image.Rect(0, 0, 1, 1),
	} {
		src := image.NewNRGBA(r)
		f := Curve(CurvePoint{0, 0.2}, CurvePoint{1, 0.8})
		dst := image.NewNRGBA(f.Bounds(r))
		f.Draw(dst, src, nil)
		if !dst.Bounds().Size().Eq(r.Size()) {
			t.Errorf("Curve empty image: bad bounds: expected %v got %v", r.Size(), dst.Bounds().Size())
		}
	}
}

func TestCurveParallelConsistency(t *testing.T) {
	// Pixel mapping does not depend on how rows are split between goroutines,
	// so the parallel and the serial result must be identical.
	src := makeCurveGrayRamp(253, 97)
	f := Curve(
		CurvePoint{0, 0.08},
		CurvePoint{0.25, 0.3},
		CurvePoint{0.75, 0.72},
		CurvePoint{1, 0.94},
	)

	g := New(f)
	g.SetParallelization(true)
	dst1 := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst1, src)

	g.SetParallelization(false)
	dst2 := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst2, src)

	for i := range dst1.Pix {
		if dst1.Pix[i] != dst2.Pix[i] {
			t.Fatalf("Curve parallel consistency: index %d: parallel %d serial %d", i, dst1.Pix[i], dst2.Pix[i])
		}
	}
}

func TestCurveInFilterChain(t *testing.T) {
	// The curve must work as a regular link of the filter chain.
	src := image.NewNRGBA(image.Rect(0, 0, 20, 12))
	for y := 0; y < 12; y++ {
		for x := 0; x < 20; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(x * 12)
			src.Pix[i+1] = uint8(y * 20)
			src.Pix[i+2] = uint8(x*7 + y*3)
			src.Pix[i+3] = uint8(255 - x*4)
		}
	}

	g := New(
		Resize(10, 8, NearestNeighborResampling),
		Curve(CurvePoint{0, 0.05}, CurvePoint{0.5, 0.55}, CurvePoint{1, 0.95}),
		Grayscale(),
	)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if !dst.Bounds().Size().Eq(image.Pt(10, 8)) {
		t.Fatalf("Curve chain: bad bounds: %v", dst.Bounds())
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 10; x++ {
			i := dst.PixOffset(x, y)
			if dst.Pix[i] != dst.Pix[i+1] || dst.Pix[i+1] != dst.Pix[i+2] {
				t.Errorf("Curve chain: (%d, %d): expected grayscale pixel", x, y)
			}
		}
	}
}
