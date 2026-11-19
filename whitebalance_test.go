package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"
)

// Warm and cool reference whites used by the tests. They are ordinary sRGB
// colors, i.e. the color a neutral card has in the source image.
var (
	testWhiteBalanceWarm = color.NRGBA{R: 0xff, G: 0xd6, B: 0xaa, A: 0xff} // 255, 214, 170
	testWhiteBalanceCool = color.NRGBA{R: 0xaa, G: 0xc8, B: 0xff, A: 0xff} // 170, 200, 255
)

// wbTestChart is a fixed color chart: the first three patches are neutral
// (light, mid, dark gray) and the rest are colors that should keep their
// relative positions after the correction.
var wbTestChart = [][3]uint8{
	{242, 242, 242},
	{128, 128, 128},
	{46, 46, 46},
	{200, 40, 40},
	{40, 180, 60},
	{40, 60, 200},
	{200, 150, 120},
	{230, 200, 60},
	{60, 190, 200},
	{150, 80, 160},
}

// wbTestChartImage renders the chart, as seen under the given illuminant
// white, into a 1 pixel high image. Each patch is rendered by multiplying the
// reflectance of the patch (the 8-bit values are treated as reflectance in
// linear light) with the illuminant white, so the result is what a camera
// would record. The card contains a semi-transparent copy of the chart in its
// second row and a fully transparent pixel in its third row.
func wbTestChartImage(white [3]uint8) *image.NRGBA {
	w := [3]float64{
		wbTestSRGBToLinear(float64(white[0]) / 255),
		wbTestSRGBToLinear(float64(white[1]) / 255),
		wbTestSRGBToLinear(float64(white[2]) / 255),
	}

	img := image.NewNRGBA(image.Rect(0, 0, len(wbTestChart), 3))
	for x, patch := range wbTestChart {
		var r, g, b uint8
		for c, v := range patch {
			lin := wbTestSRGBToLinear(float64(v)/255) * w[c]
			lin = wbTestClamp01(lin)
			enc := wbTestLinearToSRGB(lin)
			u := uint8(enc*255 + 0.5)
			switch c {
			case 0:
				r = u
			case 1:
				g = u
			case 2:
				b = u
			}
		}
		i := img.PixOffset(x, 0)
		img.Pix[i+0], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = r, g, b, 0xff

		i = img.PixOffset(x, 1)
		img.Pix[i+0], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = r, g, b, 0x80

		i = img.PixOffset(x, 2)
		img.Pix[i+0], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = r, g, b, 0x00
	}
	return img
}

func wbTestApply(t *testing.T, src image.Image, f Filter) *image.NRGBA {
	t.Helper()
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	return dst
}

func wbTestPixel(img *image.NRGBA, x, y int) (r, g, b, a uint8) {
	i := img.PixOffset(x, y)
	return img.Pix[i+0], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3]
}

func wbTestSRGBToLinear(v float64) float64 {
	if v <= 0.04045 {
		return v / 12.92
	}
	return math.Pow((v+0.055)/1.055, 2.4)
}

func wbTestLinearToSRGB(v float64) float64 {
	if v <= 0.0031308 {
		return v * 12.92
	}
	return 1.055*math.Pow(v, 1/2.4) - 0.055
}

func wbTestClamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// wbTestLab returns the CIE L*a*b* coordinates of an sRGB color. It is used
// to measure the color difference independently of the implementation.
func wbTestLab(c [3]uint8) [3]float64 {
	// Linear sRGB (D65) to CIE XYZ.
	lin := [3]float64{
		wbTestSRGBToLinear(float64(c[0]) / 255),
		wbTestSRGBToLinear(float64(c[1]) / 255),
		wbTestSRGBToLinear(float64(c[2]) / 255),
	}
	x := 0.4124564*lin[0] + 0.3575761*lin[1] + 0.1804375*lin[2]
	y := 0.2126729*lin[0] + 0.7151522*lin[1] + 0.0721750*lin[2]
	z := 0.0193339*lin[0] + 0.1191920*lin[1] + 0.9503041*lin[2]

	f := func(v float64) float64 {
		const eps = 216.0 / 24389.0
		if v > eps {
			return math.Cbrt(v)
		}
		return (841.0/108.0)*v + 4.0/29.0
	}
	fx := f(x / 0.95047)
	fy := f(y / 1.0)
	fz := f(z / 1.08883)

	return [3]float64{
		116*fy - 16,
		500 * (fx - fy),
		200 * (fy - fz),
	}
}

// wbTestDeltaE returns the CIE76 color difference between two sRGB colors.
func wbTestDeltaE(a, b [3]uint8) float64 {
	la := wbTestLab(a)
	lb := wbTestLab(b)
	var sum float64
	for i := range la {
		d := la[i] - lb[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

func TestWhiteBalanceNeutral(t *testing.T) {
	// The neutral patches of the warm and the cool chart must become truly
	// neutral (R == G == B) and, for the reference white itself, land on the
	// target white point. All other neutrals must keep their tone: the
	// corrected gray value equals the gray value of the reflectance.
	testData := []struct {
		desc  string
		white [3]uint8
	}{
		{"warm", [3]uint8{255, 214, 170}},
		{"cool", [3]uint8{170, 200, 255}},
	}

	for _, d := range testData {
		src := wbTestChartImage(d.white)

		// Neutral patches are recorded as the white multiplied by the
		// reflectance; after the correction they must be grayscale, and the
		// tone must match the reflectance.
		before := make([][3]uint8, 3)
		for x := 0; x < 3; x++ {
			r, g, b, _ := wbTestPixel(src, x, 0)
			before[x] = [3]uint8{r, g, b}
			if r == g && g == b {
				t.Fatalf("test [%s]: source neutral patch %d is already neutral: %v", d.desc, x, before[x])
			}
		}

		dst := wbTestApply(t, src, WhiteBalance(color.NRGBA{d.white[0], d.white[1], d.white[2], 0xff}, color.White))
		if !dst.Rect.Eq(src.Rect) {
			t.Fatalf("test [%s]: bad bounds: expected %v got %v", d.desc, src.Rect, dst.Rect)
		}

		for x, want := range wbTestChart[:3] {
			r, g, b, a := wbTestPixel(dst, x, 0)
			lo, hi := r, r
			for _, v := range []uint8{g, b} {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			if hi-lo > 1 {
				t.Errorf("test [%s]: neutral patch %d is not neutral: %d,%d,%d", d.desc, x, r, g, b)
			}
			gray := uint8((uint16(r) + uint16(g) + uint16(b)) / 3)
			if diff := int(gray) - int(want[0]); diff < -1 || diff > 1 {
				t.Errorf("test [%s]: neutral patch %d: expected gray %d got %d", d.desc, x, want[0], gray)
			}
			if a != 0xff {
				t.Errorf("test [%s]: neutral patch %d: alpha changed: expected 255 got %d", d.desc, x, a)
			}
		}
	}
}

func TestWhiteBalanceWhiteCard(t *testing.T) {
	// The source reference white must map exactly to the arbitrary target
	// white point (within 8-bit rounding), not only to color.White. This
	// exercises the general src -> dst adaptation, not just the two fixed
	// warm/cool charts.
	testData := []struct {
		desc     string
		src, dst [3]uint8
	}{
		{"warm to white", [3]uint8{255, 214, 170}, [3]uint8{255, 255, 255}},
		{"cool to white", [3]uint8{170, 200, 255}, [3]uint8{255, 255, 255}},
		{"warm to cool", [3]uint8{255, 214, 170}, [3]uint8{170, 200, 255}},
		{"cool to warm", [3]uint8{170, 200, 255}, [3]uint8{255, 214, 170}},
		{"white to warm", [3]uint8{255, 255, 255}, [3]uint8{255, 214, 170}},
		{"warm to moderate", [3]uint8{255, 214, 170}, [3]uint8{210, 225, 245}},
		{"mid gray to warm", [3]uint8{128, 128, 128}, [3]uint8{255, 214, 170}},
	}

	for _, d := range testData {
		src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		src.Set(0, 0, color.NRGBA{d.src[0], d.src[1], d.src[2], 0xff})

		dst := wbTestApply(t, src, WhiteBalance(
			color.NRGBA{d.src[0], d.src[1], d.src[2], 0xff},
			color.NRGBA{d.dst[0], d.dst[1], d.dst[2], 0xff},
		))
		r, g, b, a := wbTestPixel(dst, 0, 0)
		got := [3]uint8{r, g, b}
		for c := range got {
			// The adaptation maps the reference white to the target white
			// point by construction, so only 8-bit rounding remains.
			if diff := int(got[c]) - int(d.dst[c]); diff < -1 || diff > 1 {
				t.Errorf("test [%s]: expected white %v got %v", d.desc, d.dst, got)
				break
			}
		}
		if a != 0xff {
			t.Errorf("test [%s]: alpha changed: expected 255 got %d", d.desc, a)
		}
	}
}

func TestWhiteBalanceChart(t *testing.T) {
	// Fixed charts recorded under a warm and a cool illuminant, corrected to
	// the sRGB white point. The expected values were computed with an
	// independent implementation of the same Bradford model.
	testData := []struct {
		desc  string
		white [3]uint8
		want  [][3]uint8
	}{
		{
			"warm",
			[3]uint8{255, 214, 170},
			[][3]uint8{
				{242, 242, 242},
				{128, 128, 128},
				{46, 46, 46},
				{217, 48, 52},
				{0, 179, 73},
				{17, 58, 193},
				{209, 151, 125},
				{237, 201, 85},
				{0, 189, 197},
				{159, 81, 157},
			},
		},
		{
			"cool",
			[3]uint8{170, 200, 255},
			[][3]uint8{
				{242, 242, 242},
				{128, 128, 128},
				{46, 46, 46},
				{181, 37, 35},
				{86, 179, 48},
				{63, 65, 204},
				{190, 149, 118},
				{221, 198, 38},
				{106, 190, 201},
				{142, 82, 162},
			},
		},
	}

	for _, d := range testData {
		src := wbTestChartImage(d.white)
		dst := wbTestApply(t, src, WhiteBalance(
			color.NRGBA{d.white[0], d.white[1], d.white[2], 0xff}, color.White,
		))

		var sumBefore, sumAfter, maxBefore, maxAfter float64
		for x, want := range d.want {
			r, g, b, _ := wbTestPixel(dst, x, 0)
			got := [3]uint8{r, g, b}
			for c := range got {
				if diff := int(got[c]) - int(want[c]); diff < -1 || diff > 1 {
					t.Errorf("test [%s]: patch %d channel %d: expected %d got %d", d.desc, x, c, want[c], got[c])
				}
			}

			sr, sg, sb, _ := wbTestPixel(src, x, 0)
			deBefore := wbTestDeltaE([3]uint8{sr, sg, sb}, wbTestChart[x])
			deAfter := wbTestDeltaE(got, wbTestChart[x])
			sumBefore += deBefore
			sumAfter += deAfter
			maxBefore = math.Max(maxBefore, deBefore)
			maxAfter = math.Max(maxAfter, deAfter)

			if deAfter > deBefore+0.5 {
				t.Errorf("test [%s]: patch %d got worse: deltaE before %.2f after %.2f", d.desc, x, deBefore, deAfter)
			}
		}

		n := float64(len(d.want))
		if sumAfter/n > 6 {
			t.Errorf("test [%s]: mean deltaE too high: before %.2f after %.2f", d.desc, sumBefore/n, sumAfter/n)
		}
		if maxAfter > 12 {
			t.Errorf("test [%s]: max deltaE too high: before %.2f after %.2f", d.desc, maxBefore, maxAfter)
		}
		if sumAfter >= sumBefore {
			t.Errorf("test [%s]: color difference did not improve: before %.2f after %.2f", d.desc, sumBefore, sumAfter)
		}
	}
}

func TestWhiteBalanceAlpha(t *testing.T) {
	// The correction is applied to the color channels of every pixel,
	// regardless of its transparency, and the alpha channel is preserved.
	// This includes semi-transparent and fully transparent pixels.
	for _, alpha := range []uint8{0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff} {
		src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		src.Pix = []uint8{0xc8, 0x96, 0x78, alpha}

		dst := wbTestApply(t, src, WhiteBalance(testWhiteBalanceWarm, color.White))
		r, g, b, a := wbTestPixel(dst, 0, 0)
		if a != alpha {
			t.Errorf("alpha %d: expected alpha %d got %d", alpha, alpha, a)
		}
		if r != 0xcc || g != 0xb4 || b != 0xb6 {
			t.Errorf("alpha %d: expected color 204,180,182 got %d,%d,%d", alpha, r, g, b)
		}
	}
}

func TestWhiteBalanceBounds(t *testing.T) {
	// Bounds and Draw must behave like the other point filters: the result has
	// the size of the source (normalized to the origin) and the drawing is
	// anchored at the destination's Min point.
	src := image.NewNRGBA(image.Rect(-2, -3, 3, 4))
	for y := src.Rect.Min.Y; y < src.Rect.Max.Y; y++ {
		for x := src.Rect.Min.X; x < src.Rect.Max.X; x++ {
			src.Set(x, y, color.NRGBA{200, 150, 120, 0xff})
		}
	}

	f := WhiteBalance(testWhiteBalanceWarm, color.White)
	if b := f.Bounds(src.Bounds()); !b.Eq(image.Rect(0, 0, 5, 7)) {
		t.Fatalf("bad bounds: expected (0,0)-(5,7) got %v", b)
	}

	dst := image.NewNRGBA(image.Rect(10, 20, 15, 27))
	f.Draw(dst, src, nil)
	for y := 20; y < 27; y++ {
		for x := 10; x < 15; x++ {
			r, g, b, a := wbTestPixel(dst, x, y)
			if r != 0xcc || g != 0xb4 || b != 0xb6 || a != 0xff {
				t.Fatalf("bad pixel at %d,%d: %d,%d,%d,%d", x, y, r, g, b, a)
			}
		}
	}
}

func TestWhiteBalanceEmpty(t *testing.T) {
	for _, src := range []image.Image{
		image.NewNRGBA(image.Rect(0, 0, 0, 0)),
		image.NewNRGBA(image.Rect(5, 5, 5, 5)),
		image.NewNRGBA(image.Rect(0, 0, 0, 3)),
		image.NewNRGBA(image.Rect(0, 0, 3, 0)),
	} {
		f := WhiteBalance(testWhiteBalanceWarm, color.White)
		b := f.Bounds(src.Bounds())
		if b.Dx() != src.Bounds().Dx() || b.Dy() != src.Bounds().Dy() {
			t.Errorf("empty image: bad bounds %v for %v", b, src.Bounds())
		}
		dst := image.NewNRGBA(b)
		f.Draw(dst, src, nil) // must not panic
	}
}

func TestWhiteBalanceDegenerateParams(t *testing.T) {
	// A reference white that cannot define an adaptation must produce a
	// well-defined pass-through copy. Non-finite and out-of-gamut values can
	// only be expressed with WhitePoint, because color.RGBA clamps them.
	src := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	copy(src.Pix, []uint8{
		0x00, 0x10, 0x20, 0xff,
		0xc8, 0x96, 0x78, 0x80,
		0xff, 0xff, 0xff, 0x00,
		0x11, 0x22, 0x33, 0x01,
		0xfe, 0x01, 0x7f, 0xfe,
		0x80, 0x80, 0x80, 0xff,
	})

	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))
	wide := float32(math.Sqrt(2)) // above the sRGB gamut but finite

	testData := []struct {
		desc     string
		src, dst color.Color
	}{
		{"nil source", nil, color.White},
		{"nil destination", color.White, nil},
		{"black source", color.Black, color.White},
		{"black destination", color.White, color.Black},
		{"dark source", color.RGBA{1, 1, 1, 0xff}, color.White},
		{"dark destination", color.White, color.RGBA{2, 2, 2, 0xff}},
		{"transparent source", color.NRGBA{255, 214, 170, 0}, color.White},
		{"transparent destination", color.White, color.NRGBA{255, 255, 255, 0}},
		{"NaN source", WhitePoint{nan, 0.5, 0.5}, color.White},
		{"NaN destination", color.White, WhitePoint{0.5, nan, 0.5}},
		{"+Inf source", WhitePoint{inf, 1, 1}, color.White},
		{"-Inf destination", color.White, WhitePoint{0.5, ninf, 0.5}},
		{"negative source", WhitePoint{-0.1, 0.5, 0.5}, color.White},
		{"negative destination", color.White, WhitePoint{0.5, 0.5, -0.1}},
		{"out of gamut source", WhitePoint{wide, 1, 1}, color.White},
		{"out of gamut destination", color.White, WhitePoint{1, 1, wide}},
		{"nil white point", (*WhitePoint)(nil), color.White},
		{"equal whites", testWhiteBalanceWarm, testWhiteBalanceWarm},
	}

	for _, d := range testData {
		f := WhiteBalance(d.src, d.dst)
		if _, ok := f.(*copyimageFilter); !ok {
			t.Errorf("test [%s]: expected pass-through, got %T", d.desc, f)
			continue
		}
		dst := wbTestApply(t, src, f)
		for i := range src.Pix {
			if dst.Pix[i] != src.Pix[i] {
				t.Errorf("test [%s]: pass-through changed pixel %d: expected %d got %d", d.desc, i, src.Pix[i], dst.Pix[i])
				break
			}
		}
	}
}

func TestWhiteBalanceWhitePointColor(t *testing.T) {
	// WhitePoint implements color.Color and matches an equivalent color.NRGBA.
	src := wbTestChartImage([3]uint8{255, 214, 170})

	dst1 := wbTestApply(t, src, WhiteBalance(WhitePoint{1, 214.0 / 255, 170.0 / 255}, color.White))
	dst2 := wbTestApply(t, src, WhiteBalance(testWhiteBalanceWarm, color.White))
	for i := range dst1.Pix {
		if diff := int(dst1.Pix[i]) - int(dst2.Pix[i]); diff < -1 || diff > 1 {
			t.Fatalf("WhitePoint mismatch at %d: expected %d got %d", i, dst2.Pix[i], dst1.Pix[i])
		}
	}

	// The RGBA method clamps and never returns alpha other than opaque.
	r, g, b, a := WhitePoint{1.5, -1, 0.5}.RGBA()
	if a != 0xffff {
		t.Errorf("WhitePoint alpha: expected 0xffff got 0x%x", a)
	}
	if r != 0xffff || g != 0 || b != 0x8000 {
		t.Errorf("WhitePoint clamped channels: got 0x%x, 0x%x, 0x%x", r, g, b)
	}
}

func TestWhiteBalanceGamut(t *testing.T) {
	// Colors that adapt outside the sRGB gamut must be clamped, not wrapped
	// around or turned into non-finite values.
	src := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	src.Pix = []uint8{
		0x00, 0x00, 0xff, 0xff, // saturated blue
		0xff, 0x00, 0x00, 0xff, // saturated red
		0x00, 0xff, 0x00, 0xff, // saturated green
	}
	dst := wbTestApply(t, src, WhiteBalance(testWhiteBalanceCool, color.White))
	for x := 0; x < 3; x++ {
		r, g, b, a := wbTestPixel(dst, x, 0)
		if a != 0xff {
			t.Errorf("pixel %d: alpha changed: %d", x, a)
		}
		// The result must stay a valid color; clamping can only brighten
		// channels that were pushed past the gamut.
		if r == 0 && g == 0 && b == 0 {
			t.Errorf("pixel %d: out-of-gamut color wrapped to black", x)
		}
	}
}

func TestWhiteBalanceRoundTrip(t *testing.T) {
	// Adapting to the target white and back must recover the original colors
	// (for colors that stay inside the gamut at every step).
	colors := [][3]uint8{
		{120, 90, 70},
		{180, 140, 110},
		{100, 110, 100},
		{60, 80, 90},
		{200, 150, 120},
		{150, 150, 150},
		{40, 60, 80},
	}

	src := image.NewNRGBA(image.Rect(0, 0, len(colors), 1))
	for x, c := range colors {
		src.Set(x, 0, color.NRGBA{c[0], c[1], c[2], 0xff})
	}

	g := New(
		WhiteBalance(testWhiteBalanceWarm, color.White),
		WhiteBalance(color.White, testWhiteBalanceWarm),
	)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for x, c := range colors {
		r, g, b, _ := wbTestPixel(dst, x, 0)
		got := [3]uint8{r, g, b}
		for i := range got {
			if diff := int(got[i]) - int(c[i]); diff < -3 || diff > 3 {
				t.Errorf("round trip: color %v: expected %v got %v", c, c, got)
				break
			}
		}
	}
}

func TestWhiteBalanceParallelization(t *testing.T) {
	// Parallel and sequential drawing must produce identical results.
	src := wbTestChartImage([3]uint8{255, 214, 170})

	g1 := New(WhiteBalance(testWhiteBalanceWarm, color.White))
	g1.SetParallelization(false)
	dst1 := image.NewNRGBA(g1.Bounds(src.Bounds()))
	g1.Draw(dst1, src)

	g2 := New(WhiteBalance(testWhiteBalanceWarm, color.White))
	g2.SetParallelization(true)
	dst2 := image.NewNRGBA(g2.Bounds(src.Bounds()))
	g2.Draw(dst2, src)

	for i := range dst1.Pix {
		if dst1.Pix[i] != dst2.Pix[i] {
			t.Fatalf("parallelization mismatch at %d: %d != %d", i, dst1.Pix[i], dst2.Pix[i])
		}
	}
}

func TestWhiteBalanceFilterChain(t *testing.T) {
	// The filter must plug into an existing filter list. Correcting the
	// neutral patches and then inverting the image must invert the corrected
	// colors, not the source colors.
	src := wbTestChartImage([3]uint8{170, 200, 255})

	g := New(
		WhiteBalance(testWhiteBalanceCool, color.White),
		Invert(),
	)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for x, want := range wbTestChart[:3] {
		r, _, _, _ := wbTestPixel(dst, x, 0)
		// After the correction every neutral patch is gray equal to the
		// reflectance, so the inversion is the same gray for all channels.
		expected := 255 - want[0]
		if diff := int(r) - int(expected); diff < -1 || diff > 1 {
			t.Errorf("chain: patch %d: expected %d got %d", x, expected, r)
		}
	}
}

func TestWhiteBalanceImageTypes(t *testing.T) {
	// The filter works with all the image types supported by the package and
	// keeps alpha.
	src := wbTestChartImage([3]uint8{255, 214, 170})
	type imageMaker struct {
		desc string
		make func() draw.Image
	}
	makers := []imageMaker{
		{"NRGBA", func() draw.Image { return image.NewNRGBA(image.Rect(0, 0, 10, 3)) }},
		{"NRGBA64", func() draw.Image { return image.NewNRGBA64(image.Rect(0, 0, 10, 3)) }},
		{"RGBA", func() draw.Image { return image.NewRGBA(image.Rect(0, 0, 10, 3)) }},
		{"RGBA64", func() draw.Image { return image.NewRGBA64(image.Rect(0, 0, 10, 3)) }},
		{"Gray", func() draw.Image { return image.NewGray(image.Rect(0, 0, 10, 3)) }},
		{"Gray16", func() draw.Image { return image.NewGray16(image.Rect(0, 0, 10, 3)) }},
	}
	f := WhiteBalance(testWhiteBalanceWarm, color.White)
	for _, m := range makers {
		dst := m.make()
		New().Draw(dst, src) // copy the source into the destination type
		f.Draw(dst, src, nil)
		if !dst.Bounds().Size().Eq(src.Bounds().Size()) {
			t.Errorf("%s: bad size", m.desc)
		}
		if _, _, _, a := dst.At(0, 1).RGBA(); a != 0x8080 && m.desc != "Gray" && m.desc != "Gray16" {
			t.Errorf("%s: alpha changed: 0x%x", m.desc, a)
		}
	}
}
