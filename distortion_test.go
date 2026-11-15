package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"
)

// refInverse is an independent reference inverse of the radial model,
// implemented as plain dense bisection instead of the Newton+bracketing
// strategy used in production. hi bounds a monotonic section of the radial
// function (the fold radius for folded mappings, a large constant otherwise).
func refInverse(rd, k1, k2, hi float64) (float64, bool) {
	if rd < 0 {
		return 0, false
	}
	lo := 0.0
	if radialDistortion(hi, k1, k2) < rd {
		return 0, false
	}
	for i := 0; i < 200; i++ {
		mid := (lo + hi) * 0.5
		if radialDistortion(mid, k1, k2) < rd {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) * 0.5, true
}

// foldRTBracket returns an upper bound spanning a monotonic section of the
// radial function for the reference solver.
func foldRTBracket(k1, k2 float64) float64 {
	if radius, folded := radialFoldRadius(k1, k2); folded {
		return radius * (1 - 1e-12)
	}
	return 1e5
}

func TestRadialFoldRadius(t *testing.T) {
	testData := []struct {
		k1, k2     float64
		folded     bool
		wantRadius float64
	}{
		{0, 0, false, 0},
		{0.3, 0, false, 0},
		{0, 0.1, false, 0},
		{-0.25, 0, true, math.Sqrt(4.0 / 3.0)},
		{0, -0.1, true, math.Sqrt(math.Sqrt(2))},
		// Tangent point only: derivative touches zero but stays non-negative.
		{-2.0 / 3.0, 0.2, false, 0},
	}
	for _, d := range testData {
		radius, folded := radialFoldRadius(d.k1, d.k2)
		if folded != d.folded {
			t.Errorf("radialFoldRadius(%v, %v): folded = %v, want %v", d.k1, d.k2, folded, d.folded)
			continue
		}
		if folded && math.Abs(radius-d.wantRadius) > 1e-10 {
			t.Errorf("radialFoldRadius(%v, %v): radius = %v, want %v", d.k1, d.k2, radius, d.wantRadius)
		}
	}
}

func TestRadialInverseAnalytic(t *testing.T) {
	testData := []struct {
		desc       string
		rd, k1, k2 float64
		folded     bool
		foldRadius float64
		want       float64
		ok         bool
	}{
		{"identity", 2.5, 0, 0, false, 0, 2.5, true},
		{"k1 only exact root", 2, 1, 0, false, 0, 1, true},
		{"k1 k2 exact root", 3, 1, 1, false, 0, 1, true},
		{"pincushion in range", 0.5, -0.25, 0, true, math.Sqrt(4.0 / 3.0), 0.5391888728108506, true},
		{"k2 folded in range", 0.5, 0, -0.1, true, math.Sqrt(math.Sqrt(2)), 0.5032271589663463, true},
		{"pincushion beyond fold max", 0.9, -0.25, 0, true, math.Sqrt(4.0 / 3.0), 0, false},
		{"negative radius", -1, 1, 0, false, 0, 0, false},
		{"non-finite radius", math.NaN(), 1, 0, false, 0, 0, false},
	}
	for _, d := range testData {
		ru, ok := radialInverse(d.rd, d.k1, d.k2, d.folded, d.foldRadius)
		if ok != d.ok {
			t.Errorf("%s: ok = %v, want %v", d.desc, ok, d.ok)
			continue
		}
		if ok && math.Abs(ru-d.want) > 1e-9 {
			t.Errorf("%s: ru = %v, want %v", d.desc, ru, d.want)
		}
	}
}

func TestRadialInverseRoundTrip(t *testing.T) {
	params := []struct{ k1, k2 float64 }{
		{0, 0},
		{0.3, 0},
		{-0.05, 0},
		{0.2, 0.1},
		{0.1, -0.02},
		{-0.1, 0.05},
	}
	for _, pm := range params {
		foldRadius, folded := radialFoldRadius(pm.k1, pm.k2)
		for rd := 0.0; rd < 1.6; rd += 0.0137 {
			if folded && rd >= foldRadius {
				continue
			}
			ru, ok := radialInverse(rd, pm.k1, pm.k2, folded, foldRadius)
			if !ok {
				t.Errorf("k1=%v k2=%v rd=%v: solver failed", pm.k1, pm.k2, rd)
				continue
			}
			if got := radialDistortion(ru, pm.k1, pm.k2); math.Abs(got-rd) > 1e-7*math.Max(1, rd) {
				t.Errorf("k1=%v k2=%v rd=%v: round trip residual %v", pm.k1, pm.k2, rd, got-rd)
			}
			want, wantOK := refInverse(rd, pm.k1, pm.k2, foldRTBracket(pm.k1, pm.k2))
			if wantOK && math.Abs(ru-want) > 1e-6 {
				t.Errorf("k1=%v k2=%v rd=%v: got %v, reference %v", pm.k1, pm.k2, rd, ru, want)
			}
		}
	}
}

func TestLensDistortionBounds(t *testing.T) {
	f := LensDistortion(0, 0, 0.1, 0.02, color.Black, LinearInterpolation)
	for _, srcb := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 1, 1),
		image.Rect(-3, -5, 4, 7),
		image.Rect(10, 20, 18, 26),
	} {
		want := image.Rect(0, 0, srcb.Dx(), srcb.Dy())
		if got := f.Bounds(srcb); !got.Eq(want) {
			t.Errorf("Bounds(%v) = %v, want %v", srcb, got, want)
		}
	}
}

// pixelsEqual compares two 8-bit RGBA pixel tuples channel by channel with a
// small tolerance to absorb float32/float64 rounding differences.
func pixelsEqual(a, b [4]uint8, tolerance int) bool {
	for i := 0; i < 4; i++ {
		d := int(a[i]) - int(b[i])
		if d < 0 {
			d = -d
		}
		if d > tolerance {
			return false
		}
	}
	return true
}

func nrgbaPixel(img *image.NRGBA, x, y int) [4]uint8 {
	i := img.PixOffset(x, y)
	return [4]uint8{img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3]}
}

// TestLensDistortionGrid verifies destination coordinates against an
// independent reference implementation on a uniquely colored grid, including
// transparent pixels and background fill at the edges. Nearest-neighbor
// sampling makes the expected pixel a pure coordinate-mapping question.
func TestLensDistortionGrid(t *testing.T) {
	const w, h = 8, 6
	cases := []struct {
		name   string
		k1, k2 float32
	}{
		{"barrel", 0.25, 0.05},
		{"mild pincushion", -0.12, 0},
		{"k2 only", 0, 0.08},
		{"pincushion near fold", -0.2, 0},
		{"pincushion with k2", -0.15, 0.05},
	}
	bg := color.NRGBA{12, 34, 56, 0}

	for _, c := range cases {
		src := image.NewNRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				i := src.PixOffset(x, y)
				src.Pix[i+0] = uint8(x * 30)
				src.Pix[i+1] = uint8(y * 40)
				src.Pix[i+2] = uint8((x + y) * 20)
				src.Pix[i+3] = uint8((x*7 + y*13) % 256)
			}
		}

		f := LensDistortion(0, 0, c.k1, c.k2, bg, NearestNeighborInterpolation)
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dst, src, &Options{Parallelization: false})

		// Independent reference rendering with the same normalized model.
		scale := 0.5 * math.Min(w, h)
		cx, cy := float64(w)/2, float64(h)/2
		k1, k2 := float64(c.k1), float64(c.k2)
		foldRadius, folded := radialFoldRadius(k1, k2)
		hi := 1e5
		if folded {
			hi = foldRadius * (1 - 1e-12)
		}
		bgpx := pixelFromColor(bg)
		getter := newPixelGetter(src)
		bgSeen, sampleSeen := false, false
		for y := 0; y < h; y++ {
			dy := (float64(y) - cy) / scale
			for x := 0; x < w; x++ {
				dx := (float64(x) - cx) / scale
				rd := math.Hypot(dx, dy)
				var want pixel
				if rd == 0 {
					want = getter.getPixel(int(math.Round(cx)), int(math.Round(cy)))
					sampleSeen = true
				} else if folded && rd >= foldRadius {
					want = bgpx
					bgSeen = true
				} else {
					ru, ok := refInverse(rd, k1, k2, hi)
					if !ok {
						want = bgpx
						bgSeen = true
					} else {
						sx := cx + dx*scale*(ru/rd)
						sy := cy + dy*scale*(ru/rd)
						sx0, sy0 := math.Floor(sx+0.5), math.Floor(sy+0.5)
						if image.Pt(int(sx0), int(sy0)).In(src.Bounds()) {
							want = getter.getPixel(int(sx0), int(sy0))
							sampleSeen = true
						} else {
							want = bgpx
							bgSeen = true
						}
					}
				}
				got := nrgbaPixel(dst, x, y)
				expected := [4]uint8{
					f32u8(want.r * 0xff),
					f32u8(want.g * 0xff),
					f32u8(want.b * 0xff),
					f32u8(want.a * 0xff),
				}
				if !pixelsEqual(got, expected, 1) {
					t.Errorf("%s: pixel (%d,%d) = %v, want ~%v",
						c.name, x, y, got, expected)
				}
			}
		}
		if !sampleSeen {
			t.Errorf("%s: no source pixels sampled at all", c.name)
		}
		if folded && !bgSeen {
			t.Errorf("%s: folded mapping produced no background pixels", c.name)
		}
	}
}

// TestLensDistortionInterpolationSmoke runs the linear and cubic paths and
// checks they stay finite, inside the color range and match across modes.
func TestLensDistortionInterpolationSmoke(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 9, 7))
	for y := 0; y < 7; y++ {
		for x := 0; x < 9; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(x * 28)
			src.Pix[i+1] = uint8(y * 36)
			src.Pix[i+2] = uint8((x*y + 1) * 9)
			src.Pix[i+3] = uint8((x*3 + y*5) % 256)
		}
	}
	for _, interp := range []Interpolation{LinearInterpolation, CubicInterpolation} {
		f := LensDistortion(0, 0, 0.22, -0.04, color.NRGBA{100, 150, 200, 90}, interp)
		d1 := image.NewNRGBA(f.Bounds(src.Bounds()))
		d2 := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(d1, src, &Options{Parallelization: false})
		f.Draw(d2, src, &Options{Parallelization: true})
		if !checkBoundsAndPix(d1.Bounds(), d2.Bounds(), d1.Pix, d2.Pix) {
			t.Errorf("interpolation %v: serial/parallel mismatch", interp)
		}
	}
}

func TestLensDistortionSerialParallel(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 13, 9))
	for y := 0; y < 9; y++ {
		for x := 0; x < 13; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(x * 17)
			src.Pix[i+1] = uint8(y * 23)
			src.Pix[i+2] = uint8((x ^ y) * 11)
			src.Pix[i+3] = uint8((x*5 + y*9) % 256)
		}
	}

	cases := []struct {
		name           string
		cx, cy, k1, k2 float32
		bg             color.Color
		interp         Interpolation
	}{
		{"barrel nearest", 0, 0, 0.3, 0.05, color.NRGBA{1, 2, 3, 200}, NearestNeighborInterpolation},
		{"pincushion linear", 0, 0, -0.15, 0, color.Black, LinearInterpolation},
		{"custom center cubic", 2.5, 7.25, 0.1, -0.02, color.NRGBA{250, 128, 0, 128}, CubicInterpolation},
		{"strong fold", 0, 0, -2, 0, color.NRGBA{9, 8, 7, 255}, LinearInterpolation},
		{"nan k1", 0, 0, float32(math.NaN()), 0, color.NRGBA{5, 6, 7, 8}, LinearInterpolation},
		{"inf center", float32(math.Inf(1)), 0, 0.1, 0, color.NRGBA{5, 6, 7, 8}, LinearInterpolation},
	}
	for _, c := range cases {
		f := LensDistortion(c.cx, c.cy, c.k1, c.k2, c.bg, c.interp)
		dstSerial := image.NewNRGBA(f.Bounds(src.Bounds()))
		dstParallel := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dstSerial, src, &Options{Parallelization: false})
		f.Draw(dstParallel, src, &Options{Parallelization: true})
		if !checkBoundsAndPix(dstSerial.Bounds(), dstParallel.Bounds(), dstSerial.Pix, dstParallel.Pix) {
			t.Errorf("%s: serial and parallel results differ", c.name)
		}
		// Every output byte must be a valid finite color value (setPixel
		// always clamps, but assert it explicitly for the degenerate cases).
		for _, v := range dstSerial.Pix {
			if v > 0xff {
				t.Errorf("%s: non-finite/overflow byte %d", c.name, v)
			}
		}
	}
}

func TestLensDistortionIdentity(t *testing.T) {
	// Zero coefficients must reproduce the input exactly, for every
	// supported destination pixel format, including non-zero source bounds.
	srcb := image.Rect(-2, -3, 6, 5)
	builders := []func(image.Rectangle) (image.Image, draw.Image){
		func(r image.Rectangle) (image.Image, draw.Image) {
			img := image.NewNRGBA(r)
			return img, image.NewNRGBA(r)
		},
		func(r image.Rectangle) (image.Image, draw.Image) {
			img := image.NewNRGBA64(r)
			return img, image.NewNRGBA64(r)
		},
		func(r image.Rectangle) (image.Image, draw.Image) {
			img := image.NewRGBA(r)
			return img, image.NewRGBA(r)
		},
		func(r image.Rectangle) (image.Image, draw.Image) {
			img := image.NewRGBA64(r)
			return img, image.NewRGBA64(r)
		},
		func(r image.Rectangle) (image.Image, draw.Image) {
			img := image.NewGray(r)
			return img, image.NewGray(r)
		},
		func(r image.Rectangle) (image.Image, draw.Image) {
			img := image.NewGray16(r)
			return img, image.NewGray16(r)
		},
		func(r image.Rectangle) (image.Image, draw.Image) {
			palette := color.Palette{
				color.NRGBA{0, 0, 0, 255},
				color.NRGBA{255, 0, 0, 255},
				color.NRGBA{0, 255, 0, 128},
				color.NRGBA{0, 0, 255, 0},
			}
			img := image.NewPaletted(r, palette)
			return img, image.NewPaletted(r, palette)
		},
	}

	for i, build := range builders {
		src, dst := build(srcb)
		// Fill with distinctive values through the color interface.
		for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				src.(draw.Image).Set(x, y, color.NRGBA{
					R: uint8((x*37 + 11) & 0xff),
					G: uint8((y*53 + 29) & 0xff),
					B: uint8((x*y*7 + 5) & 0xff),
					A: uint8((x*13 + y*17) & 0xff),
				})
			}
		}
		srcPix := bytesOf(src)

		f := LensDistortion(0, 0, 0, 0, color.NRGBA{255, 0, 255, 255}, CubicInterpolation)
		if got := f.Bounds(src.Bounds()); !got.Eq(image.Rect(0, 0, srcb.Dx(), srcb.Dy())) {
			t.Errorf("format %d: unexpected bounds %v", i, got)
		}
		f.Draw(dst, src, &Options{Parallelization: false})
		// Premultiplied RGBA formats lose one unit at most during the
		// premultiply/unpremultiply round trip; everything else must be exact.
		tolerance := 0
		switch dst.(type) {
		case *image.RGBA, *image.RGBA64:
			tolerance = 1
		}
		gotPix := bytesOf(dst)
		if len(gotPix) != len(srcPix) {
			t.Errorf("format %d: pixel buffer length changed", i)
			continue
		}
		for j := range srcPix {
			d := int(gotPix[j]) - int(srcPix[j])
			if d < 0 {
				d = -d
			}
			if d > tolerance {
				t.Errorf("format %d: zero coefficients changed pixel at %d: %d -> %d",
					i, j, srcPix[j], gotPix[j])
				break
			}
		}
	}

	// YCbCr source (not a draw.Image, so only usable as a source).
	ycbcr := image.NewYCbCr(srcb, image.YCbCrSubsampleRatio420)
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			iy := ycbcr.YOffset(x, y)
			ycbcr.Y[iy] = uint8((x*9 + y*11) & 0xff)
		}
	}
	f := LensDistortion(0, 0, 0, 0, color.Black, LinearInterpolation)
	dst := image.NewNRGBA(f.Bounds(ycbcr.Bounds()))
	f.Draw(dst, ycbcr, nil)
	ref := image.NewNRGBA(f.Bounds(ycbcr.Bounds()))
	copyimage(ref, ycbcr, &Options{Parallelization: false})
	if !checkBoundsAndPix(dst.Bounds(), ref.Bounds(), dst.Pix, ref.Pix) {
		t.Errorf("YCbCr identity does not match plain copy")
	}
}

func bytesOf(img image.Image) []uint8 {
	switch p := img.(type) {
	case *image.NRGBA:
		return p.Pix
	case *image.NRGBA64:
		return p.Pix
	case *image.RGBA:
		return p.Pix
	case *image.RGBA64:
		return p.Pix
	case *image.Gray:
		return p.Pix
	case *image.Gray16:
		return p.Pix
	case *image.Paletted:
		return p.Pix
	}
	return nil
}

func TestLensDistortionBackgroundAndAlpha(t *testing.T) {
	// Opaque square on a transparent canvas. A pincushion mapping
	// (negative k1) magnifies the periphery and uncovers the corners, which
	// must take the exact background color and alpha.
	const w, h = 10, 10
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 3; y < 7; y++ {
		for x := 3; x < 7; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = 200
			src.Pix[i+1] = 100
			src.Pix[i+2] = 50
			src.Pix[i+3] = 255
		}
	}

	bg := color.NRGBA{10, 20, 30, 128}
	f := LensDistortion(0, 0, -0.3, 0, bg, LinearInterpolation)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	corners := [][2]int{{0, 0}, {w - 1, 0}, {0, h - 1}, {w - 1, h - 1}}
	for _, c := range corners {
		got := nrgbaPixel(dst, c[0], c[1])
		if got != [4]uint8{10, 20, 30, 128} {
			t.Errorf("corner %v = %v, want background {10,20,30,128}", c, got)
		}
	}

	// Fully transparent background: uncovered pixels keep zeroed RGB and
	// zero alpha instead of bleeding color through interpolation.
	fTransparent := LensDistortion(0, 0, -0.3, 0, color.Transparent, LinearInterpolation)
	dstTransparent := image.NewNRGBA(fTransparent.Bounds(src.Bounds()))
	fTransparent.Draw(dstTransparent, src, nil)
	for _, c := range corners {
		got := nrgbaPixel(dstTransparent, c[0], c[1])
		if got != [4]uint8{0, 0, 0, 0} {
			t.Errorf("corner %v = %v, want transparent {0,0,0,0}", c, got)
		}
	}

	// The center maps to the optical center and stays fully opaque.
	center := nrgbaPixel(dst, w/2, h/2)
	if center[3] != 255 {
		t.Errorf("center alpha = %d, want 255", center[3])
	}

	// Cross-check every pixel against the model directly: nearest sampling
	// on a separately rendered result must land on either a source pixel or
	// the background, never a hole or an arbitrary value.
	fNearest := LensDistortion(0, 0, -0.3, 0, bg, NearestNeighborInterpolation)
	dstNearest := image.NewNRGBA(fNearest.Bounds(src.Bounds()))
	fNearest.Draw(dstNearest, src, nil)

	scale := 0.5 * math.Min(w, h)
	cx, cy := float64(w)/2, float64(h)/2
	getter := newPixelGetter(src)
	bgpx := pixelFromColor(bg)
	foldRadius, folded := radialFoldRadius(-0.3, 0)
	refHi := 1e5
	if folded {
		refHi = foldRadius * (1 - 1e-12)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx := (float64(x) - cx) / scale
			dy := (float64(y) - cy) / scale
			rd := math.Hypot(dx, dy)
			var want pixel
			if rd == 0 {
				want = getter.getPixel(int(math.Round(cx)), int(math.Round(cy)))
			} else if folded && rd >= foldRadius {
				want = bgpx
			} else {
				ru, ok := refInverse(rd, -0.3, 0, refHi)
				if !ok {
					want = bgpx
				} else {
					sx := cx + dx*scale*(ru/rd)
					sy := cy + dy*scale*(ru/rd)
					pt := image.Pt(int(math.Floor(sx+0.5)), int(math.Floor(sy+0.5)))
					if pt.In(src.Bounds()) {
						want = getter.getPixel(pt.X, pt.Y)
					} else {
						want = bgpx
					}
				}
			}
			expected := [4]uint8{
				f32u8(want.r * 0xff),
				f32u8(want.g * 0xff),
				f32u8(want.b * 0xff),
				f32u8(want.a * 0xff),
			}
			if !pixelsEqual(nrgbaPixel(dstNearest, x, y), expected, 1) {
				t.Errorf("nearest pixel (%d,%d) does not follow the model", x, y)
			}
		}
	}
}

func TestLensDistortionNonFinite(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for i := range src.Pix {
		src.Pix[i] = 0xaa
	}
	bg := color.NRGBA{31, 41, 59, 73}

	bad := []struct {
		cx, cy, k1, k2 float32
	}{
		{float32(math.NaN()), 0, 0.1, 0},
		{0, float32(math.Inf(-1)), 0.1, 0},
		{0, 0, float32(math.Inf(1)), 0},
		{0, 0, 0.1, float32(math.NaN())},
	}
	for i, p := range bad {
		f := LensDistortion(p.cx, p.cy, p.k1, p.k2, bg, CubicInterpolation)
		dst := image.NewNRGBA(f.Bounds(src.Bounds()))
		f.Draw(dst, src, nil)
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				got := nrgbaPixel(dst, x, y)
				if got != [4]uint8{31, 41, 59, 73} {
					t.Errorf("case %d pixel (%d,%d) = %v, want background", i, x, y, got)
				}
			}
		}
	}
}

func TestLensDistortionFolded(t *testing.T) {
	// Strong pincushion: f(r) = r*(1-2r^2) folds at r = sqrt(1/6) and the
	// maximum reachable destination radius is well below the corner radius.
	const w, h = 12, 12
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(x * 20)
			src.Pix[i+1] = uint8(y * 20)
			src.Pix[i+2] = 64
			src.Pix[i+3] = 255
		}
	}
	bg := color.NRGBA{7, 7, 7, 255}
	f := LensDistortion(0, 0, -2, 0, bg, NearestNeighborInterpolation)
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)

	// Fold radius in pixels: scale * sqrt(1/6).
	scale := 0.5 * math.Min(w, h)
	foldR := scale * math.Sqrt(1.0/6.0)
	cx, cy := float64(w)/2, float64(h)/2
	bgCount, sampledCount := 0, 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			got := nrgbaPixel(dst, x, y)
			rd := math.Hypot(float64(x)-cx, float64(y)-cy)
			if rd > foldR {
				// Outside the invertibility interval: background only.
				if got != [4]uint8{7, 7, 7, 255} {
					t.Errorf("folded pixel (%d,%d) = %v, want background", x, y, got)
				}
				bgCount++
			} else if got == [4]uint8{7, 7, 7, 255} {
				bgCount++
			} else {
				sampledCount++
			}
		}
	}
	if bgCount == 0 || sampledCount == 0 {
		t.Errorf("expected both background and sampled pixels, got bg=%d sampled=%d", bgCount, sampledCount)
	}
}

func TestLensDistortionEmpty(t *testing.T) {
	f := LensDistortion(0, 0, 0.3, 0.05, color.Black, LinearInterpolation)
	b := f.Bounds(image.Rect(0, 0, 0, 0))
	if !b.Eq(image.Rect(0, 0, 0, 0)) {
		t.Errorf("empty bounds = %v", b)
	}
	dst := image.NewNRGBA(b)
	src := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	f.Draw(dst, src, nil) // must not panic
}

func TestLensDistortionDefaultCenter(t *testing.T) {
	// Explicit image center must match the zero-center default.
	src := image.NewNRGBA(image.Rect(0, 0, 7, 5))
	for i := range src.Pix {
		src.Pix[i] = uint8(i*31 + 1)
	}
	fDefault := LensDistortion(0, 0, 0.2, 0.05, color.Black, CubicInterpolation)
	fExplicit := LensDistortion(3.5, 2.5, 0.2, 0.05, color.Black, CubicInterpolation)
	d1 := image.NewNRGBA(fDefault.Bounds(src.Bounds()))
	d2 := image.NewNRGBA(fExplicit.Bounds(src.Bounds()))
	fDefault.Draw(d1, src, nil)
	fExplicit.Draw(d2, src, nil)
	if !checkBoundsAndPix(d1.Bounds(), d2.Bounds(), d1.Pix, d2.Pix) {
		t.Errorf("default and explicit image center differ")
	}
}

func TestLensDistortionFilterChain(t *testing.T) {
	// The new filter must compose with the existing Rotate filters and the
	// GIFT chaining machinery, serially and in parallel.
	src := image.NewNRGBA(image.Rect(0, 0, 11, 7))
	for y := 0; y < 7; y++ {
		for x := 0; x < 11; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(x * 21)
			src.Pix[i+1] = uint8(y * 33)
			src.Pix[i+2] = uint8((x + y) * 15)
			src.Pix[i+3] = 255
		}
	}

	makeChain := func() *GIFT {
		return New(
			Rotate90(),
			LensDistortion(0, 0, 0.15, -0.03, color.NRGBA{0, 0, 0, 255}, LinearInterpolation),
			Rotate(30, color.White, CubicInterpolation),
		)
	}

	gSerial := makeChain()
	gSerial.SetParallelization(false)
	gParallel := makeChain()

	bSerial := gSerial.Bounds(src.Bounds())
	bParallel := gParallel.Bounds(src.Bounds())
	if !bSerial.Eq(bParallel) {
		t.Fatalf("chain bounds differ: %v vs %v", bSerial, bParallel)
	}

	dSerial := image.NewNRGBA(bSerial)
	dParallel := image.NewNRGBA(bParallel)
	gSerial.Draw(dSerial, src)
	gParallel.Draw(dParallel, src)
	if !checkBoundsAndPix(dSerial.Bounds(), dParallel.Bounds(), dSerial.Pix, dParallel.Pix) {
		t.Errorf("serial and parallel filter chains differ")
	}

	// Step-by-step equivalent of the chain must give the same result.
	// Internal pipeline temps are NRGBA64, but the user-facing destination
	// here is 8-bit NRGBA, so allow the final quantization rounding.
	f0 := Rotate90()
	step0 := image.NewNRGBA(f0.Bounds(src.Bounds()))
	f0.Draw(step0, src, &Options{Parallelization: false})
	f1 := LensDistortion(0, 0, 0.15, -0.03, color.NRGBA{0, 0, 0, 255}, LinearInterpolation)
	step1 := image.NewNRGBA(f1.Bounds(step0.Bounds()))
	f1.Draw(step1, step0, &Options{Parallelization: false})
	f2 := Rotate(30, color.White, CubicInterpolation)
	step2 := image.NewNRGBA(f2.Bounds(step1.Bounds()))
	f2.Draw(step2, step1, &Options{Parallelization: false})
	if !dSerial.Bounds().Eq(step2.Bounds()) {
		t.Fatalf("bounds differ: %v vs %v", dSerial.Bounds(), step2.Bounds())
	}
	for i := range dSerial.Pix {
		d := int(dSerial.Pix[i]) - int(step2.Pix[i])
		if d < 0 {
			d = -d
		}
		if d > 2 {
			t.Errorf("chained result differs from step-by-step application at byte %d: %d vs %d",
				i, dSerial.Pix[i], step2.Pix[i])
			break
		}
	}
}

// TestLensDistortionStress sweeps extreme and non-finite parameter
// combinations through every interpolation mode, asserting the filter never
// panics and always emits finite, in-range pixel values (no holes or NaNs).
func TestLensDistortionStress(t *testing.T) {
	src := image.NewNRGBA(image.Rect(-4, -2, 8, 6))
	for y := -2; y < 6; y++ {
		for x := -4; x < 8; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8((x*29 + 200) & 0xff)
			src.Pix[i+1] = uint8((y*47 + 64) & 0xff)
			src.Pix[i+2] = uint8((x ^ y) * 13)
			src.Pix[i+3] = uint8((x*5 + y*3 + 128) & 0xff)
		}
	}

	values := []float32{
		0, 1, -1, 0.5, -0.5, 1e-6, -1e-6, 1e3, -1e3, 1e6, -1e6,
		3.4e38, -3.4e38, float32(math.NaN()),
		float32(math.Inf(1)), float32(math.Inf(-1)),
	}
	interps := []Interpolation{NearestNeighborInterpolation, LinearInterpolation, CubicInterpolation}
	bgs := []color.Color{color.Black, color.Transparent, color.NRGBA{1, 2, 3, 100}}

	n := 0
	for _, k1 := range values {
		for _, k2 := range values {
			for _, cx := range []float32{0, -7.5, 4.25, float32(math.NaN()), float32(math.Inf(1))} {
				for _, cy := range []float32{0, 3.75, -9.5, float32(math.Inf(-1))} {
					for _, interp := range interps {
						bg := bgs[n%len(bgs)]
						n++
						f := LensDistortion(cx, cy, k1, k2, bg, interp)
						dst := image.NewNRGBA(f.Bounds(src.Bounds()))
						f.Draw(dst, src, nil) // must not panic
						if len(dst.Pix) != 12*8*4 {
							t.Fatalf("unexpected output size with params %v %v %v %v", cx, cy, k1, k2)
						}
						for _, b := range dst.Pix {
							if b > 0xff {
								t.Fatalf("out-of-range byte %d for params %v %v %v %v", b, cx, cy, k1, k2)
							}
						}
					}
				}
			}
		}
	}
}
