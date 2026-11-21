package gift

import (
	"image"
	"math"
	"testing"
)

// gaborStripeImage builds an NRGBA test image with a cosine stripe pattern
// whose intensity varies along the given direction (degrees) with the given
// period in pixels. angleDeg=90 produces horizontal stripes.
func gaborStripeImage(w, h int, angleDeg, period, lo, hi float64, alpha uint8) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	sin, cos := math.Sincos(angleDeg * math.Pi / 180)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			phase := 2 * math.Pi * (float64(x)*cos + float64(y)*sin) / period
			v := lo + (hi-lo)*(0.5+0.5*math.Cos(phase))
			q := uint8(v*255 + 0.5)
			i := img.PixOffset(x, y)
			img.Pix[i+0] = q
			img.Pix[i+1] = q
			img.Pix[i+2] = q
			img.Pix[i+3] = alpha
		}
	}
	return img
}

// gaborMeanAbsDeviation returns the mean per-channel absolute deviation
// between two same-sized NRGBA images (in 0-255 units).
func gaborMeanAbsDeviation(a, b *image.NRGBA) float64 {
	var sum float64
	for i := 0; i < len(a.Pix); i += 4 {
		sum += math.Abs(float64(a.Pix[i+0]) - float64(b.Pix[i+0]))
		sum += math.Abs(float64(a.Pix[i+1]) - float64(b.Pix[i+1]))
		sum += math.Abs(float64(a.Pix[i+2]) - float64(b.Pix[i+2]))
	}
	return sum / float64(len(a.Pix)/4*3)
}

func gaborDraw(f Filter, src *image.NRGBA, parallel bool) *image.NRGBA {
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, &Options{Parallelization: parallel})
	return dst
}

func TestGaborOrientationSelection(t *testing.T) {
	for _, parallel := range []bool{true, false} {
		// Horizontal stripes: intensity varies along y, so the 90 degree
		// kernel must respond much stronger than the 0 degree kernel.
		src := gaborStripeImage(48, 48, 90, 6, 0.3, 0.7, 0xff)

		f0 := Gabor([]float32{0}, 6, 1, 1, GaborFirstOrientation)
		f90 := Gabor([]float32{90}, 6, 1, 1, GaborFirstOrientation)
		dev0 := gaborMeanAbsDeviation(gaborDraw(f0, src, parallel), src)
		dev90 := gaborMeanAbsDeviation(gaborDraw(f90, src, parallel), src)

		if dev90 <= dev0 {
			t.Errorf("horizontal stripes: 90 deg response (%v) should exceed 0 deg response (%v)", dev90, dev0)
		}
		if dev90 < 2*dev0 {
			t.Errorf("horizontal stripes: 90 deg response (%v) should dominate 0 deg response (%v)", dev90, dev0)
		}

		// Max-response combination of {0, 90} must be at least as strong
		// as the best single orientation (per-pixel max of absolutes).
		fall := Gabor([]float32{0, 90}, 6, 1, 1, GaborMaxResponse)
		devAll := gaborMeanAbsDeviation(gaborDraw(fall, src, parallel), src)
		if devAll+1 < dev90 {
			t.Errorf("max response combination (%v) weaker than best orientation (%v)", devAll, dev90)
		}

		// Diagonal stripes: the 45 degree kernel must win over 0 and 90.
		src45 := gaborStripeImage(48, 48, 45, 6, 0.3, 0.7, 0xff)
		f45 := Gabor([]float32{45}, 6, 1, 1, GaborFirstOrientation)
		dev45 := gaborMeanAbsDeviation(gaborDraw(f45, src45, parallel), src45)
		dev0d := gaborMeanAbsDeviation(gaborDraw(f0, src45, parallel), src45)
		dev90d := gaborMeanAbsDeviation(gaborDraw(f90, src45, parallel), src45)

		if dev45 <= dev0d || dev45 <= dev90d {
			t.Errorf("diagonal stripes: 45 deg response (%v) should exceed 0 deg (%v) and 90 deg (%v)", dev45, dev0d, dev90d)
		}
	}
}

func TestGaborConstantImage(t *testing.T) {
	// A constant semi-transparent image must pass through unchanged:
	// zero-mean kernels give zero response and alpha is preserved.
	src := image.NewNRGBA(image.Rect(0, 0, 9, 7))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i+0] = 100
		src.Pix[i+1] = 150
		src.Pix[i+2] = 200
		src.Pix[i+3] = 128
	}

	for _, parallel := range []bool{true, false} {
		f := Gabor([]float32{0, 45, 90}, 6, 1, 1, GaborMaxResponse)
		dst := gaborDraw(f, src, parallel)

		if !checkBoundsAndPix(dst.Bounds(), image.Rect(0, 0, 9, 7), dst.Pix, src.Pix) {
			t.Errorf("constant image changed: %#v, %#v", dst.Bounds(), dst.Pix)
		}
	}
}

func TestGaborAmountAttenuation(t *testing.T) {
	src := gaborStripeImage(48, 48, 90, 6, 0.3, 0.7, 0xff)

	for _, parallel := range []bool{true, false} {
		// Zero amount is a no-op.
		f0 := Gabor([]float32{90}, 6, 1, 0, GaborFirstOrientation)
		dst0 := gaborDraw(f0, src, parallel)
		if !checkBoundsAndPix(dst0.Bounds(), image.Rect(0, 0, 48, 48), dst0.Pix, src.Pix) {
			t.Errorf("zero amount should copy the image: %#v", dst0.Pix)
		}

		// The deviation scales (almost) linearly with the amount.
		f1 := Gabor([]float32{90}, 6, 1, 1, GaborFirstOrientation)
		f2 := Gabor([]float32{90}, 6, 1, 2, GaborFirstOrientation)
		dev1 := gaborMeanAbsDeviation(gaborDraw(f1, src, parallel), src)
		dev2 := gaborMeanAbsDeviation(gaborDraw(f2, src, parallel), src)

		if dev1 <= 0 {
			t.Errorf("expected non-zero response for amount=1")
		}
		if math.Abs(dev2-2*dev1) > 2 {
			t.Errorf("response should scale with amount: dev1=%v dev2=%v", dev1, dev2)
		}
	}
}

func TestGaborAlphaPreserved(t *testing.T) {
	// Semi-transparent stripes: alpha must survive even where the
	// luminance response is non-zero.
	src := gaborStripeImage(32, 32, 90, 6, 0.2, 0.8, 0x80)

	for _, parallel := range []bool{true, false} {
		f := Gabor([]float32{0, 90}, 6, 1, 1.5, GaborMaxResponse)
		dst := gaborDraw(f, src, parallel)
		for i := 3; i < len(dst.Pix); i += 4 {
			if dst.Pix[i] != 0x80 {
				t.Fatalf("alpha not preserved at pixel %d: got %d", i/4, dst.Pix[i])
			}
		}
		if gaborMeanAbsDeviation(dst, src) == 0 {
			t.Errorf("expected non-zero luminance response on stripes")
		}
	}
}

func TestGaborBounds(t *testing.T) {
	f := Gabor([]float32{0, 90}, 6, 1, 1, GaborMaxResponse)
	got := f.Bounds(image.Rect(-3, -2, 10, 20))
	want := image.Rect(0, 0, 13, 22)
	if !got.Eq(want) {
		t.Errorf("unexpected bounds: got %v want %v", got, want)
	}
}

func TestGaborSafety(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))

	filters := []Filter{
		Gabor([]float32{0}, 0, 1, 1, GaborMaxResponse),         // zero wavelength
		Gabor([]float32{0}, -5, 1, 1, GaborMaxResponse),        // negative wavelength
		Gabor([]float32{0}, nan, 1, 1, GaborMaxResponse),       // NaN wavelength
		Gabor([]float32{0}, inf, 1, 1, GaborMaxResponse),       // +Inf wavelength
		Gabor([]float32{0}, 0.5, 1, 1, GaborMaxResponse),       // sub-pixel wavelength
		Gabor([]float32{0}, 6, 0, 1, GaborMaxResponse),         // zero bandwidth
		Gabor([]float32{0}, 6, -2, 1, GaborMaxResponse),        // negative bandwidth
		Gabor([]float32{0}, 6, nan, 1, GaborMaxResponse),       // NaN bandwidth
		Gabor([]float32{0}, 6, 1000, 1, GaborMaxResponse),      // huge bandwidth / kernel clamp
		Gabor([]float32{0}, 6, 1, nan, GaborMaxResponse),       // NaN amount
		Gabor([]float32{0}, 6, 1, inf, GaborMaxResponse),       // +Inf amount
		Gabor(nil, 6, 1, 1, GaborMaxResponse),                  // nil orientations
		Gabor([]float32{}, 6, 1, 1, GaborMaxResponse),          // empty orientations
		Gabor([]float32{nan, 45}, 6, 1, 1, GaborMaxResponse),   // NaN orientation skipped
		Gabor([]float32{-30}, 6, 1, 1, GaborMaxResponse),       // negative angle
		Gabor([]float32{33.7}, 6, 1, 1, GaborMaxResponse),      // non-integer angle
		Gabor([]float32{0, 90}, 6, 1, 1, GaborCombineMode(42)), // invalid combine mode
	}

	sizes := []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 1, 1),
		image.Rect(0, 0, 2, 3),
		image.Rect(0, 0, 8, 8),
	}

	for fi, f := range filters {
		for _, r := range sizes {
			src := image.NewNRGBA(r)
			for i := 0; i < len(src.Pix); i += 4 {
				src.Pix[i+0] = 80
				src.Pix[i+1] = 120
				src.Pix[i+2] = 160
				src.Pix[i+3] = 200
			}
			for _, parallel := range []bool{true, false} {
				dst := image.NewNRGBA(f.Bounds(src.Bounds()))
				f.Draw(dst, src, &Options{Parallelization: parallel})
				want := image.Rect(0, 0, r.Dx(), r.Dy())
				if !dst.Bounds().Eq(want) {
					t.Errorf("filter %d size %v: unexpected bounds %v", fi, r, dst.Bounds())
				}
			}
		}
	}
}

func TestGaborKernel(t *testing.T) {
	// The kernel must be finite, odd-sized, zero-mean and bounded.
	for _, d := range []struct {
		theta, wavelength, bandwidth float32
	}{
		{0, 4, 1},
		{45, 6, 1},
		{90, 8, 0.5},
		{33.7, 3, 2},
		{135, 100, 10}, // hits the kernel radius clamp
	} {
		size, kernel := gaborKernel(d.theta, d.wavelength, d.bandwidth)
		if size < 1 || size%2 == 0 {
			t.Errorf("bad kernel size %d for %+v", size, d)
		}
		if size > 2*maxGaborKernelRadius+1 {
			t.Errorf("kernel size %d exceeds the safety bound", size)
		}
		if len(kernel) != size*size {
			t.Errorf("bad kernel length %d for size %d", len(kernel), size)
		}
		var sum float64
		for _, w := range kernel {
			if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				t.Fatalf("non-finite kernel weight for %+v", d)
			}
			sum += float64(w)
		}
		if math.Abs(sum) > 1e-3 {
			t.Errorf("kernel is not zero-mean: sum=%v for %+v", sum, d)
		}
	}
}
