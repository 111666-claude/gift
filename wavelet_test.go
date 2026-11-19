package gift

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// xorshift32 is a small deterministic PRNG used to generate fixed noise.
type xorshift32 uint32

func (x *xorshift32) next() uint32 {
	v := uint32(*x)
	v ^= v << 13
	v ^= v >> 17
	v ^= v << 5
	*x = xorshift32(v)
	return v
}

func (x *xorshift32) float64() float64 {
	return float64(x.next()&0xffffff) / 0xffffff
}

// TestWaveletDenoiseImpulse checks the exact output of the filter for a
// hand-computed 4x4 gray image with a single bright pixel.
//
// The luminance plane (impulse of value 1 at (1,1)) after one level of the
// normalized Haar transform is:
//
//	 0.25  0    -0.25  0
//	 0     0     0     0
//	-0.25  0     0.25  0
//	 0     0     0     0
//
// Soft-thresholding the detail coefficients with t = 0.2 shrinks the three
// 0.25 coefficients to 0.05. The inverse transform then gives:
//
//	0.2  0.2  0  0
//	0.2  0.4  0  0
//	0    0    0  0
//	0    0    0  0
//
// which corresponds to gray values 51, 51, 51 and 102.
func TestWaveletDenoiseImpulse(t *testing.T) {
	srcb := image.Rect(-1, -1, 3, 3)
	src := image.NewGray(srcb)
	src.SetGray(0, 0, color.Gray{255})

	g := New(WaveletDenoise(1, 0.2))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	expectedBounds := image.Rect(0, 0, 4, 4)
	expectedPix := []uint8{
		51, 51, 0, 0,
		51, 102, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	if !checkBoundsAndPix(dst.Bounds(), expectedBounds, dst.Pix, expectedPix) {
		t.Errorf("WaveletDenoise(1, 0.2) impulse: unexpected result: bounds=%v pix=%v", dst.Bounds(), dst.Pix)
	}
}

// TestWaveletDenoiseZeroThreshold checks that a zero, negative or NaN
// threshold leaves the image unchanged.
func TestWaveletDenoiseZeroThreshold(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 17, 9))
	rng := xorshift32(12345)
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.next())
	}

	thresholds := []float32{
		0,
		-0.5,
		float32(math.Inf(-1)),
		float32(math.NaN()),
	}
	for _, threshold := range thresholds {
		for _, levels := range []int{1, 2, 3} {
			g := New(WaveletDenoise(levels, threshold))
			dst := image.NewNRGBA(g.Bounds(src.Bounds()))
			g.Draw(dst, src)
			if !checkBoundsAndPix(dst.Bounds(), src.Bounds(), dst.Pix, src.Pix) {
				t.Errorf("WaveletDenoise(%d, %v): expected the unchanged image", levels, threshold)
			}
		}
	}
}

// TestWaveletDenoisePreservesHueAndAlpha checks that a constant
// semi-transparent color region is left unchanged by the filter: the detail
// coefficients are all zero, so hue and alpha must be preserved exactly.
func TestWaveletDenoisePreservesHueAndAlpha(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	c := color.NRGBA{153, 102, 51, 128}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			src.SetNRGBA(x, y, c)
		}
	}

	g := New(WaveletDenoise(3, 0.5))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if !checkBoundsAndPix(dst.Bounds(), src.Bounds(), dst.Pix, src.Pix) {
		t.Errorf("WaveletDenoise(3, 0.5) constant color: expected the unchanged image")
	}
}

// TestWaveletDenoiseAlpha checks that the alpha channel is passed through
// unchanged even when the luminance is modified, and that the hue of each
// pixel is preserved (the RGB channels are rescaled by the luminance ratio).
func TestWaveletDenoiseAlpha(t *testing.T) {
	w, h := 33, 21
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetNRGBA(x, y, color.NRGBA{
				uint8(40 + x*5),
				uint8(30 + y*6),
				uint8(100 + x*2),
				uint8(10 + x*7),
			})
		}
	}

	g := New(WaveletDenoise(2, 0.05))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := src.PixOffset(x, y)
			s, d := src.Pix[i:i+4], dst.Pix[i:i+4]
			if d[3] != s[3] {
				t.Fatalf("WaveletDenoise: alpha changed at (%d, %d): expected %d got %d", x, y, s[3], d[3])
			}
			h0, _, _ := convertRGBToHSL(float32(s[0])/255, float32(s[1])/255, float32(s[2])/255)
			h1, _, _ := convertRGBToHSL(float32(d[0])/255, float32(d[1])/255, float32(d[2])/255)
			dh := math.Abs(float64(h1 - h0))
			if dh > 0.5 {
				dh = 1 - dh
			}
			if dh > 0.03 {
				t.Fatalf("WaveletDenoise: hue changed at (%d, %d): expected %v got %v", x, y, h0, h1)
			}
		}
	}
}

// TestWaveletDenoiseThinLine checks that a thin bright line (a strong edge)
// survives the denoising while a plain blur would destroy it.
func TestWaveletDenoiseThinLine(t *testing.T) {
	src := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		src.SetGray(8, y, color.Gray{255})
	}

	g := New(WaveletDenoise(1, 0.1))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	for y := 0; y < 16; y++ {
		line := dst.GrayAt(8, y).Y
		neighbor := dst.GrayAt(9, y).Y
		far := dst.GrayAt(0, y).Y
		if line < 200 {
			t.Errorf("WaveletDenoise: thin line faded at y=%d: got %d", y, line)
		}
		if int(neighbor)+100 > int(line) {
			t.Errorf("WaveletDenoise: thin line leaked too much at y=%d: line=%d neighbor=%d", y, line, neighbor)
		}
		if far != 0 {
			t.Errorf("WaveletDenoise: background changed at y=%d: got %d", y, far)
		}
	}
}

// TestWaveletDenoiseNoise checks that the filter reduces a fixed noise
// pattern added to a smooth gradient.
func TestWaveletDenoiseNoise(t *testing.T) {
	w, h := 64, 64
	clean := image.NewGray(image.Rect(0, 0, w, h))
	noisy := image.NewGray(image.Rect(0, 0, w, h))
	rng := xorshift32(777)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 0.25 + 0.5*float64(x)/float64(w-1)
			clean.SetGray(x, y, color.Gray{uint8(v*255 + 0.5)})
			n := v + (rng.float64()-0.5)*0.16
			if n < 0 {
				n = 0
			}
			if n > 1 {
				n = 1
			}
			noisy.SetGray(x, y, color.Gray{uint8(n*255 + 0.5)})
		}
	}

	mse := func(img *image.Gray) float64 {
		var sum float64
		for i, v := range img.Pix {
			d := float64(v) - float64(clean.Pix[i])
			sum += d * d
		}
		return sum / float64(len(img.Pix))
	}

	g := New(WaveletDenoise(2, 0.06))
	dst := image.NewGray(g.Bounds(noisy.Bounds()))
	g.Draw(dst, noisy)

	mseNoisy := mse(noisy)
	mseDenoised := mse(dst)
	if mseDenoised >= mseNoisy {
		t.Errorf("WaveletDenoise: noise not reduced: mse before=%v after=%v", mseNoisy, mseDenoised)
	}
}

// TestWaveletDenoiseSizes checks odd and tiny image sizes, out-of-range
// levels and the consistency between parallelized and sequential processing.
func TestWaveletDenoiseSizes(t *testing.T) {
	sizes := []image.Rectangle{
		image.Rect(0, 0, 1, 1),
		image.Rect(0, 0, 1, 7),
		image.Rect(0, 0, 7, 1),
		image.Rect(0, 0, 2, 2),
		image.Rect(0, 0, 3, 5),
		image.Rect(0, 0, 5, 3),
		image.Rect(0, 0, 31, 17),
	}
	levels := []int{-2, 0, 1, 2, 3, 4, 100}

	for _, r := range sizes {
		src := image.NewNRGBA(r)
		rng := xorshift32(999)
		for i := range src.Pix {
			src.Pix[i] = uint8(rng.next())
		}
		for _, lv := range levels {
			g1 := New(WaveletDenoise(lv, 0.1))
			g1.SetParallelization(false)
			dst1 := image.NewNRGBA(g1.Bounds(src.Bounds()))
			g1.Draw(dst1, src)

			g2 := New(WaveletDenoise(lv, 0.1))
			dst2 := image.NewNRGBA(g2.Bounds(src.Bounds()))
			g2.Draw(dst2, src)

			expectedBounds := image.Rect(0, 0, r.Dx(), r.Dy())
			if !checkBoundsAndPix(dst1.Bounds(), expectedBounds, dst1.Pix, dst2.Pix) {
				t.Errorf("WaveletDenoise(%d, 0.1) size %v: parallel/sequential mismatch or bad bounds", lv, r)
			}
		}
	}

	// A 1x1 image has no detail coefficients: the output equals the input.
	src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	src.SetNRGBA(0, 0, color.NRGBA{200, 100, 50, 255})
	dst := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	New(WaveletDenoise(3, 0.9)).Draw(dst, src)
	if dst.Pix[0] != 200 || dst.Pix[1] != 100 || dst.Pix[2] != 50 || dst.Pix[3] != 255 {
		t.Errorf("WaveletDenoise(3, 0.9) 1x1: expected the unchanged pixel, got %v", dst.Pix)
	}
}

// TestWaveletDenoiseBounds checks that the filter output does not depend on
// the origin of the source image bounds.
func TestWaveletDenoiseBounds(t *testing.T) {
	w, h := 13, 9
	fill := func(src *image.NRGBA) {
		rng := xorshift32(4242)
		for i := range src.Pix {
			src.Pix[i] = uint8(rng.next())
		}
	}
	src1 := image.NewNRGBA(image.Rect(0, 0, w, h))
	fill(src1)
	src2 := image.NewNRGBA(image.Rect(-7, 11, -7+w, 11+h))
	fill(src2)

	g := New(WaveletDenoise(3, 0.08))
	dst1 := image.NewNRGBA(g.Bounds(src1.Bounds()))
	g.Draw(dst1, src1)
	dst2 := image.NewNRGBA(g.Bounds(src2.Bounds()))
	g.Draw(dst2, src2)

	if !checkBoundsAndPix(dst1.Bounds(), dst2.Bounds(), dst1.Pix, dst2.Pix) {
		t.Errorf("WaveletDenoise: the result depends on the source bounds origin")
	}
}

// TestWaveletDenoiseInfiniteThreshold checks that an infinite threshold is
// handled deterministically: all the detail coefficients are zeroed and a
// constant image is left unchanged.
func TestWaveletDenoiseInfiniteThreshold(t *testing.T) {
	inf := float32(math.Inf(1))

	src := image.NewGray(image.Rect(0, 0, 16, 16))
	for i := range src.Pix {
		src.Pix[i] = 90
	}
	dst := image.NewGray(image.Rect(0, 0, 16, 16))
	New(WaveletDenoise(2, inf)).Draw(dst, src)
	for i, v := range dst.Pix {
		if v != 90 {
			t.Fatalf("WaveletDenoise(2, +Inf) constant image: pix[%d]: expected 90 got %d", i, v)
		}
	}

	// A non-constant image must still produce a valid (finite) result.
	rng := xorshift32(31337)
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.next())
	}
	New(WaveletDenoise(3, inf)).Draw(dst, src)
	for i, v := range dst.Pix {
		if math.IsNaN(float64(v)) {
			t.Fatalf("WaveletDenoise(3, +Inf): pix[%d] is NaN", i)
		}
	}
}

// TestWaveletDenoiseChain checks that the filter integrates into a filter
// chain and produces the same result as the filters applied one by one.
func TestWaveletDenoiseChain(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 21, 13))
	rng := xorshift32(555)
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.next())
	}

	g := New(WaveletDenoise(2, 0.1), Invert())
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	tmp := image.NewNRGBA(image.Rect(0, 0, 21, 13))
	New(WaveletDenoise(2, 0.1)).Draw(tmp, src)
	expected := image.NewNRGBA(image.Rect(0, 0, 21, 13))
	New(Invert()).Draw(expected, tmp)

	if !checkBoundsAndPix(dst.Bounds(), expected.Bounds(), dst.Pix, expected.Pix) {
		t.Errorf("WaveletDenoise: filter chain result differs from the sequential application")
	}
}

// TestWaveletDenoiseLarge checks that a large noisy image is processed
// correctly (bounded memory: one luminance plane) and without data races.
func TestWaveletDenoiseLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large image test in short mode")
	}
	w, h := 2048, 1029
	src := image.NewGray(image.Rect(0, 0, w, h))
	rng := xorshift32(42)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 0.3 + 0.4*float64(x)/float64(w-1) + (rng.float64()-0.5)*0.1
			src.Pix[y*w+x] = uint8(v*255 + 0.5)
		}
	}

	g := New(WaveletDenoise(3, 0.05))
	dst := image.NewGray(g.Bounds(src.Bounds()))
	g.Draw(dst, src)

	if !dst.Bounds().Eq(image.Rect(0, 0, w, h)) {
		t.Errorf("WaveletDenoise large: bad bounds: %v", dst.Bounds())
	}
	// The noise must be reduced: the result must be smoother than the input.
	var gradSrc, gradDst float64
	for y := 0; y < h; y++ {
		for x := 0; x+1 < w; x++ {
			gradSrc += math.Abs(float64(src.Pix[y*w+x+1]) - float64(src.Pix[y*w+x]))
			gradDst += math.Abs(float64(dst.Pix[y*w+x+1]) - float64(dst.Pix[y*w+x]))
		}
	}
	if gradDst >= gradSrc {
		t.Errorf("WaveletDenoise large: noise not reduced: gradient before=%v after=%v", gradSrc, gradDst)
	}
}
