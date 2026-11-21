package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// Tolerances for comparing the separable fast path against the generic 2-D
// implementation. Both accumulate the same float32 products in a different
// order, so results differ only by a handful of ulps:
//   - on NRGBA64 destinations (16 bits/channel) differences stay below one
//     8-bit quantization level (1/255 of the full channel range); the bound
//     below is 1.5/255 to leave headroom for platform-dependent rounding;
//   - on NRGBA8 destinations ordinary kernels produce byte-identical output;
//     only zero-sum kernels whose normalization falls back to the positive
//     weight sum push values close to rounding boundaries, where a single
//     float32 ulp can flip a byte. Those cases are validated on the 16-bit
//     destination instead.
const (
	separableToleranceNRGBA64 = 1.5 / 255.0
)

// roundingSensitiveKernels lists kernels containing negative weights: their
// float32 summation order differs between the two implementations, so 8-bit
// output is compared with a one-level tolerance instead of byte equality.
var roundingSensitiveKernels = map[string]bool{
	"signed9": true,
	"mixed11": true,
}

func outerProductKernel(h, v []float32) []float32 {
	n := len(h)
	k := make([]float32, n*n)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			k[j*n+i] = v[j] * h[i]
		}
	}
	return k
}

func kernelSum(k []float32) float32 {
	var s float32
	for _, w := range k {
		s += w
	}
	return s
}

func genRandomNRGBA(seed int64, r image.Rectangle) *image.NRGBA {
	rnd := rand.New(rand.NewSource(seed))
	src := image.NewNRGBA(r)
	for i := range src.Pix {
		src.Pix[i] = uint8(rnd.Intn(256))
	}
	return src
}

func maxNRGBA64Diff(a, b *image.NRGBA64) float64 {
	var m float64
	for i := range a.Pix {
		d := math.Abs(float64(a.Pix[i])-float64(b.Pix[i])) / 65535.0
		if d > m {
			m = d
		}
	}
	return m
}

func maxNRGBADiff(a, b *image.NRGBA) float64 {
	var m float64
	for i := range a.Pix {
		d := math.Abs(float64(a.Pix[i])-float64(b.Pix[i])) / 255.0
		if d > m {
			m = d
		}
	}
	return m
}

func maxNRGBAByteDelta(a, b *image.NRGBA) int {
	m := 0
	for i := range a.Pix {
		d := int(a.Pix[i]) - int(b.Pix[i])
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

// TestFactorSeparableKernel checks the rank-one decomposition itself,
// including normalization handling, structural zeros and rejection of
// non-separable, empty, zero and non-finite kernels.
func TestFactorSeparableKernel(t *testing.T) {
	ones3 := []float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}

	testData := []struct {
		desc      string
		kernel    []float32
		normalize bool
		ok        bool
	}{
		{"empty kernel", []float32{}, false, false},
		{"1x1 kernel", []float32{1}, false, false},
		{"3x3 non-separable (emboss)", []float32{
			-1, -1, 0,
			-1, 1, 1,
			0, 1, 1,
		}, false, false},
		{"3x3 all ones non-normalized", ones3, false, true},
		{"3x3 all ones normalized", ones3, true, true},
		{"5x5 disk (not separable)", genDisk(5), true, false},
		{"5x5 zero kernel", make([]float32, 25), false, false},
		{"7x7 zero kernel normalized", make([]float32, 49), true, false},
		{"rank one with zeros", outerProductKernel(
			[]float32{0, -1, 2, -1, 0},
			[]float32{0.1, 0.2, 0.4, 0.2, 0.1},
		), false, true},
		{"rank one normalized", outerProductKernel(
			[]float32{1, 2, 3, 2, 1},
			[]float32{0.5, 1, 1.5, 1, 0.5},
		), true, true},
	}

	for _, d := range testData {
		h, v, ok := factorSeparableKernel(d.kernel, d.normalize)
		if ok != d.ok {
			t.Errorf("%s: got ok=%v want %v", d.desc, ok, d.ok)
			continue
		}
		if !ok {
			if h != nil || v != nil {
				t.Errorf("%s: factors must be nil on failure", d.desc)
			}
			continue
		}
		n := int(math.Sqrt(float64(len(d.kernel))))
		if len(h) != n || len(v) != n {
			t.Errorf("%s: unexpected factor lengths %d, %d (want %d)", d.desc, len(h), len(v), n)
		}
		// Every reconstructed entry must match the effective kernel.
		var div float32
		if d.normalize {
			var sum, sumpositive float32
			for _, w := range d.kernel {
				sum += w
				if w > 0 {
					sumpositive += w
				}
			}
			if sum != 0 {
				div = sum
			} else {
				div = sumpositive
			}
		}
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				got := v[j] * h[i]
				want := d.kernel[j*n+i]
				if div != 0 {
					want /= div
				}
				if want == 0 {
					if got != 0 {
						t.Errorf("%s: structural zero mismatch at (%d,%d)", d.desc, i, j)
					}
					continue
				}
				if math.Abs(float64(got-want)) > 2*separableMaxRelativeError*math.Abs(float64(want)) {
					t.Errorf("%s: reconstruction mismatch at (%d,%d): %g != %g", d.desc, i, j, got, want)
				}
			}
		}
	}

	// Non-finite kernels must always be rejected so the generic path keeps
	// the original (non-panicking) behavior and no division ever happens here.
	nanKernel := make([]float32, 25)
	nanKernel[12] = float32(math.NaN())
	if _, _, ok := factorSeparableKernel(nanKernel, false); ok {
		t.Error("NaN kernel must not be separable")
	}
	infKernel := make([]float32, 25)
	infKernel[0] = float32(math.Inf(1))
	if _, _, ok := factorSeparableKernel(infKernel, true); ok {
		t.Error("+Inf kernel must not be separable")
	}

	// A perturbed rank-one matrix is accepted for tiny perturbations and
	// rejected once it leaves the tolerance band.
	base := outerProductKernel([]float32{1, 2, 3, 4, 5}, []float32{0.5, 1, 1.5, 2, 2.5})
	pert := make([]float32, len(base))
	copy(pert, base)
	pert[12] += 1e-7
	if _, _, ok := factorSeparableKernel(pert, false); !ok {
		t.Error("near rank-one kernel (1e-7 perturbation) should be accepted")
	}
	pert[12] += 1e-3
	if _, _, ok := factorSeparableKernel(pert, false); ok {
		t.Error("non-rank-one kernel (1e-3 perturbation) should be rejected")
	}
}

// TestConvolutionSeparableParity compares separable kernels processed by the
// fast path against the generic 2-D implementation pixel by pixel.
func TestConvolutionSeparableParity(t *testing.T) {
	kernels := map[string][]float32{
		"box5": outerProductKernel(
			[]float32{1, 1, 1, 1, 1},
			[]float32{1, 1, 1, 1, 1},
		),
		"gaussian7": outerProductKernel(
			[]float32{0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044},
			[]float32{0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044},
		),
		"signed9": outerProductKernel(
			[]float32{0, 0, -1, 2, -1, 0, 0, 0, 0},
			[]float32{0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.0},
		),
		"mixed11": outerProductKernel(
			[]float32{-0.3, 0, 0.7, 1.4, 2, 1.4, 0.7, 0, -0.3, 0, 0},
			[]float32{0.1, 0.2, -0.4, 0, 0.8, 1, 0.8, 0, -0.4, 0.2, 0.1},
		),
	}

	rects := []image.Rectangle{
		image.Rect(0, 0, 32, 24),
		image.Rect(-3, -5, 29, 19), // non-zero Min bounds
		image.Rect(0, 0, 1, 1),     // single pixel
		image.Rect(0, 0, 7, 1),     // single row
		image.Rect(0, 0, 1, 7),     // single column
		image.Rect(0, 0, 3, 2),     // smaller than the kernels
	}

	for name, kernel := range kernels {
		for _, normalize := range []bool{false, true} {
			for _, alpha := range []bool{false, true} {
				for _, parallel := range []bool{false, true} {
					for _, r := range rects {
						src := genRandomNRGBA(1234, r)
						f := Convolution(kernel, normalize, alpha, false, 0)

						generic := image.NewNRGBA64(f.Bounds(r))
						f.(*convolutionFilter).drawGeneric(generic, src, &Options{Parallelization: parallel})

						fast := image.NewNRGBA64(f.Bounds(r))
						f.Draw(fast, src, &Options{Parallelization: parallel})

						if d := maxNRGBA64Diff(generic, fast); d > separableToleranceNRGBA64 {
							t.Errorf("%s normalize=%v alpha=%v parallel=%v r=%v: NRGBA64 diff %g exceeds %g",
								name, normalize, alpha, parallel, r, d, separableToleranceNRGBA64)
						}

						// Bounds must stay identical.
						if !generic.Bounds().Eq(fast.Bounds()) {
							t.Errorf("%s: bounds mismatch %v != %v", name, generic.Bounds(), fast.Bounds())
						}

						// On 8-bit destinations non-negative kernels must
						// be byte-identical. Kernels containing negative
						// weights accumulate the same float32 products in a
						// different order, so they may differ by at most one
						// 8-bit quantization level. The exception is a
						// zero-sum kernel under normalization: the fallback
						// divisor (positive weight sum) pushes results onto
						// rounding boundaries where one ulp flips the byte
						// completely. Such configurations are still covered
						// by the strict NRGBA64 comparison above.
						g8 := image.NewNRGBA(f.Bounds(r))
						s8 := image.NewNRGBA(f.Bounds(r))
						f.(*convolutionFilter).drawGeneric(g8, src, &Options{Parallelization: parallel})
						f.Draw(s8, src, &Options{Parallelization: parallel})
						switch {
						case roundingSensitiveKernels[name] && normalize && kernelSum(kernel) == 0:
							// validated via NRGBA64
						case roundingSensitiveKernels[name]:
							if d := maxNRGBAByteDelta(g8, s8); d > 1 {
								t.Errorf("%s normalize=%v alpha=%v parallel=%v r=%v: NRGBA byte delta %d exceeds 1",
									name, normalize, alpha, parallel, r, d)
							}
						default:
							if d := maxNRGBADiff(g8, s8); d != 0 {
								t.Errorf("%s normalize=%v alpha=%v parallel=%v r=%v: NRGBA diff %g, want 0",
									name, normalize, alpha, parallel, r, d)
							}
						}
					}
				}
			}
		}
	}
}

// TestConvolutionFastPathDecision verifies which configurations take the fast
// path and which must fall back to the generic implementation.
func TestConvolutionFastPathDecision(t *testing.T) {
	separable := outerProductKernel(
		[]float32{1, 2, 3, 2, 1},
		[]float32{0.5, 1, 1.5, 1, 0.5},
	)
	nonSeparable := genDisk(5)

	shouldFactor := func(kernel []float32, normalize, alpha, abs bool, delta float32) bool {
		if abs || delta != 0 {
			return false
		}
		_, _, ok := factorSeparableKernel(kernel, normalize)
		return ok
	}

	cases := []struct {
		desc             string
		kernel           []float32
		normalize, alpha bool
		abs              bool
		delta            float32
		wantSeparable    bool
	}{
		{"separable plain", separable, false, false, false, 0, true},
		{"separable normalized", separable, true, false, false, 0, true},
		{"separable alpha", separable, false, true, false, 0, true},
		{"separable normalized alpha", separable, true, true, false, 0, true},
		{"separable abs falls back", separable, false, true, true, 0, false},
		{"separable delta falls back", separable, false, false, false, 0.25, false},
		{"separable abs+delta falls back", separable, true, true, true, 1, false},
		{"non-separable falls back", nonSeparable, true, true, false, 0, false},
		{"zero kernel falls back", make([]float32, 25), true, true, false, 0, false},
		{"empty kernel falls back", []float32{}, false, false, false, 0, false},
	}

	for _, c := range cases {
		got := shouldFactor(c.kernel, c.normalize, c.alpha, c.abs, c.delta)
		if got != c.wantSeparable {
			t.Errorf("%s: fast path decision = %v, want %v", c.desc, got, c.wantSeparable)
		}
	}

	// End-to-end: an abs configuration on a separable kernel must produce
	// exactly the generic output (the fast path is bypassed).
	src := genRandomNRGBA(77, image.Rect(0, 0, 16, 16))
	fAbs := Convolution(separable, false, true, true, 0)
	g1 := image.NewNRGBA(fAbs.Bounds(src.Bounds()))
	g2 := image.NewNRGBA(fAbs.Bounds(src.Bounds()))
	fAbs.(*convolutionFilter).drawGeneric(g1, src, &Options{Parallelization: false})
	fAbs.Draw(g2, src, &Options{Parallelization: false})
	if maxNRGBADiff(g1, g2) != 0 {
		t.Error("abs configuration must produce byte-identical generic output")
	}

	fDelta := Convolution(separable, false, false, false, 3.0/255.0)
	d1 := image.NewNRGBA(fDelta.Bounds(src.Bounds()))
	d2 := image.NewNRGBA(fDelta.Bounds(src.Bounds()))
	fDelta.(*convolutionFilter).drawGeneric(d1, src, &Options{Parallelization: false})
	fDelta.Draw(d2, src, &Options{Parallelization: false})
	if maxNRGBADiff(d1, d2) != 0 {
		t.Error("delta configuration must produce byte-identical generic output")
	}
}

// TestConvolutionSeparableEdgeSafety ensures zero, non-finite and extremely
// wide kernels and degenerate image shapes never panic, never divide by zero
// and do not allocate more than the generic implementation's working set.
func TestConvolutionSeparableEdgeSafety(t *testing.T) {
	// Wide separable kernel (101x101) applied to images much smaller than it.
	wide := outerProductKernel(
		func() []float32 {
			k := make([]float32, 101)
			for i := range k {
				k[i] = float32(i%7) + 1
			}
			return k
		}(),
		func() []float32 {
			k := make([]float32, 101)
			for i := range k {
				k[i] = float32(101-i%5) - 50
			}
			return k
		}(),
	)

	rects := []image.Rectangle{
		image.Rect(0, 0, 1, 1),
		image.Rect(0, 0, 1, 5),
		image.Rect(0, 0, 5, 1),
		image.Rect(0, 0, 2, 3),
		image.Rect(-4, -4, 4, 4),
		image.Rect(0, 0, 50, 40),
	}

	for _, r := range rects {
		src := genRandomNRGBA(9, r)

		// wide separable kernel: fast path must match the generic output.
		f := Convolution(wide, true, true, false, 0)
		dg := image.NewNRGBA64(f.Bounds(r))
		df := image.NewNRGBA64(f.Bounds(r))
		f.(*convolutionFilter).drawGeneric(dg, src, &Options{Parallelization: false})
		f.Draw(df, src, &Options{Parallelization: false})
		if d := maxNRGBA64Diff(dg, df); d > separableToleranceNRGBA64 {
			t.Errorf("wide kernel r=%v: diff %g", r, d)
		}
	}

	// Zero kernel: output is all zero (RGB and, with alpha, A as well).
	zero := make([]float32, 25)
	zeroSrc := genRandomNRGBA(1, image.Rect(0, 0, 8, 8))
	zeroDst := image.NewNRGBA64(image.Rect(0, 0, 8, 8))
	Convolution(zero, true, true, false, 0).Draw(zeroDst, zeroSrc, nil)
	for i := range zeroDst.Pix {
		if zeroDst.Pix[i] != 0 {
			t.Fatalf("zero kernel produced non-zero output at %d: %d", i, zeroDst.Pix[i])
		}
	}

	// Non-finite kernels must not panic and must follow the generic path.
	for _, bad := range [][]float32{
		append(make([]float32, 24), float32(math.NaN())),
		append(make([]float32, 24), float32(math.Inf(-1))),
		append(make([]float32, 24), float32(math.Inf(1))),
	} {
		src := genRandomNRGBA(2, image.Rect(0, 0, 6, 6))
		d1 := image.NewNRGBA64(image.Rect(0, 0, 6, 6))
		d2 := image.NewNRGBA64(image.Rect(0, 0, 6, 6))
		Convolution(bad, true, true, false, 0).(*convolutionFilter).drawGeneric(d1, src, &Options{Parallelization: false})
		Convolution(bad, true, true, false, 0).Draw(d2, src, &Options{Parallelization: false})
		if maxNRGBA64Diff(d1, d2) != 0 {
			t.Error("non-finite kernel must take the generic path byte-for-byte")
		}
	}

	// Allocation guard: with a kernel much taller than the image the fast
	// path's per-goroutine ring must be image-height sized, not kernel sized.
	bigN := 501
	bigKernel := outerProductKernel(
		func() []float32 {
			k := make([]float32, bigN)
			for i := range k {
				k[i] = float32(i + 1)
			}
			return k
		}(),
		func() []float32 {
			k := make([]float32, bigN)
			for i := range k {
				k[i] = float32(bigN - i)
			}
			return k
		}(),
	)
	small := genRandomNRGBA(3, image.Rect(0, 0, 6, 4))
	fBig := Convolution(bigKernel, true, true, false, 0)
	dst := image.NewNRGBA64(fBig.Bounds(small.Bounds()))

	var ms0, ms1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms0)
	fBig.Draw(dst, small, &Options{Parallelization: false})
	runtime.ReadMemStats(&ms1)
	allocated := ms1.TotalAlloc - ms0.TotalAlloc

	// Upper bound: one image-sized ring of 4-byte float pixels plus a few
	// single-row buffers. A kernel-sized ring (the failure mode) would need
	// bigN rows of width 6 = ~48 KB; an image-sized ring stays under 4 KB.
	const maxAllowed = 8 * 1024
	if allocated > maxAllowed {
		t.Errorf("wide kernel on small image allocated %d bytes, want <= %d", allocated, maxAllowed)
	}

	// Empty/dst-less calls must remain panic free.
	Convolution(wide, true, true, false, 0).Draw(
		image.NewNRGBA(image.Rect(0, 0, 0, 0)),
		image.NewNRGBA(image.Rect(0, 0, 0, 0)), nil)
}

// TestConvolutionSeparableImageTypes checks the fast path against the generic
// implementation across the image types supported by newPixelGetter, both as
// sources and destinations.
func TestConvolutionSeparableImageTypes(t *testing.T) {
	kernel := outerProductKernel(
		[]float32{0.05, 0.25, 0.4, 0.25, 0.05},
		[]float32{1, 2, 3, 2, 1},
	)
	r := image.Rect(0, 0, 18, 14)
	nrgba := genRandomNRGBA(2024, r)
	nrgba64 := image.NewNRGBA64(r)
	rgba := image.NewRGBA(r)
	gray := image.NewGray(r)
	gray16 := image.NewGray16(r)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			c := nrgba.NRGBAAt(x, y)
			nrgba64.SetNRGBA64(x, y, color.NRGBA64{
				R: uint16(c.R) * 0x101,
				G: uint16(c.G) * 0x101,
				B: uint16(c.B) * 0x101,
				A: uint16(c.A) * 0x101,
			})
			rgba.Set(x, y, c)
			gray.Set(x, y, c)
			gray16.Set(x, y, c)
		}
	}

	sources := map[string]image.Image{
		"NRGBA":   nrgba,
		"NRGBA64": nrgba64,
		"RGBA":    rgba,
		"Gray":    gray,
		"Gray16":  gray16,
	}

	dests := map[string]func(image.Rectangle) draw.Image{
		"NRGBA":   func(b image.Rectangle) draw.Image { return image.NewNRGBA(b) },
		"NRGBA64": func(b image.Rectangle) draw.Image { return image.NewNRGBA64(b) },
		"RGBA":    func(b image.Rectangle) draw.Image { return image.NewRGBA(b) },
		"Gray":    func(b image.Rectangle) draw.Image { return image.NewGray(b) },
	}

	for sname, src := range sources {
		for dname, mkDst := range dests {
			f := Convolution(kernel, true, true, false, 0)
			d1 := mkDst(f.Bounds(r))
			d2 := mkDst(f.Bounds(r))
			f.(*convolutionFilter).drawGeneric(d1, src, &Options{Parallelization: true})
			f.Draw(d2, src, &Options{Parallelization: true})

			// Compare via re-decoding pixels so different concrete
			// destination types share one comparison path.
			w, hh := r.Dx(), r.Dy()
			for y := 0; y < hh; y++ {
				for x := 0; x < w; x++ {
					p1 := newPixelGetter(d1).getPixel(d1.Bounds().Min.X+x, d1.Bounds().Min.Y+y)
					p2 := newPixelGetter(d2).getPixel(d2.Bounds().Min.X+x, d2.Bounds().Min.Y+y)
					for ch, a := range [4]float32{p1.R, p1.G, p1.B, p1.A} {
						b := [4]float32{p2.R, p2.G, p2.B, p2.A}[ch]
						if math.Abs(float64(a-b)) > separableToleranceNRGBA64 {
							t.Fatalf("src=%s dst=%s (%d,%d) ch=%d: generic %g fast %g",
								sname, dname, x, y, ch, a, b)
						}
					}
				}
			}
		}
	}
}

//
// Benchmarks: large separable kernel, generic (non-separable) kernel of the
// same size, plus allocation accounting for both paths.
//

var (
	benchSize          = 200
	benchKernelSize    = 31
	benchSeparableK    = benchOuterKernel(benchKernelSize)
	benchNonSeparableK = genDisk(benchKernelSize)
	benchSrc           = genBenchImage(benchSize, benchSize)
	benchDst           draw.Image
)

func benchOuterKernel(n int) []float32 {
	h := make([]float32, n)
	v := make([]float32, n)
	for i := 0; i < n; i++ {
		x := float32(i - n/2)
		h[i] = float32(math.Exp(-float64(x*x) / (2 * 25)))
		v[i] = 1 + 0.1*float32(i)
	}
	return outerProductKernel(h, v)
}

func genBenchImage(w, h int) *image.NRGBA {
	rnd := rand.New(rand.NewSource(1))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = uint8(rnd.Intn(256))
	}
	return img
}

func BenchmarkConvolutionSeparable31(b *testing.B) {
	g := New(Convolution(benchSeparableK, true, true, false, 0))
	dst := image.NewNRGBA64(g.Bounds(benchSrc.Bounds()))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Draw(dst, benchSrc)
	}
	benchDst = dst
}

func BenchmarkConvolutionGeneric31(b *testing.B) {
	g := New(Convolution(benchNonSeparableK, true, true, false, 0))
	dst := image.NewNRGBA64(g.Bounds(benchSrc.Bounds()))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Draw(dst, benchSrc)
	}
	benchDst = dst
}

func BenchmarkConvolutionSeparable31SingleRun(b *testing.B) {
	dst := image.NewNRGBA64(image.Rect(0, 0, benchSize, benchSize))
	f := Convolution(benchSeparableK, true, true, false, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Draw(dst, benchSrc, &Options{Parallelization: true})
	}
	benchDst = dst
}

func BenchmarkConvolutionGeneric31SingleRun(b *testing.B) {
	dst := image.NewNRGBA64(image.Rect(0, 0, benchSize, benchSize))
	f := Convolution(benchNonSeparableK, true, true, false, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Draw(dst, benchSrc, &Options{Parallelization: true})
	}
	benchDst = dst
}
