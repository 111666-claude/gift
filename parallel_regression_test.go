package gift

// System-level regression tests for the parallel (multi-goroutine) image
// processing paths.
//
// The tests in this file deliberately do NOT compare against a checked-in
// golden image.  A single golden image is normally produced with one set of
// bounds and one GOMAXPROCS value; a chunk boundary bug can therefore pass it
// unnoticed (for example if the broken row happens to be overwritten later or
// lies outside the golden crop).  Instead, every case is rendered twice:
//
//   - once with parallelization disabled (the serial reference), and
//   - once with parallelization enabled (usually many small chunks).
//
// Destination bounds and the complete pixel buffers must be bit-for-bit equal.
// In addition the complete backing buffer of every source image is snapshotted
// before and after rendering to prove that no filter writes to its input.
//
// Every generated image is derived from an explicit seed.  A failure reports
// the seed, image format, source bounds, filter chain, prefix and the first
// differing coordinate, so it can be reproduced with:
//
//	GIFT_TEST_SEED=<seed> go test -run TestParallel -v
//
// Two deliberately broken chunk implementations are exercised in
// TestParallelInjectedChunkBugs.  They prove that the comparison used by the
// main tests actually detects chunk/boundary errors of the kind described
// above.

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// Deterministic image generation
// -----------------------------------------------------------------------------

type regFormat int

const (
	regRGBA regFormat = iota
	regNRGBA
	regGray16
)

func (f regFormat) String() string {
	switch f {
	case regRGBA:
		return "RGBA"
	case regNRGBA:
		return "NRGBA"
	case regGray16:
		return "Gray16"
	}
	return fmt.Sprintf("regFormat(%d)", int(f))
}

func (f regFormat) bpp() int {
	switch f {
	case regRGBA, regNRGBA:
		return 4
	case regGray16:
		return 2
	}
	return 1
}

// regVariant describes one source shape/format.  A subimage variant has a
// parent rectangle larger than bounds, which makes the Go image stride differ
// from the row width and gives the source non-zero Bounds().Min.
type regVariant struct {
	name   string
	format regFormat
	sub    bool
	parent image.Rectangle
	bounds image.Rectangle
}

func regVariants() []regVariant {
	return []regVariant{
		{
			name:   "RGBA/direct-zero-origin",
			format: regRGBA,
			bounds: image.Rect(0, 0, 23, 14),
		},
		{
			name:   "RGBA/direct-nonzero-origin",
			format: regRGBA,
			bounds: image.Rect(-3, 5, 20, 19),
		},
		{
			name:   "RGBA/subimage-nonzero-origin",
			format: regRGBA,
			sub:    true,
			parent: image.Rect(-9, 2, 31, 26),
			bounds: image.Rect(-4, 7, 19, 21),
		},
		{
			name:   "NRGBA/direct-zero-origin",
			format: regNRGBA,
			bounds: image.Rect(0, 0, 23, 14),
		},
		{
			name:   "NRGBA/direct-nonzero-origin",
			format: regNRGBA,
			bounds: image.Rect(-3, 5, 20, 19),
		},
		{
			name:   "NRGBA/subimage-nonzero-origin",
			format: regNRGBA,
			sub:    true,
			parent: image.Rect(-9, 2, 31, 26),
			bounds: image.Rect(-4, 7, 19, 21),
		},
		{
			name:   "Gray16/direct-zero-origin",
			format: regGray16,
			bounds: image.Rect(0, 0, 23, 14),
		},
		{
			name:   "Gray16/direct-nonzero-origin",
			format: regGray16,
			bounds: image.Rect(-3, 5, 20, 19),
		},
		{
			name:   "Gray16/subimage-nonzero-origin",
			format: regGray16,
			sub:    true,
			parent: image.Rect(-9, 2, 31, 26),
			bounds: image.Rect(-4, 7, 19, 21),
		},
		// Large images: parallelize uses a chunk size of datasize/(GOMAXPROCS*10),
		// so these are needed to exercise chunks that span several rows (for
		// vertical scanning filters) and several columns (for horizontal ones).
		{
			name:   "RGBA/large-nonzero-origin",
			format: regRGBA,
			bounds: image.Rect(-5, 11, 132, 102),
		},
		{
			name:   "NRGBA/large-subimage-nonzero-origin",
			format: regNRGBA,
			sub:    true,
			parent: image.Rect(-13, 1, 150, 120),
			bounds: image.Rect(-7, 6, 124, 103),
		},
		{
			name:   "Gray16/large-nonzero-origin",
			format: regGray16,
			bounds: image.Rect(3, -4, 152, 99),
		},
	}
}

// regSource keeps the complete backing buffer so that writes outside a
// subimage are detected too.
type regSource struct {
	format  regFormat
	bounds  image.Rectangle
	img     draw.Image
	backing []byte
}

func regNewImage(format regFormat, r image.Rectangle) draw.Image {
	switch format {
	case regRGBA:
		return image.NewRGBA(r)
	case regNRGBA:
		return image.NewNRGBA(r)
	case regGray16:
		return image.NewGray16(r)
	}
	panic("unknown format")
}

func regPix(img draw.Image) []byte {
	switch img := img.(type) {
	case *image.RGBA:
		return img.Pix
	case *image.NRGBA:
		return img.Pix
	case *image.Gray16:
		return img.Pix
	}
	panic("unknown image type")
}

var regAlphaPattern = [...]uint8{0, 255, 17, 128, 254, 1, 64, 200}

// regFill deterministically fills the full extent of img.  The values include
// transparent, opaque and partially transparent pixels so alpha handling is
// covered.  RGBA data is stored premultiplied, as required by image.RGBA.
func regFill(img draw.Image, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	b := img.Bounds()
	n := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			switch img := img.(type) {
			case *image.RGBA:
				i := img.PixOffset(x, y)
				a := regAlphaPattern[n%len(regAlphaPattern)]
				r := uint8(rng.Intn(256))
				g := uint8(rng.Intn(256))
				bl := uint8(rng.Intn(256))
				// Store premultiplied values.
				img.Pix[i+0] = uint8(uint32(r) * uint32(a) / 255)
				img.Pix[i+1] = uint8(uint32(g) * uint32(a) / 255)
				img.Pix[i+2] = uint8(uint32(bl) * uint32(a) / 255)
				img.Pix[i+3] = a
			case *image.NRGBA:
				i := img.PixOffset(x, y)
				img.Pix[i+0] = uint8(rng.Intn(256))
				img.Pix[i+1] = uint8(rng.Intn(256))
				img.Pix[i+2] = uint8(rng.Intn(256))
				img.Pix[i+3] = regAlphaPattern[n%len(regAlphaPattern)]
			case *image.Gray16:
				i := img.PixOffset(x, y)
				var v uint16
				switch n % 11 {
				case 0:
					v = 0
				case 1:
					v = 0xffff
				default:
					v = uint16(rng.Intn(1 << 16))
				}
				img.Pix[i+0] = uint8(v >> 8)
				img.Pix[i+1] = uint8(v & 0xff)
			default:
				panic("unknown image type")
			}
			n++
		}
	}
}

func regNewSource(v regVariant, seed int64) *regSource {
	if v.sub {
		parent := regNewImage(v.format, v.parent)
		regFill(parent, seed)
		img, ok := parent.(interface {
			SubImage(image.Rectangle) image.Image
		}).SubImage(v.bounds).(draw.Image)
		if !ok {
			panic("subimage is not draw.Image")
		}
		return &regSource{
			format:  v.format,
			bounds:  img.Bounds(),
			img:     img,
			backing: regPix(parent),
		}
	}
	img := regNewImage(v.format, v.bounds)
	regFill(img, seed)
	return &regSource{
		format:  v.format,
		bounds:  img.Bounds(),
		img:     img,
		backing: regPix(img),
	}
}

func regNewDest(format regFormat, r image.Rectangle, seed int64) draw.Image {
	img := regNewImage(format, r)
	regFill(img, seed^0x5bd1e995)
	return img
}

// -----------------------------------------------------------------------------
// Filter chains
// -----------------------------------------------------------------------------

type regFilter struct {
	name string
	f    Filter
}

type regChain struct {
	name    string
	filters []regFilter
}

var regConvolutionKernel = []float32{
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1,
}

func regChains() []regChain {
	return []regChain{
		{
			// Exercises an empty filter list (the copyimage path).
			name: "copy/empty-chain",
		},
		{
			// Every required category in one chain:
			// scale, rotate, convolution, crop, color.
			name: "scale-rotate-conv-crop-color",
			filters: []regFilter{
				{"resize/lanczos", Resize(17, 0, LanczosResampling)},
				{"rotate/17-linear", Rotate(17, color.NRGBA{R: 9, G: 17, B: 33, A: 200}, LinearInterpolation)},
				{"gaussian-blur/0.9", GaussianBlur(0.9)},
				{"crop", Crop(image.Rect(2, 2, 15, 13))},
				{"contrast/18", Contrast(18)},
				{"gamma/1.6", Gamma(1.6)},
			},
		},
		{
			// Convolution first; crop, scale, color and rotate after it.
			name: "conv-crop-scale-color-rotate",
			filters: []regFilter{
				{"convolution/sobel-like", Convolution(regConvolutionKernel, false, false, true, 0)},
				{"crop", Crop(image.Rect(3, 2, 19, 12))},
				{"resize/box", Resize(0, 9, BoxResampling)},
				{"colorize", Colorize(210, 40, 70)},
				{"rotate90", Rotate90()},
				{"sepia/60", Sepia(60)},
			},
		},
		{
			// CropToSize, resize-to-fit, rank filtering and a long color chain.
			name: "crop-size-fit-rank-colors",
			filters: []regFilter{
				{"crop-to-size", CropToSize(15, 11, CenterAnchor)},
				{"resize-to-fit", ResizeToFit(13, 13, LinearResampling)},
				{"median/3", Median(3, false)},
				{"unsharp-mask", UnsharpMask(0.7, 1.0, 0.01)},
				{"grayscale", Grayscale()},
				{"color-balance", ColorBalance(8, -5, 3)},
				{"sigmoid", Sigmoid(0.5, 3)},
			},
		},
		{
			// Nearest-neighbour scaling and pixelation, which process pixels
			// using destination/chunk-local indexing.
			name: "nearest-pixelate-blur",
			filters: []regFilter{
				{"resize/nearest", Resize(0, 61, NearestNeighborResampling)},
				{"pixelate/5", Pixelate(5)},
				{"resize/nearest-explicit", Resize(37, 29, NearestNeighborResampling)},
			},
		},
		{
			// Non-commutative pair A: resize then blur then crop.
			name: "order-A-resize-blur-crop",
			filters: []regFilter{
				{"resize/linear", Resize(15, 11, LinearResampling)},
				{"gaussian-blur/0.8", GaussianBlur(0.8)},
				{"crop", Crop(image.Rect(2, 2, 13, 9))},
			},
		},
		{
			// Non-commutative pair B: blur then resize then crop.
			name: "order-B-blur-resize-crop",
			filters: []regFilter{
				{"gaussian-blur/0.8", GaussianBlur(0.8)},
				{"resize/linear", Resize(15, 11, LinearResampling)},
				{"crop", Crop(image.Rect(2, 2, 13, 9))},
			},
		},
		{
			// Non-commutative pair C: crop then blur then resize.
			name: "order-C-crop-blur-resize",
			filters: []regFilter{
				{"crop", Crop(image.Rect(2, 2, 21, 12))},
				{"gaussian-blur/0.8", GaussianBlur(0.8)},
				{"resize/linear", Resize(15, 11, LinearResampling)},
			},
		},
	}
}

// -----------------------------------------------------------------------------
// Rendering and comparison
// -----------------------------------------------------------------------------

type regOutput struct {
	bounds       image.Rectangle
	pix          []byte
	srcBefore    []byte
	srcAfter     []byte
	srcModified  bool
	panicValue   interface{}
	panicContext string
}

func regRun(v regVariant, seed int64, filters []Filter, parallel bool, dstBounds image.Rectangle, pt image.Point, op Operator, useDrawAt bool) (out regOutput) {
	src := regNewSource(v, seed)
	out.srcBefore = append([]byte(nil), src.backing...)

	defer func() {
		if r := recover(); r != nil {
			out.panicValue = r
			out.panicContext = fmt.Sprintf("format=%s srcBounds=%v", v.format, src.bounds)
			out.srcAfter = append([]byte(nil), src.backing...)
		}
	}()

	g := New(filters...)
	g.SetParallelization(parallel)

	var dst draw.Image
	if useDrawAt {
		dst = regNewDest(v.format, dstBounds, seed)
		g.DrawAt(dst, src.img, pt, op)
	} else {
		dst = regNewDest(v.format, g.Bounds(src.img.Bounds()), seed)
		g.Draw(dst, src.img)
	}

	out.bounds = dst.Bounds()
	out.pix = append([]byte(nil), regPix(dst)...)
	out.srcAfter = append([]byte(nil), src.backing...)
	out.srcModified = !bytes.Equal(out.srcBefore, out.srcAfter)
	return out
}

type regDiff struct {
	kind    string
	x, y    int
	channel int
	want    byte
	got     byte
	msg     string
}

func (d *regDiff) String() string {
	if d.kind == "pix" {
		return fmt.Sprintf("pixel mismatch at coord=(%d,%d) byte-channel=%d: serial=%#02x parallel=%#02x", d.x, d.y, d.channel, d.want, d.got)
	}
	return d.msg
}

func regFirstByteDiff(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

func regByteCoord(bounds image.Rectangle, bpp, index int) (x, y, channel int) {
	if bounds.Empty() {
		return 0, 0, index
	}
	pixelIndex := index / bpp
	return bounds.Min.X + pixelIndex%bounds.Dx(),
		bounds.Min.Y + pixelIndex/bounds.Dx(),
		index % bpp
}

// regCompare returns the first difference between the serial reference (want)
// and the parallel render (got).  Bounds and every pixel byte must match.
func regCompare(want, got regOutput, format regFormat) *regDiff {
	if want.panicValue != nil || got.panicValue != nil {
		if fmt.Sprint(want.panicValue) != fmt.Sprint(got.panicValue) {
			return &regDiff{
				kind: "panic",
				msg: fmt.Sprintf("panic mismatch: serial=%v (%s) parallel=%v (%s)",
					want.panicValue, want.panicContext, got.panicValue, got.panicContext),
			}
		}
		return nil
	}
	if !want.bounds.Eq(got.bounds) {
		return &regDiff{
			kind: "bounds",
			msg:  fmt.Sprintf("destination bounds mismatch: serial=%v parallel=%v", want.bounds, got.bounds),
		}
	}
	if len(want.pix) != len(got.pix) {
		return &regDiff{
			kind: "pixlen",
			msg:  fmt.Sprintf("destination pixel buffer length mismatch: serial=%d parallel=%d", len(want.pix), len(got.pix)),
		}
	}
	i := regFirstByteDiff(want.pix, got.pix)
	if i >= 0 {
		x, y, ch := regByteCoord(want.bounds, format.bpp(), i)
		return &regDiff{
			kind:    "pix",
			x:       x,
			y:       y,
			channel: ch,
			want:    want.pix[i],
			got:     got.pix[i],
		}
	}
	return nil
}

func regCompareSource(want, got regOutput, v regVariant) *regDiff {
	if want.srcModified && got.srcModified {
		i := regFirstByteDiff(want.srcBefore, want.srcAfter)
		if i < 0 {
			i = regFirstByteDiff(got.srcBefore, got.srcAfter)
		}
		x, y, ch := regByteCoord(v.bounds, v.format.bpp(), i)
		return &regDiff{
			kind:    "src-modified",
			x:       x,
			y:       y,
			channel: ch,
			want:    want.srcBefore[i],
			got:     want.srcAfter[i],
			msg: fmt.Sprintf("source backing buffer was modified by both serial and parallel runs at coord=(%d,%d) byte-channel=%d: before=%#02x after=%#02x",
				x, y, ch, want.srcBefore[i], want.srcAfter[i]),
		}
	}
	if want.srcModified {
		i := regFirstByteDiff(want.srcBefore, want.srcAfter)
		x, y, ch := regByteCoord(v.bounds, v.format.bpp(), i)
		return &regDiff{
			kind:    "src-modified",
			x:       x,
			y:       y,
			channel: ch,
			want:    want.srcBefore[i],
			got:     want.srcAfter[i],
			msg: fmt.Sprintf("source backing buffer was modified by the serial run at coord=(%d,%d) byte-channel=%d: before=%#02x after=%#02x",
				x, y, ch, want.srcBefore[i], want.srcAfter[i]),
		}
	}
	if got.srcModified {
		i := regFirstByteDiff(got.srcBefore, got.srcAfter)
		x, y, ch := regByteCoord(v.bounds, v.format.bpp(), i)
		return &regDiff{
			kind:    "src-modified",
			x:       x,
			y:       y,
			channel: ch,
			want:    got.srcBefore[i],
			got:     got.srcAfter[i],
			msg: fmt.Sprintf("source backing buffer was modified by the parallel run at coord=(%d,%d) byte-channel=%d: before=%#02x after=%#02x",
				x, y, ch, got.srcBefore[i], got.srcAfter[i]),
		}
	}
	return nil
}

func regSeeds(t *testing.T) []int64 {
	t.Helper()
	if s := strings.TrimSpace(os.Getenv("GIFT_TEST_SEED")); s != "" {
		var seeds []int64
		for _, part := range strings.Split(s, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			n, err := strconv.ParseInt(part, 0, 64)
			if err != nil {
				t.Fatalf("invalid GIFT_TEST_SEED %q: %v", s, err)
			}
			seeds = append(seeds, n)
		}
		if len(seeds) == 0 {
			t.Fatalf("GIFT_TEST_SEED %q contains no seeds", s)
		}
		return seeds
	}
	return []int64{1, 0xC0FFEE}
}

// regForceMultiCore makes sure the parallel path really splits work.  With 4
// usable Ps the production chunk size becomes 1 for typical test images, so
// chunk boundaries occur on every processed row/column.
func regForceMultiCore(t *testing.T) func() {
	t.Helper()
	old := runtime.GOMAXPROCS(4)
	return func() { runtime.GOMAXPROCS(old) }
}

func regChainNames(filters []regFilter, prefix int) string {
	names := make([]string, 0, prefix)
	for i := 0; i < prefix; i++ {
		names = append(names, filters[i].name)
	}
	return strings.Join(names, " -> ")
}

// regCheckPrefix renders every prefix of a chain serially and in parallel and
// reports a fully qualified failure for the first difference.
func regCheckPrefix(t *testing.T, seed int64, v regVariant, ch regChain, prefix int, parallel bool) {
	t.Helper()
	filters := make([]Filter, prefix)
	for i := 0; i < prefix; i++ {
		filters[i] = ch.filters[i].f
	}

	serial := regRun(v, seed, filters, false, image.Rectangle{}, image.Point{}, CopyOperator, false)
	par := regRun(v, seed, filters, true, image.Rectangle{}, image.Point{}, CopyOperator, false)

	if d := regCompareSource(serial, par, v); d != nil {
		t.Errorf("seed=%d GOMAXPROCS=%d variant=%q srcBounds=%v chain=%q filter[%d]=%q: %s",
			seed, runtime.GOMAXPROCS(0), v.name, v.bounds, ch.name, prefix-1, ch.filters[prefix-1].name, d)
		return
	}
	if d := regCompare(serial, par, v.format); d != nil {
		last := "none"
		if prefix > 0 {
			last = ch.filters[prefix-1].name
		}
		t.Errorf("seed=%d GOMAXPROCS=%d variant=%q srcBounds=%v chain=%q prefix=%q filter[%d]=%q parallel=%v: %s",
			seed, runtime.GOMAXPROCS(0), v.name, v.bounds, ch.name, regChainNames(ch.filters, prefix), prefix-1, last, parallel, d)
	}
}

// -----------------------------------------------------------------------------
// Main regression matrix: formats, non-zero bounds, subimages, filter chains
// -----------------------------------------------------------------------------

func TestParallelFormatsAndBounds(t *testing.T) {
	defer regForceMultiCore(t)()
	seeds := regSeeds(t)

	for _, v := range regVariants() {
		v := v
		t.Run(v.name, func(t *testing.T) {
			for _, ch := range regChains() {
				ch := ch
				for _, seed := range seeds {
					// Prefix 0 is the empty chain too.
					for prefix := 0; prefix <= len(ch.filters); prefix++ {
						regCheckPrefix(t, seed, v, ch, prefix, true)
					}
				}
			}
		})
	}
}

// TestParallelAcrossProcCounts re-runs the main matrix at several processor
// counts.  The production chunk size is datasize/(GOMAXPROCS*10), so each
// count produces a different number of chunks and different boundary
// positions; a boundary bug that only shows for a particular chunk layout is
// still caught.  Only chains using multi-row chunks are repeated here to keep
// the runtime low; the large variants cover that case.
func TestParallelAcrossProcCounts(t *testing.T) {
	t.Helper()
	seeds := regSeeds(t)
	procs := []int{2, 3, 4, 7}
	chains := regChains()
	// Indices into regChains: convolution chain, order pairs and the
	// nearest/pixelate chain all have multi-row or multi-column work.
	chainIdx := []int{1, 2, 3, 5, 6, 7}

	for _, p := range procs {
		p := p
		t.Run(fmt.Sprintf("GOMAXPROCS=%d", p), func(t *testing.T) {
			old := runtime.GOMAXPROCS(p)
			defer runtime.GOMAXPROCS(old)

			for _, v := range regVariants() {
				if v.bounds.Dy() < p*10 || v.bounds.Dx() < p*10 {
					continue // chunks would be a single row/column
				}
				for _, ci := range chainIdx {
					if ci >= len(chains) {
						continue
					}
					ch := chains[ci]
					for _, seed := range seeds {
						regCheckPrefix(t, seed, v, ch, len(ch.filters), true)
					}
				}
			}
		})
	}
}

// TestParallelFilterChainOrder proves, independently for each ordering, that
// parallel and serial agree.  It also asserts that the chosen filter pairs are
// genuinely order-sensitive (otherwise the per-ordering test would be vacuous).
func TestParallelFilterChainOrder(t *testing.T) {
	defer regForceMultiCore(t)()
	seeds := regSeeds(t)

	pairs := []struct {
		name string
		a, b []regFilter
	}{
		{
			name: "color-contrast-then-gamma",
			a: []regFilter{
				{"contrast/30", Contrast(30)},
				{"gamma/1.7", Gamma(1.7)},
			},
			b: []regFilter{
				{"gamma/1.7", Gamma(1.7)},
				{"contrast/30", Contrast(30)},
			},
		},
		{
			name: "spatial-blur-then-crop",
			a: []regFilter{
				{"gaussian-blur/1.2", GaussianBlur(1.2)},
				{"crop", Crop(image.Rect(3, 2, 15, 11))},
			},
			b: []regFilter{
				{"crop", Crop(image.Rect(3, 2, 15, 11))},
				{"gaussian-blur/1.2", GaussianBlur(1.2)},
			},
		},
	}

	for _, v := range regVariants() {
		v := v
		t.Run(v.name, func(t *testing.T) {
			for _, pair := range pairs {
				pair := pair
				for _, seed := range seeds {
					// Each ordering is checked and reported independently.
					for _, chain := range []struct {
						label   string
						filters []regFilter
					}{
						{"order-A", pair.a},
						{"order-B", pair.b},
					} {
						fs := make([]Filter, len(chain.filters))
						for i := range chain.filters {
							fs[i] = chain.filters[i].f
						}
						serial := regRun(v, seed, fs, false, image.Rectangle{}, image.Point{}, CopyOperator, false)
						par := regRun(v, seed, fs, true, image.Rectangle{}, image.Point{}, CopyOperator, false)
						if d := regCompareSource(serial, par, v); d != nil {
							t.Errorf("seed=%d variant=%q srcBounds=%v pair=%q %s: %s",
								seed, v.name, v.bounds, pair.name, chain.label, d)
							continue
						}
						if d := regCompare(serial, par, v.format); d != nil {
							t.Errorf("seed=%d variant=%q srcBounds=%v pair=%q %s: %s",
								seed, v.name, v.bounds, pair.name, chain.label, d)
						}
					}

					// Prove that changing the order changes the result, so the
					// independent checks above actually differentiate orders.
					fa := make([]Filter, len(pair.a))
					fb := make([]Filter, len(pair.b))
					for i := range pair.a {
						fa[i] = pair.a[i].f
					}
					for i := range pair.b {
						fb[i] = pair.b[i].f
					}
					serialA := regRun(v, seed, fa, false, image.Rectangle{}, image.Point{}, CopyOperator, false)
					serialB := regRun(v, seed, fb, false, image.Rectangle{}, image.Point{}, CopyOperator, false)
					if !serialA.bounds.Eq(serialB.bounds) {
						// Different dimensions are enough for the order to matter.
						continue
					}
					if bytes.Equal(serialA.pix, serialB.pix) {
						t.Errorf("seed=%d variant=%q pair=%q: filter order had no observable effect; order test would be vacuous",
							seed, v.name, pair.name)
					}
				}
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Empty and narrow images (independent determinations)
// -----------------------------------------------------------------------------

func regSmallChains() []regChain {
	return []regChain{
		{
			name: "copy",
		},
		{
			name: "convolution",
			filters: []regFilter{
				{"convolution/3x3", Convolution(regConvolutionKernel, true, true, false, 0)},
			},
		},
		{
			name: "blur-and-rank",
			filters: []regFilter{
				{"gaussian-blur/0.8", GaussianBlur(0.8)},
				{"median/3", Median(3, false)},
			},
		},
		{
			name: "resize-rotate-color",
			filters: []regFilter{
				{"resize/linear", Resize(4, 6, LinearResampling)},
				{"rotate/23-linear", Rotate(23, color.NRGBA{A: 255}, LinearInterpolation)},
				{"grayscale", Grayscale()},
				{"invert", Invert()},
			},
		},
		{
			name: "crop-color",
			filters: []regFilter{
				{"crop", Crop(image.Rect(-1, -1, 3, 7))},
				{"hue/35", Hue(35)},
			},
		},
	}
}

func TestParallelEmptyImages(t *testing.T) {
	defer regForceMultiCore(t)()
	seeds := regSeeds(t)

	emptyBounds := []struct {
		name   string
		bounds image.Rectangle
	}{
		{"zero-origin-empty", image.Rect(0, 0, 0, 0)},
		{"nonzero-origin-empty", image.Rect(5, 7, 5, 7)},
		{"degenerate-width", image.Rect(-2, 3, -2, 9)},
		{"degenerate-height", image.Rect(-2, 3, 6, 3)},
	}
	formats := []regFormat{regRGBA, regNRGBA, regGray16}

	for _, format := range formats {
		format := format
		for _, eb := range emptyBounds {
			eb := eb
			v := regVariant{
				name:   fmt.Sprintf("%s/%s", format, eb.name),
				format: format,
				bounds: eb.bounds,
			}
			t.Run(v.name, func(t *testing.T) {
				for _, ch := range regSmallChains() {
					ch := ch
					for _, seed := range seeds {
						for prefix := 0; prefix <= len(ch.filters); prefix++ {
							regCheckPrefix(t, seed, v, ch, prefix, true)
						}
					}
				}
			})
		}
	}
}

func TestParallelNarrowImages(t *testing.T) {
	defer regForceMultiCore(t)()
	seeds := regSeeds(t)

	narrow := []struct {
		name   string
		bounds image.Rectangle
	}{
		{"one-column", image.Rect(0, 0, 1, 9)},
		{"one-row", image.Rect(0, 0, 9, 1)},
		{"single-pixel", image.Rect(-2, 3, -1, 4)},
		{"two-columns", image.Rect(0, 0, 2, 9)},
		{"two-rows", image.Rect(0, 0, 9, 2)},
		{"nonzero-one-column", image.Rect(-4, 6, -3, 15)},
		{"nonzero-one-row", image.Rect(-4, 6, 5, 7)},
	}
	formats := []regFormat{regRGBA, regNRGBA, regGray16}

	for _, format := range formats {
		format := format
		for _, n := range narrow {
			n := n
			v := regVariant{
				name:   fmt.Sprintf("%s/%s", format, n.name),
				format: format,
				bounds: n.bounds,
			}
			t.Run(v.name, func(t *testing.T) {
				for _, ch := range regSmallChains() {
					ch := ch
					for _, seed := range seeds {
						for prefix := 0; prefix <= len(ch.filters); prefix++ {
							regCheckPrefix(t, seed, v, ch, prefix, true)
						}
					}
				}
			})
		}
	}
}

// -----------------------------------------------------------------------------
// DrawAt / compositing (the "paste" path)
// -----------------------------------------------------------------------------

func TestParallelDrawAtCompositing(t *testing.T) {
	defer regForceMultiCore(t)()
	seeds := regSeeds(t)

	type atCase struct {
		name     string
		pt       image.Point
		op       Operator
		dstb     image.Rectangle
		filters  []Filter
		chainLog string
	}
	chain := regChains()[1] // scale-rotate-conv-crop-color
	fs := make([]Filter, len(chain.filters))
	for i := range chain.filters {
		fs[i] = chain.filters[i].f
	}
	fullChain := append([]Filter(nil), fs...)

	cases := []atCase{
		{"copy-inside", image.Pt(2, 3), CopyOperator, image.Rect(-7, -6, 34, 28), fullChain, chain.name},
		{"copy-negative", image.Pt(-4, 5), CopyOperator, image.Rect(-7, -6, 34, 28), fullChain, chain.name},
		{"copy-at-dst-min", image.Pt(-7, -6), CopyOperator, image.Rect(-7, -6, 34, 28), fullChain, chain.name},
		{"over-inside", image.Pt(1, 2), OverOperator, image.Rect(-7, -6, 34, 28), fullChain, chain.name},
		{"over-negative", image.Pt(-3, -2), OverOperator, image.Rect(-7, -6, 34, 28), fullChain, chain.name},
		{"copy-empty-chain", image.Pt(3, -5), CopyOperator, image.Rect(-7, -6, 34, 28), nil, "copy/empty-chain"},
		{"over-empty-chain", image.Pt(3, -5), OverOperator, image.Rect(-7, -6, 34, 28), nil, "copy/empty-chain"},
	}

	for _, v := range regVariants() {
		v := v
		t.Run(v.name, func(t *testing.T) {
			for _, c := range cases {
				c := c
				for _, seed := range seeds {
					serial := regRun(v, seed, c.filters, false, c.dstb, c.pt, c.op, true)
					par := regRun(v, seed, c.filters, true, c.dstb, c.pt, c.op, true)
					if d := regCompareSource(serial, par, v); d != nil {
						t.Errorf("seed=%d variant=%q srcBounds=%v drawAt=%q chain=%q: %s",
							seed, v.name, v.bounds, c.name, c.chainLog, d)
						continue
					}
					if d := regCompare(serial, par, v.format); d != nil {
						t.Errorf("seed=%d variant=%q srcBounds=%v drawAt=%q pt=%v op=%v chain=%q: %s",
							seed, v.name, v.bounds, c.name, c.pt, c.op, c.chainLog, d)
					}
				}
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Injected chunk / boundary errors
// -----------------------------------------------------------------------------

type regFaultMode int

const (
	regFaultNone regFaultMode = iota
	regFaultDropChunkEdgeRow
	regFaultChunkLocalHalo
)

func (m regFaultMode) String() string {
	switch m {
	case regFaultNone:
		return "none"
	case regFaultDropChunkEdgeRow:
		return "drop-last-row-of-each-chunk"
	case regFaultChunkLocalHalo:
		return "chunk-local-neighbour-clamp"
	}
	return "unknown"
}

// regBoundaryFaultFilter is a row-neighbour blend (the shape of a separable
// convolution).  Its control mode is correct; the other two modes inject
// realistic chunk-boundary bugs:
//
//   - the chunk end is treated as exclusive, dropping the last row of every
//     chunk, and
//   - neighbour lookups are clamped at the chunk edge instead of the image
//     edge, so rows adjacent to an internal chunk boundary blend the wrong
//     row ("错行" at the block boundary).
//
// Both faults write to disjoint destination rows, so the test is race-free and
// can itself be run under -race.  A fault is detected because the serial
// render (one chunk) is correct while the parallel render is not.
type regBoundaryFaultFilter struct {
	fault regFaultMode
}

func (p *regBoundaryFaultFilter) Bounds(srcBounds image.Rectangle) image.Rectangle {
	return image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
}

func (p *regBoundaryFaultFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	srcb := src.Bounds()
	dstb := dst.Bounds()
	getter := newPixelGetter(src)
	setter := newPixelSetter(dst)

	parallel := true
	if options != nil {
		parallel = options.Parallelization
	}

	// blendRow writes dst row y using rows y and y+1 (clamped to the image).
	blendRow := func(y int) {
		ny := y + 1
		if ny > srcb.Max.Y-1 {
			ny = srcb.Max.Y - 1
		}
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			px0 := getter.getPixel(x, y)
			px1 := getter.getPixel(x, ny)
			setter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, pixel{
				R: (px0.R + px1.R) / 2,
				G: (px0.G + px1.G) / 2,
				B: (px0.B + px1.B) / 2,
				A: (px0.A + px1.A) / 2,
			})
		}
	}

	if !parallel {
		for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
			blendRow(y)
		}
		return
	}

	// Use the production chunking helper so the injected faults affect exactly
	// the same chunk ranges as the real filters.
	parallelize(true, srcb.Min.Y, srcb.Max.Y, func(pmin, pmax int) {
		switch p.fault {
		case regFaultNone:
			for y := pmin; y < pmax; y++ {
				blendRow(y)
			}
		case regFaultDropChunkEdgeRow:
			// BUG: the chunk end is treated as exclusive when it is not.
			for y := pmin; y < pmax-1; y++ {
				blendRow(y)
			}
		case regFaultChunkLocalHalo:
			// BUG: the neighbour is clamped to the chunk end instead of the
			// image end, so the last row of every chunk reads itself.
			for y := pmin; y < pmax; y++ {
				ny := y + 1
				if ny > pmax-1 {
					ny = pmax - 1
				}
				for x := srcb.Min.X; x < srcb.Max.X; x++ {
					px0 := getter.getPixel(x, y)
					px1 := getter.getPixel(x, ny)
					setter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, pixel{
						R: (px0.R + px1.R) / 2,
						G: (px0.G + px1.G) / 2,
						B: (px0.B + px1.B) / 2,
						A: (px0.A + px1.A) / 2,
					})
				}
			}
		}
	})
}

// regChunkSize mirrors the chunk-size calculation performed by parallelize.
func regChunkSize(datasize int) int {
	partsize := datasize
	numProcs := runtime.GOMAXPROCS(0)
	if numProcs > 1 {
		partsize = datasize / (numProcs * 10)
		if partsize < 1 {
			partsize = 1
		}
	}
	return partsize
}

func TestParallelInjectedChunkBugs(t *testing.T) {
	defer regForceMultiCore(t)()
	seeds := regSeeds(t)

	// Tall enough that a chunk spans several rows, so both faults land on an
	// internal chunk boundary rather than degrading to one-row chunks.
	v := regVariant{
		name:   "RGBA/direct-zero-origin",
		format: regRGBA,
		bounds: image.Rect(0, 0, 19, 200),
	}
	chunk := regChunkSize(v.bounds.Dy())
	if chunk < 2 {
		t.Fatalf("effective chunk size is %d; injected-boundary test needs multi-row chunks", chunk)
	}
	// The last row of the first chunk.  Both faults are expected to become
	// visible exactly there; earlier rows are computed normally.
	boundaryRow := v.bounds.Min.Y + chunk - 1
	t.Logf("GOMAXPROCS=%d chunk=%d rows, first internal boundary row=%d", runtime.GOMAXPROCS(0), chunk, boundaryRow)

	tests := []struct {
		name      string
		fault     regFaultMode
		expectBad bool
	}{
		{"control/correct-chunking", regFaultNone, false},
		{"mutation/drop-last-row-of-each-chunk", regFaultDropChunkEdgeRow, true},
		{"mutation/chunk-local-neighbour-clamp", regFaultChunkLocalHalo, true},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			for _, seed := range seeds {
				filters := []Filter{&regBoundaryFaultFilter{fault: tc.fault}}

				serial := regRun(v, seed, filters, false, image.Rectangle{}, image.Point{}, CopyOperator, false)
				par := regRun(v, seed, filters, true, image.Rectangle{}, image.Point{}, CopyOperator, false)

				d := regCompare(serial, par, v.format)
				if !tc.expectBad {
					if d != nil {
						t.Errorf("seed=%d control chunking unexpectedly differs: %s", seed, d)
					}
					continue
				}
				if d == nil {
					t.Fatalf("seed=%d injected fault %q escaped detection: parallel output matched serial", seed, tc.fault)
				}
				if d.kind != "pix" {
					t.Errorf("seed=%d injected fault %q was detected as %q, want pixel mismatch: %s", seed, tc.fault, d.kind, d)
					continue
				}
				// The comparator must localize the problem to the chunk boundary.
				if d.y != boundaryRow {
					t.Errorf("seed=%d injected fault %q: first difference at y=%d, want the first chunk boundary row %d: %s",
						seed, tc.fault, d.y, boundaryRow, d)
				}
				if d.x != v.bounds.Min.X {
					t.Errorf("seed=%d injected fault %q: first difference at x=%d, want x=%d: %s",
						seed, tc.fault, d.x, v.bounds.Min.X, d)
				}
				if d.x < 0 || d.y < 0 || !image.Pt(d.x, d.y).In(serial.bounds) {
					t.Errorf("seed=%d injected fault %q: comparator reported an out-of-bounds coordinate: %s", seed, tc.fault, d)
				}
				t.Logf("caught injected fault %q seed=%d at coord=(%d,%d) channel=%d: serial=%#02x parallel=%#02x",
					tc.fault, seed, d.x, d.y, d.channel, d.want, d.got)
			}
		})
	}
}
