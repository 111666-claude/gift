package gift

import (
	"context"
	"errors"
	"image"
	"image/draw"
	"reflect"
	"testing"
)

// blockingFilter is a slow, fully controllable filter. Its Draw method signals
// the stage index on started and blocks until it receives a value on release,
// which makes the cancellation points deterministic.
type blockingFilter struct {
	index   int
	started chan<- int
	release <-chan struct{}
}

func (p *blockingFilter) Bounds(srcBounds image.Rectangle) image.Rectangle {
	return image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
}

func (p *blockingFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	p.started <- p.index
	<-p.release
	copyimage(dst, src, options)
}

// snapNRGBA returns a deep copy of the image pixels so later mutations of the
// image can be detected.
func snapNRGBA(img *image.NRGBA) []uint8 {
	pix := make([]uint8, len(img.Pix))
	copy(pix, img.Pix)
	return pix
}

// pixEqual reports whether two pixel buffers are identical.
func pixEqual(a, b []uint8) bool {
	return reflect.DeepEqual(a, b)
}

// pixClose reports whether every byte of a and b differs by at most tolerance.
// The context-enabled entry point renders through NRGBA64 temporary images, so
// for 8-bit destination formats a single LSB rounding difference is allowed.
func pixClose(a, b []uint8, tolerance int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
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

func newBlockingChain(n int) (filters []Filter, started <-chan int, releaseChans []chan struct{}) {
	start := make(chan int)
	for i := 0; i < n; i++ {
		rel := make(chan struct{}, 1)
		releaseChans = append(releaseChans, rel)
		filters = append(filters, &blockingFilter{
			index:   i,
			started: start,
			release: rel,
		})
	}
	return filters, start, releaseChans
}

// waitStarted blocks until the given stage starts; it fails the test if the
// draw call returns first instead.
func waitStarted(t *testing.T, started <-chan int, result <-chan error, stage int, desc string) {
	t.Helper()
	select {
	case idx := <-started:
		if idx != stage {
			t.Fatalf("%s: expected stage %d to start, got %d", desc, stage, idx)
		}
	case err := <-result:
		t.Fatalf("%s: draw returned before stage %d started: %v", desc, stage, err)
	}
}

// TestDrawContextCancelAtStages verifies the cancellation response at the start
// and at every stage boundary of a multi-stage chain, including the guarantee
// that the destination image is never modified.
func TestDrawContextCancelAtStages(t *testing.T) {
	const stages = 3

	// cancelAt == 0: canceled before the first stage runs.
	// cancelAt == k (k > 0): stages 0..k-1 finish, then the context is
	// canceled and the boundary check before stage k must abort the run.
	for _, cancelAt := range []int{0, 1, 2} {
		desc := "cancelAt=" + string(rune('0'+cancelAt))
		filters, started, releases := newBlockingChain(stages)
		g := New(filters...)

		src := image.NewNRGBA(image.Rect(2, 3, 6, 7))
		for i := range src.Pix {
			src.Pix[i] = uint8(i + 1)
		}
		// Nonzero destination bounds.
		dst := image.NewNRGBA(image.Rect(5, 7, 9, 11))
		before := snapNRGBA(dst)

		ctx, cancel := context.WithCancel(context.Background())
		result := make(chan error, 1)
		go func() {
			result <- g.DrawContext(ctx, dst, src, nil)
		}()

		for stage := 0; stage < cancelAt; stage++ {
			waitStarted(t, started, result, stage, desc)
			releases[stage] <- struct{}{}
		}
		cancel()

		err := <-result
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("%s: expected context.Canceled, got %v", desc, err)
		}

		// No further stage may start after the cancellation.
		select {
		case idx := <-started:
			t.Fatalf("%s: stage %d started after cancellation", desc, idx)
		default:
		}
		if !pixEqual(dst.Pix, before) {
			t.Fatalf("%s: destination image was modified after cancellation", desc)
		}

		for _, rel := range releases {
			close(rel)
		}
	}
}

// TestDrawContextCanceledBeforeStart makes sure an already canceled context is
// rejected without running any filter stage or touching the destination.
func TestDrawContextCanceledBeforeStart(t *testing.T) {
	filters, started, releases := newBlockingChain(2)
	g := New(filters...)

	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	dst := image.NewNRGBA(image.Rect(1, 1, 5, 5))
	before := snapNRGBA(dst)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := g.DrawContext(ctx, dst, src, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if !pixEqual(dst.Pix, before) {
		t.Fatal("destination image was modified")
	}

	select {
	case <-started:
		t.Fatal("a filter stage ran despite the pre-canceled context")
	default:
	}
	for _, rel := range releases {
		close(rel)
	}
}

// TestDrawContextProgress verifies the reported progress for a multi-stage
// chain and for an empty chain.
func TestDrawContextProgress(t *testing.T) {
	filters, started, releases := newBlockingChain(3)
	g := New(filters...)

	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	dst := image.NewNRGBA(image.Rect(0, 0, 4, 4))

	var reports [][2]int
	result := make(chan error, 1)
	go func() {
		result <- g.DrawContext(context.Background(), dst, src, func(completed, total int) error {
			reports = append(reports, [2]int{completed, total})
			return nil
		})
	}()

	for stage := 0; stage < 3; stage++ {
		waitStarted(t, started, result, stage, "progress")
		releases[stage] <- struct{}{}
	}
	if err := <-result; err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := [][2]int{{1, 3}, {2, 3}, {3, 3}}
	if !reflect.DeepEqual(reports, want) {
		t.Fatalf("unexpected progress reports: got %v, want %v", reports, want)
	}

	// Empty chain: a single (0, 0) report.
	empty := New()
	emptyDst := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	var emptyReports [][2]int
	if err := empty.DrawContext(context.Background(), emptyDst, src, func(completed, total int) error {
		emptyReports = append(emptyReports, [2]int{completed, total})
		return nil
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(emptyReports, [][2]int{{0, 0}}) {
		t.Fatalf("unexpected empty-chain progress reports: %v", emptyReports)
	}
}

// errProgress is a sentinel error thrown by a progress callback.
var errProgress = errors.New("progress callback failure")

// TestDrawContextProgressError verifies that an error thrown by the progress
// callback aborts the processing, is returned unchanged and does not leak any
// temporary result into the destination.
func TestDrawContextProgressError(t *testing.T) {
	filters, started, releases := newBlockingChain(3)
	g := New(filters...)

	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	dst := image.NewNRGBA(image.Rect(2, 2, 6, 6))
	before := snapNRGBA(dst)

	result := make(chan error, 1)
	go func() {
		result <- g.DrawContext(context.Background(), dst, src, func(completed, total int) error {
			if completed == 2 {
				return errProgress
			}
			return nil
		})
	}()

	for stage := 0; stage < 2; stage++ {
		waitStarted(t, started, result, stage, "progress-error")
		releases[stage] <- struct{}{}
	}

	err := <-result
	if !errors.Is(err, errProgress) {
		t.Fatalf("expected errProgress, got %v", err)
	}
	if !pixEqual(dst.Pix, before) {
		t.Fatal("destination image was modified by the failed callback")
	}

	// The remaining stage must never run.
	select {
	case idx := <-started:
		t.Fatalf("stage %d ran after the callback error", idx)
	default:
	}
	for _, rel := range releases {
		close(rel)
	}
}

// TestDrawContextCompletion verifies the successful run, including the final
// progress report and the produced pixels against a deterministic filter.
func TestDrawContextCompletion(t *testing.T) {
	src := image.NewNRGBA(image.Rect(1, 2, 5, 6))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*7 + 3) % 256)
	}

	g := New(FlipHorizontal())
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	dst.Rect = image.Rect(8, 9, 12, 13)

	var last [2]int
	err := g.DrawContext(context.Background(), dst, src, func(completed, total int) error {
		last = [2]int{completed, total}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if last != [2]int{1, 1} {
		t.Fatalf("unexpected final progress: %v", last)
	}

	want := image.NewNRGBA(g.Bounds(src.Bounds()))
	want.Rect = dst.Rect
	g.Draw(want, src)
	if !pixEqual(dst.Pix, want.Pix) {
		t.Fatal("DrawContext result differs from Draw result")
	}
}

// TestDrawContextEquivalence compares the cancellable entry points against the
// original Draw and DrawAt over several chains, offsets, operators and
// parallelization settings.
func TestDrawContextEquivalence(t *testing.T) {
	chains := [][]Filter{
		{},
		{FlipHorizontal()},
		{Brightness(10), Invert(), Grayscale()},
		{Resize(8, 8, NearestNeighborResampling), FlipVertical()},
		{Crop(image.Rect(1, 1, 3, 3))},
	}
	pts := []image.Point{
		image.Pt(0, 0),
		image.Pt(2, -1),
		image.Pt(-3, 2),
		image.Pt(100, 100), // fully outside the destination
	}
	ops := []Operator{CopyOperator, OverOperator}

	src := image.NewNRGBA(image.Rect(2, 3, 8, 9))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*13 + 5) % 256)
	}

	for ci, filters := range chains {
		for _, parallel := range []bool{true, false} {
			g := New(filters...)
			g.SetParallelization(parallel)

			// Draw vs DrawContext.
			dst0 := image.NewNRGBA(image.Rect(5, 7, 15, 17))
			dst1 := image.NewNRGBA(dst0.Rect)
			g.Draw(dst0, src)
			if err := g.DrawContext(context.Background(), dst1, src, nil); err != nil {
				t.Fatalf("chain %d parallel=%v: unexpected error %v", ci, parallel, err)
			}
			if !pixClose(dst0.Pix, dst1.Pix, 1) {
				t.Fatalf("chain %d parallel=%v: DrawContext output differs from Draw", ci, parallel)
			}

			for _, pt := range pts {
				for _, op := range ops {
					d0 := image.NewNRGBA(image.Rect(-4, -4, 10, 10))
					d1 := image.NewNRGBA(d0.Rect)
					g.DrawAt(d0, src, pt, op)
					if err := g.DrawAtContext(context.Background(), d1, src, pt, op, nil); err != nil {
						t.Fatalf("chain %d pt %v op %d: unexpected error %v", ci, pt, op, err)
					}
					if !pixClose(d0.Pix, d1.Pix, 1) {
						t.Fatalf("chain %d pt %v op %d parallel=%v: DrawAtContext output differs from DrawAt", ci, pt, op, parallel)
					}
				}
			}
		}
	}
}

// TestDrawContextAtCancelKeepsOffsetTarget checks the offset DrawAt path: a
// cancellation while the chain is half done must leave the offset target image
// byte-identical to its pre-call state, for both composition operators.
func TestDrawContextAtCancelKeepsOffsetTarget(t *testing.T) {
	filters, started, releases := newBlockingChain(3)
	g := New(filters...)

	src := image.NewNRGBA(image.Rect(0, 0, 6, 6))
	for i := range src.Pix {
		src.Pix[i] = uint8(i % 256)
	}

	for _, op := range []Operator{CopyOperator, OverOperator} {
		dst := image.NewNRGBA(image.Rect(-2, -2, 8, 8))
		for i := range dst.Pix {
			dst.Pix[i] = uint8((i + 11) % 256)
		}
		before := snapNRGBA(dst)

		ctx, cancel := context.WithCancel(context.Background())
		result := make(chan error, 1)
		go func() {
			result <- g.DrawAtContext(ctx, dst, src, image.Pt(-1, 2), op, nil)
		}()

		// Finish two of the three stages, then cancel at the next boundary.
		for stage := 0; stage < 2; stage++ {
			waitStarted(t, started, result, stage, "drawat-cancel")
			releases[stage] <- struct{}{}
		}
		cancel()

		if err := <-result; !errors.Is(err, context.Canceled) {
			t.Fatalf("op %d: expected context.Canceled, got %v", op, err)
		}
		select {
		case idx := <-started:
			t.Fatalf("op %d: stage %d started after cancellation", op, idx)
		default:
		}
		if !pixEqual(dst.Pix, before) {
			t.Fatalf("op %d: offset target was modified after cancellation", op)
		}
	}
	for _, rel := range releases {
		close(rel)
	}
}

// TestDrawContextEmptyImageAndChain covers empty chains, empty images and
// chains that land completely outside the destination: all must return a
// stable result without errors.
func TestDrawContextEmptyImageAndChain(t *testing.T) {
	// Empty source image.
	emptySrc := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	g := New(FlipHorizontal())
	dst := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	if err := g.DrawContext(context.Background(), dst, emptySrc, nil); err != nil {
		t.Fatalf("empty image: unexpected error %v", err)
	}

	// Empty filter chain on a normal image with nonzero bounds.
	src := image.NewNRGBA(image.Rect(3, 4, 7, 8))
	for i := range src.Pix {
		src.Pix[i] = uint8(i + 1)
	}
	emptyG := New()
	dst2 := image.NewNRGBA(emptyG.Bounds(src.Bounds()))
	dst2.Rect = image.Rect(9, 10, 13, 14)
	if err := emptyG.DrawContext(context.Background(), dst2, src, nil); err != nil {
		t.Fatalf("empty chain: unexpected error %v", err)
	}
	want := image.NewNRGBA(emptyG.Bounds(src.Bounds()))
	want.Rect = dst2.Rect
	emptyG.Draw(want, src)
	if !pixClose(dst2.Pix, want.Pix, 1) {
		t.Fatal("empty chain: DrawContext output differs from Draw")
	}

	// Chain whose result is fully outside the destination bounds.
	outG := New(Crop(image.Rect(0, 0, 4, 4)))
	dst3 := image.NewNRGBA(image.Rect(0, 0, 5, 5))
	before := snapNRGBA(dst3)
	if err := outG.DrawAtContext(
		context.Background(), dst3, src, image.Pt(50, 50), CopyOperator, nil,
	); err != nil {
		t.Fatalf("out-of-bounds draw: unexpected error %v", err)
	}
	if !pixEqual(dst3.Pix, before) {
		t.Fatal("out-of-bounds draw modified the destination")
	}
}

// TestDrawContextNilProgress ensures a nil callback is accepted.
func TestDrawContextNilProgress(t *testing.T) {
	g := New(Invert())
	src := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	if err := g.DrawContext(context.Background(), dst, src, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := g.DrawAtContext(context.Background(), dst, src, image.Pt(1, 1), OverOperator, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Compile-time assertions that the original entry points keep their
// signatures and the new ones expose the intended ones.
var (
	_ func(draw.Image, image.Image)                                                             = (*GIFT)(nil).Draw
	_ func(draw.Image, image.Image, image.Point, Operator)                                      = (*GIFT)(nil).DrawAt
	_ func(context.Context, draw.Image, image.Image, ProgressFunc) error                        = (*GIFT)(nil).DrawContext
	_ func(context.Context, draw.Image, image.Image, image.Point, Operator, ProgressFunc) error = (*GIFT)(nil).DrawAtContext
)
