package gift

import (
	"context"
	"image"
	"image/draw"
)

// ProgressFunc is the type of the function called by DrawContext and
// DrawAtContext to report the filter chain progress.
//
// completed is the number of the filter stages that have already been applied,
// and total is the total number of the filter stages. For an empty filter
// chain the callback is invoked once with completed and total both set to zero.
// The completed-to-total ratio represents the already finished fraction of the
// work.
//
// If the callback returns a non-nil error, the processing is aborted, that
// error is returned to the caller and the destination image is left untouched.
type ProgressFunc func(completed, total int) error

// DrawContext applies all the added filters to the src image and outputs the
// result to the dst image. It is the cancellable counterpart of Draw:
//
//   - The ctx cancellation is observed before the processing starts and at
//     every filter stage boundary. When the context is canceled, the function
//     returns the context error (for example context.Canceled) and the dst
//     image keeps its original content; no partially processed pixels are ever
//     written into it.
//   - The progress callback, when not nil, is invoked after every completed
//     filter stage. A non-nil error returned by the callback aborts the
//     processing in the same way as a context cancellation.
//
// On success the produced image is identical to the one produced by Draw.
func (g *GIFT) DrawContext(ctx context.Context, dst draw.Image, src image.Image, progress ProgressFunc) error {
	finalRect := g.Bounds(src.Bounds())
	// Anchor the result at the destination bounds origin, the same way Draw
	// maps the source bounds origin onto dst.Bounds().Min.
	finalRect = finalRect.Sub(finalRect.Min).Add(dst.Bounds().Min)

	tmp, err := g.renderToTemp(ctx, src, finalRect, progress)
	if err != nil {
		return err
	}
	// The context may have been canceled while the last stage was running.
	// Commit only while the work is still wanted, so the destination is never
	// overwritten after a cancellation.
	if err := ctx.Err(); err != nil {
		return err
	}

	blitCopy(dst, tmp, finalRect, g.Options.Parallelization)
	return nil
}

// DrawAtContext applies all the added filters to the src image and outputs the
// result to the dst image at the specified position pt using the specified
// composition operator op. It is the cancellable counterpart of DrawAt.
//
// Cancellation and progress reporting work exactly as in DrawContext. On
// cancellation or a callback error the dst image is never modified, including
// when pt is outside of the destination bounds or only partially intersects
// them.
func (g *GIFT) DrawAtContext(ctx context.Context, dst draw.Image, src image.Image, pt image.Point, op Operator, progress ProgressFunc) error {
	finalRect := g.Bounds(src.Bounds())
	finalRect = finalRect.Sub(finalRect.Min).Add(pt)

	tmp, err := g.renderToTemp(ctx, src, finalRect, progress)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	switch op {
	case OverOperator:
		blitOver(dst, tmp, finalRect, g.Options.Parallelization)
	default:
		blitCopy(dst, tmp, finalRect, g.Options.Parallelization)
	}
	return nil
}

// renderToTemp runs the whole filter chain into a temporary image and never
// touches the caller's destination image, so an aborted run cannot leak
// partially processed pixels. The returned image has the finalRect bounds.
//
// The context is checked before the first stage and at every stage boundary.
// The progress callback is invoked after each completed stage, and also once
// with (0, 0) for an empty filter chain.
func (g *GIFT) renderToTemp(ctx context.Context, src image.Image, finalRect image.Rectangle, progress ProgressFunc) (draw.Image, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	total := len(g.Filters)
	result := createTempImage(finalRect)

	if total == 0 {
		copyimage(result, src, &g.Options)
		if err := reportProgress(progress, 0, 0); err != nil {
			return nil, err
		}
		return result, nil
	}

	var tmpIn image.Image
	var tmpOut draw.Image

	for i, f := range g.Filters {
		// Stage boundary: respond to cancellation before any stage runs.
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if i == 0 {
			tmpIn = src
		} else {
			tmpIn = tmpOut
		}

		if i == total-1 {
			tmpOut = result
		} else {
			tmpOut = createTempImage(f.Bounds(tmpIn.Bounds()))
		}

		f.Draw(tmpOut, tmpIn, &g.Options)

		if err := reportProgress(progress, i+1, total); err != nil {
			return nil, err
		}
	}

	return result, nil
}

func reportProgress(progress ProgressFunc, completed, total int) error {
	if progress == nil {
		return nil
	}
	return progress(completed, total)
}

// blitCopy copies the pixels of src located inside rect into dst, clipping the
// copied area to the destination bounds.
func blitCopy(dst draw.Image, src image.Image, rect image.Rectangle, parallel bool) {
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)
	ib := rect.Intersect(dst.Bounds())
	parallelize(parallel, ib.Min.Y, ib.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := ib.Min.X; x < ib.Max.X; x++ {
				pixSetter.setPixel(x, y, pixGetter.getPixel(x, y))
			}
		}
	})
}

// blitOver alpha-composites the pixels of src located inside rect over dst,
// clipping the composited area to the destination bounds.
func blitOver(dst draw.Image, src image.Image, rect image.Rectangle, parallel bool) {
	pixGetterDst := newPixelGetter(dst)
	pixGetterSrc := newPixelGetter(src)
	pixSetterDst := newPixelSetter(dst)
	ib := rect.Intersect(dst.Bounds())
	parallelize(parallel, ib.Min.Y, ib.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := ib.Min.X; x < ib.Max.X; x++ {
				px0 := pixGetterDst.getPixel(x, y)
				px1 := pixGetterSrc.getPixel(x, y)
				c1 := px1.a
				c0 := (1 - c1) * px0.a
				cs := c0 + c1
				c0 /= cs
				c1 /= cs
				r := px0.r*c0 + px1.r*c1
				g := px0.g*c0 + px1.g*c1
				b := px0.b*c0 + px1.b*c1
				a := px0.a + px1.a*(1-px0.a)
				pixSetterDst.setPixel(x, y, pixel{r, g, b, a})
			}
		}
	})
}
