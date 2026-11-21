package gift

import (
	"errors"
	"image"
	"image/draw"
)

// FilterSupport is implemented by filters whose input support (the source
// rectangle that can affect any pixel of a given destination rectangle) can
// be computed analytically.
//
// Support returns the source rectangle, in the coordinate system of the
// filter's input image, needed to render the destination rectangle dstRect.
// dstRect is expressed in the coordinate system of the filter's output image.
// The returned rectangle may be larger than dstRect (for example around the
// borders of a convolution); it must never be smaller than the true support,
// otherwise region rendering produces visible seams.
//
// Only filters that preserve the image dimensions (filters whose Bounds
// method returns a rectangle of the same size as the input) are expected to
// implement this interface.
type FilterSupport interface {
	Support(dstRect image.Rectangle) (srcRect image.Rectangle)
}

// filterRegionSupport is an internal extension of FilterSupport for filters
// whose support depends on the original image origin (the pixelation block
// grid). The driver uses it when it is available and falls back to
// FilterSupport.Support otherwise.
type filterRegionSupport interface {
	supportRegion(dstRect image.Rectangle, origin image.Point) (srcRect image.Rectangle)
}

// supportOf returns the input support rectangle for f given the requested
// output rectangle dstRect and the original source image origin.
func supportOf(f Filter, dstRect image.Rectangle, origin image.Point) image.Rectangle {
	if rs, ok := f.(filterRegionSupport); ok {
		return rs.supportRegion(dstRect, origin)
	}
	return f.(FilterSupport).Support(dstRect)
}

// ErrRegionUnsupported is returned by DrawRegion when the filter chain
// contains a filter that cannot render an arbitrary sub-rectangle of its
// output without processing the whole input image. Filters that change the
// image dimensions (resize, rotation, crop, ...) or filters provided by the
// caller without a Support method fall into this category. DrawRegionOrFull
// can be used to transparently fall back to a full Draw in this case.
var ErrRegionUnsupported = errors.New("gift: region rendering is not supported by this filter chain")

// DrawRegion applies the filter chain to src and writes only the requested
// rectangular part of the result to dst, without ever allocating full-size
// intermediate images. The region parameter is a rectangle in the coordinate
// system of the full result image returned by g.Bounds(src.Bounds()).
//
// The region is rendered as a viewport: its top-left corner corresponds to
// the top-left corner (Min point) of dst, mirroring how Draw aligns the full
// result with dst. Only the pixels of dst that fit both the region cropped
// to the full result bounds and the dst bounds are modified; the rest of dst
// is left untouched. The produced pixels are bit-identical to the pixels of
// a full g.Draw call cropped to the same rectangle.
//
// If the intersection of region with the full result bounds is empty the
// function does nothing and returns nil.
//
// The function returns ErrRegionUnsupported if the chain contains a filter
// whose input support cannot be determined. It never silently renders
// seam-prone pixels: either the whole requested region is rendered exactly or
// an error is returned.
func (g *GIFT) DrawRegion(dst draw.Image, src image.Image, region image.Rectangle) error {
	srcb := src.Bounds()
	fullB := g.Bounds(srcb)

	// Clip the requested region to the full result bounds, then translate it
	// into the destination coordinate space by aligning its top-left corner
	// with dst's Min point (viewport semantics).
	win := region.Intersect(fullB)
	if win.Empty() {
		return nil
	}
	r := win.Sub(win.Min).Add(dst.Bounds().Min).Intersect(dst.Bounds())
	if r.Empty() {
		return nil
	}
	// Requested output window in source coordinates. When dst is smaller
	// than the window, shrink the window to the part that actually fits.
	win = win.Intersect(image.Rectangle{
		Min: win.Min,
		Max: win.Min.Add(r.Size()),
	})
	target := win.Sub(fullB.Min).Add(srcb.Min)

	if len(g.Filters) == 0 {
		// No filters: the result is the source image itself.
		srcRect := target.Intersect(srcb)
		if srcRect.Empty() {
			return nil
		}
		sub, _ := subImage(src, srcRect)
		copyWindow(dst, sub, sub.Bounds(), r.Min, &g.Options)
		return nil
	}

	// Validate the chain: every filter must expose its support and preserve
	// the image dimensions.
	for _, f := range g.Filters {
		if _, ok := f.(FilterSupport); !ok {
			return ErrRegionUnsupported
		}
		b := f.Bounds(srcb)
		if b.Dx() != srcb.Dx() || b.Dy() != srcb.Dy() {
			return ErrRegionUnsupported
		}
	}

	// Backward pass: starting from the requested destination rectangle,
	// compute the required source rectangle (and the per-stage rectangles).
	// All rectangles are kept in the source coordinate space; dimension-
	// preserving filters map coordinates one-to-one.
	n := len(g.Filters)
	stageRects := make([]image.Rectangle, n+1)
	stageRects[n] = target
	for i := n - 1; i >= 0; i-- {
		need := supportOf(g.Filters[i], stageRects[i+1], srcb.Min).Intersect(srcb)
		if need.Empty() {
			return nil
		}
		stageRects[i] = need
	}

	// Forward pass. Every intermediate image uses the source coordinate
	// space, so existing filters that assume aligned src/dst coordinates
	// work unchanged. Filters write their full output rectangle; the halo
	// pixels outside the next stage's requested rectangle are simply not
	// read (and discarded at the final stage).
	//
	// opts carries the original source origin so that coordinate-anchored
	// filters (pixelation) keep their grid aligned to the original image
	// rather than to the cropped sub-image.
	opts := g.Options
	opts.regionOrigin = &srcb.Min

	tmpIn, copied := subImage(src, stageRects[0])
	if copied {
		// Generic image type without SubImage: copy the needed pixels into
		// a temp image and use it as the first stage input.
		firstOut := createTempImage(stageRects[0])
		copyimage(firstOut, tmpIn, &opts)
		tmpIn = firstOut
	}

	for i, f := range g.Filters {
		tmpOut := createTempImage(stageRects[i])
		f.Draw(tmpOut, tmpIn, &opts)
		if i == n-1 {
			// Last stage: copy only the requested window (not the halo)
			// into dst.
			copyWindow(dst, tmpOut, stageRects[n], r.Min, &opts)
			break
		}
		// Crop the output to the rectangle needed by the next filter.
		tmpIn, _ = subImage(tmpOut, stageRects[i+1])
	}

	return nil
}

// DrawRegionOrFull works like DrawRegion but falls back to a full g.Draw
// when the filter chain does not support region rendering. The fallback
// allocates the full-size result image, so it never produces seams but also
// does not save memory.
//
// The fallback draws the complete result into a temporary image and copies
// only the requested rectangle into dst (aligned to dst's Min point, as in
// DrawRegion).
func (g *GIFT) DrawRegionOrFull(dst draw.Image, src image.Image, region image.Rectangle) error {
	fullB := g.Bounds(src.Bounds())
	win := region.Intersect(fullB)
	if win.Empty() {
		return nil
	}

	err := g.DrawRegion(dst, src, region)
	if !errors.Is(err, ErrRegionUnsupported) {
		return err
	}

	tmp := createTempImage(fullB)
	g.Draw(tmp, src)

	r := win.Sub(win.Min).Add(dst.Bounds().Min).Intersect(dst.Bounds())
	if r.Empty() {
		return nil
	}
	copyWindow(dst, tmp, win, r.Min, &g.Options)
	return nil
}

// subImage returns a view of img restricted to r. For the standard image
// types it returns a zero-copy sub-image (copied == false). For types that
// do not implement the SubImage method, it allocates an NRGBA64 image with
// the same rectangle and copies the pixels (copied == true).
func subImage(img image.Image, r image.Rectangle) (view image.Image, copied bool) {
	r = r.Intersect(img.Bounds())
	switch img := img.(type) {
	case *image.Gray:
		return img.SubImage(r), false
	case *image.Gray16:
		return img.SubImage(r), false
	case *image.RGBA:
		return img.SubImage(r), false
	case *image.RGBA64:
		return img.SubImage(r), false
	case *image.NRGBA:
		return img.SubImage(r), false
	case *image.NRGBA64:
		return img.SubImage(r), false
	case *image.Paletted:
		return img.SubImage(r), false
	default:
		out := image.NewNRGBA64(r)
		pixGetter := newPixelGetter(img)
		pixSetter := newPixelSetter(out)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				pixSetter.setPixel(x, y, pixGetter.getPixel(x, y))
			}
		}
		return out, true
	}
}

// copyWindow copies the srcRect window of src into dst, placing its top-left
// corner at dstMin. Only pixels inside dst's bounds are written.
func copyWindow(dst draw.Image, src image.Image, srcRect image.Rectangle, dstMin image.Point, options *Options) {
	srcb := src.Bounds()
	dstb := dst.Bounds()
	win := srcRect.Intersect(srcb)
	if win.Empty() {
		return
	}
	// Clip against the destination bounds as well.
	dstMax := dstMin.Add(win.Size())
	dstRect := image.Rectangle{Min: dstMin, Max: dstMax}.Intersect(dstb)
	if dstRect.Empty() {
		return
	}
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)
	parallelize(options.Parallelization, dstRect.Min.Y, dstRect.Max.Y, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			for x := dstRect.Min.X; x < dstRect.Max.X; x++ {
				sx := win.Min.X + x - dstRect.Min.X
				sy := win.Min.Y + y - dstRect.Min.Y
				pixSetter.setPixel(x, y, pixGetter.getPixel(sx, sy))
			}
		}
	})
}
