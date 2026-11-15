package gift

import (
	"image"
	"image/draw"
)

// maxMorphologyRadius is the upper bound for the Dilate/Erode radius.
// Larger radii are clamped to this value.
const maxMorphologyRadius = 4096

type morphologyMode int

const (
	morphDilate morphologyMode = iota
	morphErode
)

type morphologyFilter struct {
	radius int
	mode   morphologyMode
}

func (p *morphologyFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *morphologyFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	w, h := srcb.Dx(), srcb.Dy()
	if w <= 0 || h <= 0 {
		return
	}

	if p.radius <= 0 {
		copyimage(dst, src, options)
		return
	}
	r := minint(p.radius, maxMorphologyRadius)

	maximize := p.mode == morphDilate

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// Extract the alpha channel.
	alpha := make([]float32, w*h)
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := 0; x < w; x++ {
				alpha[y*w+x] = pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y).a
			}
		}
	})

	// A square (2r+1)x(2r+1) neighborhood extremum is separable:
	// it equals a horizontal (2r+1)-wide pass followed by a vertical one.
	// Pixels outside the image bounds are treated as fully transparent (alpha 0),
	// so dilation never grows past the edges and erosion eats into them.
	// Both passes run in place: each row/column is copied to a padded
	// scratch buffer before its sliding window extrema are written back.
	ksize := 2*r + 1

	// Horizontal pass.
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		row := make([]float32, w+2*r) // zero-padded on both sides
		out := make([]float32, w)
		for y := start; y < stop; y++ {
			copy(row[r:r+w], alpha[y*w:(y+1)*w])
			windowExtrema(out, row, ksize, maximize)
			copy(alpha[y*w:(y+1)*w], out)
		}
	})

	// Vertical pass.
	parallelize(options.Parallelization, 0, w, func(start, stop int) {
		col := make([]float32, h+2*r) // zero-padded on both sides
		out := make([]float32, h)
		for x := start; x < stop; x++ {
			for y := 0; y < h; y++ {
				col[r+y] = alpha[y*w+x]
			}
			windowExtrema(out, col, ksize, maximize)
			for y := 0; y < h; y++ {
				alpha[y*w+x] = out[y]
			}
		}
	})

	// Write the result: new alpha, original RGB.
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				px.a = alpha[y*w+x]
				pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, px)
			}
		}
	})
}

// windowExtrema computes the sliding window minimum (or maximum) of arr.
// out must have length len(arr)-w+1. It runs in O(len(arr)) time using a
// monotonic deque held in a ring buffer, independently of the window size w.
func windowExtrema(out, arr []float32, w int, maximize bool) {
	dq := make([]int, w) // ring buffer of indices into arr, values monotonic
	head, n := 0, 0
	for i := 0; i < len(arr); i++ {
		if n > 0 && dq[head%w] <= i-w {
			head++
			n--
		}
		if maximize {
			for n > 0 && arr[dq[(head+n-1)%w]] <= arr[i] {
				n--
			}
		} else {
			for n > 0 && arr[dq[(head+n-1)%w]] >= arr[i] {
				n--
			}
		}
		dq[(head+n)%w] = i
		n++
		if i >= w-1 {
			out[i-w+1] = arr[dq[head%w]]
		}
	}
}

// Dilate creates a filter that expands the opaque regions of the image:
// the alpha channel of each pixel is replaced by the maximum alpha value
// found in the (2*radius+1) x (2*radius+1) square neighborhood centered
// at the pixel. Only the alpha channel is modified, the original RGB
// values are preserved. Pixels outside the image bounds are treated as
// fully transparent, so the result is clipped by the image edges.
//
// The radius parameter is the neighborhood radius in pixels.
// A radius of zero or less produces an unchanged copy of the image.
// Radii greater than 4096 are clamped to 4096.
func Dilate(radius int) Filter {
	if radius > maxMorphologyRadius {
		radius = maxMorphologyRadius
	}
	return &morphologyFilter{
		radius: radius,
		mode:   morphDilate,
	}
}

// Erode creates a filter that shrinks the opaque regions of the image:
// the alpha channel of each pixel is replaced by the minimum alpha value
// found in the (2*radius+1) x (2*radius+1) square neighborhood centered
// at the pixel. Only the alpha channel is modified, the original RGB
// values are preserved. Pixels outside the image bounds are treated as
// fully transparent, so opaque regions recede from the image edges too.
//
// The radius parameter is the neighborhood radius in pixels.
// A radius of zero or less produces an unchanged copy of the image.
// Radii greater than 4096 are clamped to 4096.
func Erode(radius int) Filter {
	if radius > maxMorphologyRadius {
		radius = maxMorphologyRadius
	}
	return &morphologyFilter{
		radius: radius,
		mode:   morphErode,
	}
}
