package gift

import (
	"image"
	"image/draw"
)

// pixelTransformer is implemented by filters that transform each pixel
// independently of its neighbors and preserve the size of the image.
// Such filters can be fused with adjacent pixel transformers and applied
// in a single pass without intermediate temporary images.
type pixelTransformer interface {
	Filter
	// pixelFunc returns a function that transforms a single pixel.
	// The it parameter is the image type of the filter input and
	// numPixels is the number of pixels in the input image.
	pixelFunc(it imageType, numPixels int) func(pixel) pixel
}

// quantizeNRGBA64 replicates the loss of precision caused by writing a pixel
// to an intermediate NRGBA64 image (see createTempImage) and reading it back.
// Applying it between the fused stages preserves the per-stage quantization
// semantics of the unfused filter chain.
func quantizeNRGBA64(px pixel) pixel {
	return pixel{
		r: float32(f32u16(px.r*0xffff)) * qf16,
		g: float32(f32u16(px.g*0xffff)) * qf16,
		b: float32(f32u16(px.b*0xffff)) * qf16,
		a: float32(f32u16(px.a*0xffff)) * qf16,
	}
}

// fusedPixelFilter applies a sequence of pixel transformers in a single pass.
type fusedPixelFilter struct {
	filters []pixelTransformer
}

func (p *fusedPixelFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *fusedPixelFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// Prepare the per-pixel transformation for every stage. The first filter
	// reads from src, while every subsequent filter would read from an
	// intermediate NRGBA64 image in the unfused chain, so its result is
	// quantized accordingly to keep the output identical.
	n := len(p.filters)
	numPixels := srcb.Dx() * srcb.Dy()
	stages := make([]func(pixel) pixel, n)
	for i, f := range p.filters {
		it := itNRGBA64
		if i == 0 {
			it = pixGetter.it
		}
		fn := f.pixelFunc(it, numPixels)
		if i < n-1 {
			stages[i] = func(px pixel) pixel {
				return quantizeNRGBA64(fn(px))
			}
		} else {
			stages[i] = fn
		}
	}

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				for _, stage := range stages {
					px = stage(px)
				}
				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// fusePixelFilters replaces every maximal run of two or more adjacent
// pixel transformers with a single fused filter. All other filters keep
// their order and behavior. The original slice is returned unchanged if
// there is nothing to fuse.
func fusePixelFilters(filters []Filter) []Filter {
	// Check if there is at least one run of two or more adjacent
	// pixel transformers.
	fusable := false
	for i := 1; i < len(filters); i++ {
		if _, ok := filters[i].(pixelTransformer); ok {
			if _, ok := filters[i-1].(pixelTransformer); ok {
				fusable = true
				break
			}
		}
	}
	if !fusable {
		return filters
	}

	out := make([]Filter, 0, len(filters))
	var run []pixelTransformer

	flush := func() {
		switch len(run) {
		case 0:
		case 1:
			out = append(out, run[0])
		default:
			filters := make([]pixelTransformer, len(run))
			copy(filters, run)
			out = append(out, &fusedPixelFilter{filters: filters})
		}
		run = run[:0]
	}

	for _, f := range filters {
		if pt, ok := f.(pixelTransformer); ok {
			run = append(run, pt)
		} else {
			flush()
			out = append(out, f)
		}
	}
	flush()

	return out
}
