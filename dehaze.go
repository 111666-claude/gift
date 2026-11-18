package gift

import (
	"image"
	"image/draw"
	"math"
)

// dehazeFilter implements an adjustable dehaze filter based on the dark
// channel prior (He et al.). It estimates the atmospheric (air) light and a
// per-pixel transmission map from local dark channels, then recovers the
// scene radiance. Alpha is never modified.
type dehazeFilter struct {
	radius   int
	strength float32
	tmin     float32
}

func (p *dehazeFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *dehazeFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	if srcb.Dx() <= 0 || srcb.Dy() <= 0 {
		return
	}

	// Zero strength or a non-positive radius means no dehazing.
	if p.strength <= 0 || p.radius <= 0 {
		copyimage(dst, src, options)
		return
	}

	w, h := srcb.Dx(), srcb.Dy()
	n := w * h

	// Clamp the radius to the image dimensions so that arbitrarily large
	// radii stay cheap and well defined.
	radius := p.radius
	if radius > w {
		radius = w
	}
	if radius > h {
		radius = h
	}

	pixGetter := newPixelGetter(src)

	// Per-pixel minimum of the RGB channels. Fully transparent pixels get 1
	// so they can neither darken the dark channel nor hijack the airlight.
	dark := make([]float32, n)
	visible := 0
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			px := pixGetter.getPixel(x, y)
			i := (y-srcb.Min.Y)*w + (x - srcb.Min.X)
			if px.a <= 0 {
				dark[i] = 1
				continue
			}
			visible++
			dark[i] = minf32(px.r, minf32(px.g, px.b))
		}
	}
	if visible == 0 {
		// Nothing but transparent pixels: keep the image untouched.
		copyimage(dst, src, options)
		return
	}

	// Dark channel: local minimum of the per-pixel minima.
	dark = localMin2D(dark, w, h, radius, options)

	// Estimate the airlight from the brightest pixel among the darkest
	// 0.1% of the dark channel values.
	ar, ag, ab := estimateAirlight(pixGetter, srcb, dark, visible)

	// Normalized dark channel: local minimum of min(Ic/Ac).
	// The dark buffer is no longer needed, reuse its memory.
	norm := dark
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			px := pixGetter.getPixel(x, y)
			i := (y-srcb.Min.Y)*w + (x - srcb.Min.X)
			if px.a <= 0 {
				norm[i] = 1
				continue
			}
			mr := px.r / ar
			mg := px.g / ag
			mb := px.b / ab
			norm[i] = minf32(mr, minf32(mg, mb))
		}
	}
	norm = localMin2D(norm, w, h, radius, options)

	// Raw transmission map.
	t := make([]float32, n)
	for i := 0; i < n; i++ {
		v := 1 - p.strength*norm[i]
		if v < 0 {
			v = 0
		}
		t[i] = v
	}

	// Refine the transmission with a guided filter (grayscale guide) to
	// avoid blocky halos around strong edges.
	guide := make([]float32, n)
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			px := pixGetter.getPixel(x, y)
			i := (y-srcb.Min.Y)*w + (x - srcb.Min.X)
			guide[i] = 0.299*px.r + 0.587*px.g + 0.114*px.b
		}
	}
	t = guidedFilter(guide, t, w, h, radius, 1.0e-3, options)

	// Recover the scene radiance: J = A + (I-A)/max(t, tmin).
	pixSetter := newPixelSetter(dst)
	dstb := dst.Bounds()
	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				if px.a <= 0 {
					// Preserve fully transparent pixels verbatim.
					pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
					continue
				}
				i := (y-srcb.Min.Y)*w + (x - srcb.Min.X)
				tt := t[i]
				if tt < p.tmin {
					tt = p.tmin
				}
				px.r = minf32(maxf32(ar+(px.r-ar)/tt, 0), 1)
				px.g = minf32(maxf32(ag+(px.g-ag)/tt, 0), 1)
				px.b = minf32(maxf32(ab+(px.b-ab)/tt, 0), 1)
				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// estimateAirlight returns the RGB airlight. It selects the visible pixels
// whose dark channel value belongs to the darkest 0.1% (at least one pixel)
// and picks the one with the highest luminance. A 256-bin histogram keeps the
// selection linear in the number of pixels.
func estimateAirlight(getter *pixelGetter, srcb image.Rectangle, dark []float32, visible int) (ar, ag, ab float32) {
	const bins = 256
	var hist [bins]int
	w := srcb.Dx()
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			i := (y-srcb.Min.Y)*w + (x - srcb.Min.X)
			if getter.getPixel(x, y).a <= 0 {
				continue
			}
			b := int(dark[i] * bins)
			if b >= bins {
				b = bins - 1
			}
			hist[b]++
		}
	}

	want := visible / 1000
	if want < 1 {
		want = 1
	}
	acc, cutoff := 0, 0
	for ; cutoff < bins; cutoff++ {
		acc += hist[cutoff]
		if acc >= want {
			break
		}
	}
	threshold := float32(cutoff+1) / bins

	best := float32(-1)
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			px := getter.getPixel(x, y)
			if px.a <= 0 {
				continue
			}
			i := (y-srcb.Min.Y)*w + (x - srcb.Min.X)
			if dark[i] > threshold {
				continue
			}
			lum := 0.299*px.r + 0.587*px.g + 0.114*px.b
			if lum > best {
				best = lum
				ar, ag, ab = px.r, px.g, px.b
			}
		}
	}

	// Guard against division by zero on (near) black images.
	const eps = 1.0e-3
	ar = minf32(maxf32(ar, eps), 1)
	ag = minf32(maxf32(ag, eps), 1)
	ab = minf32(maxf32(ab, eps), 1)
	return ar, ag, ab
}

// localMinLine writes into dst the sliding-window minimum of src with the
// given radius. Out-of-range positions clamp to the line ends. The deque
// scratch buffer is reused between calls.
func localMinLine(dst, src []float32, radius int, deque []int) {
	n := len(src)
	dq := deque[:0]
	head := 0
	for i := 0; i < n; i++ {
		lo := i - radius
		if lo < 0 {
			lo = 0
		}
		for head < len(dq) && dq[head] < lo {
			head++
		}
		v := src[i]
		for len(dq) > head && src[dq[len(dq)-1]] >= v {
			dq = dq[:len(dq)-1]
		}
		dq = append(dq, i)
		dst[i] = src[dq[head]]
	}
}

// localMin2D applies a square local minimum filter (edge clamped). It runs in
// linear time regardless of the radius.
func localMin2D(src []float32, w, h, radius int, options *Options) []float32 {
	n := w * h
	tmp := make([]float32, n)
	dst := make([]float32, n)

	// Horizontal pass.
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		deque := make([]int, 0, 2*radius+1)
		for y := start; y < stop; y++ {
			off := y * w
			localMinLine(tmp[off:off+w], src[off:off+w], radius, deque)
		}
	})

	// Vertical pass.
	parallelize(options.Parallelization, 0, w, func(start, stop int) {
		srcCol := make([]float32, h)
		dstCol := make([]float32, h)
		deque := make([]int, 0, 2*radius+1)
		for x := start; x < stop; x++ {
			for y := 0; y < h; y++ {
				srcCol[y] = tmp[y*w+x]
			}
			localMinLine(dstCol, srcCol, radius, deque)
			for y := 0; y < h; y++ {
				dst[y*w+x] = dstCol[y]
			}
		}
	})

	return dst
}

// boxMeanLine writes into dst the sliding-window mean of src with the given
// radius. Out-of-range positions clamp to the line ends, so every output is a
// mean over the valid part of the window.
func boxMeanLine(dst, src []float32, radius int) {
	n := len(src)
	var sum float32
	for i := 0; i < n; i++ {
		lo := i - radius
		if lo < 0 {
			lo = 0
		}
		hi := i + radius
		if hi > n-1 {
			hi = n - 1
		}
		if i == 0 {
			for k := lo; k <= hi; k++ {
				sum += src[k]
			}
		} else {
			prevLo := i - 1 - radius
			if prevLo < 0 {
				prevLo = 0
			}
			if lo > prevLo {
				sum -= src[prevLo]
			}
			prevHi := i - 1 + radius
			if prevHi > n-1 {
				prevHi = n - 1
			}
			if hi > prevHi {
				sum += src[hi]
			}
		}
		dst[i] = sum / float32(hi-lo+1)
	}
}

// boxMean2D applies a square box mean filter (edge clamped) and writes the
// result into dst.
func boxMean2D(dst, src []float32, w, h, radius int, options *Options) {
	tmp := make([]float32, w*h)

	// Horizontal pass.
	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		line := make([]float32, w)
		for y := start; y < stop; y++ {
			off := y * w
			boxMeanLine(line, src[off:off+w], radius)
			copy(tmp[off:off+w], line)
		}
	})

	// Vertical pass.
	parallelize(options.Parallelization, 0, w, func(start, stop int) {
		srcCol := make([]float32, h)
		dstCol := make([]float32, h)
		for x := start; x < stop; x++ {
			for y := 0; y < h; y++ {
				srcCol[y] = tmp[y*w+x]
			}
			boxMeanLine(dstCol, srcCol, radius)
			for y := 0; y < h; y++ {
				dst[y*w+x] = dstCol[y]
			}
		}
	})
}

// guidedFilter refines src using guide as the guidance image (He et al.).
// The radius controls the window size, eps regularizes flat regions.
func guidedFilter(guide, src []float32, w, h, radius int, eps float32, options *Options) []float32 {
	n := w * h

	meanI := make([]float32, n)
	meanP := make([]float32, n)
	boxMean2D(meanI, guide, w, h, radius, options)
	boxMean2D(meanP, src, w, h, radius, options)

	// corrI = mean(I*I), covIP = mean(I*p).
	prodI := make([]float32, n)
	prodIP := make([]float32, n)
	for i := 0; i < n; i++ {
		prodI[i] = guide[i] * guide[i]
		prodIP[i] = guide[i] * src[i]
	}
	corrI := make([]float32, n)
	covIP := make([]float32, n)
	boxMean2D(corrI, prodI, w, h, radius, options)
	boxMean2D(covIP, prodIP, w, h, radius, options)

	// Linear coefficients a, b of the local model p = a*I + b.
	// Reuse the corrI/covIP buffers: each index is consumed before overwrite.
	a := corrI
	b := covIP
	for i := 0; i < n; i++ {
		variance := corrI[i] - meanI[i]*meanI[i]
		if variance < 0 {
			variance = 0
		}
		cov := covIP[i] - meanI[i]*meanP[i]
		ai := cov / (variance + eps)
		a[i] = ai
		b[i] = meanP[i] - ai*meanI[i]
	}

	meanA := make([]float32, n)
	meanB := make([]float32, n)
	boxMean2D(meanA, a, w, h, radius, options)
	boxMean2D(meanB, b, w, h, radius, options)

	// Reuse the src buffer for the output.
	out := src
	for i := 0; i < n; i++ {
		out[i] = meanA[i]*guide[i] + meanB[i]
	}
	return out
}

// Dehaze creates an adjustable dehaze filter based on the dark channel prior.
//
// The radius parameter is the neighborhood radius used for the local dark
// channel, the airlight estimation and the transmission refinement. It must be
// positive; values larger than the image are clamped automatically.
//
// The strength parameter controls how much haze is removed. It is clamped to
// the range (0, 1). Strength = 0 (or a non-positive radius) leaves the image
// unchanged.
//
// The tmin parameter is the lower bound of the transmission map and prevents
// excessive darkening of distant areas. It is clamped to the range (0, 1);
// a typical value is 0.1.
//
// The filter preserves the alpha channel and the image bounds and can be
// combined with other filters in a filter chain.
//
// Example:
//
//	g := gift.New(
//		gift.Dehaze(7, 0.8, 0.1),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func Dehaze(radius int, strength, tmin float32) Filter {
	if math.IsNaN(float64(strength)) {
		strength = 1
	}
	strength = minf32(maxf32(strength, 0), 1)

	if math.IsNaN(float64(tmin)) {
		tmin = 0.1
	}
	tmin = minf32(maxf32(tmin, 0), 1)

	if radius < 0 {
		radius = 0
	}

	if strength == 0 || radius == 0 {
		return &copyimageFilter{}
	}

	return &dehazeFilter{
		radius:   radius,
		strength: strength,
		tmin:     tmin,
	}
}
