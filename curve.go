package gift

import (
	"image"
	"image/draw"
	"math"
)

// CurvePoint is a control point of a tone curve created by the Curve filter.
// Both coordinates are normalized to the range [0, 1]: X is the input
// luminance and Y is the output luminance.
type CurvePoint struct {
	X, Y float32
}

// curveLUTSize is the number of entries in the luminance lookup table.
const curveLUTSize = 0xffff + 1

// curveControlPoint is a sanitized control point used internally.
type curveControlPoint struct {
	x, y float64
}

type curveFilter struct {
	// lut maps a luminance value in range [0, 1] to the adjusted value.
	lut []float32
}

func (p *curveFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *curveFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	if srcb.Empty() {
		return
	}
	dstb := dst.Bounds()

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// The curve maps the BT.601 luma of a pixel. The RGB channels are scaled
	// by the ratio of the new luma to the old one, which keeps the channel
	// ratios (and therefore the hue) unchanged. For pure black the ratio is
	// undefined, so the new luma is added to every channel instead.
	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				if px.a != 0 {
					lum := 0.299*px.r + 0.587*px.g + 0.114*px.b
					lum2 := p.lut[curveLUTIndex(lum)]
					if lum > 1.0e-6 {
						s := lum2 / lum
						px.r *= s
						px.g *= s
						px.b *= s
					} else {
						px.r += lum2
						px.g += lum2
						px.b += lum2
					}
					px.r = curveClamp01(px.r)
					px.g = curveClamp01(px.g)
					px.b = curveClamp01(px.b)
				}
				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// curveClamp01 clamps a value to the range [0, 1].
func curveClamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// curveClamp01f clamps a float64 value to the range [0, 1].
func curveClamp01f(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// curveIsFinite reports whether v is neither NaN nor an infinity.
func curveIsFinite(v float32) bool {
	f := float64(v)
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

// sortCurvePoints stably sorts the control points by X using a merge sort.
func sortCurvePoints(pts []curveControlPoint) {
	if len(pts) < 2 {
		return
	}
	buf := make([]curveControlPoint, len(pts))
	src, dst := pts, buf
	for width := 1; width < len(src); width *= 2 {
		for lo := 0; lo < len(src); lo += 2 * width {
			mid := minint(lo+width, len(src))
			hi := minint(lo+2*width, len(src))
			i, j, k := lo, mid, lo
			for i < mid && j < hi {
				// Take from the left run on ties to keep the sort stable.
				if src[i].x <= src[j].x {
					dst[k] = src[i]
					i++
				} else {
					dst[k] = src[j]
					j++
				}
				k++
			}
			k += copy(dst[k:], src[i:mid])
			copy(dst[k:], src[j:hi])
		}
		src, dst = dst, src
	}
	// After an even number of passes the sorted data may be in buf.
	copy(pts, src)
}

// prepareCurvePoints sanitizes the control points supplied by the caller:
// non-finite points are dropped, the coordinates are clamped to [0, 1],
// the points are sorted by X, and points that share an X are collapsed
// keeping the last one. Missing endpoints are added as (0, 0) and (1, 1),
// so the curve always covers the full input range and black and white stay
// black and white unless the caller explicitly overrides them.
func prepareCurvePoints(points []CurvePoint) []curveControlPoint {
	pts := make([]curveControlPoint, 0, len(points)+2)
	for _, p := range points {
		if !curveIsFinite(p.X) || !curveIsFinite(p.Y) {
			continue
		}
		pts = append(pts, curveControlPoint{
			float64(curveClamp01(p.X)),
			float64(curveClamp01(p.Y)),
		})
	}
	if len(pts) == 0 {
		return nil
	}

	sortCurvePoints(pts)

	n := 0
	for _, p := range pts {
		if n > 0 && pts[n-1].x == p.x {
			pts[n-1] = p
			continue
		}
		pts[n] = p
		n++
	}
	pts = pts[:n]

	if pts[0].x > 0 {
		pts = append(pts, curveControlPoint{})
		copy(pts[1:], pts[:len(pts)-1])
		pts[0] = curveControlPoint{0, 0}
	}
	if pts[len(pts)-1].x < 1 {
		pts = append(pts, curveControlPoint{1, 1})
	}
	return pts
}

// curveIsIdentity reports whether the curve maps every input value to itself.
// The control points always include the (0, 0) and (1, 1) endpoints, so a
// curve is the identity if and only if all of its points lie on y = x.
func curveIsIdentity(pts []curveControlPoint) bool {
	if len(pts) < 2 {
		return true
	}
	if pts[0].x != 0 || pts[0].y != 0 {
		return false
	}
	if pts[len(pts)-1].x != 1 || pts[len(pts)-1].y != 1 {
		return false
	}
	for _, p := range pts {
		if p.x != p.y {
			return false
		}
	}
	return true
}

// curveTangents calculates the PCHIP (shape-preserving monotone cubic
// Hermite) tangents for the given control points.
func curveTangents(pts []curveControlPoint) []float64 {
	n := len(pts)
	m := make([]float64, n)
	if n < 2 {
		return m
	}

	h := make([]float64, n-1)
	d := make([]float64, n-1)
	for i := 0; i < n-1; i++ {
		h[i] = pts[i+1].x - pts[i].x
		d[i] = (pts[i+1].y - pts[i].y) / h[i]
	}

	if n == 2 {
		m[0], m[1] = d[0], d[0]
		return m
	}

	// Interior points: weighted harmonic mean of the adjacent slopes.
	// A sign change means a local extremum, where the tangent is zero.
	for i := 1; i < n-1; i++ {
		if d[i-1]*d[i] <= 0 {
			m[i] = 0
			continue
		}
		w1 := 2*h[i] + h[i-1]
		w2 := h[i] + 2*h[i-1]
		m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
	}

	m[0] = curveEndTangent(h[0], h[1], d[0], d[1])
	m[n-1] = curveEndTangent(h[n-2], h[n-3], d[n-2], d[n-3])
	return m
}

// curveEndTangent calculates the shape-preserving tangent at an endpoint
// using the one-sided three-point formula and the PCHIP limiter.
func curveEndTangent(h0, h1, d0, d1 float64) float64 {
	m := ((2*h0+h1)*d0 - h0*d1) / (h0 + h1)
	if m*d0 <= 0 {
		return 0
	}
	if d0*d1 <= 0 && math.Abs(m) > 3*math.Abs(d0) {
		return 3 * d0
	}
	return m
}

// curveEval evaluates the monotone cubic Hermite curve at x.
func curveEval(pts []curveControlPoint, m []float64, x float64) float64 {
	n := len(pts)
	if x <= pts[0].x {
		return pts[0].y
	}
	if x >= pts[n-1].x {
		return pts[n-1].y
	}

	// Find the segment with pts[lo].x <= x < pts[lo+1].x.
	lo, hi := 0, n-1
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if pts[mid].x <= x {
			lo = mid
		} else {
			hi = mid
		}
	}

	h := pts[lo+1].x - pts[lo].x
	t := (x - pts[lo].x) / h
	t2 := t * t
	t3 := t2 * t
	return (2*t3-3*t2+1)*pts[lo].y +
		(t3-2*t2+t)*h*m[lo] +
		(-2*t3+3*t2)*pts[lo+1].y +
		(t3-t2)*h*m[lo+1]
}

// curveLUTIndex converts a luminance value to a lookup table index.
func curveLUTIndex(v float32) int {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return curveLUTSize - 1
	}
	return int(v*float32(curveLUTSize-1) + 0.5)
}

// Curve creates a filter that adjusts the tones of an image with a tone curve.
// The curve maps the luminance of every pixel to a new luminance; the RGB
// channels are scaled by the luminance ratio, which adjusts the tones without
// an obvious color shift. The alpha channel is preserved.
//
// Control points are normalized to the range [0, 1]: X is the input luminance
// and Y is the output luminance. The points are connected with a shape-
// preserving monotone cubic Hermite interpolation (PCHIP), so the curve never
// overshoots the control values and it is monotone wherever the control points
// are monotone. With no control points the curve is the identity.
//
// The control points may be supplied in any order. Points with the same X are
// collapsed and the last one wins. Non-finite points (NaN or infinity) are
// ignored, and the coordinates of the remaining points are clamped to [0, 1].
// If the (0, 0) or (1, 1) endpoints are missing, they are added automatically,
// so black and white stay black and white unless they are overridden
// explicitly. A curve whose points all lie on y = x is the identity and leaves
// the image unchanged.
//
// A typical use is lifting gray shadows and recovering blown highlights of a
// scanned negative. The curve can be previewed by applying the filter to a
// horizontal gray gradient.
//
// Example:
//
//	g := gift.New(
//		gift.Curve(
//			gift.CurvePoint{X: 0, Y: 0.08},   // lift the shadows
//			gift.CurvePoint{X: 0.5, Y: 0.5},
//			gift.CurvePoint{X: 1, Y: 0.92},   // recover the highlights
//		),
//	)
//	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func Curve(points ...CurvePoint) Filter {
	pts := prepareCurvePoints(points)
	if curveIsIdentity(pts) {
		return &copyimageFilter{}
	}

	m := curveTangents(pts)
	lut := make([]float32, curveLUTSize)
	q := 1 / float64(curveLUTSize-1)
	for i := range lut {
		lut[i] = float32(curveClamp01f(curveEval(pts, m, float64(i)*q)))
	}
	return &curveFilter{lut: lut}
}
