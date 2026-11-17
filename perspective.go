package gift

import (
	"errors"
	"image"
	"image/draw"
	"math"
)

// Point is a 2D point with floating-point coordinates.
type Point struct {
	X, Y float64
}

// Errors returned by the Perspective filter constructor.
var (
	// ErrPerspectiveNonFinitePoint is returned when a corner point has NaN or infinite coordinates.
	ErrPerspectiveNonFinitePoint = errors.New("gift: perspective corner points must be finite")
	// ErrPerspectiveDegenerateQuad is returned when the corner points form a degenerate
	// quadrilateral (three or more collinear points, coincident points or zero area).
	ErrPerspectiveDegenerateQuad = errors.New("gift: perspective corner points form a degenerate quadrilateral")
	// ErrPerspectiveSelfIntersectingQuad is returned when the corner points form a
	// self-intersecting (bowtie) quadrilateral.
	ErrPerspectiveSelfIntersectingQuad = errors.New("gift: perspective corner points form a self-intersecting quadrilateral")
	// ErrPerspectiveSingularMatrix is returned when the perspective transformation
	// matrix is not invertible.
	ErrPerspectiveSingularMatrix = errors.New("gift: perspective transformation matrix is not invertible")
	// ErrPerspectiveOutputTooLarge is returned when the output bounds of the
	// perspective transformation are excessively large.
	ErrPerspectiveOutputTooLarge = errors.New("gift: perspective transformation output bounds are too large")
)

const (
	// maxPerspectiveDimension is the maximum width or height of the perspective output.
	maxPerspectiveDimension = 1 << 16
	// maxPerspectivePixels is the maximum number of pixels in the perspective output.
	maxPerspectivePixels = 1 << 31
)

// mat3 is a 3x3 matrix in row-major order used for homographies.
type mat3 [9]float64

func mat3Mul(a, b mat3) mat3 {
	var r mat3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[i*3+j] = a[i*3+0]*b[0*3+j] + a[i*3+1]*b[1*3+j] + a[i*3+2]*b[2*3+j]
		}
	}
	return r
}

func mat3Det(m mat3) float64 {
	return m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6])
}

func mat3Inverse(m mat3) (mat3, float64) {
	det := mat3Det(m)
	if det == 0 {
		return mat3{}, 0
	}
	inv := mat3{
		(m[4]*m[8] - m[5]*m[7]) / det,
		(m[2]*m[7] - m[1]*m[8]) / det,
		(m[1]*m[5] - m[2]*m[4]) / det,
		(m[5]*m[6] - m[3]*m[8]) / det,
		(m[0]*m[8] - m[2]*m[6]) / det,
		(m[2]*m[3] - m[0]*m[5]) / det,
		(m[3]*m[7] - m[4]*m[6]) / det,
		(m[1]*m[6] - m[0]*m[7]) / det,
		(m[0]*m[4] - m[1]*m[3]) / det,
	}
	return inv, det
}

// orient returns the signed area of the triangle (a, b, c) multiplied by two.
// The sign indicates the orientation of the ordered triplet.
func orient(a, b, c Point) float64 {
	return (b.X-a.X)*(c.Y-a.Y) - (b.Y-a.Y)*(c.X-a.X)
}

// segmentsIntersect reports whether the segments ab and cd properly intersect
// (cross each other) with the given epsilon margin.
func segmentsIntersect(a, b, c, d Point, eps float64) bool {
	o1 := orient(a, b, c)
	o2 := orient(a, b, d)
	o3 := orient(c, d, a)
	o4 := orient(c, d, b)
	return o1*o2 < -eps && o3*o4 < -eps
}

// shoelace returns the signed area of the quadrilateral.
func shoelace(pts [4]Point) float64 {
	s := 0.0
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		s += pts[i].X*pts[j].Y - pts[j].X*pts[i].Y
	}
	return s / 2
}

// perspectiveMatrix computes the homography mapping the unit square
// (0,0), (1,0), (1,1), (0,1) onto the quadrilateral pts (tl, tr, br, bl).
// It returns the matrix and its determinant.
func perspectiveMatrix(pts [4]Point) (mat3, float64) {
	x0, y0 := pts[0].X, pts[0].Y
	x1, y1 := pts[1].X, pts[1].Y
	x2, y2 := pts[2].X, pts[2].Y
	x3, y3 := pts[3].X, pts[3].Y

	sx := x0 - x1 + x2 - x3
	sy := y0 - y1 + y2 - y3
	dx1 := x1 - x2
	dx2 := x3 - x2
	dy1 := y1 - y2
	dy2 := y3 - y2

	den := dx1*dy2 - dx2*dy1
	if den == 0 {
		return mat3{}, 0
	}

	g := (sx*dy2 - dx2*sy) / den
	h := (dx1*sy - dy1*sx) / den

	m := mat3{
		x1 - x0 + g*x1, x3 - x0 + h*x3, x0,
		y1 - y0 + g*y1, y3 - y0 + h*y3, y0,
		g, h, 1,
	}
	return m, mat3Det(m)
}

type perspectiveFilter struct {
	q             mat3 // homography: unit square -> destination quadrilateral
	bounds        image.Rectangle
	interpolation Interpolation
}

func (p *perspectiveFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	if srcBounds.Dx() <= 0 || srcBounds.Dy() <= 0 {
		return image.Rect(0, 0, 0, 0)
	}
	return p.bounds
}

func (p *perspectiveFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	w, h := srcb.Dx(), srcb.Dy()
	if w <= 0 || h <= 0 {
		return
	}

	dstb := dst.Bounds()
	b := p.bounds
	dw, dh := b.Dx(), b.Dy()

	// Forward transform: source edge coordinates -> destination edge coordinates.
	// It is the unit square -> quad homography composed with the source
	// rectangle normalization. The inverse is used for resampling.
	fw, fh := float64(w), float64(h)
	n := mat3{
		1 / fw, 0, -float64(srcb.Min.X) / fw,
		0, 1 / fh, -float64(srcb.Min.Y) / fh,
		0, 0, 1,
	}
	inv, det := mat3Inverse(mat3Mul(p.q, n))
	if det == 0 {
		return
	}

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)
	bgpx := pixel{}

	parallelize(options.Parallelization, 0, dh, func(start, stop int) {
		for y := start; y < stop; y++ {
			dy := float64(b.Min.Y+y) + 0.5
			for x := 0; x < dw; x++ {
				dx := float64(b.Min.X+x) + 0.5

				var px pixel
				den := inv[6]*dx + inv[7]*dy + inv[8]
				if den != 0 {
					sx := (inv[0]*dx + inv[1]*dy + inv[2]) / den
					sy := (inv[3]*dx + inv[4]*dy + inv[5]) / den
					if !math.IsNaN(sx) && !math.IsNaN(sy) && !math.IsInf(sx, 0) && !math.IsInf(sy, 0) {
						xf := float32(sx - 0.5)
						yf := float32(sy - 0.5)
						switch p.interpolation {
						case CubicInterpolation:
							px = interpolateCubic(xf, yf, srcb, pixGetter, bgpx)
						case LinearInterpolation:
							px = interpolateLinear(xf, yf, srcb, pixGetter, bgpx)
						default:
							px = interpolateNearest(xf, yf, srcb, pixGetter, bgpx)
						}
					}
				}

				pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, px)
			}
		}
	})
}

// Perspective creates a filter that applies a perspective (projective) transformation
// mapping the source image rectangle onto the quadrilateral defined by the four
// destination corner points:
//
//	tl - top-left, tr - top-right, br - bottom-right, bl - bottom-left.
//
// The points are specified in the destination image coordinate space where
// pixel (x, y) covers the [x, x+1) x [y, y+1) area. For example, mapping a
// source image onto the points (0,0), (w,0), (w,h), (0,h) is the identity
// transformation for a w x h source image.
//
// The output bounds are the bounding box of the four destination points.
// Pixels falling outside the destination quadrilateral are transparent.
// The interpolation parameter specifies the resampling algorithm:
// NearestNeighborInterpolation, LinearInterpolation or CubicInterpolation.
//
// An error is returned if the corner points have non-finite coordinates, form a
// self-intersecting or degenerate (collinear, zero-area) quadrilateral, define a
// non-invertible transformation, or produce excessively large output bounds.
//
// Example:
//
//	// Project a poster onto a tilted billboard quad.
//	f, err := gift.Perspective(
//		gift.Point{X: 40, Y: 30},
//		gift.Point{X: 220, Y: 10},
//		gift.Point{X: 240, Y: 150},
//		gift.Point{X: 20, Y: 170},
//		gift.LinearInterpolation,
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//	g := gift.New(f)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func Perspective(tl, tr, br, bl Point, interpolation Interpolation) (Filter, error) {
	pts := [4]Point{tl, tr, br, bl}

	for _, pt := range pts {
		if math.IsNaN(pt.X) || math.IsNaN(pt.Y) || math.IsInf(pt.X, 0) || math.IsInf(pt.Y, 0) {
			return nil, ErrPerspectiveNonFinitePoint
		}
	}

	// Scale-aware epsilon for the geometric predicates (cross products have
	// units of length squared).
	scale := 1.0
	for _, pt := range pts {
		scale = math.Max(scale, math.Max(math.Abs(pt.X), math.Abs(pt.Y)))
	}
	eps := 1e-9 * scale * scale

	// Reject degenerate quads: three consecutive collinear points (this also
	// covers coincident adjacent points).
	for i := 0; i < 4; i++ {
		if math.Abs(orient(pts[i], pts[(i+1)%4], pts[(i+2)%4])) <= eps {
			return nil, ErrPerspectiveDegenerateQuad
		}
	}

	// Reject self-intersecting (bowtie) quads: proper intersection of
	// opposite edges.
	if segmentsIntersect(pts[0], pts[1], pts[2], pts[3], eps) ||
		segmentsIntersect(pts[1], pts[2], pts[3], pts[0], eps) {
		return nil, ErrPerspectiveSelfIntersectingQuad
	}

	// Reject zero-area quads.
	if math.Abs(shoelace(pts)) <= eps {
		return nil, ErrPerspectiveDegenerateQuad
	}

	// The homography must be invertible.
	q, det := perspectiveMatrix(pts)
	if math.Abs(det) <= 1e-12*scale*scale {
		return nil, ErrPerspectiveSingularMatrix
	}

	// Deterministic output bounds: the bounding box of the destination quad.
	minx, maxx := pts[0].X, pts[0].X
	miny, maxy := pts[0].Y, pts[0].Y
	for _, pt := range pts[1:] {
		minx = math.Min(minx, pt.X)
		maxx = math.Max(maxx, pt.X)
		miny = math.Min(miny, pt.Y)
		maxy = math.Max(maxy, pt.Y)
	}
	w := math.Ceil(maxx) - math.Floor(minx)
	h := math.Ceil(maxy) - math.Floor(miny)
	if w < 1 || h < 1 {
		return nil, ErrPerspectiveDegenerateQuad
	}
	if w > maxPerspectiveDimension || h > maxPerspectiveDimension || w*h > maxPerspectivePixels {
		return nil, ErrPerspectiveOutputTooLarge
	}
	bounds := image.Rect(int(math.Floor(minx)), int(math.Floor(miny)), int(math.Ceil(maxx)), int(math.Ceil(maxy)))

	return &perspectiveFilter{
		q:             q,
		bounds:        bounds,
		interpolation: interpolation,
	}, nil
}
