package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

// LensDistortion creates a filter that applies a normalized radial lens
// distortion or correction.
//
// The radial mapping is:
//
//	r' = r * (1 + k1*r^2 + k2*r^4)
//
// where r is the distance from the optical center, normalized by half of the
// shorter side of the image, so the model is resolution independent and the
// normalization stays the same for any output size. Coordinates are mapped
// from the destination image back to the source (inverse mapping), therefore
// the result never contains holes. The same model serves both purposes:
// positive coefficients produce a barrel (fisheye-like) distortion while
// negative coefficients undo it (pincushion correction).
//
// centerX and centerY are the optical center in the source image pixel
// coordinates. When both are zero the center of the image is used.
//
// k1 and k2 are the radial distortion coefficients. Zero coefficients leave
// the image unchanged.
//
// backgroundColor is used to fill the destination pixels whose source
// coordinates fall outside of the source image bounds.
//
// interpolation selects the sampling method and must be one of
// NearestNeighborInterpolation, LinearInterpolation or CubicInterpolation.
//
// Non-finite coefficients or center, as well as a mapping that is not
// invertible over the destination (the radial function folds back because
// the coefficients are too strong), do not cause a failure: every affected
// destination pixel is filled with backgroundColor. All arithmetic is
// bounded, so the filter never produces holes, out-of-bounds accesses or
// non-finite pixel values.
//
// Example:
//
//	g := gift.New(
//		gift.LensDistortion(0, 0, -0.12, 0.02, color.Black, gift.CubicInterpolation),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func LensDistortion(centerX, centerY, k1, k2 float32, backgroundColor color.Color, interpolation Interpolation) Filter {
	return &lensDistortionFilter{
		centerX:       centerX,
		centerY:       centerY,
		k1:            k1,
		k2:            k2,
		bgcolor:       backgroundColor,
		interpolation: interpolation,
	}
}

type lensDistortionFilter struct {
	centerX, centerY float32
	k1, k2           float32
	bgcolor          color.Color
	interpolation    Interpolation
}

func (p *lensDistortionFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	return image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
}

func (p *lensDistortionFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	w, h := dstb.Dx(), dstb.Dy()
	if w <= 0 || h <= 0 {
		return
	}

	bgpx := pixelFromColor(p.bgcolor)
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	// Zero coefficients must reproduce the input exactly, even with a
	// non-finite optical center.
	if p.k1 == 0 && p.k2 == 0 {
		parallelize(options.Parallelization, 0, h, func(start, stop int) {
			for y := start; y < stop; y++ {
				for x := 0; x < w; x++ {
					pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y,
						pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y))
				}
			}
		})
		return
	}

	// Any non-finite parameter makes the whole mapping undefined.
	if !isFiniteF32(p.k1) || !isFiniteF32(p.k2) ||
		!isFiniteF32(p.centerX) || !isFiniteF32(p.centerY) {
		parallelize(options.Parallelization, 0, h, func(start, stop int) {
			for y := start; y < stop; y++ {
				for x := 0; x < w; x++ {
					pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, bgpx)
				}
			}
		})
		return
	}

	// Optical center in source image coordinates; zero means image center.
	cx := float64(p.centerX)
	cy := float64(p.centerY)
	if cx == 0 && cy == 0 {
		cx = float64(srcb.Min.X) + float64(srcb.Dx())/2
		cy = float64(srcb.Min.Y) + float64(srcb.Dy())/2
	}

	// Normalization scale: half of the shorter side.
	scale := 0.5 * float64(minint(srcb.Dx(), srcb.Dy()))
	if scale <= 0 {
		scale = 1
	}

	k1 := float64(p.k1)
	k2 := float64(p.k2)

	// If the radial function folds, only destination radii below the fold
	// radius have a source; the rest is filled with the background color.
	foldRadius, folded := radialFoldRadius(k1, k2)

	parallelize(options.Parallelization, 0, h, func(start, stop int) {
		for y := start; y < stop; y++ {
			dy := (float64(dstb.Min.Y+y) - cy) / scale
			for x := 0; x < w; x++ {
				dx := (float64(dstb.Min.X+x) - cx) / scale
				rd := math.Sqrt(dx*dx + dy*dy)

				var xf, yf float32
				valid := true
				if rd == 0 {
					xf = float32(cx)
					yf = float32(cy)
				} else if folded && rd >= foldRadius {
					valid = false
				} else {
					ru, ok := radialInverse(rd, k1, k2, folded, foldRadius)
					if !ok {
						valid = false
					} else {
						ratio := ru / rd
						sx := cx + dx*scale*ratio
						sy := cy + dy*scale*ratio
						if !isFiniteF64(sx) || !isFiniteF64(sy) {
							valid = false
						} else {
							xf = float32(sx)
							yf = float32(sy)
						}
					}
				}

				var px pixel
				if valid {
					switch p.interpolation {
					case CubicInterpolation:
						px = interpolateCubic(xf, yf, srcb, pixGetter, bgpx)
					case LinearInterpolation:
						px = interpolateLinear(xf, yf, srcb, pixGetter, bgpx)
					default:
						px = interpolateNearest(xf, yf, srcb, pixGetter, bgpx)
					}
					if !isFiniteF32(px.r) || !isFiniteF32(px.g) ||
						!isFiniteF32(px.b) || !isFiniteF32(px.a) {
						px = bgpx
					}
				} else {
					px = bgpx
				}

				pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, px)
			}
		}
	})
}

// radialDistortion evaluates r' = r*(1 + k1*r^2 + k2*r^4).
func radialDistortion(r, k1, k2 float64) float64 {
	r2 := r * r
	return r * (1 + k1*r2 + k2*r2*r2)
}

// radialFoldRadius reports whether the radial function is non-monotonic and,
// if so, returns the normalized radius of its first local extremum. The
// derivative of r*(1+k1*r^2+k2*r^4) is 1 + 3*k1*r^2 + 5*k2*r^4.
func radialFoldRadius(k1, k2 float64) (radius float64, folded bool) {
	if k2 == 0 {
		if k1 < 0 {
			r2 := -1 / (3 * k1)
			return math.Sqrt(r2), true
		}
		return 0, false
	}

	a := 5 * k2
	b := 3 * k1
	// Roots of a*u^2 + b*u + 1 = 0 with u = r^2. A zero discriminant is
	// only a tangent point where the derivative touches zero without
	// changing sign, so the mapping stays invertible.
	disc := b*b - 4*a
	if disc <= 0 {
		return 0, false
	}
	sqrtDisc := math.Sqrt(disc)
	u1 := (-b - sqrtDisc) / (2 * a)
	u2 := (-b + sqrtDisc) / (2 * a)
	// The first extremum sits at the smallest positive root; the ordering
	// of u1/u2 flips depending on the sign of a.
	radius2 := math.Inf(1)
	if u1 > 0 && u1 < radius2 {
		radius2 = u1
	}
	if u2 > 0 && u2 < radius2 {
		radius2 = u2
	}
	if math.IsInf(radius2, 1) {
		return 0, false
	}
	return math.Sqrt(radius2), true
}

// radialInverse finds the normalized source radius ru for which
// radialDistortion(ru, k1, k2) == rd. The search is fully bounded: bisection
// guarantees a result even when the Newton iterations do not converge.
func radialInverse(rd, k1, k2 float64, folded bool, foldRadius float64) (ru float64, ok bool) {
	if rd < 0 || !isFiniteF64(rd) {
		return 0, false
	}
	if rd == 0 {
		return 0, true
	}

	// Upper bound of the invertibility interval.
	var hi float64
	if folded {
		// Stay strictly inside the fold so f stays monotonic on [0, hi].
		hi = foldRadius * (1 - 1e-10)
		if hi <= 0 {
			return 0, false
		}
		if radialDistortion(hi, k1, k2) < rd {
			return 0, false
		}
	} else {
		// Monotonic mapping: grow the bracket until f(hi) reaches rd.
		hi = math.Max(rd, 1)
		for i := 0; i < 64; i++ {
			fhi := radialDistortion(hi, k1, k2)
			if !isFiniteF64(fhi) || fhi >= rd {
				break
			}
			hi *= 2
		}
		fhi := radialDistortion(hi, k1, k2)
		if !isFiniteF64(fhi) || fhi < rd {
			return 0, false
		}
	}

	lo := 0.0
	r := rd

	// Newton iterations (derivative: 1 + 3*k1*r^2 + 5*k2*r^4), kept inside
	// the bracket; fall back to bisection for the remaining precision.
	for i := 0; i < 24; i++ {
		r2 := r * r
		fr := radialDistortion(r, k1, k2) - rd
		d := 1 + 3*k1*r2 + 5*k2*r2*r2
		if !isFiniteF64(fr) || !isFiniteF64(d) || d <= 0 {
			break
		}
		if math.Abs(fr) <= 1e-12*math.Max(1, rd) {
			return r, true
		}
		next := r - fr/d
		if next <= lo || next >= hi || !isFiniteF64(next) {
			break
		}
		r = next
	}

	// Bisection refinement on [lo, hi].
	for i := 0; i < 64; i++ {
		mid := (lo + hi) * 0.5
		if radialDistortion(mid, k1, k2) < rd {
			lo = mid
		} else {
			hi = mid
		}
	}
	ru = (lo + hi) * 0.5
	if !isFiniteF64(ru) {
		return 0, false
	}
	return ru, true
}

func isFiniteF32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func isFiniteF64(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
