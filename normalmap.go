package gift

import (
	"image"
	"image/draw"
	"math"
)

// EdgeMode specifies how a filter samples pixels outside of the image bounds.
type EdgeMode int

const (
	// ClampEdge samples the nearest edge pixel for coordinates outside of the
	// image. This is the default mode.
	ClampEdge EdgeMode = iota
	// WrapEdge samples the opposite edge, treating the image as tileable.
	WrapEdge
)

// normalMapFilter implements the conversion of a grayscale height map to a
// tangent-space normal map.
type normalMapFilter struct {
	strength float32
	scaleX   float32
	scaleY   float32
	invertY  bool
	edge     EdgeMode
}

func (p *normalMapFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

// normalMapFinite reports whether x is neither NaN nor an infinity.
func normalMapFinite(x float32) bool {
	f := float64(x)
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

// normalMapEdgeOffset maps the coordinate v to the [0, size) range according
// to mode. Note that for size 1 the result is always 0, so a one-pixel-wide or
// one-pixel-high image has a zero gradient along that axis.
func normalMapEdgeOffset(v, size int, mode EdgeMode) int {
	if size <= 1 {
		return 0
	}
	if mode == WrapEdge {
		v %= size
		if v < 0 {
			v += size
		}
		return v
	}
	if v < 0 {
		return 0
	}
	if v >= size {
		return size - 1
	}
	return v
}

// encodeNormalComponent maps a component in the -1..1 range to the 0..1 range
// used by the pixel representation. It is written to an RGB channel as the
// conventional 0..255 normal map encoding.
func encodeNormalComponent(v float64) float32 {
	u := v*0.5 + 0.5
	if u < 0 {
		u = 0
	} else if u > 1 {
		u = 1
	}
	return float32(u)
}

// normalizeNormal normalizes the vector and falls back to the neutral normal
// (0, 0, 1) when the vector has no usable direction.
func normalizeNormal(x, y, z float64) (float64, float64, float64) {
	// The gradient components cannot exceed MaxFloat32^2, whose square still
	// fits into a float64, so this form cannot overflow.
	length := math.Sqrt(x*x + y*y + z*z)
	if !(length > 0) || math.IsNaN(length) || math.IsInf(length, 0) {
		return 0, 0, 1
	}
	return x / length, y / length, z / length
}

// normalMapHeight extracts the height of a source pixel. The height is the
// pixel luminance. A fully transparent pixel has no surface and returns false.
func normalMapHeight(px pixel) (float32, bool) {
	if !(px.a > 0) {
		return 0, false
	}
	h := 0.299*px.r + 0.587*px.g + 0.114*px.b
	if !normalMapFinite(h) {
		return 0, false
	}
	return h, true
}

func (p *normalMapFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	if srcb.Dx() <= 0 || srcb.Dy() <= 0 {
		return
	}
	dstb := dst.Bounds()

	w, h := srcb.Dx(), srcb.Dy()

	// Non-finite parameters cannot describe a slope. They are treated as zero
	// so that the result is always the well-defined neutral normal (0, 0, 1)
	// instead of a vector containing NaN or an infinity. Negative finite
	// values are preserved: they reverse the corresponding gradient.
	strength := 0.0
	if normalMapFinite(p.strength) {
		strength = float64(p.strength)
	}
	scaleX := 0.0
	if normalMapFinite(p.scaleX) {
		scaleX = float64(p.scaleX)
	}
	scaleY := 0.0
	if normalMapFinite(p.scaleY) {
		scaleY = float64(p.scaleY)
	}

	// Extract the height and alpha planes. The luminance of a pixel is its
	// height. A fully transparent pixel has no surface: its hidden color is not
	// allowed to contribute to any gradient, so its height is marked as NaN
	// (missing) instead of being read. The alpha plane is copied to the output
	// verbatim.
	nan := float32(math.NaN())
	pixGetter := newPixelGetter(src)
	heights := make([]float32, w*h)
	alphas := make([]float32, w*h)
	parallelize(options.Parallelization, 0, h, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			row := y * w
			for x := 0; x < w; x++ {
				px := pixGetter.getPixel(srcb.Min.X+x, srcb.Min.Y+y)
				i := row + x
				alphas[i] = px.a
				h, ok := normalMapHeight(px)
				if !ok {
					h = nan
				}
				heights[i] = h
			}
		}
	})

	pixSetter := newPixelSetter(dst)
	edge := p.edge
	invertY := p.invertY

	parallelize(options.Parallelization, 0, h, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			row := y * w
			rowUp := normalMapEdgeOffset(y-1, h, edge) * w
			rowDown := normalMapEdgeOffset(y+1, h, edge) * w

			for x := 0; x < w; x++ {
				i := row + x
				hCenter := heights[i]
				alpha := alphas[i]

				// A fully transparent pixel (or a pixel with a non-finite
				// height, which cannot occur for the supported image types but
				// is guarded anyway) has no surface. It is encoded as the
				// neutral normal while its alpha is preserved.
				if !normalMapFinite(hCenter) {
					pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, pixel{
						r: 0.5, g: 0.5, b: 1, a: alpha,
					})
					continue
				}

				xLeft := normalMapEdgeOffset(x-1, w, edge)
				xRight := normalMapEdgeOffset(x+1, w, edge)

				// A transparent neighbor contributes no height. Falling back
				// to the center height keeps the hidden color out of the
				// gradient and turns the difference into a one-sided one.
				hLeft := heights[row+xLeft]
				if !normalMapFinite(hLeft) {
					hLeft = hCenter
				}
				hRight := heights[row+xRight]
				if !normalMapFinite(hRight) {
					hRight = hCenter
				}
				hUp := heights[rowUp+x]
				if !normalMapFinite(hUp) {
					hUp = hCenter
				}
				hDown := heights[rowDown+x]
				if !normalMapFinite(hDown) {
					hDown = hCenter
				}

				// Central differences estimate the gradient at the pixel
				// center. A height change of one per pixel produces a slope of
				// one at strength and scale 1.
				dhdx := (float64(hRight) - float64(hLeft)) * 0.5
				dhdy := (float64(hDown) - float64(hUp)) * 0.5

				gx := dhdx * strength * scaleX
				gy := dhdy * strength * scaleY

				// The surface is z = h(x, y) in image coordinates, where y
				// points down. Its normal is (-dh/dx, -dh/dy, 1). Converting
				// the image Y axis to the tangent-space Y axis (which points
				// up) flips that component, so ny follows +dh/dy. invertY
				// selects the opposite (DirectX-style) green channel.
				nx := -gx
				ny := gy
				if invertY {
					ny = -ny
				}
				nz := 1.0

				nx, ny, nz = normalizeNormal(nx, ny, nz)

				pixSetter.setPixel(dstb.Min.X+x, dstb.Min.Y+y, pixel{
					r: encodeNormalComponent(nx),
					g: encodeNormalComponent(ny),
					b: encodeNormalComponent(nz),
					a: alpha,
				})
			}
		}
	})
}

// NormalMap creates a filter that converts a grayscale height map to a
// tangent-space normal map.
//
// The height of a pixel is its luminance. Normals are calculated from the
// central differences of the height at the neighboring pixels, scaled by
// strength and the per-axis gradient scales, and then normalized. The result
// is encoded in RGB using the conventional mapping of the -1..1 vector
// components to the 0..255 channel range, so the output can be passed to
// further filters in the same chain. A flat area produces the neutral normal
// (0, 0, 1), encoded as (128, 128, 255). The alpha channel is copied from the
// source unchanged.
//
// The strength parameter scales the height differences. A value of zero, as
// well as a NaN or an infinity, produces neutral normals. The scaleX and
// scaleY parameters scale the horizontal and vertical gradients
// independently, which is useful for non-square pixels and for exaggerating
// the relief along a single axis. A zero scale disables the corresponding
// gradient. Negative finite values reverse it. NaN and infinite scales are
// treated as zero.
//
// The invertY parameter selects the convention of the green channel. When it
// is false, the OpenGL convention is used: the green channel points up in
// tangent space. When it is true, the green channel is flipped, which matches
// the DirectX convention.
//
// The edge parameter selects how pixels outside the image are sampled.
// ClampEdge (the default) repeats the nearest edge pixel. WrapEdge wraps
// around to the opposite edge, so a tileable height map produces a tileable
// normal map without seams. A one-pixel-wide or one-pixel-high image
// contributes a zero gradient along the short axis. Non-zero source bounds
// are supported and produce a destination image with bounds starting at
// (0, 0). An empty source image produces no output.
//
// Fully transparent pixels are treated as having no surface: they do not
// contribute height to the gradients of their neighbors, and they are encoded
// as the neutral normal. Their alpha is preserved, so the shape of the
// transparent region remains available to later filters.
//
// Example:
//
//	g := gift.New(
//		gift.NormalMap(4, 1, 1, false, gift.WrapEdge),
//	)
//	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func NormalMap(strength, scaleX, scaleY float32, invertY bool, edge EdgeMode) Filter {
	return &normalMapFilter{
		strength: strength,
		scaleX:   scaleX,
		scaleY:   scaleY,
		invertY:  invertY,
		edge:     edge,
	}
}
