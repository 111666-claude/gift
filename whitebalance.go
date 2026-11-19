package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"sync"
)

// whiteBalanceMinLuminance is the minimum relative luminance (the Y component
// of CIE XYZ, where pure white has luminance 1) that a reference white must
// have to be used for chromatic adaptation. A reference that is black or
// darker than this threshold carries no usable chromatic information: the
// adaptation would have to amplify quantization noise by several orders of
// magnitude. In that case WhiteBalance returns a pass-through filter.
const whiteBalanceMinLuminance = 1.0e-3

// whiteBalanceEpsilon is the tolerance used to recognize an adaptation matrix
// that is (numerically) the identity matrix.
const whiteBalanceEpsilon = 1.0e-6

// Matrices are stored in row-major order: m[row*3+col].

// wbRGBToXYZ is the linear sRGB (D65) to CIE XYZ matrix.
var wbRGBToXYZ = [9]float64{
	0.4124564, 0.3575761, 0.1804375,
	0.2126729, 0.7151522, 0.0721750,
	0.0193339, 0.1191920, 0.9503041,
}

// wbXYZToRGB is the inverse of wbRGBToXYZ (CIE XYZ to linear sRGB, D65).
var wbXYZToRGB = [9]float64{
	3.2404542, -1.5371385, -0.4985314,
	-0.9692660, 1.8760108, 0.0415560,
	0.0556434, -0.2040259, 1.0572252,
}

// wbBradford is the Bradford cone response matrix. It is the transform used
// by the Bradford chromatic adaptation model.
var wbBradford = [9]float64{
	0.8951, 0.2664, -0.1614,
	-0.7502, 1.7135, 0.0367,
	0.0389, -0.0685, 1.0296,
}

// wbBradfordInv is the inverse of wbBradford.
var wbBradfordInv = [9]float64{
	0.9869929, -0.1470543, 0.1599627,
	0.4323053, 0.5183603, 0.0492912,
	-0.0085287, 0.0400428, 0.9684867,
}

// wbMatMul multiplies two 3x3 matrices.
func wbMatMul(a, b [9]float64) (c [9]float64) {
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			c[row*3+col] = a[row*3+0]*b[0*3+col] +
				a[row*3+1]*b[1*3+col] +
				a[row*3+2]*b[2*3+col]
		}
	}
	return
}

// wbMatVec multiplies a 3x3 matrix by a 3-vector.
func wbMatVec(m [9]float64, v [3]float64) (r [3]float64) {
	for row := 0; row < 3; row++ {
		r[row] = m[row*3+0]*v[0] + m[row*3+1]*v[1] + m[row*3+2]*v[2]
	}
	return
}

// wbFinite reports whether v is a finite number.
func wbFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// wbSRGBToLinear converts one sRGB-encoded channel value in [0, 1] to linear
// light.
func wbSRGBToLinear(v float64) float64 {
	if v <= 0.04045 {
		return v / 12.92
	}
	return math.Pow((v+0.055)/1.055, 2.4)
}

// wbLinearToSRGB converts one linear light channel value in [0, 1] to the
// sRGB encoding.
func wbLinearToSRGB(v float64) float64 {
	if v <= 0.0031308 {
		return v * 12.92
	}
	return 1.055*math.Pow(v, 1/2.4) - 0.055
}

// WhitePoint is an sRGB-encoded white point given by its normalized channel
// values, i.e. the same encoding as the pixels of the source image, where
// (0, 0, 0) is black and (1, 1, 1) is the sRGB white point (D65).
//
// It implements color.Color, so ordinary colors and colors sampled from an
// image can be used with WhiteBalance directly. WhitePoint exists because
// color.RGBA cannot express invalid channel values: a WhitePoint may hold
// NaN, ±Inf, negative or greater-than-one channels, and WhiteBalance rejects
// such degenerate white points in a defined way instead of silently wrapping
// them around.
type WhitePoint struct {
	R, G, B float32
}

// RGBA implements color.Color. Channel values are clamped to [0, 1] and
// encoded as 16-bit non-premultiplied alpha values; non-finite values become
// zero.
func (p WhitePoint) RGBA() (r, g, b, a uint32) {
	return color.NRGBA64{
		R: wbChannelToUint16(p.R),
		G: wbChannelToUint16(p.G),
		B: wbChannelToUint16(p.B),
		A: 0xffff,
	}.RGBA()
}

// wbChannelToUint16 converts a normalized channel value to 16 bits for the
// color.Color implementation of WhitePoint.
func wbChannelToUint16(v float32) uint16 {
	f := float64(v)
	if math.IsNaN(f) || f <= 0 {
		return 0
	}
	if f >= 1 {
		return 0xffff
	}
	return uint16(f*0xffff + 0.5)
}

// wbEncodedWhite validates an sRGB-encoded white color and converts it to
// linear sRGB. It returns false when the color cannot describe a white point:
// non-finite channels, channels outside the [0, 1] gamut, or a relative
// luminance below whiteBalanceMinLuminance (black or too dark).
func wbEncodedWhite(enc [3]float64) ([3]float64, bool) {
	var lin [3]float64
	for i, v := range enc {
		if !wbFinite(v) || v < 0 || v > 1 {
			return [3]float64{}, false
		}
		lin[i] = wbSRGBToLinear(v)
	}

	y := wbMatVec(wbRGBToXYZ, lin)[1]
	if !wbFinite(y) || y < whiteBalanceMinLuminance {
		return [3]float64{}, false
	}

	return lin, true
}

// wbLinearWhite converts a reference white color to linear sRGB. The color is
// interpreted the same way as the pixels of the source image, i.e. as an
// sRGB-encoded, non-premultiplied color. It returns false when the color
// cannot describe a white point (see wbEncodedWhite). A nil or fully
// transparent color has no usable channels and is rejected as well.
func wbLinearWhite(c color.Color) ([3]float64, bool) {
	switch c := c.(type) {
	case nil:
		return [3]float64{}, false
	case WhitePoint:
		return wbEncodedWhite([3]float64{float64(c.R), float64(c.G), float64(c.B)})
	case *WhitePoint:
		if c == nil {
			return [3]float64{}, false
		}
		return wbEncodedWhite([3]float64{float64(c.R), float64(c.G), float64(c.B)})
	}
	px := pixelFromColor(c)
	return wbEncodedWhite([3]float64{float64(px.r), float64(px.g), float64(px.b)})
}

// wbAdaptationMatrix returns the Bradford chromatic adaptation matrix that
// maps linear sRGB colors seen under the src white point to the same colors
// under the dst white point. The matrix already includes the sRGB <-> XYZ
// conversions and acts directly on linear sRGB.
//
// It returns false when either white point is unusable (see wbLinearWhite) or
// when the resulting matrix has non-finite elements.
func wbAdaptationMatrix(srcWhite, dstWhite color.Color) ([9]float64, bool) {
	srcLin, ok := wbLinearWhite(srcWhite)
	if !ok {
		return [9]float64{}, false
	}
	dstLin, ok := wbLinearWhite(dstWhite)
	if !ok {
		return [9]float64{}, false
	}

	srcCone := wbMatVec(wbBradford, wbMatVec(wbRGBToXYZ, srcLin))
	dstCone := wbMatVec(wbBradford, wbMatVec(wbRGBToXYZ, dstLin))

	var scale [9]float64
	for i := 0; i < 3; i++ {
		if !wbFinite(srcCone[i]) || !wbFinite(dstCone[i]) || math.Abs(srcCone[i]) < 1.0e-9 {
			return [9]float64{}, false
		}
		scale[i*3+i] = dstCone[i] / srcCone[i]
		if !wbFinite(scale[i*3+i]) {
			return [9]float64{}, false
		}
	}

	// Von Kries adaptation in the Bradford cone response space, composed with
	// the linear sRGB <-> XYZ conversions.
	m := wbMatMul(wbBradfordInv, wbMatMul(scale, wbBradford))
	m = wbMatMul(wbXYZToRGB, wbMatMul(m, wbRGBToXYZ))

	for _, v := range m {
		if !wbFinite(v) {
			return [9]float64{}, false
		}
	}

	return m, true
}

// wbIdentityMatrix reports whether m is the identity matrix within
// whiteBalanceEpsilon.
func wbIdentityMatrix(m [9]float64) bool {
	for i := 0; i < 9; i++ {
		want := 0.0
		if i%4 == 0 {
			want = 1.0
		}
		if math.Abs(m[i]-want) > whiteBalanceEpsilon {
			return false
		}
	}
	return true
}

// wbLutSize is the number of entries of the sRGB transfer function lookup
// tables. 16 bits cover both 8-bit and 16-bit sources, so the lookup error is
// far below the 8-bit quantization step.
const wbLutSize = 0x10000

var (
	wbLutOnce     sync.Once
	wbToLinearLut []float32
	wbToSRGBLut   []float32
)

// wbLUTs returns the lookup tables for the sRGB transfer functions. They are
// built once and shared by all the WhiteBalance filters.
func wbLUTs() (toLinear, toSRGB []float32) {
	wbLutOnce.Do(func() {
		toLinear = make([]float32, wbLutSize)
		toSRGB = make([]float32, wbLutSize)
		for i := 0; i < wbLutSize; i++ {
			v := float64(i) / (wbLutSize - 1)
			toLinear[i] = float32(wbSRGBToLinear(v))
			toSRGB[i] = float32(wbLinearToSRGB(v))
		}
		wbToLinearLut = toLinear
		wbToSRGBLut = toSRGB
	})
	return wbToLinearLut, wbToSRGBLut
}

// wbLutLookup evaluates a lookup table of wbLutSize entries at the normalized
// position v, with linear interpolation between the entries. The interpolation
// keeps the error of the transfer functions far below a single 8-bit step, so
// the lookup is indistinguishable from an exact evaluation. v must be finite;
// values outside [0, 1] are clamped.
func wbLutLookup(lut []float32, v float64) float64 {
	if !(v > 0) {
		return float64(lut[0])
	}
	if v >= 1 {
		return float64(lut[wbLutSize-1])
	}
	f := v * (wbLutSize - 1)
	i := int(f)
	if i >= wbLutSize-1 {
		return float64(lut[wbLutSize-1])
	}
	frac := f - float64(i)
	return float64(lut[i]) + (float64(lut[i+1])-float64(lut[i]))*frac
}

type whiteBalanceFilter struct {
	m [9]float64
}

func (p *whiteBalanceFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *whiteBalanceFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	if srcb.Empty() {
		return
	}

	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)
	m := p.m
	toLinear, toSRGB := wbLUTs()

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)

				r := wbLutLookup(toLinear, float64(px.r))
				g := wbLutLookup(toLinear, float64(px.g))
				b := wbLutLookup(toLinear, float64(px.b))

				rl := m[0]*r + m[1]*g + m[2]*b
				gl := m[3]*r + m[4]*g + m[5]*b
				bl := m[6]*r + m[7]*g + m[8]*b

				// Clamp the adapted color to the sRGB gamut in linear light.
				// Hard clipping is simple and well defined; colors that stay
				// inside the gamut are not modified.
				px.r = float32(wbLutLookup(toSRGB, wbClamp01(rl)))
				px.g = float32(wbLutLookup(toSRGB, wbClamp01(gl)))
				px.b = float32(wbLutLookup(toSRGB, wbClamp01(bl)))

				// The alpha channel is passed through unchanged.

				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// wbClamp01 clamps a linear light channel value to the sRGB gamut [0, 1].
// Components below zero are set to zero and components above one are set to
// one; non-finite values are set to zero. Clamping is deterministic and keeps
// every in-gamut color untouched.
func wbClamp01(v float64) float64 {
	if math.IsNaN(v) || v <= 0 {
		return 0
	}
	if v >= 1 {
		return 1
	}
	return v
}

// WhiteBalance creates a filter that performs chromatic adaptation of an image
// from the srcWhite reference white to the dstWhite white point.
//
// Both white points are ordinary sRGB colors, so they can be measured
// directly on the image: srcWhite is the color that a neutral object (a gray
// card, a white card, or the scene illuminant) has in the source image, and
// dstWhite is the white point that neutral colors should have in the result.
// For example, a photo of a white card lit by a warm lamp can be corrected to
// the sRGB D65 white point with:
//
//	g := gift.New(
//		gift.WhiteBalance(color.NRGBA{255, 214, 170, 255}, color.White),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
// The adaptation is the Bradford chromatic adaptation model, a standard and
// invertible transform defined for any pair of valid white points. It is
// applied in linear light: the sRGB encoded pixels are linearized, converted
// to CIE XYZ, adapted, converted back, clamped to the sRGB gamut, and encoded
// again. The source reference white is mapped exactly to the target white
// point, while other colors keep their relative appearance, so neutral tones
// become truly neutral instead of merely being shifted by a per-channel gain.
// The alpha channel is preserved.
//
// Degenerate inputs have defined results. A nil, transparent, black, or too
// dark (relative luminance below 0.001) reference white cannot define an
// adaptation; the same applies to reference colors with non-finite or
// out-of-range channel values. In all those cases, as well as when both white
// points are equal, the filter is a pass-through copy. Empty images produce
// empty results. Image bounds and dimensions are preserved.
//
// Note that the filter assumes the source image is sRGB encoded, exactly like
// the rest of the package.
func WhiteBalance(srcWhite, dstWhite color.Color) Filter {
	m, ok := wbAdaptationMatrix(srcWhite, dstWhite)
	if !ok || wbIdentityMatrix(m) {
		return &copyimageFilter{}
	}
	return &whiteBalanceFilter{m: m}
}
