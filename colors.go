package gift

import (
	"image"
	"image/draw"
	"math"
)

func prepareLut(lutSize int, fn func(float32) float32) []float32 {
	lut := make([]float32, lutSize)
	q := 1 / float32(lutSize-1)
	for v := 0; v < lutSize; v++ {
		u := float32(v) * q
		lut[v] = fn(u)
	}
	return lut
}

func getFromLut(lut []float32, u float32) float32 {
	v := int(u*float32(len(lut)-1) + 0.5)
	return lut[v]
}

type colorchanFilter struct {
	fn  func(float32) float32
	lut bool
}

func (p *colorchanFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *colorchanFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	var useLut bool
	var lut []float32

	useLut = false
	if p.lut {
		var lutSize int

		it := pixGetter.it
		if it == itNRGBA || it == itRGBA || it == itGray || it == itYCbCr {
			lutSize = 0xff + 1
		} else {
			lutSize = 0xffff + 1
		}

		numCalculations := srcb.Dx() * srcb.Dy() * 3
		if numCalculations > lutSize*2 {
			useLut = true
			lut = prepareLut(lutSize, p.fn)
		}
	}

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				if useLut {
					px.r = getFromLut(lut, px.r)
					px.g = getFromLut(lut, px.g)
					px.b = getFromLut(lut, px.b)
				} else {
					px.r = p.fn(px.r)
					px.g = p.fn(px.g)
					px.b = p.fn(px.b)
				}
				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// Invert creates a filter that negates the colors of an image.
func Invert() Filter {
	return &colorchanFilter{
		fn: func(x float32) float32 {
			return 1 - x
		},
		lut: false,
	}
}

// ColorspaceSRGBToLinear creates a filter that converts the colors of an image from sRGB to linear RGB.
func ColorspaceSRGBToLinear() Filter {
	return &colorchanFilter{
		fn: func(x float32) float32 {
			if x <= 0.04045 {
				return x / 12.92
			}
			return float32(math.Pow(float64((x+0.055)/1.055), 2.4))
		},
		lut: true,
	}
}

// ColorspaceLinearToSRGB creates a filter that converts the colors of an image from linear RGB to sRGB.
func ColorspaceLinearToSRGB() Filter {
	return &colorchanFilter{
		fn: func(x float32) float32 {
			if x <= 0.0031308 {
				return x * 12.92
			}
			return float32(1.055*math.Pow(float64(x), 1/2.4) - 0.055)
		},
		lut: true,
	}
}

// Gamma creates a filter that performs a gamma correction on an image.
// The gamma parameter must be positive. Gamma = 1 gives the original image.
// Gamma less than 1 darkens the image and gamma greater than 1 lightens it.
func Gamma(gamma float32) Filter {
	e := 1 / maxf32(gamma, 1.0e-5)
	return &colorchanFilter{
		fn: func(x float32) float32 {
			return powf32(x, e)
		},
		lut: true,
	}
}

func sigmoid(a, b, x float32) float32 {
	return 1 / (1 + expf32(b*(a-x)))
}

// Sigmoid creates a filter that changes the contrast of an image using a sigmoidal function and returns the adjusted image.
// It's a non-linear contrast change useful for photo adjustments as it preserves highlight and shadow detail.
// The midpoint parameter is the midpoint of contrast that must be between 0 and 1, typically 0.5.
// The factor parameter indicates how much to increase or decrease the contrast, typically in range (-10, 10).
// If the factor parameter is positive the image contrast is increased otherwise the contrast is decreased.
//
// Example:
//
//	g := gift.New(
//		gift.Sigmoid(0.5, 5),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func Sigmoid(midpoint, factor float32) Filter {
	a := minf32(maxf32(midpoint, 0), 1)
	b := absf32(factor)
	sig0 := sigmoid(a, b, 0)
	sig1 := sigmoid(a, b, 1)
	e := float32(1.0e-5)

	return &colorchanFilter{
		fn: func(x float32) float32 {
			if factor == 0 {
				return x
			} else if factor > 0 {
				sig := sigmoid(a, b, x)
				return (sig - sig0) / (sig1 - sig0)
			} else {
				arg := minf32(maxf32((sig1-sig0)*x+sig0, e), 1-e)
				return a - logf32(1/arg-1)/b
			}
		},
		lut: true,
	}
}

// Contrast creates a filter that changes the contrast of an image.
// The percentage parameter must be in range (-100, 100). The percentage = 0 gives the original image.
// The percentage = -100 gives solid grey image. The percentage = 100 gives an overcontrasted image.
func Contrast(percentage float32) Filter {
	if percentage == 0 {
		return &copyimageFilter{}
	}

	p := 1 + minf32(maxf32(percentage, -100), 100)/100

	return &colorchanFilter{
		fn: func(x float32) float32 {
			if 0 <= p && p <= 1 {
				return 0.5 + (x-0.5)*p
			} else if 1 < p && p < 2 {
				return 0.5 + (x-0.5)*(1/(2.0-p))
			} else {
				if x < 0.5 {
					return 0
				}
				return 1
			}
		},
		lut: false,
	}
}

// Brightness creates a filter that changes the brightness of an image.
// The percentage parameter must be in range (-100, 100). The percentage = 0 gives the original image.
// The percentage = -100 gives solid black image. The percentage = 100 gives solid white image.
func Brightness(percentage float32) Filter {
	if percentage == 0 {
		return &copyimageFilter{}
	}

	shift := minf32(maxf32(percentage, -100), 100) / 100

	return &colorchanFilter{
		fn: func(x float32) float32 {
			return x + shift
		},
		lut: false,
	}
}

type colorFilter struct {
	fn func(pixel) pixel
}

func (p *colorFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *colorFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(pmin, pmax int) {
		for y := pmin; y < pmax; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, p.fn(px))
			}
		}
	})
}

// Grayscale creates a filter that produces a grayscale version of an image.
func Grayscale() Filter {
	return &colorFilter{
		fn: func(px pixel) pixel {
			y := 0.299*px.r + 0.587*px.g + 0.114*px.b
			return pixel{y, y, y, px.a}
		},
	}
}

// Sepia creates a filter that produces a sepia-toned version of an image.
// The percentage parameter specifies how much the image should be adjusted. It must be in the range (0, 100)
//
// Example:
//
//	g := gift.New(
//		gift.Sepia(100),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func Sepia(percentage float32) Filter {
	adjustAmount := minf32(maxf32(percentage, 0), 100) / 100
	rr := 1 - 0.607*adjustAmount
	rg := 0.769 * adjustAmount
	rb := 0.189 * adjustAmount
	gr := 0.349 * adjustAmount
	gg := 1 - 0.314*adjustAmount
	gb := 0.168 * adjustAmount
	br := 0.272 * adjustAmount
	bg := 0.534 * adjustAmount
	bb := 1 - 0.869*adjustAmount
	return &colorFilter{
		fn: func(px pixel) pixel {
			r := px.r*rr + px.g*rg + px.b*rb
			g := px.r*gr + px.g*gg + px.b*gb
			b := px.r*br + px.g*bg + px.b*bb
			return pixel{r, g, b, px.a}
		},
	}
}

func convertHSLToRGB(h, s, l float32) (float32, float32, float32) {
	if s == 0 {
		return l, l, l
	}

	_v := func(p, q, t float32) float32 {
		if t < 0 {
			t++
		}
		if t > 1 {
			t--
		}
		if t < 1/6.0 {
			return p + (q-p)*6*t
		}
		if t < 1/2.0 {
			return q
		}
		if t < 2/3.0 {
			return p + (q-p)*(2/3.0-t)*6
		}
		return p
	}

	var p, q float32
	if l < 0.5 {
		q = l * (1 + s)
	} else {
		q = l + s - l*s
	}
	p = 2*l - q

	r := _v(p, q, h+1/3.0)
	g := _v(p, q, h)
	b := _v(p, q, h-1/3.0)

	return r, g, b
}

func convertRGBToHSL(r, g, b float32) (float32, float32, float32) {
	max := maxf32(r, maxf32(g, b))
	min := minf32(r, minf32(g, b))

	l := (max + min) / 2

	if max == min {
		return 0, 0, l
	}

	var h, s float32
	d := max - min
	if l > 0.5 {
		s = d / (2 - max - min)
	} else {
		s = d / (max + min)
	}

	if r == max {
		h = (g - b) / d
		if g < b {
			h += 6
		}
	} else if g == max {
		h = (b-r)/d + 2
	} else {
		h = (r-g)/d + 4
	}
	h /= 6

	return h, s, l
}

func normalizeHue(hue float32) float32 {
	hue = hue - float32(int(hue))
	if hue < 0 {
		hue++
	}
	return hue
}

// Hue creates a filter that rotates the hue of an image.
// The shift parameter is the hue angle shift, typically in range (-180, 180).
// The shift = 0 gives the original image.
func Hue(shift float32) Filter {
	p := normalizeHue(shift / 360)
	if p == 0 {
		return &copyimageFilter{}
	}

	return &colorFilter{
		fn: func(px pixel) pixel {
			h, s, l := convertRGBToHSL(px.r, px.g, px.b)
			h = normalizeHue(h + p)
			r, g, b := convertHSLToRGB(h, s, l)
			return pixel{r, g, b, px.a}
		},
	}
}

// Saturation creates a filter that changes the saturation of an image.
// The percentage parameter must be in range (-100, 500). The percentage = 0 gives the original image.
func Saturation(percentage float32) Filter {
	p := 1 + minf32(maxf32(percentage, -100), 500)/100
	if p == 1 {
		return &copyimageFilter{}
	}

	return &colorFilter{
		fn: func(px pixel) pixel {
			h, s, l := convertRGBToHSL(px.r, px.g, px.b)
			s *= p
			if s > 1 {
				s = 1
			}
			r, g, b := convertHSLToRGB(h, s, l)
			return pixel{r, g, b, px.a}
		},
	}
}

// HueRange describes a selective hue adjustment applied to a range of hues
// on the color wheel.
//
// MinHue and MaxHue are the range boundaries in degrees, typically in range (0, 360).
// The range spans clockwise from MinHue to MaxHue, so a range with MinHue > MaxHue
// wraps past 0° (e.g. MinHue = 350, MaxHue = 10 covers the reds around 0°).
// A range with MinHue == MaxHue (after normalization) is empty and is ignored.
//
// Feather is the width in degrees of the soft transition band outside both edges
// of the range. It must be non-negative; a negative or non-finite value is treated as 0.
//
// HueShift is the hue rotation in degrees applied to the pixels inside the range.
// SaturationShift is the saturation change in percent, it must be in range (-100, 500).
// Non-finite HueShift or SaturationShift values are treated as 0.
type HueRange struct {
	MinHue, MaxHue  float32
	Feather         float32
	HueShift        float32
	SaturationShift float32
}

type selectiveHueRange struct {
	min      float32 // normalized start of the range, in [0, 360)
	width    float32 // clockwise width of the range, in (0, 360]
	feather  float32 // soft edge width in degrees, in [0, 180]
	hueShift float32 // hue rotation as a fraction of the full circle
	satShift float32 // saturation multiplier - 1
}

func smoothstepf32(x float32) float32 {
	x = minf32(maxf32(x, 0), 1)
	return x * x * (3 - 2*x)
}

// weight returns the strength of the range at hue h (degrees, in [0, 360)).
// It is 1 inside the range and falls off smoothly to 0 across the feather bands.
func (r *selectiveHueRange) weight(h float32) float32 {
	t := h - r.min
	if t < 0 {
		t += 360
	}
	if t <= r.width {
		return 1
	}
	if r.feather <= 0 {
		return 0
	}
	w := float32(0)
	if d := t - r.width; d < r.feather {
		w = 1 - smoothstepf32(d/r.feather)
	}
	if d := 360 - t; d < r.feather {
		w = maxf32(w, 1-smoothstepf32(d/r.feather))
	}
	return w
}

// SelectiveHue creates a filter that adjusts the hue and saturation of the pixels
// whose hue falls into one of the given ranges, leaving the other pixels unchanged.
// The range edges can be feathered to get a smooth transition without hard breaks.
// If a pixel matches several ranges, the first matching range in the list wins.
// The adjustment strength is scaled by the pixel saturation, so near-gray pixels
// stay unchanged, and the pixel lightness and alpha are always preserved.
// Ranges with non-finite boundaries or with MinHue == MaxHue are ignored;
// if no valid ranges remain, the filter gives the original image.
//
// Example:
//
//	g := gift.New(
//		gift.SelectiveHue(
//			gift.HueRange{MinHue: 200, MaxHue: 260, Feather: 20, HueShift: 120}, // recolor blue buttons only
//		),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func SelectiveHue(ranges ...HueRange) Filter {
	var rs []selectiveHueRange
	for _, r := range ranges {
		if isNonFinite(r.MinHue) || isNonFinite(r.MaxHue) {
			continue
		}
		min := normalizeHue(r.MinHue/360) * 360
		max := normalizeHue(r.MaxHue/360) * 360
		width := max - min
		if width < 0 {
			width += 360
		}
		if width == 0 {
			continue
		}
		feather := r.Feather
		if isNonFinite(feather) || feather < 0 {
			feather = 0
		}
		feather = minf32(feather, 180)
		hueShift := r.HueShift
		if isNonFinite(hueShift) {
			hueShift = 0
		}
		satShift := r.SaturationShift
		if isNonFinite(satShift) {
			satShift = 0
		}
		satShift = minf32(maxf32(satShift, -100), 500) / 100
		rs = append(rs, selectiveHueRange{
			min:      min,
			width:    width,
			feather:  feather,
			hueShift: normalizeHue(hueShift / 360),
			satShift: satShift,
		})
	}
	if len(rs) == 0 {
		return &copyimageFilter{}
	}

	return &colorFilter{
		fn: func(px pixel) pixel {
			h, s, l := convertRGBToHSL(px.r, px.g, px.b)
			if s == 0 {
				return px
			}
			hd := h * 360
			for i := range rs {
				w := rs[i].weight(hd)
				if w <= 0 {
					continue
				}
				w *= s
				h = normalizeHue(h + rs[i].hueShift*w)
				s = minf32(maxf32(s*(1+rs[i].satShift*w), 0), 1)
				r, g, b := convertHSLToRGB(h, s, l)
				return pixel{r, g, b, px.a}
			}
			return px
		},
	}
}

// Colorize creates a filter that produces a colorized version of an image.
// The hue parameter is the angle on the color wheel, typically in range (0, 360).
// The saturation parameter must be in range (0, 100).
// The percentage parameter specifies the strength of the effect, it must be in range (0, 100).
//
// Example:
//
//	g := gift.New(
//		gift.Colorize(240, 50, 100), // blue colorization, 50% saturation
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func Colorize(hue, saturation, percentage float32) Filter {
	h := normalizeHue(hue / 360)
	s := minf32(maxf32(saturation, 0), 100) / 100
	p := minf32(maxf32(percentage, 0), 100) / 100
	if p == 0 {
		return &copyimageFilter{}
	}

	return &colorFilter{
		fn: func(px pixel) pixel {
			_, _, l := convertRGBToHSL(px.r, px.g, px.b)
			r, g, b := convertHSLToRGB(h, s, l)
			px.r += (r - px.r) * p
			px.g += (g - px.g) * p
			px.b += (b - px.b) * p
			return px
		},
	}
}

// ColorBalance creates a filter that changes the color balance of an image.
// The percentage parameters for each color channel (red, green, blue) must be in range (-100, 500).
//
// Example:
//
//	g := gift.New(
//		gift.ColorBalance(20, -20, 0), // +20% red, -20% green
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func ColorBalance(percentageRed, percentageGreen, percentageBlue float32) Filter {
	pr := 1 + minf32(maxf32(percentageRed, -100), 500)/100
	pg := 1 + minf32(maxf32(percentageGreen, -100), 500)/100
	pb := 1 + minf32(maxf32(percentageBlue, -100), 500)/100

	return &colorFilter{
		fn: func(px pixel) pixel {
			px.r *= pr
			px.g *= pg
			px.b *= pb
			return px
		},
	}
}

// Threshold creates a filter that applies black/white thresholding to the image.
// The percentage parameter must be in range (0, 100).
func Threshold(percentage float32) Filter {
	p := minf32(maxf32(percentage, 0), 100) / 100
	return &colorFilter{
		fn: func(px pixel) pixel {
			y := 0.299*px.r + 0.587*px.g + 0.114*px.b
			if y > p {
				return pixel{1, 1, 1, px.a}
			}
			return pixel{0, 0, 0, px.a}
		},
	}
}

// ColorFunc creates a filter that changes the colors of an image using custom function.
// The fn parameter specifies a function that takes red, green, blue and alpha channels of a pixel
// as float32 values in range (0, 1) and returns the modified channel values.
//
// Example:
//
//	g := gift.New(
//		gift.ColorFunc(
//			func(r0, g0, b0, a0 float32) (r, g, b, a float32) {
//				r = 1 - r0   // invert the red channel
//				g = g0 + 0.1 // shift the green channel by 0.1
//				b = 0        // set the blue channel to 0
//				a = a0       // preserve the alpha channel
//				return
//			},
//		),
//	)
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
//
func ColorFunc(fn func(r0, g0, b0, a0 float32) (r, g, b, a float32)) Filter {
	return &colorFilter{
		fn: func(px pixel) pixel {
			r, g, b, a := fn(px.r, px.g, px.b, px.a)
			return pixel{r, g, b, a}
		},
	}
}
