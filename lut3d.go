package gift

import (
	"bufio"
	"fmt"
	"image"
	"image/draw"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// maxLUT3DSize is the largest supported dimension of a 3D LUT.
const maxLUT3DSize = 256

// LUT3D is a three-dimensional color lookup table loaded from an
// Iridas/Adobe .cube file.
//
// The table maps an RGB color to a new RGB color. The alpha channel is not
// part of the table and is always preserved by the filter. A LUT3D is
// immutable after it has been loaded, so it can be shared between filters
// and used concurrently.
type LUT3D struct {
	size      int
	domainMin [3]float32
	domainMax [3]float32
	scale     [3]float32 // (size-1)/(domainMax-domainMin) per channel
	data      []float32  // RGB triplets, the red index changes fastest
}

// Size returns the dimension of the LUT grid. The table contains
// Size*Size*Size RGB entries.
func (l *LUT3D) Size() int {
	if l == nil {
		return 0
	}
	return l.size
}

// Domain returns the input domain of the LUT. Values of the domain default
// to (0, 0, 0) and (1, 1, 1) when the .cube file does not specify them.
func (l *LUT3D) Domain() (min, max [3]float32) {
	if l == nil {
		return
	}
	return l.domainMin, l.domainMax
}

func isFinite32(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0)
}

// validateLUT3DDomain checks the size and the domain of a LUT3D.
func validateLUT3DDomain(size int, domainMin, domainMax [3]float32) error {
	if size < 2 || size > maxLUT3DSize {
		return fmt.Errorf("invalid size %d (must be between 2 and %d)", size, maxLUT3DSize)
	}
	for i := 0; i < 3; i++ {
		if !isFinite32(domainMin[i]) || !isFinite32(domainMax[i]) {
			return fmt.Errorf("domain values must be finite")
		}
		if domainMax[i] <= domainMin[i] {
			return fmt.Errorf("invalid domain: max %v must be greater than min %v for channel %d",
				domainMax[i], domainMin[i], i)
		}
	}
	return nil
}

// newLUT3D validates the size, the domain, and the table data and constructs
// a LUT3D from them.
func newLUT3D(size int, domainMin, domainMax [3]float32, data []float32) (*LUT3D, error) {
	if err := validateLUT3DDomain(size, domainMin, domainMax); err != nil {
		return nil, fmt.Errorf("gift: LUT3D: %v", err)
	}

	numEntries := size * size * size
	if len(data) != numEntries*3 {
		return nil, fmt.Errorf("gift: LUT3D: invalid data length %d (expected %d RGB values for size %d)",
			len(data), numEntries*3, size)
	}
	for i, v := range data {
		if !isFinite32(v) {
			return nil, fmt.Errorf("gift: LUT3D: non-finite value at index %d", i)
		}
	}

	lut := &LUT3D{
		size:      size,
		domainMin: domainMin,
		domainMax: domainMax,
		data:      data,
	}
	for i := 0; i < 3; i++ {
		lut.scale[i] = float32(size-1) / (domainMax[i] - domainMin[i])
	}
	return lut, nil
}

// NewLUT3D creates a LUT3D from the given grid size, input domain, and
// RGB data and validates them. The data must contain exactly
// size*size*size RGB triplets: the red index changes fastest, then green,
// then blue. The size must be between 2 and 256, every domain range must be
// non-empty, and all numbers must be finite. The data is copied, so the
// caller may reuse the slice after the call.
func NewLUT3D(size int, domainMin, domainMax [3]float32, data []float32) (*LUT3D, error) {
	if err := validateLUT3DDomain(size, domainMin, domainMax); err != nil {
		return nil, fmt.Errorf("gift: LUT3D: %v", err)
	}
	if len(data) != size*size*size*3 {
		return nil, fmt.Errorf("gift: LUT3D: invalid data length %d (expected %d RGB values for size %d)",
			len(data), size*size*size*3, size)
	}
	return newLUT3D(size, domainMin, domainMax, append([]float32(nil), data...))
}

// ParseLUT3D reads a 3D LUT in the Iridas/Adobe .cube format from r and
// returns it as a LUT3D.
//
// The LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX, and TITLE directives are
// recognized; text after a '#' character is treated as a comment. The RGB
// values must be finite numbers, and their count must be exactly
// Size*Size*Size. Only 3D LUTs are supported.
func ParseLUT3D(r io.Reader) (*LUT3D, error) {
	if r == nil {
		return nil, fmt.Errorf("gift: LUT3D: nil reader")
	}

	var (
		size         int
		hasSize      bool
		domainMin    = [3]float32{0, 0, 0}
		domainMax    = [3]float32{1, 1, 1}
		hasDomainMin bool
		hasDomainMax bool
		data         []float32
		numRows      int
		lineNum      int
	)

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		lineNum++
		line := strings.TrimPrefix(scanner.Text(), "\ufeff")
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = line[:i]
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}

		switch keyword := strings.ToUpper(fields[0]); keyword {
		case "TITLE":
			// A title is informational only.

		case "LUT_1D_SIZE":
			return nil, fmt.Errorf("gift: LUT3D: line %d: 1D LUTs are not supported", lineNum)

		case "LUT_3D_SIZE":
			if hasSize {
				return nil, fmt.Errorf("gift: LUT3D: line %d: duplicate LUT_3D_SIZE", lineNum)
			}
			if len(fields) != 2 {
				return nil, fmt.Errorf("gift: LUT3D: line %d: LUT_3D_SIZE requires exactly one value", lineNum)
			}
			n, err := strconv.Atoi(fields[1])
			if err != nil {
				return nil, fmt.Errorf("gift: LUT3D: line %d: invalid LUT_3D_SIZE %q", lineNum, fields[1])
			}
			if n < 2 || n > maxLUT3DSize {
				return nil, fmt.Errorf("gift: LUT3D: line %d: invalid LUT_3D_SIZE %d (must be between 2 and %d)",
					lineNum, n, maxLUT3DSize)
			}
			size = n
			hasSize = true

		case "DOMAIN_MIN", "DOMAIN_MAX":
			if keyword == "DOMAIN_MIN" && hasDomainMin {
				return nil, fmt.Errorf("gift: LUT3D: line %d: duplicate DOMAIN_MIN", lineNum)
			}
			if keyword == "DOMAIN_MAX" && hasDomainMax {
				return nil, fmt.Errorf("gift: LUT3D: line %d: duplicate DOMAIN_MAX", lineNum)
			}
			domain, err := parseDomain(fields)
			if err != nil {
				return nil, fmt.Errorf("gift: LUT3D: line %d: %v", lineNum, err)
			}
			if keyword == "DOMAIN_MIN" {
				domainMin = domain
				hasDomainMin = true
			} else {
				domainMax = domain
				hasDomainMax = true
			}

		default:
			if _, err := strconv.ParseFloat(fields[0], 32); err != nil {
				return nil, fmt.Errorf("gift: LUT3D: line %d: unsupported keyword %q", lineNum, fields[0])
			}
			if len(fields) != 3 {
				return nil, fmt.Errorf("gift: LUT3D: line %d: expected 3 RGB values, got %d", lineNum, len(fields))
			}
			var row [3]float32
			for i := 0; i < 3; i++ {
				f, err := strconv.ParseFloat(fields[i], 32)
				if err != nil {
					return nil, fmt.Errorf("gift: LUT3D: line %d: invalid RGB value %q", lineNum, fields[i])
				}
				if !isFinite32(float32(f)) {
					return nil, fmt.Errorf("gift: LUT3D: line %d: non-finite RGB value %q", lineNum, fields[i])
				}
				row[i] = float32(f)
			}
			numRows++
			if hasSize && numRows > size*size*size {
				return nil, fmt.Errorf("gift: LUT3D: line %d: too many data rows for LUT_3D_SIZE %d (expected %d)",
					lineNum, size, size*size*size)
			}
			data = append(data, row[0], row[1], row[2])
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("gift: LUT3D: read error: %v", err)
	}
	if !hasSize {
		return nil, fmt.Errorf("gift: LUT3D: missing LUT_3D_SIZE")
	}
	if err := validateLUT3DDomain(size, domainMin, domainMax); err != nil {
		return nil, fmt.Errorf("gift: LUT3D: %v", err)
	}
	if numRows != size*size*size {
		return nil, fmt.Errorf("gift: LUT3D: expected %d data rows, got %d", size*size*size, numRows)
	}

	return newLUT3D(size, domainMin, domainMax, data)
}

// parseDomain parses the three numbers of a DOMAIN_MIN or DOMAIN_MAX
// directive. The directive keyword itself is fields[0].
func parseDomain(fields []string) ([3]float32, error) {
	var domain [3]float32
	if len(fields) != 4 {
		return domain, fmt.Errorf("%s requires exactly three values", strings.ToUpper(fields[0]))
	}
	for i := 0; i < 3; i++ {
		f, err := strconv.ParseFloat(fields[i+1], 32)
		if err != nil {
			return domain, fmt.Errorf("invalid domain value %q", fields[i+1])
		}
		if !isFinite32(float32(f)) {
			return domain, fmt.Errorf("domain values must be finite, got %q", fields[i+1])
		}
		domain[i] = float32(f)
	}
	return domain, nil
}

// LoadLUT3D reads a 3D LUT from the named .cube file.
func LoadLUT3D(filename string) (*LUT3D, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("gift: LUT3D: %v", err)
	}
	defer f.Close()

	lut, err := ParseLUT3D(f)
	if err != nil {
		return nil, err
	}
	return lut, nil
}

// clampLUTCoord clamps a grid coordinate to the range [0, max]. NaN values
// are clamped to zero.
func clampLUTCoord(x, max float32) float32 {
	if !(x > 0) {
		return 0
	}
	if x > max {
		return max
	}
	return x
}

// Lookup maps an RGB color through the LUT. Channel values are in the range
// [0, 1]. Colors inside the declared domain are trilinearly interpolated
// between the eight surrounding grid entries; colors outside the domain are
// clamped to the nearest domain edge before the lookup. The result is not
// clamped to [0, 1], so it can be used for further processing.
func (l *LUT3D) Lookup(r, g, b float32) (float32, float32, float32) {
	if l == nil {
		return r, g, b
	}
	return l.sample(r, g, b)
}

// sample maps an RGB color through the LUT using trilinear interpolation.
// Input values outside the declared domain are clamped to the domain edges.
func (l *LUT3D) sample(r, g, b float32) (float32, float32, float32) {
	n := l.size
	nmax := float32(n - 1)

	// Convert the input to grid coordinates. Values outside the domain are
	// clamped to the nearest edge of the grid.
	pr := clampLUTCoord((r-l.domainMin[0])*l.scale[0], nmax)
	pg := clampLUTCoord((g-l.domainMin[1])*l.scale[1], nmax)
	pb := clampLUTCoord((b-l.domainMin[2])*l.scale[2], nmax)

	r0 := int(pr)
	if r0 > n-2 {
		r0 = n - 2
	}
	g0 := int(pg)
	if g0 > n-2 {
		g0 = n - 2
	}
	b0 := int(pb)
	if b0 > n-2 {
		b0 = n - 2
	}

	fr := pr - float32(r0)
	fg := pg - float32(g0)
	fb := pb - float32(b0)

	// The red index changes fastest, then green, then blue.
	const channels = 3
	rowStride := n * channels
	sliceStride := n * n * channels
	base := (b0*n+g0)*n*channels + r0*channels

	d := l.data
	var out [channels]float32
	for c := 0; c < channels; c++ {
		v000 := d[base+c]
		v100 := d[base+channels+c]
		v010 := d[base+rowStride+c]
		v110 := d[base+rowStride+channels+c]
		v001 := d[base+sliceStride+c]
		v101 := d[base+sliceStride+channels+c]
		v011 := d[base+sliceStride+rowStride+c]
		v111 := d[base+sliceStride+rowStride+channels+c]

		v00 := v000 + (v100-v000)*fr
		v10 := v010 + (v110-v010)*fr
		v01 := v001 + (v101-v001)*fr
		v11 := v011 + (v111-v011)*fr

		v0 := v00 + (v10-v00)*fg
		v1 := v01 + (v11-v01)*fg

		out[c] = v0 + (v1-v0)*fb
	}

	return out[0], out[1], out[2]
}

type colorLUTFilter struct {
	lut *LUT3D
}

func (p *colorLUTFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	dstBounds = image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
	return
}

func (p *colorLUTFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	if options == nil {
		options = &defaultOptions
	}

	srcb := src.Bounds()
	dstb := dst.Bounds()
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)

	parallelize(options.Parallelization, srcb.Min.Y, srcb.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := srcb.Min.X; x < srcb.Max.X; x++ {
				px := pixGetter.getPixel(x, y)
				px.r, px.g, px.b = p.lut.sample(px.r, px.g, px.b)
				pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
			}
		}
	})
}

// ColorLUT creates a filter that maps the colors of an image through the
// given 3D LUT. Colors inside the LUT domain are trilinearly interpolated,
// and colors outside the domain are clamped to its edges. The alpha channel
// is not modified.
//
// If lut is nil, the filter leaves the image unchanged.
//
// Example:
//
//	lut, err := gift.LoadLUT3D("look.cube")
//	if err != nil {
//		log.Fatal(err)
//	}
//	g := gift.New(gift.ColorLUT(lut))
//	dst := image.NewRGBA(g.Bounds(src.Bounds()))
//	g.Draw(dst, src)
func ColorLUT(lut *LUT3D) Filter {
	if lut == nil {
		return &copyimageFilter{}
	}
	return &colorLUTFilter{lut: lut}
}
