package gift

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testLUT3D creates a LUT3D of the given size by evaluating fn at every grid
// point. The entries are stored in .cube order: the red index changes
// fastest, then green, then blue.
func testLUT3D(size int, fn func(r, g, b float32) (float32, float32, float32)) *LUT3D {
	data := make([]float32, 0, size*size*size*3)
	for k := 0; k < size; k++ {
		for j := 0; j < size; j++ {
			for i := 0; i < size; i++ {
				r := float32(i) / float32(size-1)
				g := float32(j) / float32(size-1)
				b := float32(k) / float32(size-1)
				rr, gg, bb := fn(r, g, b)
				data = append(data, rr, gg, bb)
			}
		}
	}
	lut, err := newLUT3D(size, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, data)
	if err != nil {
		panic(err)
	}
	return lut
}

func identityFn(r, g, b float32) (float32, float32, float32) { return r, g, b }

func invertFn(r, g, b float32) (float32, float32, float32) { return 1 - r, 1 - g, 1 - b }

func closeEnough(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-6
}

func TestLUT3DGridAndMidpoints(t *testing.T) {
	const size = 4
	lut := testLUT3D(size, func(r, g, b float32) (float32, float32, float32) {
		// A deliberately non-linear mapping. A correct trilinear sampler must
		// reproduce the node values at the grid points and the average of the
		// eight surrounding nodes at the center of each cell.
		return r * r * 0.5, g * g * g, 1 - b*b
	})

	d := lut.data
	at := func(i, j, k int) (float32, float32, float32) {
		idx := ((k*size+j)*size + i) * 3
		return d[idx], d[idx+1], d[idx+2]
	}

	// Grid points must be reproduced exactly.
	for k := 0; k < size; k++ {
		for j := 0; j < size; j++ {
			for i := 0; i < size; i++ {
				r := float32(i) / float32(size-1)
				g := float32(j) / float32(size-1)
				b := float32(k) / float32(size-1)
				gotR, gotG, gotB := lut.sample(r, g, b)
				wantR, wantG, wantB := at(i, j, k)
				if !closeEnough(gotR, wantR) || !closeEnough(gotG, wantG) || !closeEnough(gotB, wantB) {
					t.Fatalf("grid point (%d,%d,%d): expected (%v,%v,%v), got (%v,%v,%v)",
						i, j, k, wantR, wantG, wantB, gotR, gotG, gotB)
				}
			}
		}
	}

	// Cell midpoints must be the uniform average of the eight cell corners.
	for k := 0; k < size-1; k++ {
		for j := 0; j < size-1; j++ {
			for i := 0; i < size-1; i++ {
				var wantR, wantG, wantB float32
				for dk := 0; dk <= 1; dk++ {
					for dj := 0; dj <= 1; dj++ {
						for di := 0; di <= 1; di++ {
							r, g, b := at(i+di, j+dj, k+dk)
							wantR += r
							wantG += g
							wantB += b
						}
					}
				}
				wantR /= 8
				wantG /= 8
				wantB /= 8

				r := (float32(i) + 0.5) / float32(size-1)
				g := (float32(j) + 0.5) / float32(size-1)
				b := (float32(k) + 0.5) / float32(size-1)
				gotR, gotG, gotB := lut.sample(r, g, b)
				if !closeEnough(gotR, wantR) || !closeEnough(gotG, wantG) || !closeEnough(gotB, wantB) {
					t.Fatalf("cell midpoint (%d,%d,%d): expected (%v,%v,%v), got (%v,%v,%v)",
						i, j, k, wantR, wantG, wantB, gotR, gotG, gotB)
				}
			}
		}
	}
}

func TestLUT3DTrilinearVsNearest(t *testing.T) {
	// A 2x2x2 LUT of the product r*g. At (0.25, 0.25, 0.25) nearest sampling
	// would return 0, while trilinear interpolation of the eight corners
	// returns 0.0625: the corners (1,1,0), (1,0,1), and (0,1,1) each
	// contribute 0.046875 and (1,1,1) contributes 0.015625.
	lut := testLUT3D(2, func(r, g, b float32) (float32, float32, float32) {
		return r * g, g * b, b * r
	})
	r, g, b := lut.sample(0.25, 0.25, 0.25)
	if !closeEnough(r, 0.0625) || !closeEnough(g, 0.0625) || !closeEnough(b, 0.0625) {
		t.Fatalf("expected trilinear result 0.0625 for all channels, got (%v,%v,%v)", r, g, b)
	}
}

func TestLUT3DIdentityAndInvert(t *testing.T) {
	for _, size := range []int{2, 3, 17, 65} {
		identity := testLUT3D(size, identityFn)
		invert := testLUT3D(size, invertFn)
		for i := 0; i <= 32; i++ {
			v := float32(i) / 32
			for _, c := range []float32{0, 0.125, 0.25, v, 1 - v, 1} {
				r, g, b := identity.sample(c, 1-c, c)
				if !closeEnough(r, c) || !closeEnough(g, 1-c) || !closeEnough(b, c) {
					t.Fatalf("identity size %d: input (%v,%v,%v), got (%v,%v,%v)", size, c, 1-c, c, r, g, b)
				}
				r, g, b = invert.sample(c, 1-c, c)
				if !closeEnough(r, 1-c) || !closeEnough(g, c) || !closeEnough(b, 1-c) {
					t.Fatalf("invert size %d: input (%v,%v,%v), got (%v,%v,%v)", size, c, 1-c, c, r, g, b)
				}
			}
		}
	}
}

func TestLUT3DLookup(t *testing.T) {
	lut := testLUT3D(5, func(r, g, b float32) (float32, float32, float32) {
		return r, 1 - g, b * b
	})
	for _, c := range []float32{0, 0.1, 0.5, 0.9, 1} {
		r, g, b := lut.Lookup(c, c, c)
		if !closeEnough(r, c) || !closeEnough(g, 1-c) {
			t.Fatalf("Lookup(%v): got (%v,%v,%v)", c, r, g, b)
		}
		// Lookup and the internal sampler must agree.
		sr, sg, sb := lut.sample(c, c, c)
		if r != sr || g != sg || b != sb {
			t.Fatalf("Lookup(%v) does not match sample: (%v,%v,%v) != (%v,%v,%v)",
				c, r, g, b, sr, sg, sb)
		}
	}

	// Grid points must reproduce the sampled function exactly.
	for i := 0; i < 5; i++ {
		c := float32(i) / 4
		_, _, b := lut.Lookup(c, c, c)
		if !closeEnough(b, c*c) {
			t.Fatalf("Lookup grid point %v: expected %v, got %v", c, c*c, b)
		}
	}

	// A nil LUT must return its input unchanged.
	var nilLUT *LUT3D
	r, g, b := nilLUT.Lookup(0.1, 0.2, 0.3)
	if r != 0.1 || g != 0.2 || b != 0.3 {
		t.Fatalf("nil Lookup changed its input: (%v,%v,%v)", r, g, b)
	}
}

func TestLUT3DDomain(t *testing.T) {
	// Domain [0.2, 0.8] on red, [0, 0.5] on green, [0.25, 1] on blue.
	domainMin := [3]float32{0.2, 0, 0.25}
	domainMax := [3]float32{0.8, 0.5, 1}
	data := make([]float32, 2*2*2*3)
	for k := 0; k < 2; k++ {
		for j := 0; j < 2; j++ {
			for i := 0; i < 2; i++ {
				idx := ((k*2+j)*2 + i) * 3
				data[idx+0] = float32(i)
				data[idx+1] = float32(j)
				data[idx+2] = float32(k)
			}
		}
	}
	lut, err := newLUT3D(2, domainMin, domainMax, data)
	if err != nil {
		t.Fatalf("newLUT3D failed: %v", err)
	}

	// Values inside the domain interpolate normally.
	r, g, b := lut.sample(0.5, 0.25, 0.625)
	if !closeEnough(r, 0.5) || !closeEnough(g, 0.5) || !closeEnough(b, 0.5) {
		t.Fatalf("domain midpoint: expected (0.5,0.5,0.5), got (%v,%v,%v)", r, g, b)
	}

	// Values outside the domain clamp to its edges.
	r, g, b = lut.sample(-10, -10, -10)
	if !closeEnough(r, 0) || !closeEnough(g, 0) || !closeEnough(b, 0) {
		t.Fatalf("below domain: expected (0,0,0), got (%v,%v,%v)", r, g, b)
	}
	r, g, b = lut.sample(10, 10, 10)
	if !closeEnough(r, 1) || !closeEnough(g, 1) || !closeEnough(b, 1) {
		t.Fatalf("above domain: expected (1,1,1), got (%v,%v,%v)", r, g, b)
	}
	r, g, b = lut.sample(0.1, 0.1, 0.1)
	if !closeEnough(r, 0) || !closeEnough(g, 0.2) || !closeEnough(b, 0) {
		t.Fatalf("partially below domain: expected (0,0.2,0), got (%v,%v,%v)", r, g, b)
	}
}

func TestParseLUT3D(t *testing.T) {
	const cube = `# Created by a color grading tool
TITLE "Test Look"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0 # last entry
`
	lut, err := ParseLUT3D(strings.NewReader(cube))
	if err != nil {
		t.Fatalf("ParseLUT3D failed: %v", err)
	}
	if lut.Size() != 2 {
		t.Fatalf("expected size 2, got %d", lut.Size())
	}
	dmin, dmax := lut.Domain()
	if dmin != [3]float32{0, 0, 0} || dmax != [3]float32{1, 1, 1} {
		t.Fatalf("unexpected domain: %v %v", dmin, dmax)
	}
	r, g, b := lut.sample(1, 0, 1)
	if !closeEnough(r, 1) || !closeEnough(g, 0) || !closeEnough(b, 1) {
		t.Fatalf("unexpected sample at grid point: (%v,%v,%v)", r, g, b)
	}
}

func TestParseLUT3DDefaultsAndFormatting(t *testing.T) {
	// DOMAIN_MIN/MAX omitted: the default [0,1] domain must be used. The
	// directive keywords are case-insensitive, values may use exponents and
	// leading '+' signs, lines may end with CRLF, and a UTF-8 BOM may be
	// present at the start of the file.
	cube := "\ufefflut_3d_size 2\r\n" +
		"domain_min 0 0 0\r\n" +
		"TITLE Some Look\r\n" +
		"+0 0 0\r\n1 0 0\r\n0 1 0\r\n1 1 0\r\n0 0 1\r\n1 0 1\r\n0 1 1\r\n" +
		"1e0 1E0 1.0\r\n"
	lut, err := ParseLUT3D(strings.NewReader(cube))
	if err != nil {
		t.Fatalf("ParseLUT3D failed: %v", err)
	}
	dmin, dmax := lut.Domain()
	if dmin != [3]float32{0, 0, 0} || dmax != [3]float32{1, 1, 1} {
		t.Fatalf("unexpected default domain: %v %v", dmin, dmax)
	}
	if lut.Size() != 2 {
		t.Fatalf("expected size 2, got %d", lut.Size())
	}
}

func TestParseLUT3DErrors(t *testing.T) {
	validRow := "0 0 0\n"
	tests := []struct {
		name string
		cube string
		want string
	}{
		{"empty", "", "missing LUT_3D_SIZE"},
		{"no size", validRow, "missing LUT_3D_SIZE"},
		{"1d", "LUT_1D_SIZE 64\n" + validRow, "1D LUTs are not supported"},
		{"size zero", "LUT_3D_SIZE 0\n", "invalid LUT_3D_SIZE 0"},
		{"size one", "LUT_3D_SIZE 1\n", "invalid LUT_3D_SIZE 1"},
		{"size too large", "LUT_3D_SIZE 257\n", "invalid LUT_3D_SIZE 257"},
		{"size negative", "LUT_3D_SIZE -2\n", "invalid LUT_3D_SIZE -2"},
		{"size not a number", "LUT_3D_SIZE abc\n", "invalid LUT_3D_SIZE"},
		{"size missing value", "LUT_3D_SIZE\n", "requires exactly one value"},
		{"size extra value", "LUT_3D_SIZE 2 3\n", "requires exactly one value"},
		{"duplicate size", "LUT_3D_SIZE 2\nLUT_3D_SIZE 2\n", "duplicate LUT_3D_SIZE"},
		{"missing rows", "LUT_3D_SIZE 2\n" + validRow, "expected 8 data rows, got 1"},
		{"too many rows", "LUT_3D_SIZE 2\n" + strings.Repeat(validRow, 9), "too many data rows"},
		{"short row", "LUT_3D_SIZE 2\n0 0\n", "expected 3 RGB values"},
		{"long row", "LUT_3D_SIZE 2\n0 0 0 0\n", "expected 3 RGB values"},
		{"bad number", "LUT_3D_SIZE 2\n0 x 0\n", "invalid RGB value"},
		{"nan", "LUT_3D_SIZE 2\nNaN 0 0\n", "non-finite RGB value"},
		{"inf", "LUT_3D_SIZE 2\n+Inf 0 0\n", "non-finite RGB value"},
		{"unknown keyword", "LUT_3D_SIZE 2\nNONSENSE 1 2 3\n", "unsupported keyword"},
		{"domain min short", "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0\n", "requires exactly three values"},
		{"domain max long", "LUT_3D_SIZE 2\nDOMAIN_MAX 0 0 0 0\n", "requires exactly three values"},
		{"domain min bad", "LUT_3D_SIZE 2\nDOMAIN_MIN 0 x 0\n", "invalid domain value"},
		{"domain min nan", "LUT_3D_SIZE 2\nDOMAIN_MIN NaN 0 0\n", "must be finite"},
		{"duplicate domain min", "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MIN 0 0 0\n", "duplicate DOMAIN_MIN"},
		{"duplicate domain max", "LUT_3D_SIZE 2\nDOMAIN_MAX 1 1 1\nDOMAIN_MAX 1 1 1\n", "duplicate DOMAIN_MAX"},
		{"inverted domain", "LUT_3D_SIZE 2\nDOMAIN_MIN 1 0 0\nDOMAIN_MAX 0 1 1\n", "invalid domain"},
		{"domain min above default max", "LUT_3D_SIZE 2\nDOMAIN_MIN 2 0 0\n", "invalid domain"},
		{"equal domain", "LUT_3D_SIZE 2\nDOMAIN_MIN 1 0 0\nDOMAIN_MAX 1 1 1\n", "invalid domain"},
	}
	for _, d := range tests {
		_, err := ParseLUT3D(strings.NewReader(d.cube))
		if err == nil {
			t.Errorf("%s: expected an error, got nil", d.name)
			continue
		}
		if !strings.Contains(err.Error(), d.want) {
			t.Errorf("%s: expected error containing %q, got %q", d.name, d.want, err)
		}
	}
}

func TestNewLUT3D(t *testing.T) {
	data := make([]float32, 8*3)
	for k := 0; k < 2; k++ {
		for j := 0; j < 2; j++ {
			for i := 0; i < 2; i++ {
				idx := ((k*2+j)*2 + i) * 3
				data[idx+0] = float32(i)
				data[idx+1] = float32(j)
				data[idx+2] = float32(k)
			}
		}
	}
	lut, err := NewLUT3D(2, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, data)
	if err != nil {
		t.Fatalf("NewLUT3D failed: %v", err)
	}
	if lut.Size() != 2 {
		t.Fatalf("expected size 2, got %d", lut.Size())
	}

	// The caller may reuse the data slice: the LUT must keep its own copy.
	for i := range data {
		data[i] = -1
	}
	r, g, b := lut.sample(1, 0, 1)
	if !closeEnough(r, 1) || !closeEnough(g, 0) || !closeEnough(b, 1) {
		t.Fatalf("LUT data was aliased to the caller slice: (%v,%v,%v)", r, g, b)
	}
}

func TestNewLUT3DValidation(t *testing.T) {
	good := make([]float32, 8*3)
	if _, err := NewLUT3D(2, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, good); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	badData := make([]float32, len(good))
	badData[5] = float32(math.Inf(1))
	if _, err := NewLUT3D(2, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, badData); err == nil {
		t.Error("expected an error for non-finite data")
	}
	if _, err := NewLUT3D(2, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, good[:10]); err == nil {
		t.Error("expected an error for wrong data length")
	}
	if _, err := NewLUT3D(2, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, nil); err == nil {
		t.Error("expected an error for nil data")
	}
	if _, err := NewLUT3D(2, [3]float32{0, 0, 0}, [3]float32{1, 1, 0}, good); err == nil {
		t.Error("expected an error for a zero-width domain")
	}
	if _, err := NewLUT3D(2, [3]float32{0, 0, 0}, [3]float32{1, 1, float32(math.NaN())}, good); err == nil {
		t.Error("expected an error for a NaN domain")
	}
	if _, err := NewLUT3D(1, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, nil); err == nil {
		t.Error("expected an error for size 1")
	}
}

func TestLoadLUT3D(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "look.cube")
	content := "LUT_3D_SIZE 2\n" +
		"0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	lut, err := LoadLUT3D(path)
	if err != nil {
		t.Fatalf("LoadLUT3D failed: %v", err)
	}
	if lut.Size() != 2 {
		t.Fatalf("expected size 2, got %d", lut.Size())
	}

	if _, err := LoadLUT3D(filepath.Join(dir, "missing.cube")); err == nil {
		t.Error("expected an error for a missing file")
	}

	if err := os.WriteFile(path, []byte("LUT_3D_SIZE 2\n"), 0o600); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	if _, err := LoadLUT3D(path); err == nil {
		t.Error("expected an error for an incomplete file")
	}
}

func TestLUT3DNil(t *testing.T) {
	var lut *LUT3D
	if lut.Size() != 0 {
		t.Errorf("expected nil LUT size 0, got %d", lut.Size())
	}
	dmin, dmax := lut.Domain()
	if dmin != [3]float32{} || dmax != [3]float32{} {
		t.Errorf("unexpected nil LUT domain: %v %v", dmin, dmax)
	}

	src := testImageNRGBA()
	dst := image.NewNRGBA(image.Rect(0, 0, src.Bounds().Dx(), src.Bounds().Dy()))
	New(ColorLUT(nil)).Draw(dst, src)
	for i := range src.Pix {
		if dst.Pix[i] != src.Pix[i] {
			t.Fatalf("ColorLUT(nil) changed pixel data at %d", i)
		}
	}
}

func testImageNRGBA() *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, 8, 6))
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x * 31),
				G: uint8(y * 47),
				B: uint8((x*17 + y*29) & 0xff),
				A: uint8(255 - (x*13+y*11)%200),
			})
		}
	}
	return img
}

// newTestImage creates a fresh draw.Image of the given kind, filled with a
// deterministic pattern that includes partially transparent and fully
// transparent pixels.
func newTestImage(kind string) draw.Image {
	var img draw.Image
	switch kind {
	case "NRGBA":
		img = image.NewNRGBA(image.Rect(0, 0, 8, 6))
	case "RGBA":
		img = image.NewRGBA(image.Rect(0, 0, 8, 6))
	case "NRGBA64":
		img = image.NewNRGBA64(image.Rect(0, 0, 8, 6))
	case "RGBA64":
		img = image.NewRGBA64(image.Rect(0, 0, 8, 6))
	case "Gray":
		img = image.NewGray(image.Rect(0, 0, 8, 6))
	case "Gray16":
		img = image.NewGray16(image.Rect(0, 0, 8, 6))
	default:
		panic("unknown image kind: " + kind)
	}
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			a := uint8(255)
			if x%4 == 0 {
				a = 0
			} else if x%2 == 0 {
				a = 128
			}
			img.Set(x, y, color.NRGBA{
				R: uint8(x * 31),
				G: uint8(y * 47),
				B: uint8((x*17 + y*29) & 0xff),
				A: a,
			})
		}
	}
	return img
}

func TestColorLUTImage(t *testing.T) {
	identity := testLUT3D(33, identityFn)
	invert := testLUT3D(33, invertFn)

	kinds := []string{"NRGBA", "RGBA", "NRGBA64", "RGBA64", "Gray", "Gray16"}
	for _, kind := range kinds {
		src := newTestImage(kind)
		srcCopy := newTestImage(kind)

		// The identity LUT must preserve every pixel, including alpha. The
		// tolerance of one allows for the rounding done by the 8-bit
		// conversion after interpolation.
		dst := newTestImage(kind)
		New(ColorLUT(identity)).Draw(dst, src)
		for y := dst.Bounds().Min.Y; y < dst.Bounds().Max.Y; y++ {
			for x := dst.Bounds().Min.X; x < dst.Bounds().Max.X; x++ {
				got := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
				want := color.NRGBAModel.Convert(srcCopy.At(x, y)).(color.NRGBA)
				if got.A != want.A {
					t.Fatalf("%s identity: alpha changed at (%d,%d): %d -> %d",
						kind, x, y, want.A, got.A)
				}
				if !withinTolerance(got.R, want.R, 1) ||
					!withinTolerance(got.G, want.G, 1) ||
					!withinTolerance(got.B, want.B, 1) {
					t.Errorf("%s identity: pixel (%d,%d) changed: %v -> %v",
						kind, x, y, want, got)
				}
			}
		}

		// The inverting LUT must negate the color channels while leaving the
		// alpha channel untouched. Premultiplied image types lose the color
		// of fully transparent pixels, so only the alpha is checked there.
		dst = newTestImage(kind)
		New(ColorLUT(invert)).Draw(dst, src)
		for y := dst.Bounds().Min.Y; y < dst.Bounds().Max.Y; y++ {
			for x := dst.Bounds().Min.X; x < dst.Bounds().Max.X; x++ {
				got := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
				want := color.NRGBAModel.Convert(srcCopy.At(x, y)).(color.NRGBA)
				if got.A != want.A {
					t.Fatalf("%s invert: alpha changed at (%d,%d): %d -> %d", kind, x, y, want.A, got.A)
				}
				if want.A == 0 {
					continue
				}
				if !withinTolerance(got.R, 255-want.R, 2) ||
					!withinTolerance(got.G, 255-want.G, 2) ||
					!withinTolerance(got.B, 255-want.B, 2) {
					t.Errorf("%s invert: pixel (%d,%d): expected ~%v, got %v",
						kind, x, y, color.NRGBA{255 - want.R, 255 - want.G, 255 - want.B, want.A}, got)
				}
			}
		}

		// The source image must not be modified.
		for y := src.Bounds().Min.Y; y < src.Bounds().Max.Y; y++ {
			for x := src.Bounds().Min.X; x < src.Bounds().Max.X; x++ {
				if !colorsEqual(src.At(x, y), srcCopy.At(x, y)) {
					t.Errorf("%s: source image was modified at (%d,%d)", kind, x, y)
				}
			}
		}
	}
}

// colorsEqual reports whether two colors are identical in 16-bit RGBA form.
func colorsEqual(c1, c2 color.Color) bool {
	r1, g1, b1, a1 := c1.RGBA()
	r2, g2, b2, a2 := c2.RGBA()
	return r1 == r2 && g1 == g2 && b1 == b2 && a1 == a2
}

// withinTolerance reports whether two 8-bit channel values differ by at most
// maxDiff, which allows for the rounding done by different pixel paths.
func withinTolerance(a, b uint8, maxDiff int) bool {
	diff := int(a) - int(b)
	if diff < 0 {
		diff = -diff
	}
	return diff <= maxDiff
}

func TestColorLUTDifferences(t *testing.T) {
	// The same source color drawn through different destination image types
	// must produce the same result.
	lut := testLUT3D(5, func(r, g, b float32) (float32, float32, float32) {
		return r * g, g * b, b * r
	})
	src := newTestImage("NRGBA")
	ref := image.NewNRGBA(image.Rect(0, 0, 8, 6))
	New(ColorLUT(lut)).Draw(ref, src)

	for _, kind := range []string{"NRGBA", "RGBA", "NRGBA64", "RGBA64"} {
		dst := newTestImage(kind)
		New(ColorLUT(lut)).Draw(dst, src)
		for y := 0; y < 6; y++ {
			for x := 0; x < 8; x++ {
				want := color.NRGBAModel.Convert(ref.At(x, y)).(color.NRGBA)
				got := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
				if got.A != want.A {
					t.Errorf("%s: alpha at (%d,%d): expected %d, got %d", kind, x, y, want.A, got.A)
					continue
				}
				// Fully transparent pixels are stored as zero by the
				// premultiplied image types, so their color is not comparable.
				if want.A == 0 {
					continue
				}
				if !withinTolerance(got.R, want.R, 2) ||
					!withinTolerance(got.G, want.G, 2) ||
					!withinTolerance(got.B, want.B, 2) {
					t.Errorf("%s: pixel (%d,%d): expected %v, got %v", kind, x, y, want, got)
				}
			}
		}
	}
}

func TestColorLUTFiltersChain(t *testing.T) {
	// ColorLUT must work as part of a filter chain with filters that precede
	// and follow it, including filters that change the image bounds.
	lut := testLUT3D(17, func(r, g, b float32) (float32, float32, float32) {
		return b, r, g // channel rotation
	})
	src := newTestImage("NRGBA")

	g := New(
		Crop(image.Rect(1, 1, 7, 5)),
		ColorLUT(lut),
		FlipHorizontal(),
	)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)
	if !dst.Bounds().Size().Eq(image.Pt(6, 4)) {
		t.Fatalf("unexpected bounds: %v", dst.Bounds())
	}

	// The chained result must equal the result of the individual steps in the
	// same order.
	step1 := image.NewNRGBA(image.Rect(0, 0, 6, 4))
	New(Crop(image.Rect(1, 1, 7, 5))).Draw(step1, src)
	step2 := image.NewNRGBA(image.Rect(0, 0, 6, 4))
	New(ColorLUT(lut)).Draw(step2, step1)
	step3 := image.NewNRGBA(image.Rect(0, 0, 6, 4))
	New(FlipHorizontal()).Draw(step3, step2)

	for y := 0; y < 4; y++ {
		for x := 0; x < 6; x++ {
			if !colorsEqual(dst.At(x, y), step3.At(x, y)) {
				t.Errorf("chain mismatch at (%d,%d): %v != %v", x, y, dst.At(x, y), step3.At(x, y))
			}
		}
	}
}

func TestColorLUTDrawNilOptions(t *testing.T) {
	// Filters must work when Draw is called with nil options, as when used
	// directly rather than through GIFT.
	lut := testLUT3D(2, invertFn)
	src := newTestImage("NRGBA")
	dst := image.NewNRGBA(src.Bounds())
	ColorLUT(lut).Draw(dst, src, nil)

	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			got := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
			want := color.NRGBAModel.Convert(src.At(x, y)).(color.NRGBA)
			if got.A != want.A {
				t.Fatalf("alpha changed at (%d,%d): %d -> %d", x, y, want.A, got.A)
			}
			if want.A == 0 {
				continue
			}
			if !withinTolerance(got.R, 255-want.R, 2) ||
				!withinTolerance(got.G, 255-want.G, 2) ||
				!withinTolerance(got.B, 255-want.B, 2) {
				t.Errorf("pixel (%d,%d): expected ~%v, got %v",
					x, y, color.NRGBA{255 - want.R, 255 - want.G, 255 - want.B, want.A}, got)
			}
		}
	}
}

func TestColorLUTBounds(t *testing.T) {
	lut := testLUT3D(2, identityFn)
	f := ColorLUT(lut)
	b := f.Bounds(image.Rect(10, 20, 40, 60))
	if !b.Eq(image.Rect(0, 0, 30, 40)) {
		t.Fatalf("unexpected bounds: %v", b)
	}
}

func TestColorLUTParallelization(t *testing.T) {
	lut := testLUT3D(11, func(r, g, b float32) (float32, float32, float32) {
		return r * r, b * g, 1 - b
	})

	// Use a large image so that the parallel workers really split the work.
	const w, h = 257, 193
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x * 7),
				G: uint8(y * 5),
				B: uint8(x*y*3 + y),
				A: uint8((x*3 + y*7) & 0xff),
			})
		}
	}

	dsts := make([]*image.NRGBA, 2)
	for i, parallel := range []bool{true, false} {
		g := New(ColorLUT(lut))
		g.SetParallelization(parallel)
		dst := image.NewNRGBA(g.Bounds(src.Bounds()))
		g.Draw(dst, src)
		dsts[i] = dst
	}

	for i := range dsts[0].Pix {
		if dsts[0].Pix[i] != dsts[1].Pix[i] {
			t.Fatalf("parallel and sequential results differ at byte %d: %d != %d",
				i, dsts[0].Pix[i], dsts[1].Pix[i])
		}
	}
}

func TestColorLUTNonZeroOrigin(t *testing.T) {
	lut := testLUT3D(9, invertFn)

	// The source and destination bounds do not start at (0, 0). The result
	// must be written at the destination Min point and map the source pixels
	// in order, exactly as with the other color filters.
	src := image.NewNRGBA(image.Rect(-3, 7, 5, 13))
	for y := src.Bounds().Min.Y; y < src.Bounds().Max.Y; y++ {
		for x := src.Bounds().Min.X; x < src.Bounds().Max.X; x++ {
			src.SetNRGBA(x, y, color.NRGBA{
				R: uint8((x + 3) * 29),
				G: uint8((y + 7) * 17),
				B: uint8((x*11 + y*13) & 0xff),
				A: uint8((x*5 + y*3) & 0xff),
			})
		}
	}

	dst := image.NewNRGBA(image.Rect(100, -50, 108, -44))
	New(ColorLUT(lut)).Draw(dst, src)

	want := image.NewNRGBA(image.Rect(0, 0, 8, 6))
	New(ColorLUT(lut)).Draw(want, src)
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			got := dst.At(dst.Bounds().Min.X+x, dst.Bounds().Min.Y+y)
			expected := want.At(x, y)
			if !colorsEqual(got, expected) {
				t.Errorf("pixel (%d,%d): expected %v, got %v", x, y, expected, got)
			}
		}
	}
}

func TestColorLUTExistingFiltersUnchanged(t *testing.T) {
	// The classic color filters must keep their documented behavior after
	// adding the LUT filter.
	src := image.NewGray(image.Rect(0, 0, 256, 1))
	for i := 0; i <= 255; i++ {
		src.Pix[i] = uint8(i)
	}

	dst := image.NewGray(src.Bounds())
	New(Invert()).Draw(dst, src)
	for i := 0; i <= 255; i++ {
		if dst.Pix[i] != 255-src.Pix[i] {
			t.Fatalf("Invert changed behavior at %d: expected %d got %d",
				i, 255-src.Pix[i], dst.Pix[i])
		}
	}

	dst = image.NewGray(src.Bounds())
	New(Gamma(1)).Draw(dst, src)
	for i := 0; i <= 255; i++ {
		if dst.Pix[i] != src.Pix[i] {
			t.Fatalf("Gamma(1) changed behavior at %d: expected %d got %d",
				i, src.Pix[i], dst.Pix[i])
		}
	}

	rgb := image.NewNRGBA(image.Rect(0, 0, 8, 6))
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			rgb.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x * 31),
				G: uint8(y * 47),
				B: uint8((x*17 + y*29) & 0xff),
				A: uint8((x*13 + y*11) & 0xff),
			})
		}
	}
	rgbDst := image.NewNRGBA(rgb.Bounds())
	New(Brightness(0)).Draw(rgbDst, rgb)
	for i := range rgb.Pix {
		if rgbDst.Pix[i] != rgb.Pix[i] {
			t.Fatalf("Brightness(0) changed behavior at byte %d", i)
		}
	}
}

func TestLoadLUT3DFiles(t *testing.T) {
	identity, err := LoadLUT3D("testdata/identity.cube")
	if err != nil {
		t.Fatalf("LoadLUT3D(identity) failed: %v", err)
	}
	invert, err := LoadLUT3D("testdata/invert.cube")
	if err != nil {
		t.Fatalf("LoadLUT3D(invert) failed: %v", err)
	}

	// The identity file must reproduce every possible 8-bit gray value, and
	// the invert file must negate it. Both must keep the alpha channel.
	for v := 0; v <= 255; v++ {
		c := float32(v) / 255

		r, g, b := identity.sample(c, c, c)
		if !closeEnough(r, c) || !closeEnough(g, c) || !closeEnough(b, c) {
			t.Fatalf("identity cube at %v: got (%v,%v,%v)", c, r, g, b)
		}

		r, g, b = invert.sample(c, c, c)
		if !closeEnough(r, 1-c) || !closeEnough(g, 1-c) || !closeEnough(b, 1-c) {
			t.Fatalf("invert cube at %v: got (%v,%v,%v)", c, r, g, b)
		}
	}
}

func TestColorLUTGolden(t *testing.T) {
	// Apply the LUT files to the standard test image and compare against the
	// existing Invert filter, which is the exact equivalent of invert.cube.
	src := loadImage(t, "testdata/src.png")
	identity, err := LoadLUT3D("testdata/identity.cube")
	if err != nil {
		t.Fatalf("LoadLUT3D(identity) failed: %v", err)
	}
	invert, err := LoadLUT3D("testdata/invert.cube")
	if err != nil {
		t.Fatalf("LoadLUT3D(invert) failed: %v", err)
	}

	// The identity LUT must leave the image unchanged.
	dstIdentity := image.NewNRGBA(New().Bounds(src.Bounds()))
	New(ColorLUT(identity)).Draw(dstIdentity, src)
	wantIdentity := loadImageNRGBA(t, "testdata/src.png")
	if !goldenEqual(dstIdentity, wantIdentity) {
		t.Fatal("identity LUT changed the test image")
	}

	// The inverting LUT must match the built-in Invert filter.
	dstLUT := image.NewNRGBA(New().Bounds(src.Bounds()))
	New(ColorLUT(invert)).Draw(dstLUT, src)
	dstInvert := image.NewNRGBA(New(Invert()).Bounds(src.Bounds()))
	New(Invert()).Draw(dstInvert, src)
	if !goldenEqual(dstLUT, dstInvert) {
		t.Fatal("inverting LUT does not match the Invert filter")
	}
}
