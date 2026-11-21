package gift

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// normalMapTestAt returns the decoded normal and alpha of an output pixel.
// The output is always an NRGBA image, so the channels are read directly.
// At().RGBA() would premultiply them for alpha values below 255.
func normalMapTestAt(img *image.NRGBA, x, y int) (nx, ny, nz, a float64) {
	i := img.PixOffset(x, y)
	r := img.Pix[i+0]
	g := img.Pix[i+1]
	b := img.Pix[i+2]
	a = float64(img.Pix[i+3]) / 255
	return decodeNormalComponent(r), decodeNormalComponent(g), decodeNormalComponent(b), a
}

// normalMapTestLen returns the length of a decoded normal.
func normalMapTestLen(nx, ny, nz float64) float64 {
	return math.Sqrt(nx*nx + ny*ny + nz*nz)
}

// normalMapTestCheck checks direction and unit length of one output pixel.
func normalMapTestCheck(t *testing.T, desc string, img *image.NRGBA, x, y int, ex, ey, ez, tol float64) {
	t.Helper()
	nx, ny, nz, _ := normalMapTestAt(img, x, y)
	l := normalMapTestLen(nx, ny, nz)
	if math.Abs(l-1) > tol {
		t.Errorf("%s: pixel (%d,%d): expected unit length, got %f (normal %v)", desc, x, y, l, [3]float64{nx, ny, nz})
	}
	if math.Abs(nx-ex) > tol || math.Abs(ny-ey) > tol || math.Abs(nz-ez) > tol {
		t.Errorf("%s: pixel (%d,%d): expected normal (%v,%v,%v), got (%v,%v,%v)", desc, x, y, ex, ey, ez, nx, ny, nz)
	}
}

// normalMapTestGray builds an NRGBA image from a height function that returns
// values in the 0..1 range and an alpha function that returns 0..1.
func normalMapTestGray(b image.Rectangle, height func(x, y int) float64, alpha func(x, y int) float64) *image.NRGBA {
	img := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			v := uint8(math.Round(height(x, y) * 255))
			a := uint8(math.Round(alpha(x, y) * 255))
			i := img.PixOffset(x, y)
			img.Pix[i+0] = v
			img.Pix[i+1] = v
			img.Pix[i+2] = v
			img.Pix[i+3] = a
		}
	}
	return img
}

func normalMapTestDraw(f Filter, src image.Image, parallel bool) *image.NRGBA {
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, &Options{Parallelization: parallel})
	return dst
}

func TestNormalMapHorizontalSlope(t *testing.T) {
	// A slope that rises along +X with a constant height step per pixel.
	// The step is large enough (10/255) that a sign error cannot hide inside
	// the tolerance.
	const step = 10.0 / 255
	h := func(x, y int) float64 { return 0.2 + float64(x)*step }

	for _, parallel := range []bool{false, true} {
		src := normalMapTestGray(image.Rect(0, 0, 8, 5), h, func(x, y int) float64 { return 1 })
		f := NormalMap(1, 1, 1, false, ClampEdge)
		dst := normalMapTestDraw(f, src, parallel)

		// The central difference of the heights is exactly one 8-bit step.
		// The tangent-space normal must lean towards -X and have no Y
		// component.
		n := step
		l := math.Sqrt(1 + n*n)
		ex, ey, ez := -n/l, 0.0, 1/l

		for y := 0; y < 5; y++ {
			for x := 0; x < 8; x++ {
				// At the clamped edges the one-sided difference is half of
				// the central one, so the slope is half as steep.
				if x == 0 || x == 7 {
					le := math.Sqrt(1 + (n/2)*(n/2))
					normalMapTestCheck(t, "horizontal slope edge", dst, x, y, -(n/2)/le, 0, 1/le, 0.01)
					continue
				}
				normalMapTestCheck(t, "horizontal slope", dst, x, y, ex, ey, ez, 0.01)
			}
		}
	}
}

func TestNormalMapVerticalSlope(t *testing.T) {
	// A slope that rises towards the bottom of the image (+image Y) with a
	// constant height step per pixel.
	const step = 10.0 / 255
	h := func(x, y int) float64 { return 0.2 + float64(y)*step }

	for _, parallel := range []bool{false, true} {
		src := normalMapTestGray(image.Rect(0, 0, 5, 8), h, func(x, y int) float64 { return 1 })

		n := step
		l := math.Sqrt(1 + n*n)

		// Image Y points down while tangent-space Y points up. A surface that
		// rises towards the bottom therefore tilts towards +Y in tangent
		// space, which is the OpenGL convention (positive green).
		dst := normalMapTestDraw(NormalMap(1, 1, 1, false, ClampEdge), src, parallel)
		for y := 1; y < 7; y++ {
			for x := 0; x < 5; x++ {
				normalMapTestCheck(t, "vertical slope opengl", dst, x, y, 0, n/l, 1/l, 0.01)
			}
		}

		// The DirectX convention flips the green channel.
		dstInv := normalMapTestDraw(NormalMap(1, 1, 1, true, ClampEdge), src, parallel)
		for y := 1; y < 7; y++ {
			for x := 0; x < 5; x++ {
				normalMapTestCheck(t, "vertical slope directx", dstInv, x, y, 0, -n/l, 1/l, 0.01)
			}
		}
	}
}

func TestNormalMapOrthogonalSlopes(t *testing.T) {
	// A surface that rises along +X and towards the bottom of the image at the
	// same time. The horizontal and vertical scales must combine into a
	// diagonal normal.
	const step = 10.0 / 255
	h := func(x, y int) float64 { return 0.2 + float64(x+y)*step }

	for _, parallel := range []bool{false, true} {
		src := normalMapTestGray(image.Rect(0, 0, 7, 7), h, func(x, y int) float64 { return 1 })
		dst := normalMapTestDraw(NormalMap(1, 1, 1, false, ClampEdge), src, parallel)

		n := step
		l := math.Sqrt(1 + 2*n*n)
		ex, ey, ez := -n/l, n/l, 1/l
		normalMapTestCheck(t, "orthogonal slopes", dst, 3, 3, ex, ey, ez, 0.01)

		// The X and Y scales are applied independently.
		lx := math.Sqrt(1 + n*n)
		dstX := normalMapTestDraw(NormalMap(1, 2, 0, false, ClampEdge), src, parallel)
		normalMapTestCheck(t, "x scale only", dstX, 3, 3, -2*n/lx, 0, 1/lx, 0.01)

		dstY := normalMapTestDraw(NormalMap(1, 0, 2, false, ClampEdge), src, parallel)
		normalMapTestCheck(t, "y scale only", dstY, 3, 3, 0, 2*n/lx, 1/lx, 0.01)
	}
}

func TestNormalMapFlat(t *testing.T) {
	// A constant image must produce the neutral normal (128, 128, 255) in
	// every pixel and keep alpha.
	for _, parallel := range []bool{false, true} {
		src := normalMapTestGray(image.Rect(0, 0, 6, 4), func(x, y int) float64 { return 0.37 }, func(x, y int) float64 { return 0.5 })
		dst := normalMapTestDraw(NormalMap(10, 1, 1, false, ClampEdge), src, parallel)

		for y := 0; y < 4; y++ {
			for x := 0; x < 6; x++ {
				normalMapTestCheck(t, "flat", dst, x, y, 0, 0, 1, 0.01)
				if a := dst.Pix[dst.PixOffset(x, y)+3]; a != 128 {
					t.Errorf("flat: pixel (%d,%d): expected alpha 128, got %d", x, y, a)
				}
			}
		}
	}

	// A single pixel image has no neighbors and is flat too.
	src := normalMapTestGray(image.Rect(0, 0, 1, 1), func(x, y int) float64 { return 1 }, func(x, y int) float64 { return 1 })
	dst := normalMapTestDraw(NormalMap(5, 1, 1, false, ClampEdge), src, false)
	normalMapTestCheck(t, "single pixel", dst, 0, 0, 0, 0, 1, 0.01)
}

func TestNormalMapStrengthAndScales(t *testing.T) {
	// A horizontal ramp with a fixed slope. Doubling the strength or the X
	// scale doubles the tangent of the tilt angle; the vector stays unit
	// length in every case.
	const step = 1.0 / 255
	h := func(x, y int) float64 { return 0.2 + float64(x)*step }
	src := normalMapTestGray(image.Rect(0, 0, 9, 3), h, func(x, y int) float64 { return 1 })

	check := func(desc string, f Filter, slope float64) {
		dst := normalMapTestDraw(f, src, false)
		l := math.Sqrt(1 + slope*slope)
		normalMapTestCheck(t, desc, dst, 4, 1, -slope/l, 0, 1/l, 0.01)
	}

	n := 1.0 / 255
	check("strength 1", NormalMap(1, 1, 1, false, ClampEdge), n)
	check("strength 2", NormalMap(2, 1, 1, false, ClampEdge), 2*n)
	check("x scale 2", NormalMap(1, 2, 1, false, ClampEdge), 2*n)
	check("strength and x scale 2", NormalMap(2, 2, 1, false, ClampEdge), 4*n)
	check("negative strength", NormalMap(-1, 1, 1, false, ClampEdge), -n)

	// A zero strength or a zero X scale on a purely horizontal ramp yields
	// the neutral normal.
	check("zero strength", NormalMap(0, 1, 1, false, ClampEdge), 0)
	check("zero x scale", NormalMap(1, 0, 1, false, ClampEdge), 0)
}

func TestNormalMapEdgeModes(t *testing.T) {
	// A horizontal ramp. At the left edge the clamped sampling repeats the
	// edge height, which halves the one-sided difference, while the wrapped
	// sampling reads the opposite edge and produces a steep gradient in the
	// other direction.
	const step = 1.0 / 255
	src := normalMapTestGray(image.Rect(0, 0, 6, 3), func(x, y int) float64 { return 0.1 + float64(x)*step }, func(x, y int) float64 { return 1 })

	for _, parallel := range []bool{false, true} {
		clampDst := normalMapTestDraw(NormalMap(1, 1, 1, false, ClampEdge), src, parallel)
		wrapDst := normalMapTestDraw(NormalMap(1, 1, 1, false, WrapEdge), src, parallel)

		// The clamped left edge: (h1 - h0) / 2, a gentle tilt towards -X.
		n := step
		l := math.Sqrt(1 + (n/2)*(n/2))
		normalMapTestCheck(t, "clamp left edge", clampDst, 0, 1, -(n/2)/l, 0, 1/l, 0.01)

		// The wrapped left edge: (h1 - h5) / 2, a steep tilt towards +X.
		slope := (1.0 - 5.0) * step * 0.5
		lw := math.Sqrt(1 + slope*slope)
		normalMapTestCheck(t, "wrap left edge", wrapDst, 0, 1, -slope/lw, 0, 1/lw, 0.01)

		// The clamped right edge: (h5 - h4) / 2, a gentle tilt towards -X.
		normalMapTestCheck(t, "clamp right edge", clampDst, 5, 1, -(n/2)/l, 0, 1/l, 0.01)

		// The interior is identical in both modes.
		for x := 1; x < 5; x++ {
			cx, cy, cz, _ := normalMapTestAt(clampDst, x, 1)
			wx, wy, wz, _ := normalMapTestAt(wrapDst, x, 1)
			if math.Abs(cx-wx) > 0.005 || math.Abs(cy-wy) > 0.005 || math.Abs(cz-wz) > 0.005 {
				t.Errorf("edge modes: interior pixel %d differs between clamp and wrap", x)
			}
		}
	}
}

func TestNormalMapTileable(t *testing.T) {
	// A height map based on a periodic function is tileable. With WrapEdge
	// the normal map must be tileable too: its edges match the continuation
	// across the image borders, unlike the clamped result.
	const w, h = 16, 16
	period := 8
	height := func(x, y int) float64 {
		return 0.5 + 0.25*math.Sin(2*math.Pi*float64(x)/float64(period)) + 0.2*math.Cos(2*math.Pi*float64(y)/float64(period))
	}
	src := normalMapTestGray(image.Rect(0, 0, w, h), height, func(x, y int) float64 { return 1 })

	for _, parallel := range []bool{false, true} {
		wrapDst := normalMapTestDraw(NormalMap(4, 1, 1, false, WrapEdge), src, parallel)
		clampDst := normalMapTestDraw(NormalMap(4, 1, 1, false, ClampEdge), src, parallel)

		// Every wrapped normal is unit length.
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				nx, ny, nz, _ := normalMapTestAt(wrapDst, x, y)
				if l := normalMapTestLen(nx, ny, nz); math.Abs(l-1) > 0.01 {
					t.Errorf("tileable: pixel (%d,%d): expected unit length, got %f", x, y, l)
				}
			}
		}

		// The wrapped normal map is seamless: the normal at the first column
		// equals the normal one period to the left, and the first row equals
		// the row one period above. Since the height is periodic with the
		// image size, sampling across the border is the same as sampling the
		// periodic continuation, so the columns at 0 and w-1 must be
		// consistent with the gradient between them.
		//
		// Check the horizontal seam by comparing the pixel at column 0 with
		// the pixel at column 0 shifted by one period, i.e. it must match the
		// gradient implied by the periodic height rather than being flattened
		// by clamping.
		{
			// Compute the expected normal of the seam pixel from the actual
			// quantized source heights, using periodic (wrapping) neighbors.
			// The height varies along both axes here, so all three components
			// are checked.
			sh := func(x, y int) float64 {
				return float64(src.Pix[src.PixOffset((x+w)%w, (y+h)%h)]) / 255
			}
			dhdx := (sh(1, 5) - sh(w-1, 5)) * 0.5 * 4
			dhdy := (sh(0, 6) - sh(0, 4)) * 0.5 * 4
			l := math.Sqrt(1 + dhdx*dhdx + dhdy*dhdy)
			ex, ey, ez := -dhdx/l, dhdy/l, 1/l
			normalMapTestCheck(t, "tileable horizontal seam", wrapDst, 0, 5, ex, ey, ez, 0.02)
		}

		// The clamped result flattens the seam: the first column of a
		// periodic height map has a smaller horizontal gradient than the
		// wrapping one wherever the slope is non-zero.
		differs := false
		for y := 0; y < h; y++ {
			cnx, _, _, _ := normalMapTestAt(clampDst, 0, y)
			wnx, _, _, _ := normalMapTestAt(wrapDst, 0, y)
			if math.Abs(cnx-wnx) > 0.01 {
				differs = true
				break
			}
		}
		if !differs {
			t.Error("tileable: clamp and wrap edge modes produced the same first column")
		}
	}
}

func TestNormalMapOnePixelWide(t *testing.T) {
	// A one pixel wide image has a zero horizontal gradient; only the
	// vertical slope contributes.
	h := func(x, y int) float64 { return float64(y) / 4 }
	src := normalMapTestGray(image.Rect(0, 0, 1, 6), h, func(x, y int) float64 { return 1 })

	for _, mode := range []EdgeMode{ClampEdge, WrapEdge} {
		for _, parallel := range []bool{false, true} {
			dst := normalMapTestDraw(NormalMap(1, 1, 1, false, mode), src, parallel)
			for y := 0; y < 6; y++ {
				nx, ny, nz, _ := normalMapTestAt(dst, 0, y)
				if math.Abs(nx) > 0.005 {
					t.Errorf("one pixel wide (mode %v): pixel %d: expected no X component, got %v", mode, y, nx)
				}
				if l := normalMapTestLen(nx, ny, nz); math.Abs(l-1) > 0.01 {
					t.Errorf("one pixel wide (mode %v): pixel %d: expected unit length, got %v", mode, y, l)
				}
			}
			// The interior rows have a non-zero vertical tilt.
			_, ny, _, _ := normalMapTestAt(dst, 0, 3)
			if math.Abs(ny) < 0.001 {
				t.Errorf("one pixel wide (mode %v): expected a vertical tilt, got %v", mode, ny)
			}
		}
	}

	// The same holds for a one pixel high image and the vertical axis.
	src2 := normalMapTestGray(image.Rect(0, 0, 6, 1), func(x, y int) float64 { return float64(x) / 4 }, func(x, y int) float64 { return 1 })
	dst2 := normalMapTestDraw(NormalMap(1, 1, 1, false, ClampEdge), src2, false)
	for x := 0; x < 6; x++ {
		_, ny, _, _ := normalMapTestAt(dst2, x, 0)
		if math.Abs(ny) > 0.005 {
			t.Errorf("one pixel high: pixel %d: expected no Y component, got %v", x, ny)
		}
	}
}

func TestNormalMapTransparency(t *testing.T) {
	// A horizontal ramp where the middle pixel is fully transparent but
	// carries a misleading hidden color. It must not influence its
	// neighbors' gradients, and its own output must be the neutral normal
	// with alpha preserved.
	for _, parallel := range []bool{false, true} {
		src := image.NewNRGBA(image.Rect(0, 0, 5, 1))
		heights := []uint8{64, 64, 200, 64, 64}
		for x := 0; x < 5; x++ {
			i := src.PixOffset(x, 0)
			src.Pix[i+0], src.Pix[i+1], src.Pix[i+2] = heights[x], heights[x], heights[x]
			src.Pix[i+3] = 0xff
		}
		// Transparent pixel with a bright red hidden color.
		i := src.PixOffset(2, 0)
		src.Pix[i+0], src.Pix[i+1], src.Pix[i+2] = 0xff, 0x00, 0x00
		src.Pix[i+3] = 0x00

		dst := normalMapTestDraw(NormalMap(2, 1, 1, false, ClampEdge), src, parallel)

		// The transparent pixel is the neutral normal with alpha 0.
		normalMapTestCheck(t, "transparent center", dst, 2, 0, 0, 0, 1, 0.01)
		if a := dst.Pix[dst.PixOffset(2, 0)+3]; a != 0 {
			t.Errorf("transparent center: expected alpha 0, got %d", a)
		}

		// Its neighbors are flat because both of their opaque samples have
		// the same height: the hidden bright color did not leak in.
		normalMapTestCheck(t, "left of transparent", dst, 1, 0, 0, 0, 1, 0.01)
		normalMapTestCheck(t, "right of transparent", dst, 3, 0, 0, 0, 1, 0.01)

		// A ramp of opaque pixels with a transparent hole at the left edge.
		// Pixel 0 is transparent, so pixel 1 uses its own height on the left
		// side: its horizontal difference is (h2 - h1) / 2, which is half of
		// the interior difference (h3 - h1) / 2 of pixel 2. It must therefore
		// still tilt towards -X, but less steeply than pixel 2.
		src2 := image.NewNRGBA(image.Rect(0, 0, 4, 1))
		for x := 0; x < 4; x++ {
			i := src2.PixOffset(x, 0)
			src2.Pix[i+0], src2.Pix[i+1], src2.Pix[i+2] = uint8(x*60), uint8(x*60), uint8(x*60)
			src2.Pix[i+3] = 0xff
		}
		src2.Pix[src2.PixOffset(0, 0)+3] = 0
		dst2 := normalMapTestDraw(NormalMap(2, 1, 1, false, ClampEdge), src2, parallel)
		nx1, _, nz1, _ := normalMapTestAt(dst2, 1, 0)
		nx2, _, _, _ := normalMapTestAt(dst2, 2, 0)
		if !(nx1 < 0) || !(nz1 > 0.9) {
			t.Errorf("transparent edge: expected pixel 1 to tilt towards -X, got (%v,%v)", nx1, nz1)
		}
		if !(nx1 > nx2) {
			t.Errorf("transparent edge: expected pixel 1 (%v) to tilt less than pixel 2 (%v)", nx1, nx2)
		}
	}
}

func TestNormalMapNonZeroBounds(t *testing.T) {
	// A source with non-zero bounds must produce the same result as the same
	// content with zero bounds, and the destination must start at (0, 0).
	height := func(x, y int) float64 { return 0.5 + 0.1*math.Sin(float64(x)) + 0.1*float64(y)/8 }

	plainSrc := normalMapTestGray(image.Rect(0, 0, 11, 9), height, func(x, y int) float64 { return 1 })

	// Fill an offset image with the same content.
	offsetSrc := image.NewNRGBA(image.Rect(-7, 4, 4, 13))
	for y := 0; y < 9; y++ {
		for x := 0; x < 11; x++ {
			// Content coordinates relative to the source bounds origin.
			v := uint8(math.Round(height(x, y) * 255))
			i := offsetSrc.PixOffset(-7+x, 4+y)
			offsetSrc.Pix[i+0], offsetSrc.Pix[i+1], offsetSrc.Pix[i+2] = v, v, v
			offsetSrc.Pix[i+3] = 0xff
		}
	}
	before := make([]uint8, len(offsetSrc.Pix))
	copy(before, offsetSrc.Pix)

	for _, parallel := range []bool{false, true} {
		f := NormalMap(3, 1, 1, false, WrapEdge)

		plainDst := image.NewNRGBA(f.Bounds(plainSrc.Bounds()))
		f.Draw(plainDst, plainSrc, &Options{Parallelization: parallel})
		if !plainDst.Bounds().Eq(image.Rect(0, 0, 11, 9)) {
			t.Fatalf("non-zero bounds: expected plain dst bounds (0,0)-(11,9), got %v", plainDst.Bounds())
		}

		offsetDst := image.NewNRGBA(f.Bounds(offsetSrc.Bounds()))
		f.Draw(offsetDst, offsetSrc, &Options{Parallelization: parallel})
		if !offsetDst.Bounds().Eq(image.Rect(0, 0, 11, 9)) {
			t.Fatalf("non-zero bounds: expected offset dst bounds (0,0)-(11,9), got %v", offsetDst.Bounds())
		}
		if !checkBoundsAndPix(plainDst.Bounds(), offsetDst.Bounds(), plainDst.Pix, offsetDst.Pix) {
			t.Error("non-zero bounds: result differs from the zero-bounds result")
		}
	}

	// The source image must not be modified.
	if !checkBoundsAndPix(offsetSrc.Bounds(), offsetSrc.Bounds(), before, offsetSrc.Pix) {
		t.Error("non-zero bounds: source image was modified")
	}
}

func TestNormalMapNonFiniteInputs(t *testing.T) {
	// A non-finite strength or the scale of one axis cannot describe the
	// slope of that axis, so that axis is disabled. The result must be a
	// finite unit vector, never a NaN or an infinity.
	src := normalMapTestGray(image.Rect(0, 0, 5, 4), func(x, y int) float64 { return float64(x+y) / 8 }, func(x, y int) float64 { return 1 })

	params := []struct {
		desc                     string
		strength, scaleX, scaleY float32
		// zeroX and zeroY mark the axes whose gradient must vanish.
		zeroX, zeroY bool
	}{
		{"NaN strength", float32(math.NaN()), 1, 1, true, true},
		{"+Inf strength", float32(math.Inf(1)), 1, 1, true, true},
		{"-Inf strength", float32(math.Inf(-1)), 1, 1, true, true},
		{"NaN scale X", 1, float32(math.NaN()), 1, true, false},
		{"+Inf scale X", 1, float32(math.Inf(1)), 1, true, false},
		{"-Inf scale X", 1, float32(math.Inf(-1)), 1, true, false},
		{"NaN scale Y", 1, 1, float32(math.NaN()), false, true},
		{"+Inf scale Y", 1, 1, float32(math.Inf(1)), false, true},
		{"-Inf scale Y", 1, 1, float32(math.Inf(-1)), false, true},
		{"all non-finite", float32(math.NaN()), float32(math.Inf(1)), float32(math.NaN()), true, true},
	}

	for _, p := range params {
		dst := normalMapTestDraw(NormalMap(p.strength, p.scaleX, p.scaleY, false, ClampEdge), src, false)
		for y := 0; y < 4; y++ {
			for x := 0; x < 5; x++ {
				nx, ny, nz, _ := normalMapTestAt(dst, x, y)
				if math.IsNaN(nx) || math.IsNaN(ny) || math.IsNaN(nz) ||
					math.IsInf(nx, 0) || math.IsInf(ny, 0) || math.IsInf(nz, 0) {
					t.Errorf("%s: pixel (%d,%d): non-finite normal (%v,%v,%v)", p.desc, x, y, nx, ny, nz)
					continue
				}
				if l := normalMapTestLen(nx, ny, nz); math.Abs(l-1) > 0.01 {
					t.Errorf("%s: pixel (%d,%d): expected unit length, got %v", p.desc, x, y, l)
				}
				if p.zeroX && math.Abs(nx) > 0.005 {
					t.Errorf("%s: pixel (%d,%d): expected no X component, got %v", p.desc, x, y, nx)
				}
				if p.zeroY && math.Abs(ny) > 0.005 {
					t.Errorf("%s: pixel (%d,%d): expected no Y component, got %v", p.desc, x, y, ny)
				}
			}
		}
	}

	// Huge finite parameters must still produce a finite unit vector.
	for _, v := range []float32{math.MaxFloat32, -math.MaxFloat32} {
		dst := normalMapTestDraw(NormalMap(v, v, v, false, ClampEdge), src, false)
		for y := 0; y < 4; y++ {
			for x := 0; x < 5; x++ {
				nx, ny, nz, _ := normalMapTestAt(dst, x, y)
				if l := normalMapTestLen(nx, ny, nz); math.Abs(l-1) > 0.01 {
					t.Errorf("huge parameters %v: pixel (%d,%d): expected unit length, got %v", v, x, y, l)
				}
			}
		}
	}
}

func TestNormalMapEmpty(t *testing.T) {
	// An empty source must not panic and must not produce output.
	f := NormalMap(1, 1, 1, false, ClampEdge)

	bounds := []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(3, 2, 3, 7),
		image.Rect(4, 1, 9, 1),
	}
	for _, b := range bounds {
		src := image.NewNRGBA(b)
		dst := image.NewNRGBA(f.Bounds(b))
		f.Draw(dst, src, nil)
		if !dst.Bounds().Empty() {
			t.Errorf("empty source %v: expected empty bounds, got %v", b, dst.Bounds())
		}
		if len(dst.Pix) != 0 {
			t.Errorf("empty source %v: expected no pixels, got %d", b, len(dst.Pix))
		}
	}

	// A nil options value must behave like the default options.
	src := normalMapTestGray(image.Rect(0, 0, 3, 3), func(x, y int) float64 { return 0.5 }, func(x, y int) float64 { return 1 })
	dst := image.NewNRGBA(f.Bounds(src.Bounds()))
	f.Draw(dst, src, nil)
	normalMapTestCheck(t, "nil options", dst, 1, 1, 0, 0, 1, 0.01)
}

func TestNormalMapChain(t *testing.T) {
	// The output is an ordinary RGB image and must pass through a filter
	// chain unchanged by a pass-through filter.
	for _, parallel := range []bool{false, true} {
		src := normalMapTestGray(image.Rect(0, 0, 6, 5), func(x, y int) float64 { return 0.6 }, func(x, y int) float64 { return 1 })

		direct := image.NewNRGBA(image.Rect(0, 0, 6, 5))
		New(NormalMap(2, 1, 1, false, ClampEdge)).Draw(direct, src)

		chained := image.NewNRGBA(image.Rect(0, 0, 6, 5))
		g := New(NormalMap(2, 1, 1, false, ClampEdge), Brightness(0))
		g.SetParallelization(parallel)
		g.Draw(chained, src)

		if !checkBoundsAndPix(direct.Bounds(), chained.Bounds(), direct.Pix, chained.Pix) {
			t.Error("chain: expected the output to pass through a pass-through filter unchanged")
		}
	}

	// Two color filters around the normal map invert and restore the
	// encoded channels, which shows that the result is an ordinary image.
	src := normalMapTestGray(image.Rect(0, 0, 8, 8), func(x, y int) float64 {
		return 0.5 + 0.2*math.Sin(float64(x)*0.7) + 0.1*float64(y)/8
	}, func(x, y int) float64 { return 1 })

	plain := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	New(NormalMap(3, 1, 1, false, WrapEdge)).Draw(plain, src)

	roundTrip := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	New(NormalMap(3, 1, 1, false, WrapEdge), Invert(), Invert()).Draw(roundTrip, src)

	for i := 0; i < len(plain.Pix); i++ {
		if d := int(plain.Pix[i]) - int(roundTrip.Pix[i]); d < -2 || d > 2 {
			t.Fatalf("chain: invert round trip changed component %d: %d vs %d", i, plain.Pix[i], roundTrip.Pix[i])
		}
	}
}

func TestNormalMapGrayAndRGBAInputs(t *testing.T) {
	// Gray, Gray16 and NRGBA64 sources must produce the same normals when
	// they carry the same heights, and the dimensions must be preserved.
	for _, parallel := range []bool{false, true} {
		height := func(x, y int) float64 { return 0.3 + float64(x)/16 + float64(y)/40 }

		gray := image.NewGray(image.Rect(0, 0, 8, 6))
		for y := 0; y < 6; y++ {
			for x := 0; x < 8; x++ {
				gray.Set(x, y, color.Gray{uint8(math.Round(height(x, y) * 255))})
			}
		}

		nrgba := normalMapTestGray(image.Rect(0, 0, 8, 6), height, func(x, y int) float64 { return 1 })

		grayDst := normalMapTestDraw(NormalMap(2, 1, 1, false, ClampEdge), gray, parallel)
		nrgbaDst := normalMapTestDraw(NormalMap(2, 1, 1, false, ClampEdge), nrgba, parallel)

		if !grayDst.Bounds().Eq(image.Rect(0, 0, 8, 6)) || !nrgbaDst.Bounds().Eq(image.Rect(0, 0, 8, 6)) {
			t.Fatalf("input types: unexpected bounds %v, %v", grayDst.Bounds(), nrgbaDst.Bounds())
		}
	}

	// A source that is opaque everywhere must produce opaque output.
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i+3] = 0xff
	}
	dst := normalMapTestDraw(NormalMap(1, 1, 1, false, ClampEdge), src, false)
	for i := 3; i < len(dst.Pix); i += 4 {
		if dst.Pix[i] != 0xff {
			t.Errorf("input types: expected opaque output, got alpha %d at %d", dst.Pix[i], i)
		}
	}
}

func TestNormalMapBounds(t *testing.T) {
	f := NormalMap(1, 1, 1, false, ClampEdge)
	cases := []image.Rectangle{
		image.Rect(0, 0, 1, 1),
		image.Rect(-5, -5, 5, 5),
		image.Rect(10, 20, 30, 25),
		image.Rect(0, 0, 0, 0),
	}
	for _, b := range cases {
		got := f.Bounds(b)
		want := image.Rect(0, 0, b.Dx(), b.Dy())
		if !got.Eq(want) {
			t.Errorf("bounds %v: expected %v, got %v", b, want, got)
		}
	}
}

// decodeNormalComponent maps an 8-bit encoded channel back to -1..1.
func decodeNormalComponent(v uint8) float64 {
	return float64(v)*(2.0/255.0) - 1
}

func TestNormalMapUnitLength(t *testing.T) {
	// Every non-transparent output pixel must carry a unit-length vector for
	// any image, including one with alpha variation.
	src := image.NewNRGBA(image.Rect(0, 0, 13, 11))
	for y := 0; y < 11; y++ {
		for x := 0; x < 13; x++ {
			i := src.PixOffset(x, y)
			v := uint8(30 + x*7 + y*11)
			src.Pix[i+0] = v
			src.Pix[i+1] = uint8(255 - int(v))
			src.Pix[i+2] = uint8(x * y)
			src.Pix[i+3] = uint8(40 + (x*31+y*17)%215)
		}
	}

	for _, mode := range []EdgeMode{ClampEdge, WrapEdge} {
		for _, invert := range []bool{false, true} {
			for _, parallel := range []bool{false, true} {
				dst := normalMapTestDraw(NormalMap(6, 1.5, 0.75, invert, mode), src, parallel)
				for y := 0; y < 11; y++ {
					for x := 0; x < 13; x++ {
						i := dst.PixOffset(x, y)
						nx := decodeNormalComponent(dst.Pix[i+0])
						ny := decodeNormalComponent(dst.Pix[i+1])
						nz := decodeNormalComponent(dst.Pix[i+2])
						if l := normalMapTestLen(nx, ny, nz); math.Abs(l-1) > 0.02 {
							t.Errorf("unit length (mode %v, invert %v): pixel (%d,%d): length %v (rgb %d,%d,%d)",
								mode, invert, x, y, l, dst.Pix[i+0], dst.Pix[i+1], dst.Pix[i+2])
						}
						// The encoded blue channel must stay in the upper half
						// for these moderate slopes.
						if nz <= 0 {
							t.Errorf("unit length: pixel (%d,%d): expected a positive blue component, got %v", x, y, nz)
						}
					}
				}
			}
		}
	}
}

func TestNormalMapParallelMatchesSerial(t *testing.T) {
	src := image.NewNRGBA(image.Rect(7, -3, 27, 17))
	for y := -3; y < 17; y++ {
		for x := 7; x < 27; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8((x*13 + y*7) & 0xff)
			src.Pix[i+1] = uint8((x*3 + y*19) & 0xff)
			src.Pix[i+2] = uint8((x*x + y*y) & 0xff)
			src.Pix[i+3] = uint8(60 + (x+y+100)%190)
		}
	}

	for _, mode := range []EdgeMode{ClampEdge, WrapEdge} {
		f := NormalMap(5, 1, 0.5, false, mode)
		serial := normalMapTestDraw(f, src, false)
		parallel := normalMapTestDraw(f, src, true)
		if !checkBoundsAndPix(serial.Bounds(), parallel.Bounds(), serial.Pix, parallel.Pix) {
			t.Errorf("parallel (mode %v): result differs from serial", mode)
		}
	}
}

func BenchmarkNormalMap(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	for y := 0; y < 1024; y++ {
		for x := 0; x < 1024; x++ {
			i := img.PixOffset(x, y)
			v := uint8((x*7 + y*13) & 0xff)
			img.Pix[i+0], img.Pix[i+1], img.Pix[i+2] = v, v, v
			img.Pix[i+3] = 0xff
		}
	}
	dst := image.NewNRGBA(img.Bounds())
	f := NormalMap(4, 1, 1, false, WrapEdge)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		f.Draw(dst, img, nil)
	}
}
