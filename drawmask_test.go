package gift

import (
	"image"
	"image/color"
	"image/draw"
	"testing"
)

// blendRef is an independent float64 reference of the masked blend math.
func blendRef(dst, src [4]float64, m float64, op Operator) [4]float64 {
	if m <= 0 {
		return dst
	}
	if op == CopyOperator {
		return [4]float64{
			dst[0]*(1-m) + src[0]*m,
			dst[1]*(1-m) + src[1]*m,
			dst[2]*(1-m) + src[2]*m,
			dst[3]*(1-m) + src[3]*m,
		}
	}
	sa := src[3] * m
	a := dst[3] + sa*(1-dst[3])
	if a == 0 {
		return [4]float64{0, 0, 0, 0}
	}
	w1 := sa / a
	w0 := dst[3] * (1 - sa) / a
	return [4]float64{
		src[0]*w1 + dst[0]*w0,
		src[1]*w1 + dst[1]*w0,
		src[2]*w1 + dst[2]*w0,
		a,
	}
}

func nrgbaPixel(img *image.NRGBA, x, y int) [4]float64 {
	i := img.PixOffset(x, y)
	return [4]float64{
		float64(img.Pix[i+0]) / 255,
		float64(img.Pix[i+1]) / 255,
		float64(img.Pix[i+2]) / 255,
		float64(img.Pix[i+3]) / 255,
	}
}

// assertBlend checks every pixel of dst against the float64 reference.
// m=0 and m=1 cases must be exact; intermediate values allow one 8-bit LSB.
func assertBlend(t *testing.T, dst *image.NRGBA, src [4][4]float64, maskVals [4]float64, op Operator, pt image.Point) {
	t.Helper()
	for iy := 0; iy < 2; iy++ {
		for ix := 0; ix < 2; ix++ {
			k := iy*2 + ix
			got := nrgbaPixel(dst, pt.X+ix, pt.Y+iy)
			want := blendRef([4]float64{200.0 / 255, 100.0 / 255, 50.0 / 255, 1}, src[k], maskVals[k], op)
			for c := 0; c < 4; c++ {
				diff := got[c] - want[c]
				if diff < 0 {
					diff = -diff
				}
				tol := 1.5 / 255
				if maskVals[k] == 0 || maskVals[k] == 1 {
					tol = 0 // endpoints must be bit exact
				}
				if diff > tol {
					t.Errorf("op=%v pixel (%d,%d) channel %d: got %.4f want %.4f (m=%.3f)",
						op, ix, iy, c, got[c], want[c], maskVals[k])
				}
			}
		}
	}
}

// buildSrcSet builds NRGBA, RGBA and Gray sources carrying equivalent colors.
// Colors are opaque so RGBA premultiplication is an exact conversion.
func buildSrcSet(b image.Rectangle, colors [4]color.NRGBA) (map[string]image.Image, [4][4]float64) {
	nrgba := image.NewNRGBA(b)
	rgba := image.NewRGBA(b)
	gray := image.NewGray(b)
	var ref [4][4]float64
	for i, c := range colors {
		x := b.Min.X + i%b.Dx()
		y := b.Min.Y + i/b.Dx()
		nrgba.SetNRGBA(x, y, c)
		rgba.Set(x, y, c)
		gv := uint8(0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B) + 0.5)
		gray.SetGray(x, y, color.Gray{gv})
		ref[i] = [4]float64{float64(c.R) / 255, float64(c.G) / 255, float64(c.B) / 255, 1}
	}
	return map[string]image.Image{"NRGBA": nrgba, "RGBA": rgba, "Gray": gray}, ref
}

// TestDrawAtMaskedBlend verifies requirement 1: mask alpha 0 / partial / 1
// means keep original / proportional blend / full coverage, for both operators.
func TestDrawAtMaskedBlend(t *testing.T) {
	srcb := image.Rect(0, 0, 2, 2)
	colors := [4]color.NRGBA{
		{10, 20, 30, 255},
		{40, 50, 60, 128},
		{200, 200, 200, 0},
		{80, 90, 100, 255},
	}
	src := image.NewNRGBA(srcb)
	var srcRef [4][4]float64
	for i, c := range colors {
		src.SetNRGBA(srcb.Min.X+i%2, srcb.Min.Y+i/2, c)
		srcRef[i] = [4]float64{float64(c.R) / 255, float64(c.G) / 255, float64(c.B) / 255, float64(c.A) / 255}
	}
	maskVals := [4]float64{0, 128.0 / 255, 1, 64.0 / 255}
	mask := image.NewAlpha(srcb)
	for i, v := range maskVals {
		mask.Pix[i] = uint8(v*255 + 0.5)
	}

	for _, op := range []Operator{CopyOperator, OverOperator} {
		pt := image.Pt(3, 2)
		dstb := image.Rect(0, 0, 8, 8)
		dst := image.NewNRGBA(dstb)
		draw.Draw(dst, dstb, &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)

		New().DrawAtMasked(dst, src, mask, pt, op)
		assertBlend(t, dst, srcRef, maskVals, op, pt)

		// Everywhere outside the output rectangle must be untouched.
		for y := dstb.Min.Y; y < dstb.Max.Y; y++ {
			for x := dstb.Min.X; x < dstb.Max.X; x++ {
				if image.Pt(x, y).In(image.Rectangle{pt, pt.Add(srcb.Size())}) {
					continue
				}
				i := dst.PixOffset(x, y)
				if dst.Pix[i+0] != 200 || dst.Pix[i+1] != 100 || dst.Pix[i+2] != 50 || dst.Pix[i+3] != 255 {
					t.Errorf("op=%v pixel (%d,%d) outside output was modified", op, x, y)
				}
			}
		}
	}
}

// TestDrawAtMaskedMatchesDrawAt verifies requirement 3 (semantics unchanged):
// an opaque mask and a nil mask must reproduce DrawAt byte for byte, for both
// operators and including partially transparent sources.
func TestDrawAtMaskedMatchesDrawAt(t *testing.T) {
	srcb := image.Rect(-1, -1, 2, 2)
	colors := [9]color.NRGBA{
		{10, 20, 30, 255},
		{40, 50, 60, 128},
		{200, 200, 200, 0},
		{80, 90, 100, 255},
		{1, 2, 3, 64},
		{7, 8, 9, 200},
		{30, 40, 50, 1},
		{60, 70, 80, 192},
		{90, 100, 110, 255},
	}
	src := image.NewNRGBA(srcb)
	for i, c := range colors {
		src.SetNRGBA(srcb.Min.X+i%3, srcb.Min.Y+i/3, c)
	}
	outSize := New(Resize(5, 5, NearestNeighborResampling)).Bounds(srcb).Size()
	fullMask := image.NewAlpha(image.Rect(0, 0, outSize.X, outSize.Y))
	for i := range fullMask.Pix {
		fullMask.Pix[i] = 255
	}

	for _, op := range []Operator{CopyOperator, OverOperator} {
		for _, pt := range []image.Point{{0, 0}, {2, -1}, {-3, 4}} {
			dstb := image.Rect(-2, -2, 4, 4)

			ref := image.NewNRGBA(dstb)
			draw.Draw(ref, dstb, &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
			New(Resize(5, 5, NearestNeighborResampling)).DrawAt(ref, src, pt, op)

			for _, masked := range []bool{false, true} {
				got := image.NewNRGBA(dstb)
				draw.Draw(got, dstb, &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
				g := New(Resize(5, 5, NearestNeighborResampling))
				var m image.Image
				if masked {
					m = fullMask
				}
				g.DrawAtMasked(got, src, m, pt, op)
				if !checkBoundsAndPix(got.Bounds(), ref.Bounds(), got.Pix, ref.Pix) {
					t.Errorf("op=%v pt=%v masked=%v: masked result differs from DrawAt", op, pt, masked)
				}
			}
		}
	}
}

// TestDrawAtMaskedInputTypes verifies requirement 3: equivalent RGBA, NRGBA and
// Gray inputs (and Alpha/Gray/NRGBA masks) produce identical results. The source
// colors are grayscale so all three source representations are exact conversions.
func TestDrawAtMaskedInputTypes(t *testing.T) {
	srcb := image.Rect(0, 0, 2, 2)
	grays := [4]uint8{10, 60, 200, 100}
	colors := [4]color.NRGBA{}
	for i, v := range grays {
		colors[i] = color.NRGBA{v, v, v, 255}
	}
	srcs, _ := buildSrcSet(srcb, colors)

	coverage := []uint8{0, 128, 255, 64}
	alphaMask := image.NewAlpha(srcb)
	grayMask := image.NewGray(srcb)
	nrgbaMask := image.NewNRGBA(srcb)
	for i, v := range coverage {
		alphaMask.Pix[i] = v
		grayMask.Pix[i] = v
		j := nrgbaMask.PixOffset(srcb.Min.X+i%2, srcb.Min.Y+i/2)
		nrgbaMask.Pix[j+0] = 17
		nrgbaMask.Pix[j+1] = 99
		nrgbaMask.Pix[j+2] = 231
		nrgbaMask.Pix[j+3] = v
	}
	masks := map[string]image.Image{"Alpha": alphaMask, "Gray": grayMask, "NRGBA": nrgbaMask}

	var reference []uint8
	for srcName, src := range srcs {
		for maskName, mask := range masks {
			dst := image.NewNRGBA(image.Rect(0, 0, 6, 6))
			draw.Draw(dst, dst.Bounds(), &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
			New().DrawAtMasked(dst, src, mask, image.Pt(2, 1), CopyOperator)

			if reference == nil {
				reference = append([]uint8(nil), dst.Pix...)
			} else if !checkBoundsAndPix(dst.Bounds(), dst.Bounds(), dst.Pix, reference) {
				t.Errorf("src=%s mask=%s differs from reference result", srcName, maskName)
			}
		}
	}

	// Explicit endpoints on the Gray source against a zeroed destination.
	dst := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	New().DrawAtMasked(dst, srcs["Gray"], alphaMask, image.Point{}, CopyOperator)
	for i, v := range grays {
		j := i * 4
		var want []uint8
		switch coverage[i] {
		case 255:
			want = []uint8{v, v, v, 255} // full coverage: gray copied
		case 0:
			want = []uint8{0, 0, 0, 0} // zero coverage: zeroed dst untouched
		default:
			continue // partial values are covered by the reference comparison above
		}
		for k, w := range want {
			if dst.Pix[j+k] != w {
				t.Errorf("gray src pixel %d channel %d: got %d want %d", i, k, dst.Pix[j+k], w)
			}
		}
	}
}

// TestDrawAtMaskedAlignment verifies requirement 2: after resize or rotation the
// mask is aligned with the filter output placed at pt, not the source image.
func TestDrawAtMaskedAlignment(t *testing.T) {
	// Rotation: 2x3 source becomes a 3x2 output.
	src := image.NewNRGBA(image.Rect(0, 0, 2, 3))
	draw.Draw(src, src.Bounds(), &image.Uniform{color.NRGBA{10, 20, 30, 255}}, image.Point{}, draw.Src)
	pt := image.Pt(10, 20)

	outb := New(Rotate90()).Bounds(src.Bounds()).Size() // 3x2
	mask := image.NewAlpha(image.Rect(0, 0, outb.X, outb.Y))
	hit := image.Pt(2, 1) // mask-local opaque pixel
	mask.SetAlpha(hit.X, hit.Y, color.Alpha{255})

	dst := image.NewNRGBA(image.Rect(0, 0, 20, 30))
	draw.Draw(dst, dst.Bounds(), &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
	New(Rotate90()).DrawAtMasked(dst, src, mask, pt, CopyOperator)

	changed := 0
	for y := 0; y < 30; y++ {
		for x := 0; x < 20; x++ {
			c := dst.NRGBAAt(x, y)
			isModified := c.R != 200 || c.G != 100 || c.B != 50
			expected := image.Pt(x, y).Eq(pt.Add(hit))
			if isModified != expected {
				t.Errorf("rotate: pixel (%d,%d) modified=%v want %v", x, y, isModified, expected)
			}
			if isModified {
				changed++
			}
		}
	}
	if changed != 1 {
		t.Errorf("rotate: expected exactly 1 changed pixel, got %d", changed)
	}

	// Resize: 2x2 source becomes 4x4 output; mask with non-zero origin must
	// still map its Min onto pt. The point is kept fully inside the canvas.
	pt2 := image.Pt(2, 3)
	src2 := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	draw.Draw(src2, src2.Bounds(), &image.Uniform{color.NRGBA{10, 20, 30, 255}}, image.Point{}, draw.Src)
	outb2 := New(Resize(4, 4, NearestNeighborResampling)).Bounds(src2.Bounds()).Size()
	mask2 := image.NewAlpha(image.Rect(-5, -5, -5+outb2.X, -5+outb2.Y))
	hit2 := image.Pt(3, 0)
	mask2.SetAlpha(-5+hit2.X, -5+hit2.Y, color.Alpha{255})

	dst2 := image.NewNRGBA(image.Rect(0, 0, 20, 20))
	draw.Draw(dst2, dst2.Bounds(), &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
	New(Resize(4, 4, NearestNeighborResampling)).DrawAtMasked(dst2, src2, mask2, pt2, CopyOperator)

	target := dst2.NRGBAAt(pt2.X+hit2.X, pt2.Y+hit2.Y)
	if target.R != 10 || target.G != 20 || target.B != 30 || target.A != 255 {
		t.Errorf("resize: masked target pixel wrong: %v", target)
	}
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			if image.Pt(x, y).Eq(pt2.Add(hit2)) {
				continue
			}
			c := dst2.NRGBAAt(x, y)
			if c.R != 200 || c.G != 100 || c.B != 50 {
				t.Errorf("resize: unexpected modified pixel (%d,%d)", x, y)
			}
		}
	}
}

// TestDrawAtMaskedClipping verifies requirement 2: output or mask extending
// beyond the destination is safely clipped, and a smaller mask restricts coverage.
func TestDrawAtMaskedClipping(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	draw.Draw(src, src.Bounds(), &image.Uniform{color.NRGBA{10, 20, 30, 255}}, image.Point{}, draw.Src)

	// Output straddles the top and left edges of the destination while fully
	// covering it. The mask covers the whole output region in destination
	// coordinates, so destination clipping is exercised alone; any out-of-bounds
	// write into the temporary image would panic instead of being silently clipped.
	dstb := image.Rect(0, 0, 3, 3)
	dst := image.NewNRGBA(dstb)
	draw.Draw(dst, dstb, &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
	bigMask := image.NewAlpha(image.Rect(0, 0, 5, 5))
	for i := range bigMask.Pix {
		bigMask.Pix[i] = 255
	}
	New().DrawAtMasked(dst, src, bigMask, image.Pt(-1, -1), OverOperator)
	for y := dstb.Min.Y; y < dstb.Max.Y; y++ {
		for x := dstb.Min.X; x < dstb.Max.X; x++ {
			i := dst.PixOffset(x, y)
			if dst.Pix[i+0] != 10 || dst.Pix[i+1] != 20 || dst.Pix[i+2] != 30 || dst.Pix[i+3] != 255 {
				t.Errorf("clipped pixel (%d,%d) wrong: %v", x, y, dst.Pix[i:i+4])
			}
		}
	}

	// A 1x1 mask over a 4x4 output: only one destination pixel may change.
	dst2 := image.NewNRGBA(image.Rect(0, 0, 6, 6))
	draw.Draw(dst2, dst2.Bounds(), &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
	smallMask := image.NewAlpha(image.Rect(0, 0, 1, 1))
	smallMask.Pix[0] = 255
	New().DrawAtMasked(dst2, src, smallMask, image.Pt(1, 1), CopyOperator)
	for y := 0; y < 6; y++ {
		for x := 0; x < 6; x++ {
			i := dst2.PixOffset(x, y)
			wantSrc := x == 1 && y == 1
			if wantSrc {
				if dst2.Pix[i+0] != 10 || dst2.Pix[i+3] != 255 {
					t.Errorf("small-mask target pixel wrong: %v", dst2.Pix[i:i+4])
				}
				continue
			}
			if dst2.Pix[i+0] != 200 || dst2.Pix[i+1] != 100 || dst2.Pix[i+2] != 50 {
				t.Errorf("pixel (%d,%d) outside small mask was modified", x, y)
			}
		}
	}

	// Fully disjoint output and mask: must be a no-op without panicking.
	dst3 := image.NewNRGBA(dstb)
	draw.Draw(dst3, dstb, &image.Uniform{color.NRGBA{200, 100, 50, 255}}, image.Point{}, draw.Src)
	before := append([]uint8(nil), dst3.Pix...)
	New().DrawAtMasked(dst3, src, smallMask, image.Pt(100, 100), CopyOperator)
	if !checkBoundsAndPix(dst3.Bounds(), dst3.Bounds(), dst3.Pix, before) {
		t.Error("disjoint draw must leave dst untouched")
	}
}

// TestDrawAtMaskedParallel verifies requirement 4: the parallel toggle does not
// change any pixel, and the source and mask images are never written to.
func TestDrawAtMaskedParallel(t *testing.T) {
	srcb := image.Rect(0, 0, 5, 7)
	src := image.NewNRGBA(srcb)
	for i := range src.Pix {
		src.Pix[i] = uint8((i*37 + 11) % 256)
	}
	mask := image.NewAlpha(srcb)
	for i := range mask.Pix {
		mask.Pix[i] = uint8((i*53 + 7) % 256)
	}
	srcSnap := append([]uint8(nil), src.Pix...)
	maskSnap := append([]uint8(nil), mask.Pix...)

	results := map[bool][]uint8{}
	for _, parallel := range []bool{true, false} {
		dst := image.NewNRGBA(image.Rect(0, 0, 9, 11))
		draw.Draw(dst, dst.Bounds(), &image.Uniform{color.NRGBA{170, 85, 42, 255}}, image.Point{}, draw.Src)
		g := New(Resize(8, 10, NearestNeighborResampling))
		g.SetParallelization(parallel)
		g.DrawAtMasked(dst, src, mask, image.Pt(1, -1), OverOperator)
		results[parallel] = append([]uint8(nil), dst.Pix...)
	}
	if !checkBoundsAndPix(image.Rect(0, 0, 9, 11), image.Rect(0, 0, 9, 11), results[true], results[false]) {
		t.Error("parallel and serial runs produced different pixels")
	}
	if !checkBoundsAndPix(src.Bounds(), src.Bounds(), src.Pix, srcSnap) {
		t.Error("source image was modified")
	}
	if !checkBoundsAndPix(mask.Bounds(), mask.Bounds(), mask.Pix, maskSnap) {
		t.Error("mask image was modified")
	}
}
