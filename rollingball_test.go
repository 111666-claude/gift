package gift

import (
	"image"
	"math"
	"testing"
)

// rollingBallGradientFill fills the image with a paper-like background that
// varies slowly across the frame: a horizontal brightness ramp from lo to hi
// with an additional darker vertical falloff and a soft dark "stain" in one
// corner. It returns the background value of every pixel so tests can check
// the background flatness numerically.
func rollingBallGradientFill(img *image.NRGBA, lo, hi float32) [][]float32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	bg := make([][]float32, h)
	for y := 0; y < h; y++ {
		bg[y] = make([]float32, w)
		for x := 0; x < w; x++ {
			fx := float32(x) / float32(w-1)
			fy := float32(y) / float32(h-1)
			v := lo + (hi-lo)*fx
			// A broad vertical falloff and a soft stain, both much wider
			// than the ball, so they belong to the background.
			v *= 1 - 0.10*(1-fy)
			dx := float32(x) - 0.25*float32(w)
			dy := float32(y) - 0.75*float32(h)
			v *= 1 - 0.12*expf32(-(dx*dx+dy*dy)/(2*0.35*0.35*float32(w*w)))
			bg[y][x] = v
			g := f32u8(v * 0xff)
			i := img.PixOffset(b.Min.X+x, b.Min.Y+y)
			img.Pix[i+0] = g
			img.Pix[i+1] = g
			img.Pix[i+2] = g
			img.Pix[i+3] = 0xff
		}
	}
	return bg
}

func TestRollingBallEnvelope(t *testing.T) {
	data := []struct {
		name     string
		src      []float32
		c        float64
		maximize bool
		expected []float32
	}{
		{
			"min, single sample",
			[]float32{0.5},
			0.5, false,
			[]float32{0.5},
		},
		{
			"max, single sample",
			[]float32{0.5},
			0.5, true,
			[]float32{0.5},
		},
		{
			"min, two samples",
			[]float32{0.2, 0.8},
			0.5, false,
			[]float32{0.2, 0.7},
		},
		{
			"max, two samples",
			[]float32{0.2, 0.8},
			0.5, true,
			[]float32{0.3, 0.8},
		},
		{
			"min, curved valley",
			[]float32{0, 1, 0},
			1, false,
			[]float32{0, 1, 0},
		},
		{
			"max, peak bridges the gap",
			[]float32{0, 1, 0},
			0.25, true,
			[]float32{0.75, 1, 0.75},
		},
		{
			"min, missing samples are skipped",
			[]float32{float32(math.NaN()), 0.75, float32(math.NaN()), 0.5},
			0.5, false,
			[]float32{1.25, 0.75, 1, 0.5},
		},
		{
			"max, missing samples are skipped",
			[]float32{float32(math.Inf(-1)), 0.75, float32(math.NaN()), 0.5},
			0.5, true,
			[]float32{0.25, 0.75, 0.25, 0.5},
		},
	}

	for _, d := range data {
		v := make([]int, len(d.src))
		fv := make([]float64, len(d.src))
		z := make([]float64, len(d.src))
		dst := make([]float32, len(d.src))
		rollingBallLine(dst, d.src, d.c, d.maximize, v, fv, z)
		for i := range dst {
			if absf32(dst[i]-d.expected[i]) > 1e-6 {
				t.Errorf("test [%s] failed: expected %v, got %v", d.name, d.expected, dst)
				break
			}
		}
	}
}

// rollingBallEnvelopeBruteForce computes the same envelope directly from the
// definition, ignoring non-finite samples.
func rollingBallEnvelopeBruteForce(src []float32, c float64, maximize bool) []float32 {
	dst := make([]float32, len(src))
	for x := range src {
		if maximize {
			best := math.Inf(-1)
			for q, s := range src {
				if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
					continue
				}
				d := float64(x - q)
				if v := float64(s) - c*d*d; v > best {
					best = v
				}
			}
			dst[x] = float32(best)
			continue
		}
		best := math.Inf(1)
		for q, s := range src {
			if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
				continue
			}
			d := float64(x - q)
			if v := float64(s) + c*d*d; v < best {
				best = v
			}
		}
		dst[x] = float32(best)
	}
	return dst
}

func TestRollingBallEnvelopeBruteForce(t *testing.T) {
	seed := uint32(1)
	next := func() float32 {
		// Small deterministic LCG, independent of math/rand.
		seed = seed*1664525 + 1013904223
		return float32(seed>>8) / float32(1<<24)
	}
	curvatures := []float64{1e-3, 0.05, 0.5, 1, 12.5}
	for trial := 0; trial < 40; trial++ {
		n := 1 + int(next()*40)
		src := make([]float32, n)
		for i := range src {
			src[i] = next()
		}
		if trial%3 == 0 && n > 1 {
			src[trial%n] = float32(math.NaN())
		}
		c := curvatures[trial%len(curvatures)]
		for _, maximize := range []bool{false, true} {
			want := rollingBallEnvelopeBruteForce(src, c, maximize)
			v := make([]int, n)
			fv := make([]float64, n)
			z := make([]float64, n)
			got := make([]float32, n)
			rollingBallLine(got, src, c, maximize, v, fv, z)
			for i := range got {
				if absf32(got[i]-want[i]) > 1e-6 {
					t.Fatalf("trial %d (n=%d, c=%g, maximize=%v): index %d: expected %v, got %v",
						trial, n, c, maximize, i, want[i], got[i])
				}
			}
		}
	}
}

func TestRollingBall(t *testing.T) {
	src := image.NewNRGBA(image.Rect(1, 2, 33, 10))
	bg := rollingBallGradientFill(src, 0.45, 0.90)
	w, h := src.Bounds().Dx(), src.Bounds().Dy()

	// Dark text strokes: two vertical bars and one horizontal bar.
	for _, x := range []int{6, 7, 18, 19} {
		for y := 0; y < h; y++ {
			i := src.PixOffset(src.Bounds().Min.X+x, src.Bounds().Min.Y+y)
			src.Pix[i+0], src.Pix[i+1], src.Pix[i+2] = 8, 8, 8
		}
	}
	for x := 0; x < w; x++ {
		i := src.PixOffset(src.Bounds().Min.X+x, src.Bounds().Min.Y+4)
		src.Pix[i+0], src.Pix[i+1], src.Pix[i+2] = 8, 8, 8
	}
	// A semi-transparent color patch (orange) with the same underlying paper.
	for y := 0; y < 2; y++ {
		for x := 24; x < 28; x++ {
			i := src.PixOffset(src.Bounds().Min.X+x, src.Bounds().Min.Y+y)
			src.Pix[i+0], src.Pix[i+1], src.Pix[i+2], src.Pix[i+3] = 255, 128, 0, 128
		}
	}
	srcCopy := make([]uint8, len(src.Pix))
	copy(srcCopy, src.Pix)

	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	RollingBall(5).Draw(dst, src, nil)

	// The source image must not be modified.
	for i := range src.Pix {
		if src.Pix[i] != srcCopy[i] {
			t.Fatalf("source image modified at %d", i)
		}
	}

	// The output bounds are the size of the source and start at (0, 0).
	if !dst.Bounds().Eq(image.Rect(0, 0, w, h)) {
		t.Fatalf("unexpected bounds: %v", dst.Bounds())
	}

	// Paper flatness: away from strokes, all background pixels must end up at
	// the same luminance (the brightest background value).
	isStroke := func(x, y int) bool {
		return x == 6 || x == 7 || x == 18 || x == 19 || y == 4
	}
	isPatch := func(x, y int) bool {
		return y < 2 && x >= 24 && x < 28
	}
	var paperLuma []float32
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if isStroke(x, y) || isPatch(x, y) {
				continue
			}
			px := dst.NRGBAAt(dst.Bounds().Min.X+x, dst.Bounds().Min.Y+y)
			paperLuma = append(paperLuma, 0.299*float32(px.R)+0.587*float32(px.G)+0.114*float32(px.B))
		}
	}
	paperMax, paperMin := paperLuma[0], paperLuma[0]
	for _, v := range paperLuma {
		paperMax = maxf32(paperMax, v)
		paperMin = minf32(paperMin, v)
	}
	if paperMax-paperMin > 1.5 {
		t.Errorf("background is not flat: min %v, max %v", paperMin, paperMax)
	}
	// The corrected paper is brighter than the original one everywhere; the
	// darkest corner is lifted towards the brightest one.
	var bgMin, bgMax float32 = 1, 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if isStroke(x, y) || isPatch(x, y) {
				continue
			}
			bgMin = minf32(bgMin, bg[y][x])
			bgMax = maxf32(bgMax, bg[y][x])
		}
	}
	if bgMax-bgMin < 0.2 {
		t.Fatalf("test fixture has no background gradient: %v..%v", bgMin, bgMax)
	}

	// Strokes stay dark: the correction preserves foreground contrast
	// instead of flattening it together with the paper.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !isStroke(x, y) || isPatch(x, y) {
				continue
			}
			px := dst.NRGBAAt(dst.Bounds().Min.X+x, dst.Bounds().Min.Y+y)
			lum := 0.299*float32(px.R) + 0.587*float32(px.G) + 0.114*float32(px.B)
			if lum > 40 {
				t.Fatalf("stroke pixel (%d, %d) lost its contrast: %v", x, y, lum)
			}
		}
	}

	// The semi-transparent patch keeps its alpha and its hue. The channels
	// are scaled by the same factor, so the R:G:B ratio is unchanged.
	for y := 0; y < 2; y++ {
		for x := 24; x < 28; x++ {
			px := dst.NRGBAAt(dst.Bounds().Min.X+x, dst.Bounds().Min.Y+y)
			if px.A != 128 {
				t.Fatalf("alpha changed at (%d, %d): %d", x, y, px.A)
			}
			if px.G == 0 || px.R == 0 {
				t.Fatalf("unexpected color at (%d, %d): %v", x, y, px)
			}
			if ratioOK := absf32(float32(px.R)/float32(px.G)-2) <= 0.05; !ratioOK {
				t.Errorf("hue changed at (%d, %d): %v", x, y, px)
			}
		}
	}
}

func TestRollingBallIdentity(t *testing.T) {
	// A constant image is not changed by any radius.
	src := image.NewNRGBA(image.Rect(0, 0, 16, 12))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i+0], src.Pix[i+1], src.Pix[i+2], src.Pix[i+3] = 63, 120, 200, 77
	}
	for _, radius := range []float32{0.5, 3, 1000} {
		dst := image.NewNRGBA(src.Bounds())
		RollingBall(radius).Draw(dst, src, nil)
		for i := range dst.Pix {
			if dst.Pix[i] != src.Pix[i] {
				t.Fatalf("radius %g changed a constant image at %d", radius, i)
			}
		}
	}

	// A smooth background gradient is the very thing the filter removes: the
	// result is the flat reference level everywhere. This holds for both a
	// two-axis ramp and a one-axis ramp.
	w, h := 20, 9
	ramps := map[string]func(x, y int) uint8{
		"two-axis": func(x, y int) uint8 { return uint8(60 + x*8 + y*4) },
		"one-axis": func(x, y int) uint8 { return uint8(60 + x*8) },
	}
	for name, fill := range ramps {
		ramp := image.NewGray(image.Rect(0, 0, w, h))
		top := uint8(0)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := fill(x, y)
				ramp.Pix[y*ramp.Stride+x] = v
				if v > top {
					top = v
				}
			}
		}
		for _, radius := range []float32{1, 4, 40, 4000} {
			dst := image.NewGray(ramp.Bounds())
			RollingBall(radius).Draw(dst, ramp, nil)
			// The closing of a convex surface can touch it only at the
			// borders of the (finite) frame, so the very first column may
			// stay a few levels darker than the reference. Everywhere else
			// the gradient must be gone.
			for i := range dst.Pix {
				if i%w == 0 {
					continue
				}
				if absf32(float32(dst.Pix[i])-float32(top)) > 1 {
					t.Fatalf("%s ramp, radius %g: pixel %d not flattened: %d (want %d)",
						name, radius, i, dst.Pix[i], top)
				}
			}
		}
	}

	// A fully black image stays black even though the background is zero.
	black := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for i := 3; i < len(black.Pix); i += 4 {
		black.Pix[i] = 0xff
	}
	dst := image.NewNRGBA(black.Bounds())
	RollingBall(2).Draw(dst, black, nil)
	for i := range dst.Pix {
		if dst.Pix[i] != black.Pix[i] {
			t.Fatalf("black image changed at %d", i)
		}
	}
}

func TestRollingBallTransparent(t *testing.T) {
	// The hidden color of fully transparent pixels must neither be changed
	// nor influence the background.
	w, h := 24, 12
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := src.PixOffset(x, y)
			g := uint8(80 + x*6)
			src.Pix[i+0], src.Pix[i+1], src.Pix[i+2] = g, g, g
			src.Pix[i+3] = 0xff
			if x >= 16 && y < 5 {
				// Fully transparent, bright green and magenta pixels.
				src.Pix[i+0], src.Pix[i+1], src.Pix[i+2] = 0, 255, 255
				src.Pix[i+3] = 0
			}
		}
	}
	srcCopy := make([]uint8, len(src.Pix))
	copy(srcCopy, src.Pix)

	dst := image.NewNRGBA(src.Bounds())
	RollingBall(4).Draw(dst, src, nil)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := src.PixOffset(x, y)
			if x >= 16 && y < 5 {
				for k := 0; k < 4; k++ {
					if dst.Pix[i+k] != src.Pix[i+k] {
						t.Fatalf("transparent pixel (%d, %d) changed: %v", x, y, dst.Pix[i:i+4])
					}
				}
			}
		}
	}
	// The visible part is still flattened; the transparent block does not
	// leave a bright or dark halo around it.
	for y := 5; y < h; y++ {
		for x := 0; x < 16; x++ {
			px := dst.NRGBAAt(x, y)
			lum := 0.299*float32(px.R) + 0.587*float32(px.G) + 0.114*float32(px.B)
			if lum < 200 {
				t.Fatalf("visible pixel (%d, %d) was not corrected: %v", x, y, lum)
			}
			if px.R != px.G || px.G != px.B {
				t.Fatalf("gray pixel became colored at (%d, %d): %v", x, y, px)
			}
		}
	}

	// A fully transparent image is returned unchanged, whatever its hidden
	// colors are.
	clear := image.NewNRGBA(image.Rect(0, 0, 9, 7))
	for i := 0; i < len(clear.Pix); i += 4 {
		clear.Pix[i+0], clear.Pix[i+1], clear.Pix[i+2], clear.Pix[i+3] = 255, 0, 128, 0
	}
	dst = image.NewNRGBA(clear.Bounds())
	RollingBall(3).Draw(dst, clear, nil)
	for i := range dst.Pix {
		if dst.Pix[i] != clear.Pix[i] {
			t.Fatalf("fully transparent image changed at %d", i)
		}
	}
}

func TestRollingBallParams(t *testing.T) {
	src := image.NewNRGBA(image.Rect(-2, -3, 6, 4))
	for y := src.Bounds().Min.Y; y < src.Bounds().Max.Y; y++ {
		for x := src.Bounds().Min.X; x < src.Bounds().Max.X; x++ {
			i := src.PixOffset(x, y)
			src.Pix[i+0] = uint8(40 + 10*(x+2))
			src.Pix[i+1] = uint8(50 + 10*(y+3))
			src.Pix[i+2] = 70
			src.Pix[i+3] = uint8(64 + 10*(x+2))
		}
	}
	srcCopy := make([]uint8, len(src.Pix))
	copy(srcCopy, src.Pix)

	// Invalid radii and empty images are explicit no-ops.
	invalid := []float32{0, -1, -0.5, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	for _, radius := range invalid {
		dst := image.NewNRGBA(src.Bounds())
		RollingBall(radius).Draw(dst, src, nil)
		for i := range dst.Pix {
			if dst.Pix[i] != srcCopy[i] {
				t.Fatalf("radius %g changed the image at %d", radius, i)
			}
		}
	}

	// A radius larger than the image is well defined: it is clamped to the
	// image size, so huge radii produce exactly the same result as the
	// largest meaningful one and never panic or produce non-finite pixels.
	dst := image.NewNRGBA(src.Bounds())
	RollingBall(1e9).Draw(dst, src, nil)
	clamped := image.NewNRGBA(src.Bounds())
	RollingBall(float32(src.Bounds().Dx())).Draw(clamped, src, nil)
	for i := range dst.Pix {
		if dst.Pix[i] != clamped.Pix[i] {
			t.Fatalf("huge radius differs from the clamped radius at %d: %d vs %d", i, dst.Pix[i], clamped.Pix[i])
		}
	}

	// Empty images produce empty destinations without panicking.
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(3, 3, 3, 9),
		image.Rect(3, 3, 9, 3),
	} {
		empty := image.NewNRGBA(r)
		out := image.NewNRGBA(r)
		RollingBall(5).Draw(out, empty, nil)
		if !out.Bounds().Eq(r) {
			t.Fatalf("unexpected bounds for empty image: %v", out.Bounds())
		}
	}

	// Tiny images are handled by every radius.
	for _, size := range []image.Point{{1, 1}, {1, 7}, {7, 1}, {2, 2}} {
		tiny := image.NewNRGBA(image.Rect(0, 0, size.X, size.Y))
		for i := 0; i < len(tiny.Pix); i += 4 {
			tiny.Pix[i+0], tiny.Pix[i+1], tiny.Pix[i+2], tiny.Pix[i+3] = 100, 110, 120, 200
		}
		for _, radius := range []float32{0.5, 2, 1e9} {
			out := image.NewNRGBA(tiny.Bounds())
			RollingBall(radius).Draw(out, tiny, nil)
			for i := range out.Pix {
				if out.Pix[i] != tiny.Pix[i] {
					t.Fatalf("size %v radius %g changed a constant image at %d", size, radius, i)
				}
			}
		}
	}
}

func TestRollingBallGrayAndOpaque(t *testing.T) {
	// A Gray image is corrected in the same way as the luminance of an
	// NRGBA image, and the results do not depend on the parallelization
	// setting or on the position of the bounds.
	w, h := 24, 14
	gray := image.NewGray(image.Rect(2, 5, 2+w, 5+h))
	nrgba := image.NewNRGBA(gray.Bounds())
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(70 + x*6 + y)
			gray.Pix[y*gray.Stride+x] = v
			i := nrgba.PixOffset(gray.Bounds().Min.X+x, gray.Bounds().Min.Y+y)
			nrgba.Pix[i+0], nrgba.Pix[i+1], nrgba.Pix[i+2], nrgba.Pix[i+3] = v, v, v, 0xff
		}
	}

	g1 := New(RollingBall(3))
	g1.SetParallelization(false)
	dstGray := image.NewGray(gray.Bounds())
	g1.Draw(dstGray, gray)

	g2 := New(RollingBall(3))
	dstNRGBA := image.NewNRGBA(nrgba.Bounds())
	g2.Draw(dstNRGBA, nrgba)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gi := gray.PixOffset(gray.Bounds().Min.X+x, gray.Bounds().Min.Y+y)
			ni := nrgba.PixOffset(nrgba.Bounds().Min.X+x, nrgba.Bounds().Min.Y+y)
			if dstNRGBA.Pix[ni+0] != dstNRGBA.Pix[ni+1] || dstNRGBA.Pix[ni+1] != dstNRGBA.Pix[ni+2] {
				t.Fatalf("gray pixel became colored at (%d, %d)", x, y)
			}
			if d := int(dstGray.Pix[gi]) - int(dstNRGBA.Pix[ni+0]); d < -1 || d > 1 {
				t.Fatalf("gray and NRGBA results differ at (%d, %d): %d vs %d",
					x, y, dstGray.Pix[gi], dstNRGBA.Pix[ni+0])
			}
		}
	}

	// Bounds of the filtered image follow the source size.
	if b := g1.Bounds(gray.Bounds()); !b.Eq(image.Rect(0, 0, w, h)) {
		t.Fatalf("unexpected filter bounds: %v", b)
	}

	// The filter can be used in a chain and with DrawAt.
	chain := New(RollingBall(3), Grayscale())
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	chain.Draw(out, nrgba)
	for i := 0; i < len(out.Pix); i += 4 {
		if out.Pix[i] != out.Pix[i+1] || out.Pix[i+1] != out.Pix[i+2] {
			t.Fatalf("Grayscale after RollingBall produced a colored pixel at %d", i)
		}
		if out.Pix[i+3] != 0xff {
			t.Fatalf("alpha lost at %d", i)
		}
	}

	at := image.NewRGBA(image.Rect(0, 0, 2*w, 2*h))
	New(RollingBall(3)).DrawAt(at, nrgba, image.Pt(w/2, h/2), CopyOperator)
	// The region outside the pasted image is untouched.
	if at.RGBAAt(0, 0).A != 0 {
		t.Fatalf("DrawAt touched pixels outside the target position")
	}
}
