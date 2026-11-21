package gift

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"testing"
)

// newRandomNRGBA fills an NRGBA image with deterministic pseudo-random pixels
// including random alpha values so that the non-opaque code paths are
// exercised too.
func newRandomNRGBA(r image.Rectangle, seed int64) *image.NRGBA {
	img := image.NewNRGBA(r)
	rnd := rand.New(rand.NewSource(seed))
	rnd.Read(img.Pix)
	return img
}

// drawFull applies the chain and returns the complete result as an NRGBA image
// with origin (0, 0).
func drawFull(g *GIFT, src image.Image) *image.NRGBA {
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)
	return dst
}

// drawRegion applies the chain through DrawRegion into a viewport-sized
// destination aligned at the origin. The returned image contains exactly the
// pixels of the region (clipped to the full result bounds).
func drawRegion(t *testing.T, g *GIFT, src image.Image, region image.Rectangle) *image.NRGBA {
	t.Helper()
	fullB := g.Bounds(src.Bounds())
	win := region.Intersect(fullB)
	dst := image.NewNRGBA(image.Rect(0, 0, win.Dx(), win.Dy()))
	if err := g.DrawRegion(dst, src, region); err != nil {
		t.Fatalf("DrawRegion failed: %v", err)
	}
	return dst
}

// viewportDiff compares a viewport-sized image got (origin at 0,0) against
// the win window of the full-result image want.
func viewportDiff(got, want image.Image, win image.Rectangle) (mismatches int, firstX, firstY int) {
	firstX, firstY = -1, -1
	for y := 0; y < win.Dy(); y++ {
		for x := 0; x < win.Dx(); x++ {
			ca := color.NRGBAModel.Convert(got.At(x, y)).(color.NRGBA)
			cb := color.NRGBAModel.Convert(want.At(win.Min.X+x, win.Min.Y+y)).(color.NRGBA)
			if ca != cb {
				if mismatches == 0 {
					firstX, firstY = x, y
				}
				mismatches++
			}
		}
	}
	return
}

// pixelDiff compares two images pixel by pixel and returns the number of
// mismatching pixels and the first mismatch coordinates.
func pixelDiff(a, b image.Image, r image.Rectangle) (mismatches int, firstX, firstY int) {
	firstX, firstY = -1, -1
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			ca := color.NRGBAModel.Convert(a.At(x, y)).(color.NRGBA)
			cb := color.NRGBAModel.Convert(b.At(x, y)).(color.NRGBA)
			if ca != cb {
				if mismatches == 0 {
					firstX, firstY = x, y
				}
				mismatches++
			}
		}
	}
	return
}

func regionChains() map[string][]Filter {
	return map[string][]Filter{
		"pointwise": {
			Invert(),
			Grayscale(),
			ColorBalance(12, -7, 3),
		},
		"convolution": {
			Convolution([]float32{
				-1, -1, 0,
				-1, 5, 1,
				0, 1, 1,
			}, false, true, false, 0),
			Invert(),
		},
		"gaussian_chain": {
			GaussianBlur(2.5),
			Saturation(20),
			GaussianBlur(1),
		},
		"unsharp_chain": {
			UnsharpMask(2, 1.2, 0.01),
			Contrast(15),
		},
		"rank_chain": {
			Median(5, false),
			Maximum(3, true),
			Grayscale(),
		},
		"mean_sobel": {
			Mean(5, true),
			Sobel(),
		},
		"pixelate_chain": {
			Pixelate(7),
			Invert(),
		},
		"mixed": {
			Grayscale(),
			Convolution([]float32{
				1, 2, 1,
				2, 4, 2,
				1, 2, 1,
			}, true, false, false, 0),
			GaussianBlur(1.5),
			Threshold(45),
		},
		"empty":            {},
		"single_pointwise": {Brightness(10)},
		"single_blur":      {GaussianBlur(3)},
	}
}

// TestDrawRegionExact runs every supported chain on a reasonably large random
// image and verifies that region rendering produces bytes identical to the
// full draw cropped to the same rectangle. The image is large enough that
// LUT-based color filters take the same code path in both runs.
func TestDrawRegionExact(t *testing.T) {
	const w, h = 160, 120
	src := newRandomNRGBA(image.Rect(0, 0, w, h), 42)

	regions := []image.Rectangle{
		image.Rect(0, 0, w, h),           // whole image
		image.Rect(0, 0, 37, 29),         // top-left corner
		image.Rect(w-41, h-33, w, h),     // bottom-right corner (viewport case)
		image.Rect(53, 27, 111, 88),      // interior rectangle
		image.Rect(0, 40, w, 80),         // full-width strip
		image.Rect(60, 0, 100, h),        // full-height strip
		image.Rect(-20, -10, 10, 5),      // partially outside, top-left
		image.Rect(w-5, h-7, w+50, h+50), // partially outside, bottom-right
		image.Rect(-100, -100, -50, -50), // fully outside
	}

	for name, filters := range regionChains() {
		filters := filters
		t.Run(name, func(t *testing.T) {
			for _, parallel := range []bool{true, false} {
				g := New(filters...)
				g.SetParallelization(parallel)
				full := drawFull(g, src)

				for ri, region := range regions {
					win := region.Intersect(full.Bounds())
					if win.Empty() {
						// Empty intersection must be a no-op; drawRegion
						// would create a zero-size viewport, check directly.
						dst := image.NewNRGBA(image.Rect(0, 0, 3, 3))
						if err := g.DrawRegion(dst, src, region); err != nil {
							t.Fatalf("parallel=%v region #%d: %v", parallel, ri, err)
						}
						continue
					}
					got := drawRegion(t, g, src, region)

					// The viewport must have exactly the window size and its
					// pixels must match the full draw cropped to the window.
					if got.Bounds().Dx() != win.Dx() || got.Bounds().Dy() != win.Dy() {
						t.Fatalf("parallel=%v region #%d: unexpected viewport size %v, want %v",
							parallel, ri, got.Bounds().Size(), win.Size())
					}
					mismatches, fx, fy := viewportDiff(got, full, win)
					if mismatches != 0 {
						t.Fatalf("parallel=%v region #%d: %d/%d pixels differ from full draw, first mismatch at viewport (%d,%d): got %v want %v",
							parallel, ri, mismatches, win.Dx()*win.Dy(),
							fx, fy, got.At(fx, fy), full.At(win.Min.X+fx, win.Min.Y+fy))
					}
				}
			}
		})
	}
}

// TestDrawRegionNonZeroBounds checks images and destinations with non-zero
// bounds origins, including a Gray source (zero-copy sub-image) and a Paletted
// source (generic path, copied sub-image).
func TestDrawRegionNonZeroBounds(t *testing.T) {
	srcb := image.Rect(-30, -20, 50, 40)

	sources := map[string]image.Image{}
	nrgba := image.NewNRGBA(srcb)
	rnd := rand.New(rand.NewSource(7))
	rnd.Read(nrgba.Pix)
	sources["NRGBA"] = nrgba

	gray := image.NewGray(srcb)
	rnd.Read(gray.Pix)
	sources["Gray"] = gray

	paletted := image.NewPaletted(srcb, color.Palette{
		color.NRGBA{0, 0, 0, 255},
		color.NRGBA{255, 0, 0, 255},
		color.NRGBA{0, 255, 0, 255},
		color.NRGBA{0, 0, 255, 255},
		color.NRGBA{255, 255, 255, 255},
	})
	rnd.Read(paletted.Pix)
	for i := range paletted.Pix {
		paletted.Pix[i] %= uint8(len(paletted.Palette))
	}
	sources["Paletted"] = paletted

	filters := []Filter{
		GaussianBlur(1.5),
		Grayscale(),
		Median(3, false),
	}
	regions := []image.Rectangle{
		image.Rect(-30, -20, 50, 40),
		image.Rect(-30, -20, -10, 0),
		image.Rect(30, 20, 50, 40),
		image.Rect(-5, -7, 25, 13),
	}

	for sname, src := range sources {
		src := src
		t.Run(sname, func(t *testing.T) {
			g := New(filters...)
			full := drawFull(g, src)

			// Destination with its own non-zero origin, smaller than the result.
			dstb := image.Rect(100, 200, 100+srcb.Dx(), 200+srcb.Dy())
			for ri, region := range regions {
				dst := image.NewNRGBA(dstb)
				if err := g.DrawRegion(dst, src, region); err != nil {
					t.Fatalf("region #%d DrawRegion failed: %v", ri, err)
				}
				// Viewport semantics: the clipped region is placed with its
				// top-left corner at dst's Min point.
				win := region.Intersect(full.Bounds())
				if win.Empty() {
					continue
				}
				wantR := win.Sub(win.Min).Add(dstb.Min).Intersect(dstb)
				for y := wantR.Min.Y; y < wantR.Max.Y; y++ {
					for x := wantR.Min.X; x < wantR.Max.X; x++ {
						got := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
						// full has origin (0,0); map back through the window.
						want := color.NRGBAModel.Convert(full.At(
							win.Min.X+x-wantR.Min.X,
							win.Min.Y+y-wantR.Min.Y,
						)).(color.NRGBA)
						if got != want {
							t.Fatalf("region #%d pixel (%d,%d): got %v want %v", ri, x, y, got, want)
						}
					}
				}
			}
		})
	}
}

// TestDrawRegionTiles renders adjacent local requests that cover the whole
// image and verifies that the stitched result is identical to the full draw,
// including at tile borders.
func TestDrawRegionTiles(t *testing.T) {
	const w, h = 130, 90
	src := newRandomNRGBA(image.Rect(0, 0, w, h), 99)

	chains := map[string][]Filter{
		"blur": {GaussianBlur(2), Invert()},
		"convolve": {Convolution([]float32{
			-1, -1, -1,
			-1, 9, -1,
			-1, -1, -1,
		}, false, true, false, 0)},
		"median":   {Median(7, true)},
		"pixelate": {Pixelate(6)},
	}

	// Tile grid with offsets that are not multiples of any kernel size or
	// block size, so borders cut through halo neighborhoods.
	tileW, tileH := 32, 24

	for name, filters := range chains {
		filters := filters
		t.Run(name, func(t *testing.T) {
			for _, parallel := range []bool{true, false} {
				g := New(filters...)
				g.SetParallelization(parallel)
				full := drawFull(g, src)
				stitched := image.NewNRGBA(full.Bounds())

				for ty := 0; ty < h; ty += tileH {
					for tx := 0; tx < w; tx += tileW {
						region := image.Rect(tx, ty,
							minint(tx+tileW, w), minint(ty+tileH, h))
						tile := image.NewNRGBA(image.Rect(0, 0, region.Dx(), region.Dy()))
						if err := g.DrawRegion(tile, src, region); err != nil {
							t.Fatalf("DrawRegion(%v) failed: %v", region, err)
						}
						// Paste the viewport tile back into its result position.
						pixGetter := newPixelGetter(tile)
						pixSetter := newPixelSetter(stitched)
						for y := 0; y < region.Dy(); y++ {
							for x := 0; x < region.Dx(); x++ {
								pixSetter.setPixel(region.Min.X+x, region.Min.Y+y,
									pixGetter.getPixel(x, y))
							}
						}
					}
				}

				mismatches, fx, fy := pixelDiff(stitched, full, full.Bounds())
				if mismatches != 0 {
					t.Fatalf("parallel=%v: stitched tiles differ at (%d,%d): got %v want %v, mismatches=%d",
						parallel, fx, fy, stitched.At(fx, fy), full.At(fx, fy), mismatches)
				}
			}
		})
	}
}

// TestDrawRegionEmptyIntersection checks that empty intersections are no-ops
// and never return an error for otherwise supported chains.
func TestDrawRegionEmptyIntersection(t *testing.T) {
	src := newRandomNRGBA(image.Rect(0, 0, 40, 30), 5)
	g := New(GaussianBlur(1), Invert())

	emptyRegions := []image.Rectangle{
		image.Rect(40, 0, 50, 10), // right of the image
		image.Rect(0, 30, 10, 40), // below the image
		image.Rect(-20, -20, -1, -1),
		image.Rect(100, 100, 200, 200),
	}
	for i, region := range emptyRegions {
		dst := image.NewNRGBA(g.Bounds(src.Bounds()))
		sentinel := image.NewNRGBA(g.Bounds(src.Bounds()))
		for k := range sentinel.Pix {
			sentinel.Pix[k] = 0xab
		}
		copy(dst.Pix, sentinel.Pix)

		if err := g.DrawRegion(dst, src, region); err != nil {
			t.Fatalf("region #%d returned error: %v", i, err)
		}
		for k := range dst.Pix {
			if dst.Pix[k] != sentinel.Pix[k] {
				t.Fatalf("region #%d modified dst despite empty intersection", i)
			}
		}
	}

	// Region non-empty in result space but fully outside dst bounds.
	dst := image.NewNRGBA(image.Rect(0, 0, 5, 5))
	if err := g.DrawRegion(dst, src, image.Rect(20, 20, 25, 25)); err != nil {
		t.Fatalf("out-of-dst region returned error: %v", err)
	}
}

// TestDrawRegionDstClipping renders regions larger than the destination and
// verifies that only the intersecting pixels are written and that they match
// the full draw. Regions are viewports aligned with dst's Min point.
func TestDrawRegionDstClipping(t *testing.T) {
	src := newRandomNRGBA(image.Rect(0, 0, 40, 30), 17)
	g := New(GaussianBlur(1.5), Invert())
	full := drawFull(g, src)

	// Region covers the whole result but dst only holds a small window with
	// a non-zero origin. The dst receives the top-left corner of the result.
	dstb := image.Rect(7, 9, 15, 14)
	dst := image.NewNRGBA(dstb)
	region := image.Rect(-100, -100, 100, 100)
	if err := g.DrawRegion(dst, src, region); err != nil {
		t.Fatal(err)
	}
	for y := dstb.Min.Y; y < dstb.Max.Y; y++ {
		for x := dstb.Min.X; x < dstb.Max.X; x++ {
			got := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
			want := color.NRGBAModel.Convert(full.At(x-dstb.Min.X, y-dstb.Min.Y)).(color.NRGBA)
			if got != want {
				t.Fatalf("clipped pixel (%d,%d): got %v want %v", x, y, got, want)
			}
		}
	}

	// Partial overlap at a corner: the requested region extends past the
	// result; the visible window is the 4x4 bottom-right part of the result
	// and dst (the viewport) receives exactly those pixels.
	win := image.Rect(36, 26, 40, 30)
	dstb2 := image.Rect(0, 0, 4, 4)
	dst2 := image.NewNRGBA(dstb2)
	if err := g.DrawRegion(dst2, src, image.Rect(36, 26, 46, 36)); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			got := color.NRGBAModel.Convert(dst2.At(x, y)).(color.NRGBA)
			wantPx := color.NRGBAModel.Convert(full.At(win.Min.X+x, win.Min.Y+y)).(color.NRGBA)
			if got != wantPx {
				t.Fatalf("corner overlap pixel (%d,%d): got %v want %v", x, y, got, wantPx)
			}
		}
	}

	// Dst smaller than the visible window: only its top-left sub-window is
	// written.
	dstb3 := image.Rect(10, 10, 12, 12)
	dst3 := image.NewNRGBA(dstb3)
	if err := g.DrawRegion(dst3, src, win); err != nil {
		t.Fatal(err)
	}
	for y := dstb3.Min.Y; y < dstb3.Max.Y; y++ {
		for x := dstb3.Min.X; x < dstb3.Max.X; x++ {
			got := color.NRGBAModel.Convert(dst3.At(x, y)).(color.NRGBA)
			wantPx := color.NRGBAModel.Convert(full.At(
				win.Min.X+x-dstb3.Min.X, win.Min.Y+y-dstb3.Min.Y,
			)).(color.NRGBA)
			if got != wantPx {
				t.Fatalf("small dst pixel (%d,%d): got %v want %v", x, y, got, wantPx)
			}
		}
	}
}

// unsupportedFilter is a dimension-preserving filter without FilterSupport.
type unsupportedFilter struct{}

func (p *unsupportedFilter) Bounds(srcBounds image.Rectangle) image.Rectangle {
	return image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
}
func (p *unsupportedFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	copyimage(dst, src, options)
}

// TestDrawRegionUnsupported checks that filters with unknown input support
// produce an explicit error instead of silently rendering seam-prone output.
func TestDrawRegionUnsupported(t *testing.T) {
	src := newRandomNRGBA(image.Rect(0, 0, 32, 24), 11)

	unsupportedChains := map[string][]Filter{
		"resize":       {Resize(64, 48, LinearResampling)},
		"rotate":       {Rotate90()},
		"flip":         {FlipHorizontal()},
		"crop":         {Crop(image.Rect(0, 0, 10, 10))},
		"custom":       {&unsupportedFilter{}},
		"mixed_bad":    {Invert(), Resize(40, 0, LinearResampling), GaussianBlur(1)},
		"rotate_angle": {Rotate(30, color.Black, LinearInterpolation)},
	}

	for name, filters := range unsupportedChains {
		filters := filters
		t.Run(name, func(t *testing.T) {
			g := New(filters...)
			dst := image.NewNRGBA(g.Bounds(src.Bounds()))
			before := append([]byte(nil), dst.Pix...)
			err := g.DrawRegion(dst, src, image.Rect(2, 2, 10, 8))
			if !errors.Is(err, ErrRegionUnsupported) {
				t.Fatalf("expected ErrRegionUnsupported, got %v", err)
			}
			// On error the destination must not have been touched.
			for i := range dst.Pix {
				if dst.Pix[i] != before[i] {
					t.Fatalf("dst was modified after an unsupported-chain error")
				}
			}
		})
	}
}

// TestDrawRegionOrFullFallback checks that the fallback helper renders exact
// pixels (cropped full draw, viewport-aligned) for chains that cannot do
// region rendering.
func TestDrawRegionOrFullFallback(t *testing.T) {
	src := newRandomNRGBA(image.Rect(0, 0, 48, 36), 13)
	region := image.Rect(10, 8, 30, 26)

	// Unsupported chain (resize): must take the full-draw fallback.
	g := New(Resize(96, 72, LanczosResampling), GaussianBlur(1))
	full := drawFull(g, src)

	dst := image.NewNRGBA(image.Rect(0, 0, region.Dx(), region.Dy()))
	if err := g.DrawRegionOrFull(dst, src, region); err != nil {
		t.Fatalf("DrawRegionOrFull failed: %v", err)
	}
	for y := 0; y < region.Dy(); y++ {
		for x := 0; x < region.Dx(); x++ {
			got := color.NRGBAModel.Convert(dst.At(x, y)).(color.NRGBA)
			want := color.NRGBAModel.Convert(full.At(region.Min.X+x, region.Min.Y+y)).(color.NRGBA)
			if got != want {
				t.Fatalf("fallback pixel (%d,%d): got %v want %v", x, y, got, want)
			}
		}
	}

	// A supported chain must go through the region path and still match.
	g2 := New(GaussianBlur(1.5), Invert())
	full2 := drawFull(g2, src)
	dst2 := image.NewNRGBA(image.Rect(0, 0, region.Dx(), region.Dy()))
	if err := g2.DrawRegionOrFull(dst2, src, region); err != nil {
		t.Fatalf("DrawRegionOrFull failed: %v", err)
	}
	for y := 0; y < region.Dy(); y++ {
		for x := 0; x < region.Dx(); x++ {
			got := color.NRGBAModel.Convert(dst2.At(x, y)).(color.NRGBA)
			want := color.NRGBAModel.Convert(full2.At(region.Min.X+x, region.Min.Y+y)).(color.NRGBA)
			if got != want {
				t.Fatalf("region path pixel (%d,%d): got %v want %v", x, y, got, want)
			}
		}
	}
}

// TestDrawRegionHaloCumulation checks multi-stage halo back-propagation with
// an all-convolution chain: the required source rectangle must grow through
// every stage and boundary pixels still match the full draw exactly.
func TestDrawRegionHaloCumulation(t *testing.T) {
	src := newRandomNRGBA(image.Rect(0, 0, 60, 60), 21)
	// Three 3x3 convolutions: total halo 3 pixels.
	kernel := []float32{
		1, 1, 1,
		1, 2, 1,
		1, 1, 1,
	}
	g := New(
		Convolution(kernel, true, true, false, 0),
		Convolution(kernel, true, true, false, 0),
		Convolution(kernel, true, true, false, 0),
	)
	full := drawFull(g, src)

	// One-pixel target region at the image border: back-propagation reaches
	// 3 pixels outside the target but still inside the source.
	region := image.Rect(0, 0, 1, 1)
	got := drawRegion(t, g, src, region)
	if m, _, _ := viewportDiff(got, full, region); m != 0 {
		t.Fatalf("border pixel mismatch after cumulative halo: %d", m)
	}

	// Interior target region, check all four edges.
	region = image.Rect(20, 20, 25, 25)
	got = drawRegion(t, g, src, region)
	if m, fx, fy := viewportDiff(got, full, region); m != 0 {
		t.Fatalf("interior region mismatch at (%d,%d): %d", fx, fy, m)
	}
}

// TestDrawRegionEachFilter checks every dimension-preserving filter
// individually: region output must equal the full draw on both an interior
// region and a border region.
func TestDrawRegionEachFilter(t *testing.T) {
	src := newRandomNRGBA(image.Rect(0, 0, 73, 57), 77)
	filters := map[string]Filter{
		"invert":           Invert(),
		"srgb_to_linear":   ColorspaceSRGBToLinear(),
		"linear_to_srgb":   ColorspaceLinearToSRGB(),
		"gamma":            Gamma(1.7),
		"sigmoid":          Sigmoid(0.5, 4),
		"contrast":         Contrast(25),
		"brightness":       Brightness(-15),
		"grayscale":        Grayscale(),
		"sepia":            Sepia(60),
		"hue":              Hue(120),
		"saturation":       Saturation(40),
		"colorize":         Colorize(90, 40, 80),
		"color_balance":    ColorBalance(10, -5, 20),
		"threshold":        Threshold(50),
		"color_func":       ColorFunc(func(r, g, b, a float32) (float32, float32, float32, float32) { return g, b, r, a }),
		"convolution":      Convolution([]float32{1, 0, -1, 2, 0, -2, 1, 0, -1}, false, true, false, 0),
		"convolution_even": Convolution([]float32{1, 1, 1, 1}, true, false, false, 0),
		"gaussian_blur":    GaussianBlur(2),
		"gaussian_zero":    GaussianBlur(0),
		"unsharp_mask":     UnsharpMask(1.5, 1.1, 0.02),
		"mean_square":      Mean(5, false),
		"mean_disk":        Mean(5, true),
		"mean_tiny":        Mean(1, false),
		"median":           Median(5, false),
		"minimum":          Minimum(3, false),
		"maximum_disk":     Maximum(5, true),
		"sobel":            Sobel(),
		"pixelate":         Pixelate(8),
		"pixelate_tiny":    Pixelate(1),
	}
	regions := []image.Rectangle{
		image.Rect(40, 30, 73, 57), // bottom-right border
		image.Rect(0, 0, 13, 9),    // top-left border
		image.Rect(20, 15, 45, 40), // interior
	}

	for name, f := range filters {
		f := f
		t.Run(name, func(t *testing.T) {
			g := New(f)
			full := drawFull(g, src)
			for ri, region := range regions {
				got := drawRegion(t, g, src, region)
				if m, fx, fy := viewportDiff(got, full, region); m != 0 {
					t.Fatalf("region #%d: %d mismatches, first at viewport (%d,%d): got %v want %v",
						ri, m, fx, fy, got.At(fx, fy), full.At(region.Min.X+fx, region.Min.Y+fy))
				}
			}
		})
	}
}

// TestDrawRegionPixelateNegativeOrigin checks pixelation block alignment at
// negative coordinates where truncated (Go) division would miscompute blocks.
func TestDrawRegionPixelateNegativeOrigin(t *testing.T) {
	srcb := image.Rect(-23, -17, 13, 19)
	src := image.NewNRGBA(srcb)
	rnd := rand.New(rand.NewSource(3))
	rnd.Read(src.Pix)

	g := New(Pixelate(6), Invert())
	full := drawFull(g, src)

	regions := []image.Rectangle{
		image.Rect(-23, -17, 13, 19),
		image.Rect(-23, -17, -10, -5),
		image.Rect(0, 0, 13, 19),
		image.Rect(-7, -3, 5, 8),
	}
	for ri, region := range regions {
		got := drawRegion(t, g, src, region)
		win := region.Intersect(full.Bounds())
		if m, fx, fy := viewportDiff(got, full, win); m != 0 {
			t.Fatalf("region #%d: %d mismatches, first at viewport (%d,%d)", ri, m, fx, fy)
		}
	}

	// Directly verify the block math on a few negative coordinates.
	p := Pixelate(6).(*pixelateFilter)
	s := p.Support(image.Rect(-1, -1, 1, 1))
	if s.Min.X != -6 || s.Min.Y != -6 || s.Max.X != 6 || s.Max.Y != 6 {
		t.Fatalf("unexpected support at negative origin: %v", s)
	}
}

// TestDrawRegionParallelConsistency verifies that the parallel and serial
// modes produce identical output (the parallelization switch must not change
// region rendering results).
func TestDrawRegionParallelConsistency(t *testing.T) {
	src := newRandomNRGBA(image.Rect(0, 0, 100, 80), 33)
	region := image.Rect(17, 13, 73, 61)
	filters := []Filter{GaussianBlur(3), Median(5, true), UnsharpMask(1.5, 1, 0)}

	gP := New(filters...)
	gP.SetParallelization(true)
	dstP := image.NewNRGBA(image.Rect(0, 0, region.Dx(), region.Dy()))
	if err := gP.DrawRegion(dstP, src, region); err != nil {
		t.Fatal(err)
	}

	gS := New(filters...)
	gS.SetParallelization(false)
	dstS := image.NewNRGBA(image.Rect(0, 0, region.Dx(), region.Dy()))
	if err := gS.DrawRegion(dstS, src, region); err != nil {
		t.Fatal(err)
	}

	if m, fx, fy := pixelDiff(dstP, dstS, dstP.Bounds()); m != 0 {
		t.Fatalf("parallel/serial mismatch at (%d,%d) count=%d", fx, fy, m)
	}
}

// TestDrawRegionMemoryComparison is the "big image, small viewport" check:
// it records peak heap usage of a full draw versus region rendering of a
// small bottom-right viewport and writes the measurements to artifacts/.
func TestDrawRegionMemoryComparison(t *testing.T) {
	const w, h = 4000, 3000
	const vw, vh = 320, 240

	src := newRandomNRGBA(image.Rect(0, 0, w, h), 123)
	region := image.Rect(w-vw, h-vh, w, h)
	filters := []Filter{
		GaussianBlur(2),
		Convolution([]float32{
			0, -1, 0,
			-1, 5, -1,
			0, -1, 0,
		}, false, true, false, 0),
		UnsharpMask(1, 1.2, 0.01),
		Grayscale(),
	}

	// measure runs fn with the garbage collector disabled so that all
	// allocations stay live on the heap, and returns both the heap growth
	// caused by the run (a deterministic peak lower bound) and the total
	// bytes allocated during the run.
	measure := func(fn func()) (peakInuse, totalAlloc uint64) {
		runtime.GC()
		oldPercent := debug.SetGCPercent(-1)
		defer debug.SetGCPercent(oldPercent)

		var base runtime.MemStats
		runtime.ReadMemStats(&base)
		fn()
		var after runtime.MemStats
		runtime.ReadMemStats(&after)
		return after.HeapInuse - base.HeapInuse, after.TotalAlloc - base.TotalAlloc
	}

	fullPeak, fullBytes := measure(func() {
		g := New(filters...)
		dst := image.NewNRGBA(g.Bounds(src.Bounds()))
		g.Draw(dst, src)
	})

	regionPeak, regionBytes := measure(func() {
		g := New(filters...)
		dst := image.NewNRGBA(image.Rect(0, 0, vw, vh))
		if err := g.DrawRegion(dst, src, region); err != nil {
			t.Fatal(err)
		}
	})
	// The region path must not allocate anything close to full-image scale:
	// allow at most 10% of the full draw's allocations.
	ratio := float64(regionBytes) / float64(fullBytes)
	peakRatio := float64(regionPeak) / float64(fullPeak)
	t.Logf("big image %dx%d, viewport %dx%d:", w, h, vw, vh)
	t.Logf("  full draw:   peak heap %d bytes, total alloc %d bytes", fullPeak, fullBytes)
	t.Logf("  region draw: peak heap %d bytes, total alloc %d bytes", regionPeak, regionBytes)
	t.Logf("  ratios: peak %.2f%%, total %.2f%%", peakRatio*100, ratio*100)
	if ratio > 0.1 {
		t.Fatalf("region rendering allocated %.1f%% of the full draw, expected < 10%%", ratio*100)
	}

	// Correctness on the same chain, small image (the 120MP image above is
	// too expensive to run twice for an exact comparison).
	small := newRandomNRGBA(image.Rect(0, 0, 200, 150), 123)
	g := New(filters...)
	full := drawFull(g, small)
	smallRegion := image.Rect(200-vw, 150-vh, 200, 150)
	got := drawRegion(t, g, small, smallRegion)
	smallWin := smallRegion.Intersect(full.Bounds())
	if m, fx, fy := viewportDiff(got, full, smallWin); m != 0 {
		t.Fatalf("viewport pixels differ from full draw at viewport (%d,%d), count=%d", fx, fy, m)
	}

	if err := os.MkdirAll("artifacts", 0o755); err != nil {
		t.Fatal(err)
	}
	report, err := os.Create("artifacts/memory_report.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer report.Close()
	if _, err := report.WriteString(
		"region rendering memory comparison\n" +
			"source image: 4000x3000 NRGBA (48.0 MB)\n" +
			"viewport: 320x240 bottom-right\n" +
			"chain: GaussianBlur(2) -> Convolution(3x3) -> UnsharpMask(1) -> Grayscale\n\n" +
			"measurements taken with GC disabled (debug.SetGCPercent(-1));\n" +
			"peak heap is the HeapInuse growth over the baseline, a lower\n" +
			"bound on the true peak.\n\n" +
			"full draw peak heap:          " + strconv.FormatUint(fullPeak, 10) + " bytes\n" +
			"region draw peak heap:        " + strconv.FormatUint(regionPeak, 10) + " bytes\n" +
			"peak ratio:                   " + strconv.FormatFloat(peakRatio, 'f', 6, 64) + "\n\n" +
			"full draw total allocations:   " + strconv.FormatUint(fullBytes, 10) + " bytes\n" +
			"region draw total allocations: " + strconv.FormatUint(regionBytes, 10) + " bytes\n" +
			"total ratio:                   " + strconv.FormatFloat(ratio, 'f', 6, 64) + "\n",
	); err != nil {
		t.Fatal(err)
	}
}
