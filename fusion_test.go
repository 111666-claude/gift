package gift

import (
	"image"
	"image/color"
	"image/draw"
	"testing"
)

// makeFusionTestNRGBA creates a deterministic test image with varied colors
// and alpha values (including fully and partially transparent pixels).
func makeFusionTestNRGBA(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	seed := uint32(0x12345678)
	rnd := func() uint8 {
		seed = seed*1664525 + 1013904223
		return uint8(seed >> 24)
	}
	for i := range img.Pix {
		img.Pix[i] = rnd()
	}
	return img
}

func makeFusionTestPaletted(w, h int) *image.Paletted {
	pal := make(color.Palette, 0, 256)
	seed := uint32(0x87654321)
	for i := 0; i < 256; i++ {
		seed = seed*1664525 + 1013904223
		r := uint8(seed >> 24)
		seed = seed*1664525 + 1013904223
		g := uint8(seed >> 24)
		seed = seed*1664525 + 1013904223
		b := uint8(seed >> 24)
		seed = seed*1664525 + 1013904223
		a := uint8(seed >> 24)
		pal = append(pal, color.NRGBA{r, g, b, a})
	}
	img := image.NewPaletted(image.Rect(0, 0, w, h), pal)
	seed = uint32(0xdeadbeef)
	for i := range img.Pix {
		seed = seed*1664525 + 1013904223
		img.Pix[i] = uint8(seed >> 24)
	}
	return img
}

func makeFusionTestYCbCr(w, h int) *image.YCbCr {
	r := image.Rect(0, 0, w, h)
	img := image.NewYCbCr(r, image.YCbCrSubsampleRatio420)
	seed := uint32(0x0badf00d)
	rnd := func() uint8 {
		seed = seed*1664525 + 1013904223
		return uint8(seed >> 24)
	}
	for i := range img.Y {
		img.Y[i] = rnd()
	}
	for i := range img.Cb {
		img.Cb[i] = rnd()
	}
	for i := range img.Cr {
		img.Cr[i] = rnd()
	}
	return img
}

func fusionTestSources(w, h int) map[string]image.Image {
	nrgba := makeFusionTestNRGBA(w, h)

	rgba := image.NewRGBA(nrgba.Bounds())
	draw.Draw(rgba, rgba.Bounds(), nrgba, image.Point{}, draw.Src)

	nrgba64 := image.NewNRGBA64(nrgba.Bounds())
	draw.Draw(nrgba64, nrgba64.Bounds(), nrgba, image.Point{}, draw.Src)

	rgba64 := image.NewRGBA64(nrgba.Bounds())
	draw.Draw(rgba64, rgba64.Bounds(), nrgba, image.Point{}, draw.Src)

	gray := image.NewGray(nrgba.Bounds())
	draw.Draw(gray, gray.Bounds(), nrgba, image.Point{}, draw.Src)

	gray16 := image.NewGray16(nrgba.Bounds())
	draw.Draw(gray16, gray16.Bounds(), nrgba, image.Point{}, draw.Src)

	return map[string]image.Image{
		"NRGBA":    nrgba,
		"RGBA":     rgba,
		"NRGBA64":  nrgba64,
		"RGBA64":   rgba64,
		"Gray":     gray,
		"Gray16":   gray16,
		"YCbCr":    makeFusionTestYCbCr(w, h),
		"Paletted": makeFusionTestPaletted(w, h),
	}
}

// customFilter is a user-defined filter that is not a pixel transformer.
// The fusion optimization must not reorder or merge filters around it.
type customFilter struct{}

func (p *customFilter) Bounds(srcBounds image.Rectangle) (dstBounds image.Rectangle) {
	return image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
}

func (p *customFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	srcb := src.Bounds()
	dstb := dst.Bounds()
	pixGetter := newPixelGetter(src)
	pixSetter := newPixelSetter(dst)
	for y := srcb.Min.Y; y < srcb.Max.Y; y++ {
		for x := srcb.Min.X; x < srcb.Max.X; x++ {
			px := pixGetter.getPixel(x, y)
			px.r, px.b = px.b, px.r
			pixSetter.setPixel(dstb.Min.X+x-srcb.Min.X, dstb.Min.Y+y-srcb.Min.Y, px)
		}
	}
}

func fusionTestChains() map[string][]Filter {
	return map[string][]Filter{
		"colorchan": {
			Brightness(10),
			Contrast(30),
			Gamma(1.5),
			Invert(),
			ColorspaceSRGBToLinear(),
			ColorspaceLinearToSRGB(),
			Sigmoid(0.5, 4),
		},
		"color": {
			Saturation(50),
			Hue(45),
			Sepia(60),
			Colorize(200, 40, 50),
			ColorBalance(10, -20, 30),
			Grayscale(),
			ColorFunc(func(r0, g0, b0, a0 float32) (r, g, b, a float32) {
				return 1 - r0, g0 + 0.1, b0 * 0.5, a0
			}),
		},
		"identity_stages": {
			Brightness(0),
			Contrast(25),
			Saturation(0),
			Hue(0),
			Saturation(-30),
		},
		"mixed_resize": {
			Brightness(10),
			Contrast(20),
			Resize(40, 30, LanczosResampling),
			Saturation(30),
			Invert(),
		},
		"mixed_convolution": {
			Gamma(2.2),
			Saturation(40),
			GaussianBlur(1),
			Brightness(-10),
			Contrast(15),
		},
		"mixed_custom": {
			Brightness(10),
			Saturation(20),
			&customFilter{},
			Contrast(10),
			Invert(),
		},
		"single": {
			Contrast(40),
		},
	}
}

func imagesIdentical(a, b image.Image) bool {
	if !a.Bounds().Eq(b.Bounds()) {
		return false
	}
	bnd := a.Bounds()
	for y := bnd.Min.Y; y < bnd.Max.Y; y++ {
		for x := bnd.Min.X; x < bnd.Max.X; x++ {
			r0, g0, b0, a0 := a.At(x, y).RGBA()
			r1, g1, b1, a1 := b.At(x, y).RGBA()
			if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
				return false
			}
		}
	}
	return true
}

func TestPixelFilterFusion(t *testing.T) {
	chains := fusionTestChains()

	// Small images take the direct per-pixel path, larger images also
	// exercise the LUT-based path of the color channel filters.
	sizes := []image.Point{
		{16, 16},
		{300, 200},
	}

	for _, size := range sizes {
		sources := fusionTestSources(size.X, size.Y)
		for srcName, src := range sources {
			for chainName, filters := range chains {
				for _, parallel := range []bool{false, true} {
					gRef := New(filters...)
					gRef.SetParallelization(parallel)
					gRef.SetPixelFilterFusion(false)

					gFused := New(filters...)
					gFused.SetParallelization(parallel)
					gFused.SetPixelFilterFusion(true)

					if !gFused.PixelFilterFusion() {
						t.Fatal("pixel filter fusion is not enabled by default")
					}

					for _, dstKind := range []string{"NRGBA", "RGBA"} {
						bnd := gRef.Bounds(src.Bounds())
						var dstRef, dstFused draw.Image
						if dstKind == "NRGBA" {
							dstRef = image.NewNRGBA(bnd)
							dstFused = image.NewNRGBA(bnd)
						} else {
							dstRef = image.NewRGBA(bnd)
							dstFused = image.NewRGBA(bnd)
						}

						gRef.Draw(dstRef, src)
						gFused.Draw(dstFused, src)

						if !imagesIdentical(dstRef, dstFused) {
							t.Errorf(
								"fused output differs: src=%s chain=%s dst=%s size=%v parallel=%v",
								srcName, chainName, dstKind, size, parallel,
							)
						}
					}
				}
			}
		}
	}
}

func TestPixelFilterFusionEmptyAndShort(t *testing.T) {
	src := makeFusionTestNRGBA(32, 24)

	// Empty chain.
	g := New()
	g.SetPixelFilterFusion(true)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)
	if !imagesIdentical(dst, src) {
		t.Error("empty chain with fusion enabled does not copy the source")
	}

	// Fusion can be toggled.
	g.SetPixelFilterFusion(false)
	if g.PixelFilterFusion() {
		t.Error("pixel filter fusion is not disabled")
	}
	g.SetPixelFilterFusion(true)
	if !g.PixelFilterFusion() {
		t.Error("pixel filter fusion is not enabled")
	}

	// A single fusable filter keeps the original filter instance.
	filters := []Filter{Brightness(10)}
	if got := fusePixelFilters(filters); len(got) != 1 || got[0] != filters[0] {
		t.Error("single filter chain must not be fused")
	}

	// A chain without adjacent fusable filters is returned unchanged.
	filters = []Filter{Brightness(10), Resize(10, 10, NearestNeighborResampling), Contrast(10)}
	got := fusePixelFilters(filters)
	if len(got) != 3 {
		t.Fatalf("unexpected fusion of non-adjacent filters: %d filters", len(got))
	}
	for i := range filters {
		if got[i] != filters[i] {
			t.Error("fusion reordered or replaced non-adjacent filters")
		}
	}
}

func benchmarkChain4K() []Filter {
	return []Filter{
		Brightness(5),
		Contrast(10),
		Saturation(20),
		Gamma(1.2),
		ColorBalance(5, -5, 2),
		Sigmoid(0.5, 3),
	}
}

func BenchmarkPixelFilterChain4K(b *testing.B) {
	src := makeFusionTestNRGBA(3840, 2160)

	for _, fusion := range []bool{false, true} {
		name := "separate"
		if fusion {
			name = "fused"
		}
		b.Run(name, func(b *testing.B) {
			g := New(benchmarkChain4K()...)
			g.SetPixelFilterFusion(fusion)
			dst := image.NewNRGBA(g.Bounds(src.Bounds()))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g.Draw(dst, src)
			}
		})
	}
}
