# GO IMAGE FILTERING TOOLKIT (GIFT)

[![GoDoc](https://godoc.org/github.com/disintegration/gift?status.svg)](https://godoc.org/github.com/disintegration/gift)
[![Build Status](https://travis-ci.org/disintegration/gift.svg?branch=master)](https://travis-ci.org/disintegration/gift)
[![Coverage Status](https://coveralls.io/repos/github/disintegration/gift/badge.svg?branch=master)](https://coveralls.io/github/disintegration/gift?branch=master)

*Package gift provides a set of useful image processing filters.*

Pure Go. No external dependencies outside of the Go standard library.


### INSTALLATION / UPDATING

    go get -u github.com/disintegration/gift


### DOCUMENTATION

http://godoc.org/github.com/disintegration/gift


### QUICK START

```go
// 1. Create a new GIFT filter list and add some filters:
g := gift.New(
	gift.Resize(800, 0, gift.LanczosResampling),
	gift.UnsharpMask(1, 1, 0),
)

// 2. Create a new image of the corresponding size.
// dst is a new target image, src is the original image
dst := image.NewRGBA(g.Bounds(src.Bounds()))

// 3. Use Draw func to apply the filters to src and store the result in dst:
g.Draw(dst, src)
``` 

### USAGE

To create a sequence of filters, the `New` function is used:
```go
g := gift.New(
	gift.Grayscale(),
	gift.Contrast(10),
)
```
Filters also can be added using the `Add` method:
```go
g.Add(GaussianBlur(2)) 
```

The `Bounds` method takes the bounds of the source image and returns appropriate bounds for the destination image to fit the result (for example, after using `Resize` or `Rotate` filters). 

```go
dst := image.NewRGBA(g.Bounds(src.Bounds()))
```

There are two methods available to apply these filters to an image:

- `Draw` applies all the added filters to the src image and outputs the result to the dst image starting from the top-left corner (Min point).
```go
g.Draw(dst, src)
```

- `DrawAt` provides more control. It outputs the filtered src image to the dst image at the specified position using the specified image composition operator. This example is equivalent to the previous:
```go
g.DrawAt(dst, src, dst.Bounds().Min, gift.CopyOperator)
```

Two image composition operators are supported by now:
- `CopyOperator` - Replaces pixels of the dst image with pixels of the filtered src image. This mode is used by the Draw method.
- `OverOperator` - Places the filtered src image on top of the dst image. This mode makes sence if the filtered src image has transparent areas.

Empty filter list can be used to create a copy of an image or to paste one image to another. For example:
```go
// Create a new image with dimensions of the bgImage.
dstImage := image.NewRGBA(bgImage.Bounds())
// Copy the bgImage to the dstImage.
gift.New().Draw(dstImage, bgImage)
// Draw the fgImage over the dstImage at the (100, 100) position.
gift.New().DrawAt(dstImage, fgImage, image.Pt(100, 100), gift.OverOperator)
```


### SUPPORTED FILTERS

+ Transformations

    - Crop(rect image.Rectangle)
    - CropToSize(width, height int, anchor Anchor)
    - FlipHorizontal()
    - FlipVertical()
    - Resize(width, height int, resampling Resampling)
    - ResizeToFill(width, height int, resampling Resampling, anchor Anchor)
    - ResizeToFit(width, height int, resampling Resampling)
    - Rotate(angle float32, backgroundColor color.Color, interpolation Interpolation)
    - Rotate180()
    - Rotate270()
    - Rotate90()
    - Transpose()
    - Transverse()
    
+ Adjustments & effects

    - Brightness(percentage float32)
    - ColorBalance(percentageRed, percentageGreen, percentageBlue float32)
    - ColorFunc(fn func(r0, g0, b0, a0 float32) (r, g, b, a float32))
    - Colorize(hue, saturation, percentage float32)
    - ColorspaceLinearToSRGB()
    - ColorspaceSRGBToLinear()
    - Contrast(percentage float32)
    - Convolution(kernel []float32, normalize, alpha, abs bool, delta float32)
    - Gamma(gamma float32)
    - GaussianBlur(sigma float32)
    - Grayscale()
    - Hue(shift float32)
    - Invert()
    - Maximum(ksize int, disk bool)
    - Mean(ksize int, disk bool)
    - Median(ksize int, disk bool)
    - Minimum(ksize int, disk bool)
    - Pixelate(size int)
    - Saturation(percentage float32)
    - Sepia(percentage float32)
    - Sigmoid(midpoint, factor float32)
    - Sobel()
    - UnsharpMask(sigma, amount, threshold float32)


### FILTER EXAMPLES

The original image:

![](testdata/src.png) 

Resulting images after applying some of the filters:

 name / result                              | name / result                              | name / result                              | name / result
--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------
resize                                      | crop_to_size                               | rotate_180                                 | rotate_30
![](testdata/dst_resize.png)                | ![](testdata/dst_crop_to_size.png)         | ![](testdata/dst_rotate_180.png)           | ![](testdata/dst_rotate_30.png)
brightness_increase                         | brightness_decrease                        | contrast_increase                          | contrast_decrease
![](testdata/dst_brightness_increase.png)   | ![](testdata/dst_brightness_decrease.png)  | ![](testdata/dst_contrast_increase.png)    | ![](testdata/dst_contrast_decrease.png)
saturation_increase                         | saturation_decrease                        | gamma_1.5                                  | gamma_0.5
![](testdata/dst_saturation_increase.png)   | ![](testdata/dst_saturation_decrease.png)  | ![](testdata/dst_gamma_1.5.png)            | ![](testdata/dst_gamma_0.5.png)
gaussian_blur                               | unsharp_mask                               | sigmoid                                    | pixelate
![](testdata/dst_gaussian_blur.png)         | ![](testdata/dst_unsharp_mask.png)         | ![](testdata/dst_sigmoid.png)              | ![](testdata/dst_pixelate.png)
colorize                                    | grayscale                                  | sepia                                      | invert
![](testdata/dst_colorize.png)              | ![](testdata/dst_grayscale.png)            | ![](testdata/dst_sepia.png)                | ![](testdata/dst_invert.png)
mean                                        | median                                     | minimum                                    | maximum
![](testdata/dst_mean.png)                  | ![](testdata/dst_median.png)               | ![](testdata/dst_minimum.png)              | ![](testdata/dst_maximum.png)
hue_rotate                                  | color_balance                              | color_func                                 | convolution_emboss
![](testdata/dst_hue_rotate.png)            | ![](testdata/dst_color_balance.png)        | ![](testdata/dst_color_func.png)           | ![](testdata/dst_convolution_emboss.png)

Here's the code that produces the above images:

```go
package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"os"

	"github.com/disintegration/gift"
)

func main() {
	src := loadImage("testdata/src.png")

	filters := map[string]gift.Filter{
		"resize":               gift.Resize(100, 0, gift.LanczosResampling),
		"crop_to_size":         gift.CropToSize(100, 100, gift.LeftAnchor),
		"rotate_180":           gift.Rotate180(),
		"rotate_30":            gift.Rotate(30, color.Transparent, gift.CubicInterpolation),
		"brightness_increase":  gift.Brightness(30),
		"brightness_decrease":  gift.Brightness(-30),
		"contrast_increase":    gift.Contrast(30),
		"contrast_decrease":    gift.Contrast(-30),
		"saturation_increase":  gift.Saturation(50),
		"saturation_decrease":  gift.Saturation(-50),
		"gamma_1.5":            gift.Gamma(1.5),
		"gamma_0.5":            gift.Gamma(0.5),
		"gaussian_blur":        gift.GaussianBlur(1),
		"unsharp_mask":         gift.UnsharpMask(1, 1, 0),
		"sigmoid":              gift.Sigmoid(0.5, 7),
		"pixelate":             gift.Pixelate(5),
		"colorize":             gift.Colorize(240, 50, 100),
		"grayscale":            gift.Grayscale(),
		"sepia":                gift.Sepia(100),
		"invert":               gift.Invert(),
		"mean":                 gift.Mean(5, true),
		"median":               gift.Median(5, true),
		"minimum":              gift.Minimum(5, true),
		"maximum":              gift.Maximum(5, true),
		"hue_rotate":           gift.Hue(45),
		"color_balance":        gift.ColorBalance(10, -10, -10),
		"color_func": gift.ColorFunc(
			func(r0, g0, b0, a0 float32) (r, g, b, a float32) {
				r = 1 - r0   // invert the red channel
				g = g0 + 0.1 // shift the green channel by 0.1
				b = 0        // set the blue channel to 0
				a = a0       // preserve the alpha channel
				return
			},
		),
		"convolution_emboss": gift.Convolution(
			[]float32{
				-1, -1, 0,
				-1, 1, 1,
				0, 1, 1,
			},
			false, false, false, 0.0,
		),
	}

	for name, filter := range filters {
		g := gift.New(filter)
		dst := image.NewNRGBA(g.Bounds(src.Bounds()))
		g.Draw(dst, src)
		saveImage("testdata/dst_"+name+".png", dst)
	}
}

func loadImage(filename string) image.Image {
	f, err := os.Open(filename)
	if err != nil {
		log.Fatalf("os.Open failed: %v", err)
	}
	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatalf("image.Decode failed: %v", err)
	}
	return img
}

func saveImage(filename string, img image.Image) {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatalf("os.Create failed: %v", err)
	}
	err = png.Encode(f, img)
	if err != nil {
		log.Fatalf("png.Encode failed: %v", err)
	}
}
```

### TESTING

The package keeps the original PNG golden tests (`TestGolden`, `testdata/`) and
adds a system-level regression suite for the parallel (multi-goroutine) image
processing paths in `parallel_regression_test.go`.

A single golden image is not sufficient to catch chunk-boundary bugs: it is
produced with one set of bounds and one processor count, so a row that is
dropped or computed from the wrong neighbour at an internal chunk edge can
still pass. The parallel suite instead renders each case twice — once with
`SetParallelization(false)` (the serial reference) and once with
parallelization enabled — and requires identical destination bounds and
identical pixels. It also snapshots the complete backing buffer of every
source image before and after drawing to prove the input is never modified.

Covered by the matrix:

+ Image formats: RGBA, NRGBA and Gray16.
+ Source geometry: direct images and `SubImage`s, with zero and non-zero
  `Bounds().Min`, plus large images whose chunks span several rows/columns.
+ Filters and combinations: resize (nearest, box, linear, Lanczos), rotate
  (nearest, linear, cubic), 90/180/270 rotations, flips, convolution, Gaussian
  blur, unsharp mask, rank filters, crop/crop-to-size, pixelate and the color
  filters (`Contrast`, `Gamma`, `Hue`, `Colorize`, `Grayscale`, `Invert`,
  `Sepia`, `ColorBalance`, `Sigmoid`).
+ Empty, one-pixel, one-column and one-row images are checked independently.
+ Non-commutative filter pairs are checked in each order, and the test asserts
  that the orders actually produce different pixels (so the check is not
  vacuous).
+ `DrawAt` with `CopyOperator` and `OverOperator`, at positive, negative and
  destination-minimum positions.

All inputs are generated deterministically from a seed. A failure names the
seed, image format, source bounds, filter chain, the exact filter index, and
the first differing coordinate, for example:

    seed=1 variant="RGBA/large-nonzero-origin" srcBounds=(-5,11)-(132,19)
      chain="conv-crop-scale-color-rotate" prefix="convolution/sobel-like"
      filter[0]="convolution/sobel-like" parallel=true:
      pixel mismatch at coord=(1,1) byte-channel=0: serial=0x0c parallel=0x12

Reproduce that exact case with the seed from the message:

    GIFT_TEST_SEED=1 go test -run TestParallel -v

`TestParallelInjectedChunkBugs` proves the comparison is not vacuous: it runs a
correctly chunked control plus two deliberately broken chunk implementations
(an off-by-one exclusive chunk end that drops the last row of every chunk, and
a neighbour lookup clamped to the chunk edge instead of the image edge). Both
must be detected and localized to the first internal chunk boundary row.

Run the whole suite:

    go test ./...
    go test -race ./...

Reproducible parallel runs (the tests pin GOMAXPROCS to 4 so the chunk size is
deterministic; override it in production code paths with the standard
GOMAXPROCS environment variable):

    GO111MODULE=off go test -count=1 -race ./...
    GOMAXPROCS=8 GO111MODULE=off go test -count=1 -race -run TestParallel ./...

The `-race` run is required for any change to `parallelize`, to a filter's
chunking, or to `pixelGetter`/`pixelSetter`. The injected-fault tests are
written to be race-free so the race detector can be used on the same run.
