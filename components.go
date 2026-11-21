package gift

import (
	"errors"
	"image"
)

// Connectivity specifies which neighbors are considered connected by
// connected component analysis.
type Connectivity int

// Pixel connectivity modes.
const (
	// Connectivity4 connects pixels that share an edge (horizontal or vertical
	// neighbors).
	Connectivity4 Connectivity = 4

	// Connectivity8 connects pixels that share an edge or a corner (horizontal,
	// vertical or diagonal neighbors).
	Connectivity8 Connectivity = 8
)

// Errors returned by Components and LabelComponents.
var (
	// ErrNilImage is returned when the source image is nil.
	ErrNilImage = errors.New("gift: nil source image")

	// ErrInvalidAlphaThreshold is returned when the alpha threshold is outside
	// the range [0, 255].
	ErrInvalidAlphaThreshold = errors.New("gift: alpha threshold must be in range [0, 255]")

	// ErrInvalidConnectivity is returned when the connectivity is neither
	// Connectivity4 nor Connectivity8.
	ErrInvalidConnectivity = errors.New("gift: connectivity must be Connectivity4 or Connectivity8")

	// ErrTooManyComponents is returned by LabelComponents when the number of
	// components cannot be represented by an image.Gray16 label image.
	ErrTooManyComponents = errors.New("gift: too many components for a 16-bit label image")
)

// maxComponentLabel is the largest label that fits in image.Gray16.
const maxComponentLabel = 0xffff

// Component describes a connected region of pixels with non-zero alpha.
type Component struct {
	// Label is the 1-based label of the component. It is the value stored for
	// the component's pixels in the image returned by LabelComponents.
	Label int

	// Bounds is the component's bounding box in the source image coordinates.
	Bounds image.Rectangle

	// NumPixels is the number of pixels that belong to the component.
	NumPixels int

	// CentroidX and CentroidY are the coordinates of the component centroid in
	// the source image coordinate space. The centroid is the mean of the
	// coordinates of all pixels in the component.
	CentroidX, CentroidY float64
}

// Components finds connected components of visible pixels. A pixel belongs to
// a component when its 8-bit alpha value is greater than or equal to
// alphaThreshold. Fully transparent pixels are never part of a component, so a
// threshold of 0 selects every pixel with non-zero alpha and a threshold of 255
// selects only fully opaque pixels.
//
// The function inspects only the alpha channel of the source image. Images
// without an alpha channel (such as image.Gray or image.YCbCr) are treated as
// fully opaque.
//
// The connectivity parameter selects 4-way (edge) or 8-way (edge or corner)
// adjacency. Both parameters are validated; an invalid value results in an
// error.
//
// Components are numbered in the order in which their first pixel is
// encountered in a top-to-bottom, left-to-right scan of the source image.
// The returned slice is in the same order and each Component.Label is its
// 1-based position in the slice. The result is stable for a given source image
// and set of parameters.
//
// Bounds and centroids are expressed in source image coordinates, including
// images whose Bounds do not start at (0, 0). An empty or fully transparent
// image returns an empty slice and a nil error.
func Components(src image.Image, alphaThreshold int, connectivity Connectivity) ([]Component, error) {
	runs, err := analyzeComponents(src, alphaThreshold, connectivity)
	if err != nil {
		return nil, err
	}
	return buildComponents(runs), nil
}

// LabelComponents finds connected components in the same way as Components and
// also returns an image.Gray16 where pixel values are the component labels.
//
// The label image has the same bounds as the source image. Background pixels
// (including fully transparent pixels) have label 0 and component pixels have
// labels from 1 to N, where N is the number of returned components. Component
// labels follow the same top-to-bottom, left-to-right first-pixel ordering as
// Components.
//
// image.Gray16 can represent at most 65535 components. If the source image
// contains more, ErrTooManyComponents is returned.
func LabelComponents(src image.Image, alphaThreshold int, connectivity Connectivity) (*image.Gray16, []Component, error) {
	runs, err := analyzeComponents(src, alphaThreshold, connectivity)
	if err != nil {
		return nil, nil, err
	}

	comps := buildComponents(runs)
	if len(comps) > maxComponentLabel {
		return nil, nil, ErrTooManyComponents
	}

	dst := image.NewGray16(normalizedBounds(src.Bounds()))
	writeComponentLabels(dst, runs)

	return dst, comps, nil
}

// ccRun is a horizontal run of foreground pixels. Components are represented by
// merging runs using a union-find structure, so no per-pixel objects are
// allocated and sparse images only pay for the runs they contain.
type ccRun struct {
	parent    int32
	order     int32 // component index, -1 until assigned
	x0, x1    int   // pixel range [x0, x1)
	y         int
	numPixels int
	sumX      int64
	sumY      int64
	minX      int
	minY      int
	maxX      int
	maxY      int
}

func analyzeComponents(src image.Image, alphaThreshold int, connectivity Connectivity) ([]ccRun, error) {
	if src == nil {
		return nil, ErrNilImage
	}
	if alphaThreshold < 0 || alphaThreshold > 255 {
		return nil, ErrInvalidAlphaThreshold
	}
	if connectivity != Connectivity4 && connectivity != Connectivity8 {
		return nil, ErrInvalidConnectivity
	}

	bounds := src.Bounds()
	if bounds.Empty() {
		return nil, nil
	}

	return scanComponents(newPixelGetter(src), bounds, uint32(alphaThreshold), connectivity), nil
}

// alphaVisible reports whether an alpha value is part of a component. The
// threshold is an 8-bit value, so a 16-bit alpha is reduced to its most
// significant byte before the comparison; this makes 8-bit and 16-bit images
// behave consistently. Fully transparent pixels are never visible, which also
// makes a zero threshold select every pixel that is at least partially visible.
func alphaVisible(a uint16, minAlpha uint32) bool {
	return a != 0 && uint32(a>>8) >= minAlpha
}

func scanComponents(pg *pixelGetter, bounds image.Rectangle, minAlpha uint32, connectivity Connectivity) []ccRun {
	var runs []ccRun
	var prev, cur []int

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		cur = cur[:0]

		for x := bounds.Min.X; x < bounds.Max.X; {
			for x < bounds.Max.X && !alphaVisible(pg.getAlpha16(x, y), minAlpha) {
				x++
			}
			if x >= bounds.Max.X {
				break
			}

			x0 := x
			x++
			for x < bounds.Max.X && alphaVisible(pg.getAlpha16(x, y), minAlpha) {
				x++
			}
			x1 := x

			count := x1 - x0
			runs = append(runs, ccRun{
				parent:    int32(len(runs)),
				order:     -1,
				x0:        x0,
				x1:        x1,
				y:         y,
				numPixels: count,
				sumX:      (int64(x0) + int64(x1) - 1) * int64(count) / 2,
				sumY:      int64(y) * int64(count),
				minX:      x0,
				minY:      y,
				maxX:      x1 - 1,
				maxY:      y,
			})
			cur = append(cur, len(runs)-1)
		}

		unionAdjacentRows(runs, prev, cur, connectivity)
		prev, cur = cur, prev
	}

	return runs
}

// unionAdjacentRows merges the runs of the current row with overlapping runs of
// the previous row. Both slices are sorted by x0 and contain disjoint runs.
func unionAdjacentRows(runs []ccRun, prev, cur []int, connectivity Connectivity) {
	switch connectivity {
	case Connectivity4:
		pi := 0
		for _, ci := range cur {
			c := &runs[ci]
			for pi < len(prev) && runs[prev[pi]].x1 <= c.x0 {
				pi++
			}
			for pj := pi; pj < len(prev) && runs[prev[pj]].x0 < c.x1; pj++ {
				unionRuns(runs, ci, prev[pj])
			}
		}

	case Connectivity8:
		pi := 0
		for _, ci := range cur {
			c := &runs[ci]
			for pi < len(prev) && runs[prev[pi]].x1 < c.x0 {
				pi++
			}
			for pj := pi; pj < len(prev) && runs[prev[pj]].x0 <= c.x1; pj++ {
				unionRuns(runs, ci, prev[pj])
			}
		}
	}
}

func findRun(runs []ccRun, i int) int {
	root := i
	for runs[root].parent != int32(root) {
		root = int(runs[root].parent)
	}
	for runs[i].parent != int32(i) {
		next := int(runs[i].parent)
		runs[i].parent = int32(root)
		i = next
	}
	return root
}

func unionRuns(runs []ccRun, a, b int) {
	ra := findRun(runs, a)
	rb := findRun(runs, b)
	if ra == rb {
		return
	}

	// Union by pixel count keeps the trees shallow.
	if runs[ra].numPixels < runs[rb].numPixels {
		ra, rb = rb, ra
	}
	runs[rb].parent = int32(ra)

	dst := &runs[ra]
	src := &runs[rb]
	dst.numPixels += src.numPixels
	dst.sumX += src.sumX
	dst.sumY += src.sumY
	if src.minX < dst.minX {
		dst.minX = src.minX
	}
	if src.minY < dst.minY {
		dst.minY = src.minY
	}
	if src.maxX > dst.maxX {
		dst.maxX = src.maxX
	}
	if src.maxY > dst.maxY {
		dst.maxY = src.maxY
	}
}

// buildComponents assigns component numbers in run creation order, which is the
// top-to-bottom, left-to-right order of each component's first pixel.
func buildComponents(runs []ccRun) []Component {
	if len(runs) == 0 {
		return nil
	}

	var comps []Component
	for i := range runs {
		root := findRun(runs, i)
		if runs[root].order >= 0 {
			continue
		}

		runs[root].order = int32(len(comps))
		r := &runs[root]
		comps = append(comps, Component{
			Label:     len(comps) + 1,
			Bounds:    image.Rect(r.minX, r.minY, r.maxX+1, r.maxY+1),
			NumPixels: r.numPixels,
			CentroidX: float64(r.sumX) / float64(r.numPixels),
			CentroidY: float64(r.sumY) / float64(r.numPixels),
		})
	}
	return comps
}

func writeComponentLabels(dst *image.Gray16, runs []ccRun) {
	for i := range runs {
		root := findRun(runs, i)
		label := uint16(runs[root].order + 1)
		hi, lo := byte(label>>8), byte(label)

		r := &runs[i]
		offset := dst.PixOffset(r.x0, r.y)
		for x := r.x0; x < r.x1; x++ {
			dst.Pix[offset] = hi
			dst.Pix[offset+1] = lo
			offset += 2
		}
	}
}

// normalizedBounds returns bounds that are safe to pass to image constructors.
// A normal empty rectangle is preserved, while a malformed rectangle with a
// negative dimension is collapsed to an empty rectangle at its Min point.
func normalizedBounds(r image.Rectangle) image.Rectangle {
	if r.Dx() < 0 {
		r.Max.X = r.Min.X
	}
	if r.Dy() < 0 {
		r.Max.Y = r.Min.Y
	}
	return r
}
