package gift

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
)

// PresetVersion is the version of the JSON filter preset format produced and
// accepted by this package.
const PresetVersion = 1

// Limits applied to explicit output sizes stored in a preset. They protect
// callers from presets that would allocate an unreasonably large image.
const (
	// MaxPresetDimension is the maximum width or height allowed for a filter
	// in a preset.
	MaxPresetDimension = 1 << 16
	// MaxPresetPixels is the maximum number of pixels allowed for a filter
	// with an explicitly defined width and height.
	MaxPresetPixels = 1 << 28
	// maxPresetCoordinate bounds crop coordinates so arithmetic on them can
	// never overflow.
	maxPresetCoordinate = 1 << 30
)

// Filters names used in the JSON representation.
const (
	presetBrightness     = "brightness"
	presetContrast       = "contrast"
	presetGamma          = "gamma"
	presetResize         = "resize"
	presetResizeToFit    = "resize_to_fit"
	presetResizeToFill   = "resize_to_fill"
	presetCrop           = "crop"
	presetCropToSize     = "crop_to_size"
	presetRotate         = "rotate"
	presetRotate90       = "rotate90"
	presetRotate180      = "rotate180"
	presetRotate270      = "rotate270"
	presetFlipHorizontal = "flip_horizontal"
	presetFlipVertical   = "flip_vertical"
	presetTranspose      = "transpose"
	presetTransverse     = "transverse"
	presetGaussianBlur   = "gaussian_blur"
)

// Errors returned (possibly wrapped in *PresetError) when a preset cannot be
// used. They can be tested with errors.Is.
var (
	ErrUnknownPresetVersion = errors.New("unknown preset version")
	ErrUnknownFilter        = errors.New("unknown filter name")
	ErrUnsupportedFilter    = errors.New("unsupported filter")
	ErrMissingField         = errors.New("missing field")
	ErrNonFiniteParameter   = errors.New("non-finite parameter")
	ErrInvalidParameter     = errors.New("invalid parameter")
	ErrPresetSizeLimit      = errors.New("preset would produce an oversized image")
)

// PresetError describes an error in a filter preset. For errors that belong to
// a specific filter, Index is the zero-based position of that filter in the
// chain and Name is its preset name. For document level errors Index is -1.
type PresetError struct {
	Index int
	Name  string
	Err   error
}

func (e *PresetError) Error() string {
	var loc string
	switch {
	case e.Index >= 0 && e.Name != "":
		loc = fmt.Sprintf("preset filter %d (%q)", e.Index, e.Name)
	case e.Index >= 0:
		loc = fmt.Sprintf("preset filter %d", e.Index)
	default:
		loc = "preset"
	}
	return "gift: " + loc + ": " + e.Err.Error()
}

func (e *PresetError) Unwrap() error {
	return e.Err
}

// FilterSpec is a single serialized filter. Name identifies the filter and
// Params holds its JSON encoded parameters.
type FilterSpec struct {
	Name   string          `json:"name"`
	Params json.RawMessage `json:"params"`
}

// Preset is a versioned, JSON serializable description of a filter chain. A
// preset preserves the order of the filters and all of their parameters.
type Preset struct {
	Version int          `json:"version"`
	Filters []FilterSpec `json:"filters"`
}

// NewPreset creates a preset that describes the given filters in the given
// order. It returns an error if any filter cannot be represented (for example
// a custom Filter implementation) or if a parameter is invalid.
func NewPreset(filters ...Filter) (*Preset, error) {
	specs := make([]FilterSpec, 0, len(filters))
	for i, f := range filters {
		spec, err := presetFilterSpec(i, f)
		if err != nil {
			return nil, err
		}
		specs = append(specs, spec)
	}
	return &Preset{Version: PresetVersion, Filters: specs}, nil
}

// Preset returns a preset that describes the filters of g in their current
// order.
func (g *GIFT) Preset() (*Preset, error) {
	if g == nil {
		return nil, &PresetError{Index: -1, Err: errors.New("nil GIFT")}
	}
	return NewPreset(g.Filters...)
}

// GIFT creates a filter chain from the preset. The original filter order and
// parameters are restored. Unsupported filters, unknown versions, malformed
// parameters and unsafe output sizes are reported as *PresetError.
func (p *Preset) GIFT() (*GIFT, error) {
	if p == nil {
		return nil, &PresetError{Index: -1, Err: errors.New("nil preset")}
	}
	if p.Version != PresetVersion {
		return nil, &PresetError{
			Index: -1,
			Err:   fmt.Errorf("%w %d (supported version is %d)", ErrUnknownPresetVersion, p.Version, PresetVersion),
		}
	}
	filters := make([]Filter, 0, len(p.Filters))
	for i, spec := range p.Filters {
		f, err := presetBuildFilter(i, spec)
		if err != nil {
			return nil, err
		}
		filters = append(filters, f)
	}
	return New(filters...), nil
}

// MarshalPreset serializes the filter chain of g to versioned JSON.
func MarshalPreset(g *GIFT) ([]byte, error) {
	p, err := g.Preset()
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(p)
	if err != nil {
		return nil, &PresetError{Index: -1, Err: err}
	}
	return data, nil
}

// UnmarshalPreset parses a versioned JSON preset and creates a filter chain
// from it.
func UnmarshalPreset(data []byte) (*GIFT, error) {
	var p Preset
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, &PresetError{Index: -1, Err: err}
	}
	return p.GIFT()
}

// MarshalPreset serializes the filter chain of g to versioned JSON.
func (g *GIFT) MarshalPreset() ([]byte, error) {
	return MarshalPreset(g)
}

// UnmarshalPreset replaces the filters of g with the filters described by the
// versioned JSON preset. The options of g (such as parallelization) are kept.
func (g *GIFT) UnmarshalPreset(data []byte) error {
	restored, err := UnmarshalPreset(data)
	if err != nil {
		return err
	}
	g.Filters = restored.Filters
	return nil
}

// Parameter types used in the JSON representation.
type (
	presetPercentParams struct {
		Percentage *float32 `json:"percentage"`
	}

	presetGammaParams struct {
		Gamma *float32 `json:"gamma"`
	}

	presetSigmaParams struct {
		Sigma *float32 `json:"sigma"`
	}

	presetResizeParams struct {
		Width      *int64  `json:"width"`
		Height     *int64  `json:"height"`
		Resampling *string `json:"resampling"`
	}

	presetResizeToFillParams struct {
		Width      *int64  `json:"width"`
		Height     *int64  `json:"height"`
		Resampling *string `json:"resampling"`
		Anchor     *string `json:"anchor"`
	}

	presetPoint struct {
		X *int64 `json:"x"`
		Y *int64 `json:"y"`
	}

	presetRectParams struct {
		Min *presetPoint `json:"min"`
		Max *presetPoint `json:"max"`
	}

	presetCropToSizeParams struct {
		Width  *int64  `json:"width"`
		Height *int64  `json:"height"`
		Anchor *string `json:"anchor"`
	}

	presetColor struct {
		R *int64 `json:"r"`
		G *int64 `json:"g"`
		B *int64 `json:"b"`
		A *int64 `json:"a"`
	}

	presetRotateParams struct {
		Angle         *float32     `json:"angle"`
		Background    *presetColor `json:"background"`
		Interpolation *string      `json:"interpolation"`
	}
)

// presetFilterSpec converts a filter to its serializable representation.
func presetFilterSpec(index int, f Filter) (FilterSpec, error) {
	switch f := f.(type) {
	case *colorchanFilter:
		if f.preset == nil {
			return FilterSpec{}, unsupportedFilterError(index, f)
		}
		switch f.preset.name {
		case presetBrightness:
			if err := checkFinite(index, presetBrightness, "percentage", f.preset.percentage); err != nil {
				return FilterSpec{}, err
			}
			return newFilterSpec(index, presetBrightness, presetPercentParams{Percentage: float32Ptr(f.preset.percentage)})
		case presetContrast:
			if err := checkFinite(index, presetContrast, "percentage", f.preset.percentage); err != nil {
				return FilterSpec{}, err
			}
			return newFilterSpec(index, presetContrast, presetPercentParams{Percentage: float32Ptr(f.preset.percentage)})
		case presetGamma:
			if err := checkFinite(index, presetGamma, "gamma", f.preset.gamma); err != nil {
				return FilterSpec{}, err
			}
			return newFilterSpec(index, presetGamma, presetGammaParams{Gamma: float32Ptr(f.preset.gamma)})
		}
		return FilterSpec{}, unsupportedFilterError(index, f)

	case *resizeFilter:
		return presetResizeSpec(index, presetResize, f.width, f.height, f.resampling, nil)

	case *resizeToFitFilter:
		return presetResizeSpec(index, presetResizeToFit, f.width, f.height, f.resampling, nil)

	case *resizeToFillFilter:
		anchor, err := presetAnchorName(f.anchor)
		if err != nil {
			return FilterSpec{}, itemError(index, presetResizeToFill, err)
		}
		return presetResizeSpec(index, presetResizeToFill, f.width, f.height, f.resampling, &anchor)

	case *cropFilter:
		r := f.rect
		if err := checkPresetPoint(index, presetCrop, int64(r.Min.X), int64(r.Min.Y)); err != nil {
			return FilterSpec{}, err
		}
		if err := checkPresetPoint(index, presetCrop, int64(r.Max.X), int64(r.Max.Y)); err != nil {
			return FilterSpec{}, err
		}
		if err := checkPresetSize(index, presetCrop, int64(r.Dx()), int64(r.Dy())); err != nil {
			return FilterSpec{}, err
		}
		params := presetRectParams{
			Min: &presetPoint{X: int64Ptr(int64(r.Min.X)), Y: int64Ptr(int64(r.Min.Y))},
			Max: &presetPoint{X: int64Ptr(int64(r.Max.X)), Y: int64Ptr(int64(r.Max.Y))},
		}
		return newFilterSpec(index, presetCrop, params)

	case *cropToSizeFilter:
		if err := checkPresetSize(index, presetCropToSize, int64(f.w), int64(f.h)); err != nil {
			return FilterSpec{}, err
		}
		anchor, err := presetAnchorName(f.anchor)
		if err != nil {
			return FilterSpec{}, itemError(index, presetCropToSize, err)
		}
		return newFilterSpec(index, presetCropToSize, presetCropToSizeParams{
			Width:  int64Ptr(int64(f.w)),
			Height: int64Ptr(int64(f.h)),
			Anchor: stringPtr(anchor),
		})

	case *rotateFilter:
		if err := checkFinite(index, presetRotate, "angle", f.angle); err != nil {
			return FilterSpec{}, err
		}
		interpolation, err := presetInterpolationName(f.interpolation)
		if err != nil {
			return FilterSpec{}, itemError(index, presetRotate, err)
		}
		var r, g, b, a uint32
		if f.bgcolor != nil {
			r, g, b, a = f.bgcolor.RGBA()
		}
		return newFilterSpec(index, presetRotate, presetRotateParams{
			Angle: float32Ptr(f.angle),
			Background: &presetColor{
				R: int64Ptr(int64(r)),
				G: int64Ptr(int64(g)),
				B: int64Ptr(int64(b)),
				A: int64Ptr(int64(a)),
			},
			Interpolation: stringPtr(interpolation),
		})

	case *transformFilter:
		var name string
		switch f.tt {
		case ttRotate90:
			name = presetRotate90
		case ttRotate180:
			name = presetRotate180
		case ttRotate270:
			name = presetRotate270
		case ttFlipHorizontal:
			name = presetFlipHorizontal
		case ttFlipVertical:
			name = presetFlipVertical
		case ttTranspose:
			name = presetTranspose
		case ttTransverse:
			name = presetTransverse
		default:
			return FilterSpec{}, unsupportedFilterError(index, f)
		}
		return newFilterSpec(index, name, struct{}{})

	case *gausssianBlurFilter:
		if err := checkFinite(index, presetGaussianBlur, "sigma", f.sigma); err != nil {
			return FilterSpec{}, err
		}
		return newFilterSpec(index, presetGaussianBlur, presetSigmaParams{Sigma: float32Ptr(f.sigma)})

	default:
		return FilterSpec{}, unsupportedFilterError(index, f)
	}
}

func presetResizeSpec(index int, name string, width, height int, resampling Resampling, anchor *string) (FilterSpec, error) {
	if err := checkPresetSize(index, name, int64(width), int64(height)); err != nil {
		return FilterSpec{}, err
	}
	resamplingName, err := presetResamplingName(resampling)
	if err != nil {
		return FilterSpec{}, itemError(index, name, err)
	}
	if name == presetResizeToFill {
		return newFilterSpec(index, name, presetResizeToFillParams{
			Width:      int64Ptr(int64(width)),
			Height:     int64Ptr(int64(height)),
			Resampling: stringPtr(resamplingName),
			Anchor:     anchor,
		})
	}
	return newFilterSpec(index, name, presetResizeParams{
		Width:      int64Ptr(int64(width)),
		Height:     int64Ptr(int64(height)),
		Resampling: stringPtr(resamplingName),
	})
}

// presetBuildFilter restores a filter from its serialized representation.
func presetBuildFilter(index int, spec FilterSpec) (Filter, error) {
	if spec.Name == "" {
		return nil, missingField(index, "", "name")
	}
	if len(bytes.TrimSpace(spec.Params)) == 0 || bytes.Equal(bytes.TrimSpace(spec.Params), []byte("null")) {
		spec.Params = json.RawMessage("{}")
	}

	switch spec.Name {
	case presetBrightness:
		var params presetPercentParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		v, err := requireFloat32(index, spec.Name, "percentage", params.Percentage)
		if err != nil {
			return nil, err
		}
		return Brightness(v), nil

	case presetContrast:
		var params presetPercentParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		v, err := requireFloat32(index, spec.Name, "percentage", params.Percentage)
		if err != nil {
			return nil, err
		}
		return Contrast(v), nil

	case presetGamma:
		var params presetGammaParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		v, err := requireFloat32(index, spec.Name, "gamma", params.Gamma)
		if err != nil {
			return nil, err
		}
		return Gamma(v), nil

	case presetGaussianBlur:
		var params presetSigmaParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		v, err := requireFloat32(index, spec.Name, "sigma", params.Sigma)
		if err != nil {
			return nil, err
		}
		return GaussianBlur(v), nil

	case presetResize:
		var params presetResizeParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		width, height, resampling, err := presetResizeValues(index, spec.Name, params.Width, params.Height, params.Resampling)
		if err != nil {
			return nil, err
		}
		return Resize(width, height, resampling), nil

	case presetResizeToFit:
		var params presetResizeParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		width, height, resampling, err := presetResizeValues(index, spec.Name, params.Width, params.Height, params.Resampling)
		if err != nil {
			return nil, err
		}
		return ResizeToFit(width, height, resampling), nil

	case presetResizeToFill:
		var params presetResizeToFillParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		width, height, resampling, err := presetResizeValues(index, spec.Name, params.Width, params.Height, params.Resampling)
		if err != nil {
			return nil, err
		}
		anchorName, err := requireString(index, spec.Name, "anchor", params.Anchor)
		if err != nil {
			return nil, err
		}
		anchor, err := presetAnchorByName(anchorName)
		if err != nil {
			return nil, itemError(index, spec.Name, err)
		}
		return ResizeToFill(width, height, resampling, anchor), nil

	case presetCrop:
		var params presetRectParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		if params.Min == nil {
			return nil, missingField(index, spec.Name, "min")
		}
		if params.Max == nil {
			return nil, missingField(index, spec.Name, "max")
		}
		minX, err := requireInt64(index, spec.Name, "min.x", params.Min.X)
		if err != nil {
			return nil, err
		}
		minY, err := requireInt64(index, spec.Name, "min.y", params.Min.Y)
		if err != nil {
			return nil, err
		}
		maxX, err := requireInt64(index, spec.Name, "max.x", params.Max.X)
		if err != nil {
			return nil, err
		}
		maxY, err := requireInt64(index, spec.Name, "max.y", params.Max.Y)
		if err != nil {
			return nil, err
		}
		if err := checkPresetPoint(index, spec.Name, minX, minY); err != nil {
			return nil, err
		}
		if err := checkPresetPoint(index, spec.Name, maxX, maxY); err != nil {
			return nil, err
		}
		if maxX < minX || maxY < minY {
			return nil, itemError(index, spec.Name, fmt.Errorf("%w: max must not be less than min", ErrInvalidParameter))
		}
		if err := checkPresetSize(index, spec.Name, maxX-minX, maxY-minY); err != nil {
			return nil, err
		}
		rect := image.Rectangle{
			Min: image.Pt(int(minX), int(minY)),
			Max: image.Pt(int(maxX), int(maxY)),
		}
		return Crop(rect), nil

	case presetCropToSize:
		var params presetCropToSizeParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		width, err := requireInt64(index, spec.Name, "width", params.Width)
		if err != nil {
			return nil, err
		}
		height, err := requireInt64(index, spec.Name, "height", params.Height)
		if err != nil {
			return nil, err
		}
		anchorName, err := requireString(index, spec.Name, "anchor", params.Anchor)
		if err != nil {
			return nil, err
		}
		anchor, err := presetAnchorByName(anchorName)
		if err != nil {
			return nil, itemError(index, spec.Name, err)
		}
		if err := checkPresetSize(index, spec.Name, width, height); err != nil {
			return nil, err
		}
		return CropToSize(int(width), int(height), anchor), nil

	case presetRotate:
		var params presetRotateParams
		if err := decodePresetParams(index, spec, &params); err != nil {
			return nil, err
		}
		angle, err := requireFloat32(index, spec.Name, "angle", params.Angle)
		if err != nil {
			return nil, err
		}
		if params.Background == nil {
			return nil, missingField(index, spec.Name, "background")
		}
		channels, err := presetColorValues(index, spec.Name, params.Background)
		if err != nil {
			return nil, err
		}
		interpolationName, err := requireString(index, spec.Name, "interpolation", params.Interpolation)
		if err != nil {
			return nil, err
		}
		interpolation, err := presetInterpolationByName(interpolationName)
		if err != nil {
			return nil, itemError(index, spec.Name, err)
		}
		background := color.RGBA64{
			R: uint16(channels[0]),
			G: uint16(channels[1]),
			B: uint16(channels[2]),
			A: uint16(channels[3]),
		}
		return Rotate(angle, background, interpolation), nil

	case presetRotate90:
		return Rotate90(), nil
	case presetRotate180:
		return Rotate180(), nil
	case presetRotate270:
		return Rotate270(), nil
	case presetFlipHorizontal:
		return FlipHorizontal(), nil
	case presetFlipVertical:
		return FlipVertical(), nil
	case presetTranspose:
		return Transpose(), nil
	case presetTransverse:
		return Transverse(), nil

	default:
		return nil, itemError(index, spec.Name, fmt.Errorf("%w %q", ErrUnknownFilter, spec.Name))
	}
}

func presetResizeValues(index int, name string, widthPtr, heightPtr *int64, resamplingPtr *string) (int, int, Resampling, error) {
	width, err := requireInt64(index, name, "width", widthPtr)
	if err != nil {
		return 0, 0, nil, err
	}
	height, err := requireInt64(index, name, "height", heightPtr)
	if err != nil {
		return 0, 0, nil, err
	}
	if err := checkPresetSize(index, name, width, height); err != nil {
		return 0, 0, nil, err
	}
	resamplingName, err := requireString(index, name, "resampling", resamplingPtr)
	if err != nil {
		return 0, 0, nil, err
	}
	resampling, err := presetResamplingByName(resamplingName)
	if err != nil {
		return 0, 0, nil, itemError(index, name, err)
	}
	return int(width), int(height), resampling, nil
}

func presetColorValues(index int, name string, c *presetColor) ([4]int64, error) {
	var values [4]int64
	fields := []struct {
		field string
		value *int64
	}{
		{"background.r", c.R},
		{"background.g", c.G},
		{"background.b", c.B},
		{"background.a", c.A},
	}
	for i, f := range fields {
		v, err := requireInt64(index, name, f.field, f.value)
		if err != nil {
			return values, err
		}
		if v < 0 || v > 0xffff {
			return values, itemError(index, name, fmt.Errorf("%w %q: value %d is outside range [0, 65535]", ErrInvalidParameter, f.field, v))
		}
		values[i] = v
	}
	return values, nil
}

func decodePresetParams(index int, spec FilterSpec, dst interface{}) error {
	if err := json.Unmarshal(spec.Params, dst); err != nil {
		return itemError(index, spec.Name, err)
	}
	return nil
}

func newFilterSpec(index int, name string, params interface{}) (FilterSpec, error) {
	data, err := json.Marshal(params)
	if err != nil {
		return FilterSpec{}, itemError(index, name, err)
	}
	return FilterSpec{Name: name, Params: data}, nil
}

func requireFloat32(index int, name, field string, value *float32) (float32, error) {
	if value == nil {
		return 0, missingField(index, name, field)
	}
	if err := checkFinite(index, name, field, *value); err != nil {
		return 0, err
	}
	return *value, nil
}

func checkFinite(index int, name, field string, value float32) error {
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return itemError(index, name, fmt.Errorf("%w %q: %v", ErrNonFiniteParameter, field, value))
	}
	return nil
}

func requireInt64(index int, name, field string, value *int64) (int64, error) {
	if value == nil {
		return 0, missingField(index, name, field)
	}
	return *value, nil
}

func requireString(index int, name, field string, value *string) (string, error) {
	if value == nil {
		return "", missingField(index, name, field)
	}
	return *value, nil
}

func checkPresetPoint(index int, name string, x, y int64) error {
	if x < -maxPresetCoordinate || x > maxPresetCoordinate || y < -maxPresetCoordinate || y > maxPresetCoordinate {
		return itemError(index, name, fmt.Errorf("%w: coordinate (%d, %d) is out of range", ErrPresetSizeLimit, x, y))
	}
	return nil
}

func checkPresetSize(index int, name string, width, height int64) error {
	if width < 0 || height < 0 {
		return itemError(index, name, fmt.Errorf("%w: negative size %dx%d", ErrInvalidParameter, width, height))
	}
	// A zero dimension means "derived from the source" and cannot lead to an
	// oversized allocation.
	if width == 0 || height == 0 {
		return nil
	}
	if width > MaxPresetDimension || height > MaxPresetDimension || width*height > MaxPresetPixels {
		return itemError(index, name, fmt.Errorf("%w: size %dx%d exceeds limits (%d per side, %d pixels)",
			ErrPresetSizeLimit, width, height, MaxPresetDimension, MaxPresetPixels))
	}
	return nil
}

func missingField(index int, name, field string) error {
	return itemError(index, name, fmt.Errorf("%w %q", ErrMissingField, field))
}

func itemError(index int, name string, err error) error {
	return &PresetError{Index: index, Name: name, Err: err}
}

func unsupportedFilterError(index int, f Filter) error {
	return &PresetError{Index: index, Err: fmt.Errorf("%w: %T", ErrUnsupportedFilter, f)}
}

func float32Ptr(v float32) *float32 { return &v }
func int64Ptr(v int64) *int64       { return &v }
func stringPtr(v string) *string    { return &v }

func presetResamplingName(r Resampling) (string, error) {
	rs, ok := r.(resamp)
	if !ok {
		return "", fmt.Errorf("%w: resampling %T is not a built-in resampling filter", ErrInvalidParameter, r)
	}
	switch rs.name {
	case "NearestNeighborResampling":
		return "nearest_neighbor", nil
	case "BoxResampling":
		return "box", nil
	case "LinearResampling":
		return "linear", nil
	case "CubicResampling":
		return "cubic", nil
	case "LanczosResampling":
		return "lanczos", nil
	default:
		return "", fmt.Errorf("%w: resampling %q", ErrInvalidParameter, rs.name)
	}
}

func presetResamplingByName(name string) (Resampling, error) {
	switch name {
	case "nearest_neighbor":
		return NearestNeighborResampling, nil
	case "box":
		return BoxResampling, nil
	case "linear":
		return LinearResampling, nil
	case "cubic":
		return CubicResampling, nil
	case "lanczos":
		return LanczosResampling, nil
	default:
		return nil, fmt.Errorf("%w: resampling %q", ErrInvalidParameter, name)
	}
}

func presetInterpolationName(i Interpolation) (string, error) {
	switch i {
	case NearestNeighborInterpolation:
		return "nearest_neighbor", nil
	case LinearInterpolation:
		return "linear", nil
	case CubicInterpolation:
		return "cubic", nil
	default:
		return "", fmt.Errorf("%w: interpolation %d", ErrInvalidParameter, i)
	}
}

func presetInterpolationByName(name string) (Interpolation, error) {
	switch name {
	case "nearest_neighbor":
		return NearestNeighborInterpolation, nil
	case "linear":
		return LinearInterpolation, nil
	case "cubic":
		return CubicInterpolation, nil
	default:
		return 0, fmt.Errorf("%w: interpolation %q", ErrInvalidParameter, name)
	}
}

func presetAnchorName(a Anchor) (string, error) {
	switch a {
	case CenterAnchor:
		return "center", nil
	case TopLeftAnchor:
		return "top_left", nil
	case TopAnchor:
		return "top", nil
	case TopRightAnchor:
		return "top_right", nil
	case LeftAnchor:
		return "left", nil
	case RightAnchor:
		return "right", nil
	case BottomLeftAnchor:
		return "bottom_left", nil
	case BottomAnchor:
		return "bottom", nil
	case BottomRightAnchor:
		return "bottom_right", nil
	default:
		return "", fmt.Errorf("%w: anchor %d", ErrInvalidParameter, a)
	}
}

func presetAnchorByName(name string) (Anchor, error) {
	switch name {
	case "center":
		return CenterAnchor, nil
	case "top_left":
		return TopLeftAnchor, nil
	case "top":
		return TopAnchor, nil
	case "top_right":
		return TopRightAnchor, nil
	case "left":
		return LeftAnchor, nil
	case "right":
		return RightAnchor, nil
	case "bottom_left":
		return BottomLeftAnchor, nil
	case "bottom":
		return BottomAnchor, nil
	case "bottom_right":
		return BottomRightAnchor, nil
	default:
		return 0, fmt.Errorf("%w: anchor %q", ErrInvalidParameter, name)
	}
}
