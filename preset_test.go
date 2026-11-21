package gift

import (
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"math"
	"strings"
	"testing"
)

// presetMixed returns a chain combining geometry and color filters that the
// preset format must be able to round-trip.
func presetMixed() []Filter {
	return []Filter{
		Brightness(-12.5),
		Contrast(24.25),
		Gamma(1.75),
		Resize(20, 13, LanczosResampling),
		Crop(image.Rect(2, 3, 18, 12)),
		Rotate(30, color.NRGBA{R: 10, G: 20, B: 30, A: 255}, CubicInterpolation),
		GaussianBlur(1.25),
		Rotate90(),
		FlipVertical(),
	}
}

func presetTestSource() *image.NRGBA {
	src := image.NewNRGBA(image.Rect(0, 0, 7, 5))
	for y := 0; y < 5; y++ {
		for x := 0; x < 7; x++ {
			src.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x*31 + y*7),
				G: uint8(y*43 + x*3),
				B: uint8(x*x*5 + y*11),
				A: uint8(200 + (x+y)%56),
			})
		}
	}
	return src
}

func renderFilters(filters []Filter) *image.NRGBA {
	src := presetTestSource()
	g := New(filters...)
	dst := image.NewNRGBA(g.Bounds(src.Bounds()))
	g.Draw(dst, src)
	return dst
}

func TestPresetRoundTrip(t *testing.T) {
	filters := presetMixed()

	preset, err := NewPreset(filters...)
	if err != nil {
		t.Fatalf("NewPreset failed: %v", err)
	}
	if preset.Version != PresetVersion {
		t.Fatalf("expected version %d, got %d", PresetVersion, preset.Version)
	}
	if len(preset.Filters) != len(filters) {
		t.Fatalf("expected %d filters, got %d", len(filters), len(preset.Filters))
	}
	names := []string{
		presetBrightness, presetContrast, presetGamma, presetResize, presetCrop,
		presetRotate, presetGaussianBlur, presetRotate90, presetFlipVertical,
	}
	for i, name := range names {
		if preset.Filters[i].Name != name {
			t.Errorf("filter %d: expected name %q, got %q", i, name, preset.Filters[i].Name)
		}
	}

	data, err := json.Marshal(preset)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	restored, err := UnmarshalPreset(data)
	if err != nil {
		t.Fatalf("UnmarshalPreset failed: %v", err)
	}
	if len(restored.Filters) != len(filters) {
		t.Fatalf("expected %d filters, got %d", len(filters), len(restored.Filters))
	}

	want := renderFilters(filters)
	got := renderFilters(restored.Filters)

	if want.Bounds() != got.Bounds() {
		t.Fatalf("bounds differ: want %v, got %v", want.Bounds(), got.Bounds())
	}
	if len(want.Pix) != len(got.Pix) {
		t.Fatalf("pixel counts differ: want %d, got %d", len(want.Pix), len(got.Pix))
	}
	fixedSamples := []struct{ x, y int }{
		{0, 0},
		{want.Bounds().Dx() - 1, 0},
		{0, want.Bounds().Dy() - 1},
		{want.Bounds().Dx() / 2, want.Bounds().Dy() / 2},
		{want.Bounds().Dx() - 1, want.Bounds().Dy() - 1},
		{3, 2},
	}
	for _, pt := range fixedSamples {
		if !image.Pt(pt.x, pt.y).In(want.Bounds()) {
			continue
		}
		if want.NRGBAAt(pt.x, pt.y) != got.NRGBAAt(pt.x, pt.y) {
			t.Errorf("pixel (%d, %d) differs: want %v, got %v",
				pt.x, pt.y, want.NRGBAAt(pt.x, pt.y), got.NRGBAAt(pt.x, pt.y))
		}
	}
}

func TestPresetJSONShape(t *testing.T) {
	data, err := MarshalPreset(New(Brightness(-10), GaussianBlur(2)))
	if err != nil {
		t.Fatalf("MarshalPreset failed: %v", err)
	}

	var doc map[string]interface{}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if doc["version"] != float64(PresetVersion) {
		t.Errorf("expected version %d, got %v", PresetVersion, doc["version"])
	}
	items, ok := doc["filters"].([]interface{})
	if !ok || len(items) != 2 {
		t.Fatalf("expected 2 filters, got %v", doc["filters"])
	}
	first, _ := items[0].(map[string]interface{})
	if first["name"] != presetBrightness {
		t.Errorf("expected first filter %q, got %v", presetBrightness, first["name"])
	}
	params, _ := first["params"].(map[string]interface{})
	if v, ok := params["percentage"].(float64); !ok || v != -10 {
		t.Errorf("expected percentage -10, got %v", params["percentage"])
	}

	gift, err := UnmarshalPreset(data)
	if err != nil {
		t.Fatalf("UnmarshalPreset failed: %v", err)
	}
	if len(gift.Filters) != 2 {
		t.Fatalf("expected 2 filters, got %d", len(gift.Filters))
	}
	if _, ok := gift.Filters[0].(*colorchanFilter); !ok {
		t.Errorf("expected brightness filter, got %T", gift.Filters[0])
	}
	if _, ok := gift.Filters[1].(*gausssianBlurFilter); !ok {
		t.Errorf("expected gaussian blur filter, got %T", gift.Filters[1])
	}
}

func TestPresetMethods(t *testing.T) {
	g := New(Brightness(5))
	g.SetParallelization(false)

	preset, err := g.Preset()
	if err != nil {
		t.Fatalf("GIFT.Preset failed: %v", err)
	}
	data, err := json.Marshal(preset)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	if err := g.UnmarshalPreset(data); err != nil {
		t.Fatalf("GIFT.UnmarshalPreset failed: %v", err)
	}
	if len(g.Filters) != 1 {
		t.Fatalf("expected 1 filter, got %d", len(g.Filters))
	}
	if g.Parallelization() {
		t.Error("UnmarshalPreset must not reset the options")
	}

	restored, err := UnmarshalPreset(data)
	if err != nil {
		t.Fatalf("UnmarshalPreset failed: %v", err)
	}
	if len(restored.Filters) != 1 {
		t.Fatalf("expected 1 filter, got %d", len(restored.Filters))
	}
}

func TestPresetUnsupportedFilter(t *testing.T) {
	cases := []struct {
		desc    string
		filters []Filter
	}{
		{"custom filter", []Filter{&testFilter{1}}},
		{"custom between builtins", []Filter{Brightness(1), &testFilter{2}, Gamma(1)}},
		{"color func", []Filter{ColorFunc(func(r, g, b, a float32) (float32, float32, float32, float32) { return r, g, b, a })}},
		{"sigmoid", []Filter{Sigmoid(0.5, 2)}},
		{"invert", []Filter{Invert()}},
		{"grayscale", []Filter{Grayscale()}},
		{"convolution", []Filter{Convolution([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1}, true, false, false, 0)}},
	}

	for _, d := range cases {
		_, err := NewPreset(d.filters...)
		if err == nil {
			t.Errorf("%s: expected an error, got nil", d.desc)
			continue
		}
		if !errors.Is(err, ErrUnsupportedFilter) {
			t.Errorf("%s: expected ErrUnsupportedFilter, got %v", d.desc, err)
		}
		var perr *PresetError
		if !errors.As(err, &perr) {
			t.Errorf("%s: expected *PresetError, got %T", d.desc, err)
			continue
		}
		if perr.Index < 0 {
			t.Errorf("%s: expected the failing filter index, got %d", d.desc, perr.Index)
		}
		if !strings.Contains(err.Error(), "preset filter") {
			t.Errorf("%s: error does not identify the item: %v", d.desc, err)
		}
	}

	// The index must point at the unsupported filter, not the first one.
	_, err := NewPreset(Brightness(1), &testFilter{2})
	var perr *PresetError
	if !errors.As(err, &perr) || perr.Index != 1 {
		t.Fatalf("expected an error at index 1, got %v", err)
	}

	// A matching unsupported filter must be reported on the way in as well.
	doc := `{"version":1,"filters":[{"name":"brightness","params":{"percentage":1}},{"name":"color_func","params":{}}]}`
	_, err = UnmarshalPreset([]byte(doc))
	if !errors.Is(err, ErrUnknownFilter) {
		t.Errorf("expected ErrUnknownFilter, got %v", err)
	}
	if !errors.As(err, &perr) || perr.Index != 1 || perr.Name != "color_func" {
		t.Fatalf("expected an error for filter 1 (color_func), got %v", err)
	}
}

func TestPresetBadInput(t *testing.T) {
	cases := []struct {
		desc    string
		doc     string
		wantErr error
		index   int
		field   string
	}{
		{
			"unknown version",
			`{"version":2,"filters":[]}`,
			ErrUnknownPresetVersion, -1, "version",
		},
		{
			"missing version",
			`{"filters":[]}`,
			ErrUnknownPresetVersion, -1, "version",
		},
		{
			"unknown filter name",
			`{"version":1,"filters":[{"name":"brightness","params":{"percentage":1}},{"name":"sharpen","params":{}}]}`,
			ErrUnknownFilter, 1, "sharpen",
		},
		{
			"missing filter name",
			`{"version":1,"filters":[{"params":{}}]}`,
			ErrMissingField, 0, "name",
		},
		{
			"missing percentage",
			`{"version":1,"filters":[{"name":"brightness","params":{}}]}`,
			ErrMissingField, 0, "percentage",
		},
		{
			"missing params",
			`{"version":1,"filters":[{"name":"gamma"}]}`,
			ErrMissingField, 0, "gamma",
		},
		{
			"missing resize width",
			`{"version":1,"filters":[{"name":"resize","params":{"height":10,"resampling":"linear"}}]}`,
			ErrMissingField, 0, "width",
		},
		{
			"missing crop max",
			`{"version":1,"filters":[{"name":"crop","params":{"min":{"x":0,"y":0}}}]}`,
			ErrMissingField, 0, "max",
		},
		{
			"missing rotate background channel",
			`{"version":1,"filters":[{"name":"rotate","params":{"angle":10,"background":{"r":0,"g":0,"b":0},"interpolation":"linear"}}]}`,
			ErrMissingField, 0, "background.a",
		},
		{
			"non-finite gamma",
			`{"version":1,"filters":[{"name":"gamma","params":{"gamma":1e999}}]}`,
			nil, 0, "gamma",
		},
		{
			"non-finite brightness",
			`{"version":1,"filters":[{"name":"brightness","params":{"percentage":-1e999}}]}`,
			nil, 0, "percentage",
		},
		{
			"float64 overflow",
			`{"version":1,"filters":[{"name":"gaussian_blur","params":{"sigma":1e300}}]}`,
			nil, 0, "sigma",
		},
		{
			"fractional integer parameter",
			`{"version":1,"filters":[{"name":"resize","params":{"width":10.5,"height":10,"resampling":"linear"}}]}`,
			nil, 0, "width",
		},
		{
			"unknown resampling",
			`{"version":1,"filters":[{"name":"resize","params":{"width":10,"height":10,"resampling":"mitchell"}}]}`,
			ErrInvalidParameter, 0, "mitchell",
		},
		{
			"unknown interpolation",
			`{"version":1,"filters":[{"name":"rotate","params":{"angle":10,"background":{"r":0,"g":0,"b":0,"a":0},"interpolation":"lanczos"}}]}`,
			ErrInvalidParameter, 0, "lanczos",
		},
		{
			"unknown anchor",
			`{"version":1,"filters":[{"name":"crop_to_size","params":{"width":10,"height":10,"anchor":"middle"}}]}`,
			ErrInvalidParameter, 0, "middle",
		},
		{
			"color channel out of range",
			`{"version":1,"filters":[{"name":"rotate","params":{"angle":10,"background":{"r":70000,"g":0,"b":0,"a":0},"interpolation":"linear"}}]}`,
			ErrInvalidParameter, 0, "background.r",
		},
		{
			"negative resize",
			`{"version":1,"filters":[{"name":"resize","params":{"width":-3,"height":10,"resampling":"linear"}}]}`,
			ErrInvalidParameter, 0, "size",
		},
		{
			"inverted crop",
			`{"version":1,"filters":[{"name":"crop","params":{"min":{"x":10,"y":0},"max":{"x":2,"y":10}}}]}`,
			ErrInvalidParameter, 0, "max",
		},
		{
			"oversized resize width",
			`{"version":1,"filters":[{"name":"brightness","params":{"percentage":1}},{"name":"resize","params":{"width":100000,"height":10,"resampling":"linear"}}]}`,
			ErrPresetSizeLimit, 1, "resize",
		},
		{
			"oversized resize area",
			`{"version":1,"filters":[{"name":"resize","params":{"width":30000,"height":30000,"resampling":"linear"}}]}`,
			ErrPresetSizeLimit, 0, "resize",
		},
		{
			"oversized crop",
			`{"version":1,"filters":[{"name":"crop","params":{"min":{"x":0,"y":0},"max":{"x":70000,"y":10}}}]}`,
			ErrPresetSizeLimit, 0, "crop",
		},
		{
			"oversized crop coordinate",
			`{"version":1,"filters":[{"name":"crop","params":{"min":{"x":-2000000000,"y":0},"max":{"x":2000000000,"y":10}}}]}`,
			ErrPresetSizeLimit, 0, "crop",
		},
		{
			"oversized crop to size",
			`{"version":1,"filters":[{"name":"crop_to_size","params":{"width":100000,"height":10,"anchor":"center"}}]}`,
			ErrPresetSizeLimit, 0, "crop_to_size",
		},
		{
			"oversized resize to fill",
			`{"version":1,"filters":[{"name":"resize_to_fill","params":{"width":30000,"height":30000,"resampling":"lanczos","anchor":"center"}}]}`,
			ErrPresetSizeLimit, 0, "resize_to_fill",
		},
		{
			"malformed params",
			`{"version":1,"filters":[{"name":"gamma","params":"nope"}]}`,
			nil, 0, "gamma",
		},
		{
			"malformed document",
			`{"version":1,"filters":`,
			nil, -1, "",
		},
	}

	for _, d := range cases {
		g, err := UnmarshalPreset([]byte(d.doc))
		if err == nil {
			t.Errorf("%s: expected an error, got nil", d.desc)
			continue
		}
		if g != nil {
			t.Errorf("%s: expected a nil GIFT, got %#v", d.desc, g)
		}
		var perr *PresetError
		if !errors.As(err, &perr) {
			t.Errorf("%s: expected *PresetError, got %T (%v)", d.desc, err, err)
			continue
		}
		if perr.Index != d.index {
			t.Errorf("%s: expected index %d, got %d (%v)", d.desc, d.index, perr.Index, err)
		}
		if d.wantErr != nil && !errors.Is(err, d.wantErr) {
			t.Errorf("%s: expected %v, got %v", d.desc, d.wantErr, err)
		}
		if d.field != "" && !strings.Contains(err.Error(), d.field) {
			t.Errorf("%s: expected the error to mention %q, got %v", d.desc, d.field, err)
		}
		if d.index >= 0 && !strings.Contains(err.Error(), "preset filter") {
			t.Errorf("%s: error does not identify the item: %v", d.desc, err)
		}
	}
}

func TestPresetAllBuiltinFilters(t *testing.T) {
	// Every supported filter must survive a round trip on its own.
	filters := map[string]Filter{
		"brightness":      Brightness(0),
		"contrast":        Contrast(0),
		"gamma":           Gamma(1),
		"resize":          Resize(0, 0, BoxResampling),
		"resize_to_fit":   ResizeToFit(10, 20, LinearResampling),
		"resize_to_fill":  ResizeToFill(10, 20, NearestNeighborResampling, BottomRightAnchor),
		"crop":            Crop(image.Rect(-5, -6, 7, 8)),
		"crop_to_size":    CropToSize(4, 3, TopLeftAnchor),
		"rotate":          Rotate(-45, color.RGBA64{0x1234, 0x5678, 0x9abc, 0xdef0}, LinearInterpolation),
		"rotate90":        Rotate90(),
		"rotate180":       Rotate180(),
		"rotate270":       Rotate270(),
		"flip_horizontal": FlipHorizontal(),
		"flip_vertical":   FlipVertical(),
		"transpose":       Transpose(),
		"transverse":      Transverse(),
		"gaussian_blur":   GaussianBlur(0),
	}

	for name, f := range filters {
		g := New(f)
		data, err := MarshalPreset(g)
		if err != nil {
			t.Errorf("%s: MarshalPreset failed: %v", name, err)
			continue
		}
		restored, err := UnmarshalPreset(data)
		if err != nil {
			t.Errorf("%s: UnmarshalPreset failed: %v", name, err)
			continue
		}
		src := image.NewNRGBA(image.Rect(0, 0, 5, 4))
		want := New(f)
		dstWant := image.NewNRGBA(want.Bounds(src.Bounds()))
		want.Draw(dstWant, src)
		dstGot := image.NewNRGBA(restored.Bounds(src.Bounds()))
		restored.Draw(dstGot, src)
		if dstWant.Bounds() != dstGot.Bounds() {
			t.Errorf("%s: bounds differ: want %v, got %v", name, dstWant.Bounds(), dstGot.Bounds())
			continue
		}
		for i := range dstWant.Pix {
			if dstWant.Pix[i] != dstGot.Pix[i] {
				t.Errorf("%s: pixels differ", name)
				break
			}
		}
	}
}

func TestPresetOriginalFiltersUnchanged(t *testing.T) {
	// Serializing and rebuilding filters must not change the behavior of the
	// originals, and the two chains must be independent.
	filters := presetMixed()
	original := append([]Filter(nil), filters...)

	_, err := NewPreset(filters...)
	if err != nil {
		t.Fatalf("NewPreset failed: %v", err)
	}
	if len(filters) != len(original) {
		t.Fatalf("NewPreset modified the input slice")
	}
	for i := range filters {
		if filters[i] != original[i] {
			t.Fatalf("NewPreset replaced filter %d", i)
		}
	}
	if New(filters...).Bounds(image.Rect(0, 0, 40, 30)) != New(original...).Bounds(image.Rect(0, 0, 40, 30)) {
		t.Error("NewPreset changed the bounds of the originals")
	}

	// A restored chain must not share filter state with the original one.
	data, err := MarshalPreset(New(filters...))
	if err != nil {
		t.Fatalf("MarshalPreset failed: %v", err)
	}
	restored, err := UnmarshalPreset(data)
	if err != nil {
		t.Fatalf("UnmarshalPreset failed: %v", err)
	}
	rebuilt, err := MarshalPreset(restored)
	if err != nil {
		t.Fatalf("MarshalPreset (restored) failed: %v", err)
	}
	if string(data) != string(rebuilt) {
		t.Errorf("second round trip differs:\nfirst:  %s\nsecond: %s", data, rebuilt)
	}
}

func TestPresetRejectsInvalidFilters(t *testing.T) {
	// Filters built with non-finite parameters cannot be serialized, because
	// JSON has no representation for them.
	cases := []struct {
		desc    string
		filter  Filter
		wantErr error
	}{
		{"nan brightness", Brightness(float32(math.NaN())), ErrNonFiniteParameter},
		{"inf gamma", Gamma(float32(math.Inf(1))), ErrNonFiniteParameter},
		{"inf blur", GaussianBlur(float32(math.Inf(-1))), ErrNonFiniteParameter},
		{"nan rotate", Rotate(float32(math.NaN()), color.Transparent, LinearInterpolation), ErrNonFiniteParameter},
	}
	for _, d := range cases {
		_, err := NewPreset(Brightness(1), d.filter)
		if !errors.Is(err, d.wantErr) {
			t.Errorf("%s: expected %v, got %v", d.desc, d.wantErr, err)
			continue
		}
		var perr *PresetError
		if !errors.As(err, &perr) || perr.Index != 1 {
			t.Errorf("%s: expected an error at index 1, got %v", d.desc, err)
		}
	}
}

func TestPresetNil(t *testing.T) {
	var p *Preset
	if _, err := p.GIFT(); err == nil {
		t.Error("expected an error for a nil preset")
	}
	var g *GIFT
	if _, err := g.Preset(); err == nil {
		t.Error("expected an error for a nil GIFT")
	}
}
