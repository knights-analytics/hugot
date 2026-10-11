package pipelines

import (
	"context"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
)

const addressParseTestTokenizerJSON = `{
  "version": "1.0", "added_tokens": [],
  "normalizer": {"type": "Sequence", "normalizers": [
    {"type": "Lowercase"},
    {"type": "Replace", "pattern": {"Regex": "\\p{Nd}"}, "content": "0"},
    {"type": "NFKC"},
    {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
    {"type": "Replace", "pattern": {"Regex": "^"}, "content": "▁"}
  ]},
  "model": {"type": "BPE", "unk_token": "<unk>", "fuse_unk": true,
    "vocab": {"<unk>": 0, "▁": 1, "0": 2, "▁0": 3, "a": 4, "b": 5, "ab": 6},
    "merges": [["▁", "0"], ["a", "b"]]}
}`

func TestCleanAddressParseAddress(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		splitHyphen bool
		want        string
	}{
		{name: "default cleaning", input: "  3-305 MAIN St.,  Apt 4  ", want: "3-305 main st. apt 4"},
		{name: "leading unit and number", input: "3a-305b Main Street", splitHyphen: true, want: "3a 305b main street"},
		{name: "city hyphen retained", input: "Saint-Jean Road", splitHyphen: true, want: "saint-jean road"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := cleanAddressParseAddress(test.input, test.splitHyphen); got != test.want {
				t.Fatalf("cleanAddressParseAddress(%q, %t) = %q, want %q", test.input, test.splitHyphen, got, test.want)
			}
		})
	}
}

func TestAddressParseTokenizerNormalization(t *testing.T) {
	tokenizer, err := hftokenizer.NewFromContent(nil, []byte(addressParseTestTokenizerJSON))
	if err != nil {
		t.Fatal(err)
	}
	for input, want := range map[string]string{
		"123B":       "▁000b",
		"１２٣":        "▁000",
		"²":          "▁2", // Digit replacement precedes NFKC; superscripts are not decimal digits.
		"Cafe\u0301": "▁café",
	} {
		if got := tokenizer.Normalize(input); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestAddressParseBPEmbWordPreprocessing(t *testing.T) {
	for input, want := range map[string]string{
		"1B6":  "0b0",
		"１２٣":  "000",
		"²":    "²", // BPEmb replaces decimal digits before SentencePiece's NFKC normalization.
		"CAFÉ": "café",
	} {
		if got := preprocessAddressParseWord(input); got != want {
			t.Errorf("preprocessAddressParseWord(%q) = %q, want %q", input, got, want)
		}
	}
}

type addressParseReadOnlyFileSystem struct {
	fileutil.FileSystem // Any unexpected filesystem operation fails the test.
	files               map[string]string
	opened              []string
}

func (fs *addressParseReadOnlyFileSystem) FileExists(_ context.Context, filename string) (bool, error) {
	_, exists := fs.files[filename]
	return exists, nil
}

func (fs *addressParseReadOnlyFileSystem) OpenFile(_ context.Context, filename string) (io.ReadCloser, error) {
	fs.opened = append(fs.opened, filename)
	return io.NopCloser(strings.NewReader(fs.files[filename])), nil
}

func TestAddressParseAssetsUseInjectedFileSystem(t *testing.T) {
	const modelPath = "s3://models/address-parser"
	fs := &addressParseReadOnlyFileSystem{files: map[string]string{
		modelPath + "/tokenizer.json":       addressParseTestTokenizerJSON,
		modelPath + "/metadata/parser.json": `{"format_version":2,"model":"retrained"}`,
	}}
	ctx := fileutil.WithFileSystem(context.Background(), fs)
	model := &backends.Model{Path: modelPath, InputsMeta: []backends.InputOutputInfo{
		{Name: "piece_ids"},
		{Name: "piece_lengths"},
		{Name: "forward_hidden"},
		{Name: "forward_cell"},
		{Name: "backward_hidden"},
		{Name: "backward_cell"},
	}}
	if err := backends.LoadTokenizer(ctx, model); err != nil {
		t.Fatal(err)
	}
	p := &AddressParsePipeline{tokenizer: model.Tokenizer.GoTokenizer.Tokenizer, unknownTokenID: 0}
	if err := readNativeConfig(ctx, modelPath, "metadata/parser.json", &p.manifest); err != nil {
		t.Fatal(err)
	}
	if p.manifest.Model != "retrained" {
		t.Fatal("custom manifest was not read")
	}
	if got := p.encodeWord("42"); !reflect.DeepEqual(got, []int{3, 2}) {
		t.Fatalf("Encode(42) = %v", got)
	}
	p.manifest.BPEmb.FuseUnknownPieces = true
	if got := p.encodeWord("🦄🚀"); !reflect.DeepEqual(got, []int{1, 0}) {
		t.Fatalf("adjacent unknown pieces = %v", got)
	}
	wantPaths := []string{modelPath + "/tokenizer.json", modelPath + "/metadata/parser.json"}
	if !reflect.DeepEqual(fs.opened, wantPaths) {
		t.Fatalf("opened %v, want %v", fs.opened, wantPaths)
	}
}

func TestAddressParseManifestAllowsCustomModel(t *testing.T) {
	p := &AddressParsePipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{
			Path: "s3://models/address-parser", OnnxFilename: "graphs/retrained.onnx", OnnxPath: "graphs/retrained.onnx",
		}},
		manifest: addressParseManifest{
			FormatVersion: 2, Pipeline: "deepparse", Model: "retrained-fr",
			Architecture: addressParseArchitecture{
				InputSize: 3, OutputSize: 2, EmbeddingHiddenSize: 4, EmbeddingNumLayers: 1,
				EncoderHiddenSize: 4, EncoderNumLayers: 1, DecoderHiddenSize: 4,
				DecoderNumLayers: 1, DecoderInputSize: 1,
			},
			BPEmb: addressParseBPEmb{Language: "fr", VocabularySize: 7, EmbeddingDimension: 3},
			Tags:  []addressParseTag{{ID: 0, Name: "HouseNumber"}, {ID: 1, Name: "Finish"}},
			Graphs: addressParseGraphs{
				Primary: "graphs/retrained.onnx", Encoder: "graphs/context.onnx", DecoderStep: "graphs/tagger.onnx",
				EncoderBatching: "equal_address_lengths", DecoderInitialTag: 0, DecoderSteps: "max_words_plus_eos",
			},
		},
	}
	if err := p.validateManifest(); err != nil {
		t.Fatal(err)
	}
	if err := p.validatePrimaryGraph(); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name      string
		attention bool
		inputSize int
		valid     bool
	}{
		{name: "without attention", inputSize: 1, valid: true},
		{name: "with attention", attention: true, inputSize: 5, valid: true},
		{name: "missing attention context", attention: true, inputSize: 1},
		{name: "wrong attention context size", attention: true, inputSize: 6},
		{name: "unexpected attention context", inputSize: 5},
	} {
		t.Run(test.name, func(t *testing.T) {
			variant := *p
			variant.manifest.Architecture.Attention = test.attention
			variant.manifest.Architecture.DecoderInputSize = test.inputSize
			if err := variant.validateManifest(); (err == nil) != test.valid {
				t.Fatalf("validateManifest() = %v, want valid=%t", err, test.valid)
			}
		})
	}
	p.Model.OnnxPath = "different.onnx"
	if err := p.validatePrimaryGraph(); err == nil {
		t.Fatal("accepted a model that differs from the manifest primary graph")
	}
	for _, name := range []string{"../escape.onnx", "/absolute.onnx", `..\escape.onnx`, "s3://another/model.onnx", "C:/model.onnx"} {
		p.manifest.Graphs.Primary = name
		if err := p.validateManifest(); err == nil {
			t.Errorf("accepted unsafe graph path %q", name)
		}
	}
}

func TestGroupAddressParseComponents(t *testing.T) {
	words := []AddressParseWord{
		{Word: "12", Tag: "StreetNumber", TagID: 1, Score: 0.9},
		{Word: "main", Tag: "StreetName", TagID: 2, Score: 0.8},
		{Word: "street", Tag: "StreetName", TagID: 2, Score: 0.7},
		{Word: "apt", Tag: "Unit", TagID: 3, Score: 0.6},
		{Word: "4", Tag: "StreetNumber", TagID: 1, Score: 0.5},
	}
	want := []AddressParseComponent{
		{Tag: "StreetNumber", Value: "12", Scores: []float32{0.9}},
		{Tag: "StreetName", Value: "main street", Scores: []float32{0.8, 0.7}},
		{Tag: "Unit", Value: "apt", Scores: []float32{0.6}},
		{Tag: "StreetNumber", Value: "4", Scores: []float32{0.5}},
	}
	if got := groupAddressParseComponents(words); !reflect.DeepEqual(got, want) {
		t.Fatalf("groupAddressParseComponents() = %#v, want %#v", got, want)
	}
}
