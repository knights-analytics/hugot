package embedded

import _ "embed"

//go:embed tokenExpected.json
var TokenExpectedByte []byte

//go:embed vectors.json
var ResultsByte []byte

//go:embed viltReference.json
var ViltReferenceByte []byte

//go:embed pipelineReference.json
var PipelineReferenceByte []byte

//go:embed captionReference.json
var CaptionReferenceByte []byte

//go:embed tapasReference.json
var TapasReferenceByte []byte

//go:embed tapasAggregationReference.json
var TapasAggregationReferenceByte []byte

//go:embed featureReference.json
var FeatureReferenceByte []byte

//go:embed captionGenerationReference.json
var CaptionGenerationReferenceByte []byte

//go:embed caption_cat_reference.png
var CaptionCatReferenceByte []byte

//go:embed caption_portrait_reference.png
var CaptionPortraitReferenceByte []byte
