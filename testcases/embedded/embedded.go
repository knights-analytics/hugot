package embedded

import _ "embed"

//go:embed tokenExpected.json
var TokenExpectedByte []byte

//go:embed vectors.json
var ResultsByte []byte

//go:embed pipelineReference.json
var PipelineReferenceByte []byte

//go:embed tapasReference.json
var TapasReferenceByte []byte

//go:embed tapasAggregationReference.json
var TapasAggregationReferenceByte []byte

//go:embed featureReference.json
var FeatureReferenceByte []byte
