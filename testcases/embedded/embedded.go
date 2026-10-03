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
