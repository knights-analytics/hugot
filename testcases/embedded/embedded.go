package embedded

import _ "embed"

//go:embed textClassification.jsonl
var TextClassificationData []byte

//go:embed tokenClassification.jsonl
var TokenClassificationData []byte

//go:embed tokenExpected.json
var TokenExpectedByte []byte

//go:embed vectors.json
var ResultsByte []byte
