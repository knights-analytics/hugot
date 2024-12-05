package pipelineBackends

type Model struct {
	Path         string
	OnnxFilename string
	OnnxBytes    []byte
	ORTModel     *ORTModel
	XLAModel     *XLAModel
	Tokenizer    *Tokenizer
	InputsMeta   []InputOutputInfo
	OutputsMeta  []InputOutputInfo
	Destroy      func() error
}

func ReshapeOutput(input *[]float32, meta InputOutputInfo, paddingMask [][]bool, sequenceLength int) OutputArray {
	outArray := OutputArray{}

	dimensions := meta.Dimensions.ValuesInt()
	lenDimensions := len(dimensions)
	switch lenDimensions {
	case 2:
		outArray.Result2D = flatDataTo2D(input, paddingMask, dimensions[lenDimensions-1])
	case 3:
		outArray.Result3D = flatDataTo3D(input, paddingMask, sequenceLength, dimensions[lenDimensions-1])
	}
	return outArray
}

func flatDataTo2D(input *[]float32, paddingMask [][]bool, dimension int) [][]float32 {
	// Input string, token, dimension
	output := make([][]float32, len(paddingMask))

	counter := 0
	for batchIndex := range paddingMask {
		inputEmbedding := make([]float32, dimension)

		for i := 0; i < dimension; i++ {
			inputEmbedding[i] = (*input)[counter]
			counter++
		}

		output[batchIndex] = inputEmbedding
	}

	return output
}

func flatDataTo3D(input *[]float32, paddingMask [][]bool, sequenceLength int, dimension int) [][][]float32 {
	// Input string, token, dimension
	output := make([][][]float32, len(paddingMask))

	counter := 0

	for batchIndex, mask := range paddingMask {
		tokenEmbeddings := make([][]float32, 0, sequenceLength)

		for _, isValid := range mask {
			if !isValid {
				// skip whole token
				counter = counter + dimension
				continue
			}

			// valid token, create embedding
			embedding := make([]float32, dimension)

			for i := 0; i < dimension; i++ {
				embedding[i] = (*input)[counter]
				counter++
			}

			tokenEmbeddings = append(tokenEmbeddings, embedding)
		}

		output[batchIndex] = tokenEmbeddings
	}

	return output
}
