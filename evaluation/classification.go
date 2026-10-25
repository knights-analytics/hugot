package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/pipelines"
)

// ClassScore holds precision, recall and F1 for one class or label.
type ClassScore struct {
	Precision float64
	Recall    float64
	F1        float64
	Support   int // examples whose gold labels include this class
	Predicted int // examples predicted to have this class
	Correct   int // examples predicted to have this class that do
}

// ClassificationReport scores a classifier against gold labels.
//
// Accuracy is the share of examples whose predicted label set equals the gold set exactly: plain
// accuracy for single-label data, subset accuracy for multi-label data. MacroF1 is the unweighted
// mean of the per-class F1, so rare classes count as much as common ones; WeightedF1 weights each
// class by its support; MicroF1 pools every decision, and for single-label data equals Accuracy.
// The averages run over every class seen in the gold or the predicted labels, as in scikit-learn.
type ClassificationReport struct {
	Accuracy   float64
	MacroF1    float64
	WeightedF1 float64
	MicroF1    float64
	PerClass   map[string]ClassScore
}

// ScoreClassification scores single-label predictions, one label per example.
func ScoreClassification(gold, predicted []string) (ClassificationReport, error) {
	if len(gold) != len(predicted) {
		return ClassificationReport{}, fmt.Errorf("%d gold labels but %d predicted", len(gold), len(predicted))
	}
	goldSets := make([][]string, len(gold))
	predictedSets := make([][]string, len(predicted))
	for i := range gold {
		goldSets[i] = []string{gold[i]}
		predictedSets[i] = []string{predicted[i]}
	}
	return ScoreMultiLabel(goldSets, predictedSets)
}

// ScoreMultiLabel scores multi-label predictions: for each example, the set of labels it has.
// Duplicate labels within an example count once.
func ScoreMultiLabel(gold, predicted [][]string) (ClassificationReport, error) {
	if len(gold) != len(predicted) {
		return ClassificationReport{}, fmt.Errorf("%d gold examples but %d predicted", len(gold), len(predicted))
	}
	if len(gold) == 0 {
		return ClassificationReport{}, errors.New("no examples to score")
	}
	counts := map[string]*ClassScore{}
	count := func(class string) *ClassScore {
		if counts[class] == nil {
			counts[class] = &ClassScore{}
		}
		return counts[class]
	}
	exact := 0
	var micro ClassScore
	for i := range gold {
		goldSet := toSet(gold[i])
		predictedSet := toSet(predicted[i])
		for class := range goldSet {
			count(class).Support++
			micro.Support++
		}
		for class := range predictedSet {
			c := count(class)
			c.Predicted++
			micro.Predicted++
			if goldSet[class] {
				c.Correct++
				micro.Correct++
			}
		}
		if len(goldSet) == len(predictedSet) && subsetOf(goldSet, predictedSet) {
			exact++
		}
	}

	report := ClassificationReport{
		Accuracy: float64(exact) / float64(len(gold)),
		MicroF1:  classRates(micro).F1,
		PerClass: make(map[string]ClassScore, len(counts)),
	}
	totalSupport := 0
	for class, c := range counts {
		scored := classRates(*c)
		report.PerClass[class] = scored
		report.MacroF1 += scored.F1
		report.WeightedF1 += scored.F1 * float64(scored.Support)
		totalSupport += scored.Support
	}
	report.MacroF1 /= float64(len(counts))
	if totalSupport > 0 {
		report.WeightedF1 /= float64(totalSupport)
	}
	return report, nil
}

func classRates(c ClassScore) ClassScore {
	if c.Predicted > 0 {
		c.Precision = float64(c.Correct) / float64(c.Predicted)
	}
	if c.Support > 0 {
		c.Recall = float64(c.Correct) / float64(c.Support)
	}
	if c.Precision+c.Recall > 0 {
		c.F1 = 2 * c.Precision * c.Recall / (c.Precision + c.Recall)
	}
	return c
}

func toSet(labels []string) map[string]bool {
	set := make(map[string]bool, len(labels))
	for _, label := range labels {
		set[label] = true
	}
	return set
}

func subsetOf(a, b map[string]bool) bool {
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

// MultiLabelThreshold is the score above which EvaluateTextClassification counts a label as
// predicted for a multi-label pipeline (whose scores are per-label sigmoids).
const MultiLabelThreshold = 0.5

// EvaluateTextClassification runs a text classification pipeline over examples and scores its
// predictions against their labels. A single-label pipeline is scored on its top label against
// each example's Label; a multi-label pipeline (see pipelines.WithMultiLabel) on the labels scoring
// above MultiLabelThreshold against each example's Labels.
func EvaluateTextClassification(ctx context.Context, pipeline *pipelines.TextClassificationPipeline, examples []datasets.TextClassificationExample, batchSize int) (ClassificationReport, error) {
	if pipeline == nil {
		return ClassificationReport{}, errors.New("pipeline is required")
	}
	if batchSize <= 0 {
		return ClassificationReport{}, fmt.Errorf("batch size must be greater than 0, got %d", batchSize)
	}
	multiLabel := pipeline.ProblemType == "multiLabel"
	problem := datasets.SingleLabel
	if multiLabel {
		problem = datasets.MultiLabel
	}
	// Prodigy annotations become plain labels, and examples without a usable label are left out,
	// as in training.
	labels := datasets.LabelNames(pipeline.Model.IDLabelMap)
	resolved := make([]datasets.TextClassificationExample, 0, len(examples))
	for _, example := range examples {
		example, keep, err := example.Resolve(problem, labels)
		if err != nil {
			return ClassificationReport{}, err
		}
		if keep {
			resolved = append(resolved, example)
		}
	}
	examples = resolved
	gold := make([][]string, 0, len(examples))
	predicted := make([][]string, 0, len(examples))
	for start := 0; start < len(examples); start += batchSize {
		batch := examples[start:min(start+batchSize, len(examples))]
		texts := make([]string, len(batch))
		for i, example := range batch {
			texts[i] = example.Text
		}
		output, err := pipeline.RunPipeline(ctx, texts)
		if err != nil {
			return ClassificationReport{}, err
		}
		for i, example := range batch {
			var labels []string
			for _, out := range output.ClassificationOutputs[i] {
				if !multiLabel || out.Score > MultiLabelThreshold {
					labels = append(labels, out.Label)
				}
			}
			if multiLabel {
				gold = append(gold, slices.Clone(example.Labels))
			} else {
				gold = append(gold, []string{example.Label})
			}
			predicted = append(predicted, labels)
		}
	}
	return ScoreMultiLabel(gold, predicted)
}
