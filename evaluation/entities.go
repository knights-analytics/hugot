// Package evaluation scores the predictions of fine-tuned pipelines.
package evaluation

import (
	"fmt"
	"strings"
)

// EntityScore holds entity-level precision, recall and F1. An entity counts as correct only when
// its type and its exact span both match a gold entity.
type EntityScore struct {
	Precision float64
	Recall    float64
	F1        float64
	Gold      int // entities in the gold annotation
	Predicted int // entities predicted
	Correct   int // predicted entities that match a gold entity exactly
}

// EntityReport is the entity-level score over all entity types (micro-averaged), and per type.
type EntityReport struct {
	EntityScore
	PerType map[string]EntityScore
}

type entity struct {
	kind       string
	start, end int // word indices, end exclusive
}

// ScoreEntities compares gold and predicted IOB tag sequences, one pair per sentence, at the entity
// level, as seqeval and conlleval do. Tags are "O", or a "B-" or "I-" prefix followed by the
// entity type. An "I-" tag that does not continue an entity of its type starts a new one, as in
// conlleval, so IOB1 and IOB2 tagging both score correctly.
func ScoreEntities(gold, predicted [][]string) (EntityReport, error) {
	if len(gold) != len(predicted) {
		return EntityReport{}, fmt.Errorf("%d gold sentences but %d predicted", len(gold), len(predicted))
	}
	counts := map[string]*EntityScore{}
	count := func(kind string) *EntityScore {
		if counts[kind] == nil {
			counts[kind] = &EntityScore{}
		}
		return counts[kind]
	}
	for i := range gold {
		if len(gold[i]) != len(predicted[i]) {
			return EntityReport{}, fmt.Errorf("sentence %d has %d gold tags but %d predicted", i, len(gold[i]), len(predicted[i]))
		}
		goldEntities, err := entities(gold[i])
		if err != nil {
			return EntityReport{}, fmt.Errorf("sentence %d, gold tags: %w", i, err)
		}
		predictedEntities, err := entities(predicted[i])
		if err != nil {
			return EntityReport{}, fmt.Errorf("sentence %d, predicted tags: %w", i, err)
		}
		goldSet := make(map[entity]bool, len(goldEntities))
		for _, e := range goldEntities {
			goldSet[e] = true
			count(e.kind).Gold++
		}
		for _, e := range predictedEntities {
			c := count(e.kind)
			c.Predicted++
			if goldSet[e] {
				c.Correct++
			}
		}
	}
	report := EntityReport{PerType: make(map[string]EntityScore, len(counts))}
	for kind, c := range counts {
		report.Gold += c.Gold
		report.Predicted += c.Predicted
		report.Correct += c.Correct
		report.PerType[kind] = withRates(*c)
	}
	report.EntityScore = withRates(report.EntityScore)
	return report, nil
}

// withRates fills in precision, recall and F1 from the counts. A rate with nothing to divide by
// is 0, as in seqeval.
func withRates(s EntityScore) EntityScore {
	if s.Predicted > 0 {
		s.Precision = float64(s.Correct) / float64(s.Predicted)
	}
	if s.Gold > 0 {
		s.Recall = float64(s.Correct) / float64(s.Gold)
	}
	if s.Precision+s.Recall > 0 {
		s.F1 = 2 * s.Precision * s.Recall / (s.Precision + s.Recall)
	}
	return s
}

// entities extracts the entities of one IOB tag sequence.
func entities(tags []string) ([]entity, error) {
	var found []entity
	var current *entity
	closeCurrent := func(end int) {
		if current != nil {
			current.end = end
			found = append(found, *current)
			current = nil
		}
	}
	for i, tag := range tags {
		if tag == "O" {
			closeCurrent(i)
			continue
		}
		prefix, kind, ok := strings.Cut(tag, "-")
		if !ok || kind == "" || (prefix != "B" && prefix != "I") {
			return nil, fmt.Errorf("tag %q is not an IOB tag (O, B-TYPE or I-TYPE)", tag)
		}
		if prefix == "I" && current != nil && current.kind == kind {
			continue // the entity goes on
		}
		closeCurrent(i)
		current = &entity{kind: kind, start: i}
	}
	closeCurrent(len(tags))
	return found, nil
}
