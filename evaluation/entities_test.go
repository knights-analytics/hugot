package evaluation

import (
	"math"
	"slices"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// The example from seqeval's README: f1_score(y_true, y_pred) == 0.50.
func TestScoreEntitiesSeqevalExample(t *testing.T) {
	gold := [][]string{{"O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"}, {"B-PER", "I-PER", "O"}}
	predicted := [][]string{{"O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"}, {"B-PER", "I-PER", "O"}}
	report, err := ScoreEntities(gold, predicted)
	if err != nil {
		t.Fatal(err)
	}
	if !near(report.Precision, 0.5) || !near(report.Recall, 0.5) || !near(report.F1, 0.5) {
		t.Fatalf("got %+v, want precision, recall and F1 of 0.5", report.EntityScore)
	}
	if misc := report.PerType["MISC"]; misc.F1 != 0 || misc.Gold != 1 || misc.Predicted != 1 {
		t.Fatalf("MISC %+v: a span off by one word must not count", misc)
	}
	if per := report.PerType["PER"]; !near(per.F1, 1) {
		t.Fatalf("PER %+v, want F1 1", per)
	}
}

func TestScoreEntitiesTypeMustMatch(t *testing.T) {
	report, err := ScoreEntities([][]string{{"B-LOC", "O"}}, [][]string{{"B-ORG", "O"}})
	if err != nil {
		t.Fatal(err)
	}
	if report.Correct != 0 || report.F1 != 0 {
		t.Fatalf("got %+v: the right span with the wrong type is wrong", report.EntityScore)
	}
	if report.PerType["LOC"].Recall != 0 || report.PerType["ORG"].Precision != 0 {
		t.Fatalf("per type %+v", report.PerType)
	}
}

func TestEntitiesIOBParsing(t *testing.T) {
	for name, tc := range map[string]struct {
		tags []string
		want []entity
	}{
		"IOB2":                         {[]string{"B-PER", "I-PER", "O", "B-LOC"}, []entity{{"PER", 0, 2}, {"LOC", 3, 4}}},
		"I- opening an entity (IOB1)":  {[]string{"I-PER", "I-PER", "O"}, []entity{{"PER", 0, 2}}},
		"type change splits":           {[]string{"B-PER", "I-LOC"}, []entity{{"PER", 0, 1}, {"LOC", 1, 2}}},
		"B- after B- of the same type": {[]string{"B-PER", "B-PER"}, []entity{{"PER", 0, 1}, {"PER", 1, 2}}},
		"entity running to the end":    {[]string{"O", "B-ORG", "I-ORG"}, []entity{{"ORG", 1, 3}}},
		"only O":                       {[]string{"O", "O"}, nil},
		"hyphenated type is one type":  {[]string{"B-WORK-OF-ART", "I-WORK-OF-ART"}, []entity{{"WORK-OF-ART", 0, 2}}},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := entities(tc.tags)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("entities %v, want %v", got, tc.want)
			}
		})
	}
}

func TestScoreEntitiesErrors(t *testing.T) {
	if _, err := ScoreEntities([][]string{{"O"}}, nil); err == nil {
		t.Fatal("expected an error for a different number of sentences")
	}
	if _, err := ScoreEntities([][]string{{"O"}}, [][]string{{"O", "O"}}); err == nil {
		t.Fatal("expected an error for a different number of tags")
	}
	for _, tag := range []string{"PER", "X-PER", "B-"} {
		if _, err := ScoreEntities([][]string{{tag}}, [][]string{{"O"}}); err == nil {
			t.Fatalf("expected an error for tag %q", tag)
		}
	}
}

func TestScoreEntitiesNothingToScore(t *testing.T) {
	report, err := ScoreEntities([][]string{{"O"}}, [][]string{{"O"}})
	if err != nil {
		t.Fatal(err)
	}
	if report.F1 != 0 || len(report.PerType) != 0 {
		t.Fatalf("got %+v, want zero scores and no types", report)
	}
}
