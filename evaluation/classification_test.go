package evaluation

import "testing"

// The expected values match scikit-learn's accuracy_score and f1_score on the same labels.
func TestScoreClassification(t *testing.T) {
	gold := []string{"cat", "cat", "cat", "dog", "dog", "bird"}
	predicted := []string{"cat", "cat", "dog", "dog", "cat", "cat"}
	report, err := ScoreClassification(gold, predicted)
	if err != nil {
		t.Fatal(err)
	}
	// cat: P 2/4, R 2/3, F1 4/7 · dog: P 1/2, R 1/2, F1 1/2 · bird: P 0, R 0, F1 0
	if !near(report.Accuracy, 0.5) || !near(report.MicroF1, 0.5) {
		t.Fatalf("accuracy %v, micro F1 %v, want 0.5 both", report.Accuracy, report.MicroF1)
	}
	if want := (4.0/7 + 0.5 + 0) / 3; !near(report.MacroF1, want) {
		t.Fatalf("macro F1 %v, want %v", report.MacroF1, want)
	}
	if want := (4.0/7*3 + 0.5*2 + 0*1) / 6; !near(report.WeightedF1, want) {
		t.Fatalf("weighted F1 %v, want %v", report.WeightedF1, want)
	}
	if cat := report.PerClass["cat"]; !near(cat.Precision, 0.5) || !near(cat.Recall, 2.0/3) || cat.Support != 3 {
		t.Fatalf("cat %+v", cat)
	}
	if bird := report.PerClass["bird"]; bird.F1 != 0 || bird.Support != 1 || bird.Predicted != 0 {
		t.Fatalf("bird %+v", bird)
	}
}

// The case accuracy hides: always predicting the majority class.
func TestScoreClassificationMajorityClass(t *testing.T) {
	gold := []string{"ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok", "fraud"}
	predicted := []string{"ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok"}
	report, err := ScoreClassification(gold, predicted)
	if err != nil {
		t.Fatal(err)
	}
	if !near(report.Accuracy, 0.9) {
		t.Fatalf("accuracy %v, want 0.9", report.Accuracy)
	}
	if report.PerClass["fraud"].Recall != 0 || report.MacroF1 > 0.5 {
		t.Fatalf("fraud recall %v, macro F1 %v: macro F1 must expose the missed class", report.PerClass["fraud"].Recall, report.MacroF1)
	}
}

// The expected values match scikit-learn on the binarized labels (zero_division=0).
func TestScoreMultiLabel(t *testing.T) {
	gold := [][]string{{"delivery", "sizing"}, {"price"}, {}}
	predicted := [][]string{{"delivery"}, {"price"}, {}}
	report, err := ScoreMultiLabel(gold, predicted)
	if err != nil {
		t.Fatal(err)
	}
	// Two of three label sets are exactly right; sizing was missed.
	if !near(report.Accuracy, 2.0/3) {
		t.Fatalf("subset accuracy %v, want 2/3", report.Accuracy)
	}
	// Micro: 2 correct of 2 predicted, of 3 gold: P 1, R 2/3, F1 0.8.
	if !near(report.MicroF1, 0.8) {
		t.Fatalf("micro F1 %v, want 0.8", report.MicroF1)
	}
	if sizing := report.PerClass["sizing"]; sizing.Recall != 0 || sizing.Support != 1 {
		t.Fatalf("sizing %+v", sizing)
	}
	if !near(report.MacroF1, 2.0/3) { // delivery 1, price 1, sizing 0
		t.Fatalf("macro F1 %v, want 2/3", report.MacroF1)
	}
}

func TestScoreClassificationErrors(t *testing.T) {
	if _, err := ScoreClassification([]string{"a"}, nil); err == nil {
		t.Fatal("expected an error for mismatched lengths")
	}
	if _, err := ScoreMultiLabel(nil, nil); err == nil {
		t.Fatal("expected an error for no examples")
	}
}
