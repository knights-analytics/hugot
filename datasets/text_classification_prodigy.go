package datasets

import (
	"fmt"
	"slices"
)

// Text classification examples can also be read in the formats Prodigy exports:
//
//   - choice annotations (textcat.manual with several labels): the chosen labels in "accept",
//     e.g. {"text": "...", "options": [...], "accept": ["SPORT"], "answer": "accept"};
//   - binary annotations (textcat.manual with one label, textcat.teach): one "label" and an
//     "answer", e.g. {"text": "...", "label": "INSULT", "answer": "reject"}.
//
// resolve turns each into the dataset's own form (Label or Labels), or reports that it should be
// skipped.

// IsChoiceFormat reports whether the example is a choice annotation, with its labels in Accept.
func (e TextClassificationExample) IsChoiceFormat() bool {
	return e.Accept != nil
}

// resolve returns the example with Label (single-label) or Labels (multi-label) set from its
// annotation, or keep false for an example that carries no usable training signal: answered
// "ignore", a rejected choice, or a binary rejection that the model's labels cannot express.
// labels are the model's labels; they are only needed for binary rejections and may be nil before
// the model is known, in which case such examples are kept.
func (e TextClassificationExample) resolve(problem ClassificationProblem, labels []string) (resolved TextClassificationExample, keep bool, err error) {
	switch {
	case e.Answer == "ignore":
		return e, false, nil
	case e.Answer != "" && e.Answer != AnswerAccept && e.Answer != "reject":
		return e, false, fmt.Errorf("example %q has unknown answer %q (want accept, reject or ignore)", e.Text, e.Answer)
	}
	if e.IsChoiceFormat() {
		if e.Label != "" || len(e.Labels) > 0 {
			return e, false, fmt.Errorf("example %q has both accept and label(s); give one or the other", e.Text)
		}
		if e.Answer == "reject" {
			return e, false, nil // the annotator rejected the whole example
		}
		if problem == SingleLabel {
			if len(e.Accept) != 1 {
				return e, false, fmt.Errorf("example %q accepts %d labels %q, but the dataset is single-label and needs exactly one (annotate with --exclusive, or train multi-label)", e.Text, len(e.Accept), e.Accept)
			}
			e.Label = e.Accept[0]
		} else {
			e.Labels = slices.Clone(e.Accept)
		}
		e.Accept = nil
		return e, true, nil
	}
	if e.Answer == "" || (e.Answer == AnswerAccept && problem == SingleLabel) {
		return e, true, nil // the dataset's own format; an accepted binary label is the class
	}
	// A binary annotation says whether one label applies. Other labels stay unknown, so it is only
	// a full labelling when the model's labels leave no doubt.
	if e.Label == "" {
		return e, false, fmt.Errorf("example %q has answer %q but no label", e.Text, e.Answer)
	}
	if problem == MultiLabel {
		if labels == nil {
			return e, true, nil
		}
		if len(labels) != 1 || labels[0] != e.Label {
			return e, false, fmt.Errorf("example %q is a binary %s annotation of %q, which cannot train a multi-label model with labels %q: the other labels are unknown. Use a choice annotation (textcat.manual with all labels)", e.Text, e.Answer, e.Label, labels)
		}
		if e.Answer == AnswerAccept {
			e.Labels = []string{e.Label}
		} else {
			e.Labels = []string{}
		}
		e.Label = ""
		return e, true, nil
	}
	// A single-label rejection names the class it is not: only a two-class model can turn that
	// into the class it is.
	if labels == nil {
		return e, true, nil
	}
	if len(labels) != 2 || !slices.Contains(labels, e.Label) {
		return e, false, nil
	}
	other := labels[0]
	if other == e.Label {
		other = labels[1]
	}
	e.Label = other
	return e, true, nil
}
