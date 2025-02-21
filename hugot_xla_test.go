//go:build XLA || ALL

package hugot

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot/options"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewXLASession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewXLASession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineMultiXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewXLASession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewXLASession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewXLASession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipelineValidation(t, session)
}

// No same name

func TestNoSameNamePipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	noSameNamePipeline(t, session)
}

func TestDestroyPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	destroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyXLA(t *testing.T) {
	session, err := NewXLASession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	threadSafety(t, session)
}

func TestThreadSafetyXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewXLASession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	threadSafety(t, session)
}
