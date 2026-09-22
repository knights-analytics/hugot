//go:build cgo && (ORT || ALL)

package backends

import "github.com/knights-analytics/ortgenai"

// generativeORTAdapter isolates the legacy GenAI binding until an official Go
// GenAI API provides equivalent session, engine, and multimodal behavior.
type generativeORTAdapter struct {
	session *ortgenai.Session
	engine  *ortgenai.Engine
}

func (a *generativeORTAdapter) Close() error {
	if a == nil {
		return nil
	}
	if a.session != nil {
		a.session.Destroy()
		a.session = nil
	}
	if a.engine != nil {
		a.engine.Destroy()
		a.engine = nil
	}
	return nil
}

func (a *generativeORTAdapter) Statistics() PipelineStatistics {
	if a == nil {
		return PipelineStatistics{}
	}
	if a.engine != nil {
		stats := a.engine.GetStatistics()
		return PipelineStatistics{AvgPrefillSeconds: stats.AvgPrefillSeconds, TokensPerSecond: stats.TokensPerSecond, CumulativePrefillSum: stats.CumulativePrefillSum, CumulativePrefillCount: stats.CumulativePrefillCount, CumulativeTokens: stats.CumulativeTokens, CumulativeTokenDurationSeconds: stats.CumulativeTokenDurationSeconds}
	}
	if a.session != nil {
		stats := a.session.GetStatistics()
		return PipelineStatistics{AvgPrefillSeconds: stats.AvgPrefillSeconds, TokensPerSecond: stats.TokensPerSecond, CumulativePrefillSum: stats.CumulativePrefillSum, CumulativePrefillCount: stats.CumulativePrefillCount, CumulativeTokens: stats.CumulativeTokens, CumulativeTokenDurationSeconds: stats.CumulativeTokenDurationSeconds}
	}
	return PipelineStatistics{}
}
