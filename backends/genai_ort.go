//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/ortgenai"
)

type generativeORTProvider struct {
	session *ortgenai.Session
	engine  *engineORTOwner
}

func (a *generativeORTProvider) Close() error {
	if a == nil {
		return nil
	}
	var closeErr error
	if a.session != nil {
		a.session.Destroy()
		a.session = nil
	}
	if a.engine != nil {
		closeErr = errors.Join(closeErr, a.engine.Close())
		a.engine = nil
	}
	return closeErr
}

func (a *generativeORTProvider) Statistics() PipelineStatistics {
	if a == nil {
		return PipelineStatistics{}
	}
	if a.session != nil {
		stats := a.session.GetStatistics()
		return PipelineStatistics{AvgPrefillSeconds: stats.AvgPrefillSeconds, TokensPerSecond: stats.TokensPerSecond, CumulativePrefillSum: stats.CumulativePrefillSum, CumulativePrefillCount: stats.CumulativePrefillCount, CumulativeTokens: stats.CumulativeTokens, CumulativeTokenDurationSeconds: stats.CumulativeTokenDurationSeconds}
	}
	if a.engine != nil {
		return a.engine.Statistics()
	}
	return PipelineStatistics{}
}

type engineORTState struct {
	engine       *ortgenai.Engine
	eventBuffer  *ortgenai.EventBuffer
	tokenizer    *ortgenai.Tokenizer
	capabilities ortgenai.EngineCapabilities
	turns        map[*ortgenai.Request]*engineORTTurn
	jobs         map[*engineORTJob]struct{}
}

type engineORTCommand struct {
	ctx  context.Context
	run  func(*engineORTState) error
	done chan error
}

// engineORTOwner keeps every engine, request, and event-buffer operation on
// one locked OS thread. Native handles are only exposed inside owner commands.
type engineORTOwner struct {
	commands chan engineORTCommand
	done     chan struct{}
	mu       sync.Mutex
	closed   bool
	tokens   atomic.Uint64
}

type engineORTJob struct {
	ctx         context.Context
	batch       *PipelineBatch
	inputs      [][]ortgenai.Message
	tools       []string
	settings    engineORTTurnSettings
	tokenStream chan SequenceDelta
	errorStream chan error
	nextInput   int
	errors      []error
	cancelled   bool
	cancelErr   error
	cancel      context.CancelFunc
	stopSession func() bool
}

type engineORTTurn struct {
	job        *engineORTJob
	sequence   int
	request    *ortgenai.Request
	options    *ortgenai.TurnOptions
	stream     *ortgenai.TokenizerStream
	id         uint64
	cancelling bool
}

type engineORTTurnSettings struct {
	maxGeneratedTokens uint64
	doSample           bool
	temperature        *float32
	topP               *float32
	seed               *uint64
	stopStrings        []string
	guidance           *Guidance
}

func newEngineORTOwner(modelPath string) (*engineORTOwner, error) {
	owner := &engineORTOwner{
		commands: make(chan engineORTCommand),
		done:     make(chan struct{}),
	}
	ready := make(chan error, 1)
	go owner.run(modelPath, ready)
	if err := <-ready; err != nil {
		<-owner.done
		return nil, err
	}
	return owner, nil
}

func (o *engineORTOwner) run(modelPath string, ready chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(o.done)

	state := &engineORTState{
		turns: make(map[*ortgenai.Request]*engineORTTurn),
		jobs:  make(map[*engineORTJob]struct{}),
	}
	engine, err := ortgenai.CreateEngine(modelPath)
	if err == nil {
		state.engine = engine
		state.capabilities, err = engine.Capabilities()
		if err == nil && state.capabilities.ConfiguredMaxBatchSize < 1 {
			err = errors.New("engine reported an invalid maximum batch size")
		}
	}
	if err == nil {
		state.eventBuffer, err = engine.CreateEventBuffer(256)
	}
	if err == nil {
		state.tokenizer, err = ortgenai.CreateTokenizer(modelPath)
	}
	if err != nil {
		if state.tokenizer != nil {
			state.tokenizer.Destroy()
		}
		if state.eventBuffer != nil {
			state.eventBuffer.Destroy()
		}
		if state.engine != nil {
			state.engine.Destroy()
		}
		ready <- fmt.Errorf("creating ORT GenAI engine owner: %w", err)
		return
	}
	ready <- nil

	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case command, ok := <-o.commands:
			if !ok {
				state.shutdown()
				state.destroy()
				return
			}
			runEngineORTCommand(state, command)
		case <-ticker.C:
			state.advance(o)
		}
	}
}

func (o *engineORTOwner) do(ctx context.Context, run func(*engineORTState) error) error {
	if o == nil {
		return errors.New("ORT GenAI engine owner is not initialized")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	command := engineORTCommand{ctx: ctx, run: run, done: make(chan error, 1)}
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return errors.New("ORT GenAI engine owner is closed")
	}
	select {
	case o.commands <- command:
		o.mu.Unlock()
	case <-ctx.Done():
		o.mu.Unlock()
		return ctx.Err()
	case <-o.done:
		o.mu.Unlock()
		return errors.New("ORT GenAI engine owner has stopped")
	}
	select {
	case err := <-command.done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	case <-o.done:
		return errors.New("ORT GenAI engine owner has stopped")
	}
}

func (o *engineORTOwner) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	if !o.closed {
		o.closed = true
		close(o.commands)
	}
	o.mu.Unlock()
	<-o.done
	return nil
}

func (o *engineORTOwner) Statistics() PipelineStatistics {
	if o == nil {
		return PipelineStatistics{}
	}
	return PipelineStatistics{CumulativeTokens: int(o.tokens.Load())}
}

func (o *engineORTOwner) generate(ctx, sessionCtx context.Context, batch *PipelineBatch, inputs [][]ortgenai.Message, maxLength int, stopStrings []string, temperature, topP *float64, seed *int, tools []string, guidance *Guidance) (chan SequenceDelta, chan error, error) {
	settings, err := makeEngineORTTurnSettings(maxLength, stopStrings, temperature, topP, seed, guidance)
	if err != nil {
		return nil, nil, err
	}
	if len(inputs) == 0 {
		tokenStream := make(chan SequenceDelta)
		errorStream := make(chan error, 1)
		if err := batch.Destroy(); err != nil {
			errorStream <- err
		}
		close(tokenStream)
		close(errorStream)
		return tokenStream, errorStream, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	jobCtx, cancel := context.WithCancel(ctx)
	stopSession := func() bool { return true }
	if sessionCtx != nil {
		stopSession = context.AfterFunc(sessionCtx, cancel)
	}
	job := &engineORTJob{
		ctx:         jobCtx,
		batch:       batch,
		inputs:      inputs,
		tools:       append([]string(nil), tools...),
		settings:    settings,
		tokenStream: make(chan SequenceDelta, 32),
		errorStream: make(chan error, 1),
		cancel:      cancel,
		stopSession: stopSession,
	}
	if err := o.do(jobCtx, func(state *engineORTState) error {
		state.jobs[job] = struct{}{}
		return nil
	}); err != nil {
		stopSession()
		cancel()
		return nil, nil, err
	}
	return job.tokenStream, job.errorStream, nil
}

func makeEngineORTTurnSettings(maxLength int, stopStrings []string, temperature, topP *float64, seed *int, guidance *Guidance) (engineORTTurnSettings, error) {
	if maxLength <= 0 {
		return engineORTTurnSettings{}, errors.New("maximum generated token count must be greater than zero")
	}
	settings := engineORTTurnSettings{maxGeneratedTokens: uint64(maxLength)}
	if temperature != nil {
		value := float32(*temperature)
		settings.temperature = &value
		settings.doSample = *temperature > 0
	}
	if topP != nil {
		value := float32(*topP)
		settings.topP = &value
		settings.doSample = settings.doSample || *topP < 1
	}
	if seed != nil {
		if *seed < 0 {
			return engineORTTurnSettings{}, errors.New("engine mode does not support negative generation seeds")
		}
		value := uint64(*seed)
		settings.seed = &value
	}
	for _, stop := range stopStrings {
		if stop != "" {
			settings.stopStrings = append(settings.stopStrings, stop)
		}
	}
	if guidance != nil {
		if guidance.EnableFFTokens {
			return engineORTTurnSettings{}, errors.New("WithGenerativeEngine does not support guidance with EnableFFTokens")
		}
		guidanceCopy := *guidance
		settings.guidance = &guidanceCopy
	}
	return settings, nil
}

func marshalORTEngineTemplateInput(messages []ortgenai.Message, tools []string) (string, string, error) {
	messageJSON, err := json.Marshal(messages)
	if err != nil {
		return "", "", fmt.Errorf("marshalling chat messages: %w", err)
	}
	rawTools := make([]json.RawMessage, len(tools))
	for i, tool := range tools {
		rawTools[i] = json.RawMessage(tool)
	}
	toolsJSON, err := json.Marshal(rawTools)
	if err != nil {
		return "", "", fmt.Errorf("marshalling tools: %w", err)
	}
	return string(messageJSON), string(toolsJSON), nil
}

func (s *engineORTState) schedule() error {
	for len(s.turns) < s.capabilities.ConfiguredMaxBatchSize {
		var job *engineORTJob
		for candidate := range s.jobs {
			if !candidate.cancelled && candidate.nextInput < len(candidate.inputs) {
				job = candidate
				break
			}
		}
		if job == nil {
			return nil
		}
		if err := job.ctx.Err(); err != nil {
			s.cancelJob(job, err)
			continue
		}
		sequence := job.nextInput
		job.nextInput++
		if err := s.startTurn(job, sequence); err != nil {
			job.errors = append(job.errors, fmt.Errorf("sequence %d: %w", sequence, err))
		}
	}
	return nil
}

func (s *engineORTState) startTurn(job *engineORTJob, sequence int) error {
	messageJSON, toolsJSON, err := marshalORTEngineTemplateInput(job.inputs[sequence], job.tools)
	if err != nil {
		return err
	}
	prompt, err := s.tokenizer.ApplyChatTemplate("", messageJSON, toolsJSON, true)
	if err != nil {
		return fmt.Errorf("applying chat template: %w", err)
	}
	inputIDs, err := s.tokenizer.Encode(prompt)
	if err != nil {
		return fmt.Errorf("encoding chat template: %w", err)
	}
	if s.capabilities.MaxRequestLength > 0 && uint64(len(inputIDs)) > s.capabilities.MaxRequestLength {
		return fmt.Errorf("encoded prompt has %d tokens, exceeding engine maximum request length %d", len(inputIDs), s.capabilities.MaxRequestLength)
	}
	request, err := s.engine.CreateRequest(nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	turn := &engineORTTurn{job: job, sequence: sequence, request: request}
	cleanup := func() {
		if turn.stream != nil {
			turn.stream.Destroy()
		}
		if turn.options != nil {
			turn.options.Destroy()
		}
		request.Destroy()
	}
	turn.options, err = request.CreateTurnOptions()
	if err != nil {
		cleanup()
		return fmt.Errorf("creating turn options: %w", err)
	}
	settings := job.settings
	if err = turn.options.SetMaxGeneratedTokens(settings.maxGeneratedTokens); err == nil {
		err = turn.options.SetDoSample(settings.doSample)
	}
	if err == nil && settings.temperature != nil {
		err = turn.options.SetTemperature(*settings.temperature)
	}
	if err == nil && settings.topP != nil {
		err = turn.options.SetTopP(*settings.topP)
	}
	if err == nil && settings.seed != nil {
		err = turn.options.SetSeed(*settings.seed)
	}
	if err == nil && len(settings.stopStrings) != 0 {
		err = turn.options.SetStopStrings(settings.stopStrings)
	}
	if err == nil && settings.guidance != nil {
		err = turn.options.SetGuidance(ortgenai.GuidanceType(settings.guidance.Type), settings.guidance.Data)
	}
	if err != nil {
		cleanup()
		return fmt.Errorf("configuring turn options: %w", err)
	}
	turn.stream, err = s.tokenizer.NewStream()
	if err != nil {
		cleanup()
		return fmt.Errorf("creating tokenizer stream: %w", err)
	}
	turn.id, err = request.BeginTurn(inputIDs, turn.options)
	if err != nil {
		cleanup()
		return fmt.Errorf("beginning turn: %w", err)
	}
	s.turns[request] = turn
	return nil
}

func (s *engineORTState) advance(owner *engineORTOwner) {
	for job := range s.jobs {
		if !job.cancelled {
			if err := job.ctx.Err(); err != nil {
				s.cancelJob(job, err)
			}
		}
	}
	_ = s.schedule()
	if len(s.turns) != 0 {
		events, err := s.engine.Run(s.eventBuffer)
		if err != nil {
			s.failAll(fmt.Errorf("running ORT GenAI engine: %w", err))
		} else {
			for _, event := range events {
				s.handleEvent(owner, event)
			}
		}
	}
	s.finishCompletedJobs()
}

func (s *engineORTState) handleEvent(owner *engineORTOwner, event ortgenai.Event) {
	turn, err := findORTEventTurn(s.turns, event)
	if err != nil {
		s.failAll(err)
		return
	}
	if event.Flags&ortgenai.EventFlagToken != 0 && !turn.cancelling {
		text, err := turn.stream.Decode(event.Token)
		if err != nil {
			turn.job.errors = append(turn.job.errors, fmt.Errorf("sequence %d: decoding generated token: %w", turn.sequence, err))
			turn.cancelling = true
			_, cancelErr := turn.request.CancelTurn(turn.id)
			if cancelErr != nil {
				turn.job.errors = append(turn.job.errors, cancelErr)
			}
		} else if text != "" {
			select {
			case turn.job.tokenStream <- SequenceDelta{Token: text, Sequence: turn.sequence}:
			default:
				turn.job.errors = append(turn.job.errors, fmt.Errorf("sequence %d: generation consumer is not keeping up with the engine stream", turn.sequence))
				turn.cancelling = true
				_, cancelErr := turn.request.CancelTurn(turn.id)
				if cancelErr != nil {
					turn.job.errors = append(turn.job.errors, cancelErr)
				}
			}
		}
	}
	if event.Flags&(ortgenai.EventFlagFailed|ortgenai.EventFlagRetryable) != 0 {
		turn.job.errors = append(turn.job.errors, fmt.Errorf("sequence %d: ORT GenAI engine failed (error code %d)", turn.sequence, event.ErrorCode))
		s.finishTurn(turn, true)
		return
	}
	if event.Flags&ortgenai.EventFlagTurnFinished != 0 {
		if event.FinishReason == ortgenai.FinishReasonFailed {
			turn.job.errors = append(turn.job.errors, fmt.Errorf("sequence %d: ORT GenAI turn failed", turn.sequence))
		} else if event.FinishReason == ortgenai.FinishReasonCancelled && !turn.cancelling && !turn.job.cancelled {
			turn.job.errors = append(turn.job.errors, fmt.Errorf("sequence %d: ORT GenAI turn was cancelled", turn.sequence))
		}
		if event.Usage != nil {
			owner.tokens.Add(event.Usage.GeneratedTokens)
		}
		s.finishTurn(turn, false)
	}
}

func findORTEventTurn(turns map[*ortgenai.Request]*engineORTTurn, event ortgenai.Event) (*engineORTTurn, error) {
	turn := turns[event.Request]
	if turn == nil {
		return nil, errors.New("ORT GenAI engine returned an event for an unknown request")
	}
	if event.TurnID != turn.id {
		return nil, fmt.Errorf("ORT GenAI engine returned turn %d for request turn %d", event.TurnID, turn.id)
	}
	return turn, nil
}

func runEngineORTCommand(state *engineORTState, command engineORTCommand) {
	if err := command.ctx.Err(); err != nil {
		command.done <- err
		return
	}
	err := command.run(state)
	if err == nil {
		err = state.schedule()
	}
	command.done <- err
}

func (s *engineORTState) finishTurn(turn *engineORTTurn, force bool) {
	if _, exists := s.turns[turn.request]; !exists {
		return
	}
	if turn.stream != nil {
		turn.stream.Destroy()
	}
	if turn.options != nil {
		turn.options.Destroy()
	}
	if !force {
		if err := turn.request.Close(); err != nil {
			turn.job.errors = append(turn.job.errors, fmt.Errorf("closing request: %w", err))
		}
	}
	turn.request.Destroy()
	delete(s.turns, turn.request)
}

func (s *engineORTState) cancelJob(job *engineORTJob, err error) {
	if job.cancelled {
		return
	}
	job.cancelled = true
	job.cancelErr = err
	job.nextInput = len(job.inputs)
	if err != nil {
		job.errors = append(job.errors, err)
	}
	for _, turn := range s.turns {
		if turn.job == job && !turn.cancelling {
			turn.cancelling = true
			if _, cancelErr := turn.request.CancelTurn(turn.id); cancelErr != nil {
				job.errors = append(job.errors, fmt.Errorf("cancelling sequence %d: %w", turn.sequence, cancelErr))
			}
		}
	}
}

func (s *engineORTState) failAll(err error) {
	for job := range s.jobs {
		job.errors = append(job.errors, err)
		job.cancelled = true
		job.nextInput = len(job.inputs)
	}
	for _, turn := range s.turns {
		s.finishTurn(turn, true)
	}
	s.finishCompletedJobs()
}

func (s *engineORTState) finishCompletedJobs() {
	for job := range s.jobs {
		if job.nextInput < len(job.inputs) {
			continue
		}
		active := false
		for _, turn := range s.turns {
			if turn.job == job {
				active = true
				break
			}
		}
		if active {
			continue
		}
		if destroyErr := job.batch.Destroy(); destroyErr != nil {
			job.errors = append(job.errors, destroyErr)
		}
		if len(job.errors) != 0 {
			select {
			case job.errorStream <- errors.Join(job.errors...):
			default:
			}
		}
		job.stopSession()
		job.cancel()
		close(job.tokenStream)
		close(job.errorStream)
		delete(s.jobs, job)
	}
}

func (s *engineORTState) shutdown() {
	for job := range s.jobs {
		s.cancelJob(job, context.Canceled)
	}
	for len(s.turns) != 0 {
		events, err := s.engine.Run(s.eventBuffer)
		if err != nil {
			s.failAll(fmt.Errorf("stopping ORT GenAI engine: %w", err))
			break
		}
		for _, event := range events {
			turn := s.turns[event.Request]
			if turn != nil && event.TurnID == turn.id && event.Flags&(ortgenai.EventFlagTurnFinished|ortgenai.EventFlagFailed|ortgenai.EventFlagRetryable) != 0 {
				s.finishTurn(turn, event.Flags&(ortgenai.EventFlagFailed|ortgenai.EventFlagRetryable) != 0)
			}
		}
		runtime.Gosched()
	}
	s.finishCompletedJobs()
}

func (s *engineORTState) destroy() {
	if s.tokenizer != nil {
		s.tokenizer.Destroy()
	}
	if s.eventBuffer != nil {
		s.eventBuffer.Destroy()
	}
	if s.engine != nil {
		s.engine.Destroy()
	}
}
