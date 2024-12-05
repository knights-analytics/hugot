//go:build ALL

package hugot

import (
	"errors"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/knights-analytics/hugot/options"
)

var organisations = []string{
	"Tech Innovators Inc.",
	"Green Energy Solutions",
	"Global Ventures Co.",
	"Creative Minds Studio",
	"Healthcare Partners Ltd.",
	"Future Finance Group",
	"Premier Logistics LLC",
	"Dynamic Marketing Agency",
	"Eco-Friendly Products Corp.",
	"Blue Ocean Enterprises",
	"NextGen Software Solutions",
	"Innovative Construction Co.",
	"Precision Engineering Ltd.",
	"Elite Consulting Group",
	"Urban Development LLC",
	"The goal of this library is to provide an easy, scalable, and hassle-free way to run hugging face transformer pipelines in golang applications.",
	"Digital Media Concepts",
	"Community Builders Inc.",
	"Trusted Insurance Brokers",
	"Advanced Manufacturing Corp.",
	"Visionary Design Studio",
	"Strategic Investment Partners",
	"Modern Retail Solutions",
	"Efficient Energy Systems",
	"High-Tech Components Inc.",
	"Education Outreach Network",
	"Healthcare Innovations Ltd.",
	"Creative Film Productions",
	"Global Trade Services",
	"NextLevel Sports Management",
	"Sustainable Agriculture Group",
	"Cloud Based Solutions",
}

func checkBench(b *testing.B, err error) {
	b.Helper()
	if err != nil {
		b.Fatalf("Test failed with error %s", err.Error())
	}
}

func threadBenchmark(b *testing.B, session *Session, multiThread bool, randomBatches bool) {

	// Minimum of 32 iterations to minimise spikes due to graph compilations
	if b.N < 32 {
		b.N = 32
	}

	b.Helper()
	b.StopTimer()

	numWorkers := 1
	if multiThread {
		numWorkers = runtime.NumCPU()
	}

	modelPath := "./models/sentence-transformers_all-MiniLM-L6-v2"
	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	checkBench(b, err)

	inputChannel := make(chan []string, b.N)

	wg := sync.WaitGroup{}
	worker := func() {
		for inputs := range inputChannel {
			outputs, threadErr := pipeline.RunPipeline(inputs)
			if threadErr != nil {
				b.Error(threadErr)
			}
			if len(outputs.Embeddings) != len(inputs) {
				b.Error(errors.New("number of outputs does not match number of inputs"))
			}
		}
		wg.Done()
	}

	for range numWorkers {
		wg.Add(1)
		go worker()
	}

	// warmup
	_, threadErr := pipeline.RunPipeline(organisations)
	if threadErr != nil {
		b.Error(threadErr)
	}

	b.StartTimer()
	for range b.N {
		if !randomBatches {
			inputChannel <- organisations
		} else {
			randomNumber := rand.Intn(32) + 1
			inputChannel <- organisations[0:randomNumber]
		}
	}
	close(inputChannel)
	wg.Wait()
	b.StopTimer()
}

func BenchmarkXLAThreadBenchmarkSingle(b *testing.B) {
	session, err := NewXLASession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, false, false)
}

func BenchmarkORTThreadBenchmarkSingle(b *testing.B) {
	session, err := NewORTSession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, false, false)
}

func BenchmarkXLAThreadBenchmarkMulti(b *testing.B) {
	session, err := NewXLASession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, true, false)
}

func BenchmarkORTThreadBenchmarkMulti(b *testing.B) {
	session, err := NewORTSession(
		options.WithInterOpNumThreads(1),
		options.WithIntraOpNumThreads(1),
		options.WithCpuMemArena(false),
		options.WithMemPattern(false),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, true, false)
}

func BenchmarkXLAThreadBenchmarkRandomBatchSize(b *testing.B) {
	session, err := NewXLASession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, true, true)
}

func BenchmarkORTThreadBenchmarkRandomBatchSize(b *testing.B) {
	session, err := NewORTSession(
		options.WithInterOpNumThreads(1),
		options.WithIntraOpNumThreads(1),
		options.WithCpuMemArena(false),
		options.WithMemPattern(false),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, true, true)
}

func BenchmarkXLAThreadBenchmarkCuda(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	session, err := NewXLASession(
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, true, false)
}

func BenchmarkORTThreadBenchmarkCuda(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	session, err := NewORTSession(
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	threadBenchmark(b, session, true, false)
}
