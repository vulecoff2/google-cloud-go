// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package storage

import (
	"bytes"
	"context"
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

type blockingReader struct {
	mu      sync.Mutex
	closed  atomic.Bool
	unblock chan struct{}
	started chan struct{}
	onClose func(p []byte) (int, error)
}

func newBlockingReader() *blockingReader {
	return &blockingReader{
		unblock: make(chan struct{}),
		started: make(chan struct{}, 1),
	}
}

func (b *blockingReader) Read(p []byte) (int, error) {
	select {
	case b.started <- struct{}{}:
	default:
	}
	if b.closed.Load() {
		if b.onClose != nil {
			return b.onClose(p)
		}
		return 0, io.ErrClosedPipe
	}
	<-b.unblock
	if b.closed.Load() {
		if b.onClose != nil {
			return b.onClose(p)
		}
		return 0, io.ErrClosedPipe
	}
	return 0, io.EOF
}

func (b *blockingReader) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.closed.Swap(true) {
		close(b.unblock)
	}
	return nil
}

type scriptedReadResult struct {
	data  []byte
	delay time.Duration
	block bool
	err   error
}

type scriptedReader struct {
	steps   []scriptedReadResult
	idx     int
	unblock chan struct{}
	once    sync.Once
}

func (s *scriptedReader) Read(p []byte) (int, error) {
	if s.idx >= len(s.steps) {
		return 0, io.EOF
	}
	step := s.steps[s.idx]
	s.idx++
	if step.block && s.unblock != nil {
		<-s.unblock
		return 0, io.ErrClosedPipe
	}
	if step.delay > 0 {
		time.Sleep(step.delay)
	}
	n := copy(p, step.data)
	return n, step.err
}

func (s *scriptedReader) Close() error {
	if s.unblock != nil {
		s.once.Do(func() { close(s.unblock) })
	}
	return nil
}

func TestStallReader_NormalRead(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 500*time.Millisecond, 100*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	data := bytes.Repeat([]byte("a"), int(defaultStallBlockSize*2+512))
	body := io.NopCloser(bytes.NewReader(data))

	sr := newStallReader(context.Background(), body, "test-bucket", bdm, nil, "localhost")
	defer sr.Close()

	buf := make([]byte, 64*1024)
	var totalRead int64
	for {
		n, err := sr.Read(buf)
		totalRead += int64(n)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("sr.Read() error = %v, want nil", err)
		}
	}

	if totalRead != int64(len(data)) {
		t.Errorf("total bytes read = %d, want %d", totalRead, len(data))
	}
}

func TestStallReader_StallDetected(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 30*time.Millisecond, 30*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	initialDelay := bdm.getValue("test-bucket")
	br := newBlockingReader()
	defer br.Close()

	sr := newStallReader(context.Background(), br, "test-bucket", bdm, nil, "localhost")
	defer sr.Close()

	buf := make([]byte, 1024)
	start := time.Now()
	_, err = sr.Read(buf)
	duration := time.Since(start)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("sr.Read() error = %v, want %v", err, context.DeadlineExceeded)
	}
	if duration < 25*time.Millisecond {
		t.Errorf("sr.Read() duration = %v, want >= 25ms", duration)
	}

	if newDelay := bdm.getValue("test-bucket"); newDelay <= initialDelay {
		t.Errorf("bdm.getValue(\"test-bucket\") = %v, want > %v", newDelay, initialDelay)
	}

	// Subsequent reads on a stalled reader immediately return context.DeadlineExceeded.
	if _, err := sr.Read(buf); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("subsequent sr.Read() error = %v, want %v", err, context.DeadlineExceeded)
	}
}

func TestStallReader_ApplicationPauseDoesNotStall(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 50*time.Millisecond, 50*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	data := []byte("hello world")
	body := io.NopCloser(bytes.NewReader(data))

	sr := newStallReader(context.Background(), body, "test-bucket", bdm, nil, "localhost")
	defer sr.Close()

	buf := make([]byte, 5)
	n, err := sr.Read(buf)
	if err != nil || n != 5 {
		t.Fatalf("first sr.Read() = (%d, %v), want (5, nil)", n, err)
	}

	// Simulate application pausing longer than the 50ms stall timeout between reads.
	time.Sleep(100 * time.Millisecond)

	n, err = sr.Read(buf)
	if err != nil || n != 5 {
		t.Fatalf("second sr.Read() after pause = (%d, %v), want (5, nil)", n, err)
	}
}

func TestStallReader_MultiReadChunkBudget(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 300*time.Millisecond, 50*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	initialDelay := bdm.getValue("test-bucket")
	body := &scriptedReader{
		unblock: make(chan struct{}),
		steps: []scriptedReadResult{
			{data: bytes.Repeat([]byte("a"), 16), delay: 100 * time.Millisecond},
			{data: bytes.Repeat([]byte("b"), 16), delay: 100 * time.Millisecond},
			{block: true},
		},
	}
	defer body.Close()

	// Two 100ms reads consume 200ms of the 300ms 2 MiB block budget; the third
	// read blocks and stalls once the remaining ~100ms elapses.
	sr := newStallReader(context.Background(), body, "test-bucket", bdm, nil, "localhost")
	defer sr.Close()

	buf := make([]byte, 16)
	for i := 0; i < 2; i++ {
		if n, err := sr.Read(buf); err != nil || n != 16 {
			t.Fatalf("sr.Read() #%d = (%d, %v), want (16, nil)", i+1, n, err)
		}
	}

	start := time.Now()
	_, err = sr.Read(buf)
	thirdDuration := time.Since(start)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("third sr.Read() error = %v, want %v", err, context.DeadlineExceeded)
	}
	// If budget hadn't carried over, the third read would wait the full 300ms.
	if thirdDuration >= 250*time.Millisecond {
		t.Errorf("third sr.Read() duration = %v, want < 250ms (inherited remaining budget)", thirdDuration)
	}
	if got := bdm.getValue("test-bucket"); got <= initialDelay {
		t.Errorf("bdm.getValue(\"test-bucket\") = %v, want > %v", got, initialDelay)
	}
}

func TestStallReader_RemainingTimeExhaustedBeforeRead(t *testing.T) {
	t.Parallel()
	// targetPercentile=0.5, increaseRate=1.0 => increaseFactor=2.0, decreaseFactor=0.5.
	bdm, err := newBucketDelayManager(0.5, 1.0, 120*time.Millisecond, 20*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	body := &scriptedReader{
		steps: []scriptedReadResult{
			{data: []byte("first-8b"), delay: 50 * time.Millisecond},
			{data: []byte("next-8b!")},
		},
	}
	sr := newStallReader(context.Background(), body, "test-bucket", bdm, nil, "localhost")
	defer sr.Close()

	buf := make([]byte, 8)
	if n, err := sr.Read(buf); err != nil || n != 8 {
		t.Fatalf("first sr.Read() = (%d, %v), want (8, nil)", n, err)
	}

	// Simulate concurrent fast reads lowering the bucket timeout (120ms -> 60ms -> 30ms),
	// below the 50ms already spent in the current 2 MiB block.
	bdm.decrease("test-bucket")
	bdm.decrease("test-bucket")
	beforeSecondRead := bdm.getValue("test-bucket")

	n, err := sr.Read(buf)
	if n != 0 || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second sr.Read() = (%d, %v), want (0, %v)", n, err, context.DeadlineExceeded)
	}
	if got := bdm.getValue("test-bucket"); got <= beforeSecondRead {
		t.Errorf("bdm.getValue(\"test-bucket\") = %v, want > %v", got, beforeSecondRead)
	}
}

func TestStallReader_BlockBoundaryCappedAndResets(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 80*time.Millisecond, 20*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	// Make total read time 4 * 35ms = 140ms > 80ms while each 2 MiB block takes
	// 35ms < 80ms. Passing a 4 MiB caller buffer verifies both that each Read is
	// capped at defaultStallBlockSize (2 MiB) and that timeSpentInChunk resets
	// on every block boundary.
	blockLen := int(defaultStallBlockSize)
	block := bytes.Repeat([]byte("x"), blockLen)
	steps := []scriptedReadResult{
		{data: block, delay: 35 * time.Millisecond},
		{data: block, delay: 35 * time.Millisecond},
		{data: block, delay: 35 * time.Millisecond},
		{data: block, delay: 35 * time.Millisecond},
	}
	sr := newStallReader(context.Background(), &scriptedReader{steps: steps}, "test-bucket", bdm, nil, "localhost")
	defer sr.Close()

	largeBuf := make([]byte, blockLen*2)
	var totalRead int
	for {
		n, err := sr.Read(largeBuf)
		if n > blockLen {
			t.Fatalf("sr.Read() byte count = %d, want <= %d (defaultStallBlockSize)", n, blockLen)
		}
		totalRead += n
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("sr.Read() error = %v, want nil", err)
		}
	}
	if want := blockLen * len(steps); totalRead != want {
		t.Errorf("total bytes read = %d, want %d", totalRead, want)
	}
}

func TestStallReader_CloseCancelsTimer(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 60*time.Millisecond, 30*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}
	initialDelay := bdm.getValue("test-bucket")

	br := newBlockingReader()
	sr := newStallReader(context.Background(), br, "test-bucket", bdm, nil, "localhost")

	readDone := make(chan error, 1)
	go func() {
		_, err := sr.Read(make([]byte, 128))
		readDone <- err
	}()

	<-br.started
	if err := sr.Close(); err != nil {
		t.Fatalf("sr.Close() error = %v, want nil", err)
	}

	if err := <-readDone; !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("sr.Read() after concurrent Close() = %v, want %v", err, io.ErrClosedPipe)
	}

	// Wait past the original stall timeout to confirm the canceled timer never fired.
	time.Sleep(80 * time.Millisecond)
	if got := bdm.getValue("test-bucket"); got != initialDelay {
		t.Errorf("bdm.getValue(\"test-bucket\") = %v, want %v", got, initialDelay)
	}

	// Reads after Close fail immediately.
	if _, err := sr.Read(make([]byte, 16)); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("sr.Read() on closed reader = %v, want %v", err, io.ErrClosedPipe)
	}
}

func TestStallReader_EOFDuringTimerRaceReturnsEOF(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 25*time.Millisecond, 25*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	br := newBlockingReader()
	want := []byte("final bytes")
	br.onClose = func(p []byte) (int, error) {
		return copy(p, want), io.EOF
	}
	defer br.Close()

	sr := newStallReader(context.Background(), br, "test-bucket", bdm, nil, "localhost")
	defer sr.Close()

	buf := make([]byte, 64)
	n, err := sr.Read(buf)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("sr.Read() error = %v, want %v", err, io.EOF)
	}
	if !bytes.Equal(buf[:n], want) {
		t.Errorf("sr.Read() bytes = %q, want %q", buf[:n], want)
	}

	// Subsequent reads after EOF must continue returning (0, io.EOF), not
	// context.DeadlineExceeded from the cleared stalled flag.
	if n2, err2 := sr.Read(buf); n2 != 0 || !errors.Is(err2, io.EOF) {
		t.Errorf("second sr.Read() after EOF = (%d, %v), want (0, %v)", n2, err2, io.EOF)
	}
	if n3, err3 := sr.Read(nil); n3 != 0 || !errors.Is(err3, io.EOF) {
		t.Errorf("sr.Read(nil) after EOF = (%d, %v), want (0, %v)", n3, err3, io.EOF)
	}
}

type ctxBlockingReader struct {
	ctx     context.Context
	started chan struct{}
	unblock chan struct{}
	once    sync.Once
}

func newCtxBlockingReader(ctx context.Context) *ctxBlockingReader {
	return &ctxBlockingReader{
		ctx:     ctx,
		started: make(chan struct{}, 1),
		unblock: make(chan struct{}),
	}
}

func (r *ctxBlockingReader) Read([]byte) (int, error) {
	select {
	case r.started <- struct{}{}:
	default:
	}
	select {
	case <-r.ctx.Done():
		return 0, r.ctx.Err()
	case <-r.unblock:
		return 0, io.ErrClosedPipe
	}
}

func (r *ctxBlockingReader) Close() error {
	r.once.Do(func() { close(r.unblock) })
	return nil
}

func TestStallReader_ContextCanceledDoesNotPenalizeOrRecordMetric(t *testing.T) {
	t.Parallel()

	t.Run("body_unblocks_before_timer", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		mr := sdkmetric.NewManualReader()
		provider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(mr))
		defer provider.Shutdown(context.Background())

		cm, _, err := initMetrics(t.Context(), "project-id", &storageConfig{
			enableOtelMetrics:      true,
			enableOtelDebugMetrics: true,
			meterProvider:          provider,
		})
		if err != nil {
			t.Fatalf("initMetrics: %v", err)
		}

		// Long timeout ensures the context-aware body unblocks well before the
		// stall timer fires, exercising the post-Read context cancellation path.
		bdm, err := newBucketDelayManager(0.99, 1.0, 500*time.Millisecond, 500*time.Millisecond, 2*time.Second)
		if err != nil {
			t.Fatalf("newBucketDelayManager: %v", err)
		}
		initialDelay := bdm.getValue("test-bucket")

		br := newCtxBlockingReader(ctx)
		defer br.Close()
		sr := newStallReader(ctx, br, "test-bucket", bdm, cm, "storage.googleapis.com")
		defer sr.Close()

		errCh := make(chan error, 1)
		start := time.Now()
		go func() {
			_, err := sr.Read(make([]byte, 128))
			errCh <- err
		}()

		<-br.started
		cancel()

		err = <-errCh
		elapsed := time.Since(start)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("sr.Read() error = %v, want %v", err, context.Canceled)
		}
		if elapsed >= 250*time.Millisecond {
			t.Errorf("sr.Read() wakeup elapsed = %v, want < 250ms", elapsed)
		}
		if got := bdm.getValue("test-bucket"); got != initialDelay {
			t.Errorf("bdm.getValue(\"test-bucket\") = %v, want %v", got, initialDelay)
		}

		var rm metricdata.ResourceMetrics
		if err := mr.Collect(t.Context(), &rm); err != nil {
			t.Fatalf("mr.Collect() error = %v, want nil", err)
		}
		for _, sm := range rm.ScopeMetrics {
			for _, m := range sm.Metrics {
				if m.Name == "gcp.storage.client.stall.duration" {
					t.Errorf("recorded metric %q on caller context cancellation, want none", m.Name)
				}
			}
		}
	})

	t.Run("timer_fires_after_cancel", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		mr := sdkmetric.NewManualReader()
		provider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(mr))
		defer provider.Shutdown(context.Background())

		cm, _, err := initMetrics(t.Context(), "project-id", &storageConfig{
			enableOtelMetrics:      true,
			enableOtelDebugMetrics: true,
			meterProvider:          provider,
		})
		if err != nil {
			t.Fatalf("initMetrics: %v", err)
		}

		bdm, err := newBucketDelayManager(0.99, 1.0, 30*time.Millisecond, 30*time.Millisecond, 2*time.Second)
		if err != nil {
			t.Fatalf("newBucketDelayManager: %v", err)
		}
		initialDelay := bdm.getValue("test-bucket")

		br := newBlockingReader()
		defer br.Close()
		sr := newStallReader(ctx, br, "test-bucket", bdm, cm, "storage.googleapis.com")
		defer sr.Close()

		errCh := make(chan error, 1)
		go func() {
			_, err := sr.Read(make([]byte, 128))
			errCh <- err
		}()

		<-br.started
		// Cancel caller context before the 30ms stall timer fires on a body that
		// only unblocks when Close() is called by the timer callback.
		cancel()

		if err := <-errCh; !errors.Is(err, context.Canceled) {
			t.Fatalf("sr.Read() error = %v, want %v", err, context.Canceled)
		}
		if got := bdm.getValue("test-bucket"); got != initialDelay {
			t.Errorf("bdm.getValue(\"test-bucket\") = %v, want %v", got, initialDelay)
		}

		var rm metricdata.ResourceMetrics
		if err := mr.Collect(t.Context(), &rm); err != nil {
			t.Fatalf("mr.Collect() error = %v, want nil", err)
		}
		for _, sm := range rm.ScopeMetrics {
			for _, m := range sm.Metrics {
				if m.Name == "gcp.storage.client.stall.duration" {
					t.Errorf("recorded metric %q on caller context cancellation, want none", m.Name)
				}
			}
		}
	})
}

func TestStallReader_SubBlockReadsDecayDelay(t *testing.T) {
	t.Parallel()
	// targetPercentile=0.5, increaseRate=1.0 => decreaseFactor=0.5 so any
	// update() call is immediately observable.
	bdm, err := newBucketDelayManager(0.5, 1.0, 160*time.Millisecond, 20*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	const quarterBlock = defaultStallBlockSize / 4 // 512 KiB (< 2 MiB)
	payload := bytes.Repeat([]byte("s"), int(quarterBlock))
	initialDelay := bdm.getValue("small-bucket")

	// Three 512 KiB reads (1.5 MiB total) should accumulate without yet
	// triggering a 2 MiB-normalized decay update. Mix EOF reads and Close-after-read
	// (the HTTP/2 httpReader completion pattern).
	for i := 0; i < 3; i++ {
		sr := newStallReader(t.Context(), io.NopCloser(bytes.NewReader(payload)), "small-bucket", bdm, nil, "localhost")
		if i == 1 {
			buf := make([]byte, len(payload))
			if _, err := io.ReadFull(sr, buf); err != nil {
				t.Fatalf("io.ReadFull() #%d error = %v, want nil", i+1, err)
			}
		} else {
			if _, err := io.ReadAll(sr); err != nil {
				t.Fatalf("io.ReadAll() #%d error = %v, want nil", i+1, err)
			}
		}
		if err := sr.Close(); err != nil {
			t.Fatalf("sr.Close() #%d error = %v, want nil", i+1, err)
		}
	}
	if got := bdm.getValue("small-bucket"); got != initialDelay {
		t.Errorf("bdm.getValue(\"small-bucket\") after 1.5 MiB = %v, want %v", got, initialDelay)
	}

	// Fourth 512 KiB read crosses 2 MiB cumulative bytes and decays the delay.
	sr := newStallReader(t.Context(), io.NopCloser(bytes.NewReader(payload)), "small-bucket", bdm, nil, "localhost")
	if _, err := io.ReadAll(sr); err != nil {
		t.Fatalf("io.ReadAll() #4 error = %v, want nil", err)
	}
	if err := sr.Close(); err != nil {
		t.Fatalf("sr.Close() #4 error = %v, want nil", err)
	}
	if got := bdm.getValue("small-bucket"); got >= initialDelay {
		t.Errorf("bdm.getValue(\"small-bucket\") after 2 MiB cumulative = %v, want < %v", got, initialDelay)
	}
}

func TestStallReader_RecordsMetric(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mr := sdkmetric.NewManualReader()
	provider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(mr))
	defer provider.Shutdown(ctx)

	cm, _, err := initMetrics(ctx, "project-id", &storageConfig{
		enableOtelMetrics:      true,
		enableOtelDebugMetrics: true,
		meterProvider:          provider,
	})
	if err != nil {
		t.Fatalf("initMetrics: %v", err)
	}

	bdm, err := newBucketDelayManager(0.99, 1.0, 25*time.Millisecond, 25*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	br := newBlockingReader()
	defer br.Close()
	sr := newStallReader(ctx, br, "test-bucket", bdm, cm, "storage.googleapis.com")
	defer sr.Close()

	if _, err := sr.Read(make([]byte, 128)); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("sr.Read() error = %v, want %v", err, context.DeadlineExceeded)
	}

	var rm metricdata.ResourceMetrics
	if err := mr.Collect(ctx, &rm); err != nil {
		t.Fatalf("mr.Collect() error = %v, want nil", err)
	}

	found := false
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			if m.Name == "gcp.storage.client.stall.duration" {
				found = true
				hist := m.Data.(metricdata.Histogram[float64])
				if len(hist.DataPoints) == 0 {
					t.Fatalf("stallDuration DataPoints len = 0, want >= 1")
				}
				dp := hist.DataPoints[0]
				if want := (25 * time.Millisecond).Seconds(); dp.Sum != want {
					t.Errorf("stallDuration Sum = %v, want %v", dp.Sum, want)
				}
				for k, want := range map[string]string{
					"rpc.system.name":       "http",
					"gcp.storage.operation": "ReadObject",
					"rpc.method":            "google.storage.v2.Storage/ReadObject",
					"network.path":          "cloudpath",
					"server.address":        "storage.googleapis.com",
				} {
					if got := getHistAttr(dp, k); got != want {
						t.Errorf("stallDuration attr %q = %q, want %q", k, got, want)
					}
				}
			}
		}
	}
	if !found {
		t.Errorf("metric %q not found", "gcp.storage.client.stall.duration")
	}
}

func TestStallReader_ZeroLengthRead(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.99, 1.0, 25*time.Millisecond, 25*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	sr := newStallReader(context.Background(), io.NopCloser(bytes.NewReader([]byte("abc"))), "test-bucket", bdm, nil, "localhost")
	n, err := sr.Read(nil)
	if n != 0 || err != nil {
		t.Fatalf("sr.Read(nil) on open reader = (%d, %v), want (0, nil)", n, err)
	}
	if err := sr.Close(); err != nil {
		t.Fatalf("sr.Close() error = %v, want nil", err)
	}
	if _, err := sr.Read(nil); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("sr.Read(nil) on closed reader = %v, want %v", err, io.ErrClosedPipe)
	}
}

type errAfterBytesReader struct {
	remaining int
	err       error
}

func (r *errAfterBytesReader) Read(p []byte) (int, error) {
	if r.remaining > 0 {
		n := len(p)
		if n > r.remaining {
			n = r.remaining
		}
		r.remaining -= n
		return n, nil
	}
	return 0, r.err
}

func (r *errAfterBytesReader) Close() error { return nil }

func TestStallReader_NonStallErrorDoesNotFlushPartialOnClose(t *testing.T) {
	t.Parallel()
	bdm, err := newBucketDelayManager(0.5, 1.0, 200*time.Millisecond, 50*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}
	sr := newStallReader(context.Background(), &errAfterBytesReader{
		remaining: 512 * 1024,
		err:       io.ErrUnexpectedEOF,
	}, "test-bucket", bdm, nil, "localhost")

	buf := make([]byte, 512*1024)
	if n, err := sr.Read(buf); n != len(buf) || err != nil {
		t.Fatalf("first sr.Read() = (%d, %v), want (%d, nil)", n, err, len(buf))
	}
	if _, err := sr.Read(buf); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("second sr.Read() error = %v, want %v", err, io.ErrUnexpectedEOF)
	}
	if err := sr.Close(); err != nil {
		t.Fatalf("sr.Close() error = %v, want nil", err)
	}

	d := bdm.getDelay("test-bucket")
	d.mu.RLock()
	gotPartial := d.partialBytes
	d.mu.RUnlock()
	if gotPartial != 0 {
		t.Errorf("partialBytes after non-stall error Close() = %d, want 0", gotPartial)
	}
}

type delayedEOFReader struct {
	delay  time.Duration
	closed chan struct{}
	once   sync.Once
}

func (r *delayedEOFReader) Read(p []byte) (int, error) {
	timer := time.NewTimer(r.delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return 0, io.EOF
	case <-r.closed:
		return 0, io.ErrClosedPipe
	}
}

func (r *delayedEOFReader) Close() error {
	r.once.Do(func() { close(r.closed) })
	return nil
}

func TestStallReader_DrainEOF(t *testing.T) {
	t.Parallel()
	t.Run("healthy_delayed_eof", func(t *testing.T) {
		t.Parallel()
		bdm, err := newBucketDelayManager(0.5, 1.0, 200*time.Millisecond, 50*time.Millisecond, 2*time.Second)
		if err != nil {
			t.Fatalf("newBucketDelayManager: %v", err)
		}
		dr := &delayedEOFReader{delay: 5 * time.Millisecond, closed: make(chan struct{})}
		sr := newStallReader(context.Background(), dr, "test-bucket", bdm, nil, "localhost")
		sr.mu.Lock()
		sr.bytesInChunk = 512 * 1024
		sr.timeSpentInChunk = 2 * time.Millisecond
		sr.mu.Unlock()

		before := bdm.getValue("test-bucket")
		sr.drainEOF()

		if got := bdm.getValue("test-bucket"); got != before {
			t.Errorf("delay changed on healthy drainEOF: got %v, want %v", got, before)
		}
		d := bdm.getDelay("test-bucket")
		d.mu.RLock()
		gotPartial := d.partialBytes
		d.mu.RUnlock()
		if gotPartial != 512*1024 {
			t.Errorf("partialBytes after drainEOF = %d, want %d", gotPartial, 512*1024)
		}
		if n, err := sr.Read(make([]byte, 8)); n != 0 || !errors.Is(err, io.EOF) {
			t.Errorf("sr.Read() after drainEOF = (%d, %v), want (0, io.EOF)", n, err)
		}
	})

	t.Run("stalled_eof_no_penalty", func(t *testing.T) {
		t.Parallel()
		ctx := context.Background()
		mr := sdkmetric.NewManualReader()
		provider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(mr))
		defer provider.Shutdown(ctx)

		cm, _, err := initMetrics(ctx, "test-project", &storageConfig{
			enableOtelMetrics:      true,
			enableOtelDebugMetrics: true,
			meterProvider:          provider,
		})
		if err != nil {
			t.Fatalf("initMetrics: %v", err)
		}

		bdm, err := newBucketDelayManager(0.99, 1.0, 25*time.Millisecond, 25*time.Millisecond, 2*time.Second)
		if err != nil {
			t.Fatalf("newBucketDelayManager: %v", err)
		}
		br := newBlockingReader()
		sr := newStallReader(ctx, br, "test-bucket", bdm, cm, "storage.googleapis.com")
		sr.mu.Lock()
		sr.bytesInChunk = 64 * 1024
		sr.timeSpentInChunk = 2 * time.Millisecond
		sr.mu.Unlock()

		before := bdm.getValue("test-bucket")
		sr.drainEOF()

		if got := bdm.getValue("test-bucket"); got != before {
			t.Errorf("delay changed on stalled drainEOF: got %v, want %v", got, before)
		}
		var rm metricdata.ResourceMetrics
		if err := mr.Collect(ctx, &rm); err != nil {
			t.Fatalf("mr.Collect: %v", err)
		}
		for _, sm := range rm.ScopeMetrics {
			for _, m := range sm.Metrics {
				if m.Name == "gcp.storage.client.stall.duration" {
					t.Errorf("unexpected stall.duration metric recorded during drainEOF")
				}
			}
		}
	})
}
