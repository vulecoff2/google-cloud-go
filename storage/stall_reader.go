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
	"context"
	"io"
	"sync"
	"time"
)

const defaultStallBlockSize int64 = 2 * 1024 * 1024 // 2 MiB

// stallReader wraps an HTTP response body to detect mid-stream network stalls.
// It tracks the active network read duration (ignoring application processing pauses
// between Read calls) against a per-2 MiB chunk timeout budget sourced from
// bucketDelayManager. If a chunk exceeds its timeout budget, the underlying body
// is closed and context.DeadlineExceeded is returned so httpReader can transparently
// reopen the range request from the last read offset.
type stallReader struct {
	body    io.ReadCloser
	delay   *dynamicDelay
	metrics *clientMetrics
	target  string
	ctx     context.Context

	mu               sync.Mutex
	inRead           bool
	readID           uint64
	timer            *time.Timer
	bytesInChunk     int64
	timeSpentInChunk time.Duration
	stalled          bool
	hasErr           bool
	eof              bool
	closed           bool

	closeOnce sync.Once
	closeErr  error
}

func newStallReader(ctx context.Context, body io.ReadCloser, bucket string, delayManager *bucketDelayManager, metrics *clientMetrics, target string) *stallReader {
	var delay *dynamicDelay
	if delayManager != nil {
		delay = delayManager.getDelay(bucket)
	}
	return &stallReader{
		body:    body,
		delay:   delay,
		metrics: metrics,
		target:  target,
		ctx:     ctx,
	}
}

func (s *stallReader) Read(p []byte) (int, error) {
	if s.delay == nil {
		return s.body.Read(p)
	}

	if len(p) == 0 {
		s.mu.Lock()
		closed := s.closed
		eof := s.eof
		stalled := s.stalled
		s.mu.Unlock()
		if closed {
			return 0, io.ErrClosedPipe
		}
		if eof {
			return 0, io.EOF
		}
		if stalled {
			return 0, context.DeadlineExceeded
		}
		return s.body.Read(p)
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return 0, io.ErrClosedPipe
	}
	if s.eof {
		s.mu.Unlock()
		return 0, io.EOF
	}
	if s.stalled {
		s.mu.Unlock()
		return 0, context.DeadlineExceeded
	}
	if s.ctx != nil {
		if err := s.ctx.Err(); err != nil {
			s.mu.Unlock()
			return 0, err
		}
	}

	currentTimeout := s.delay.getValue()
	remTime := currentTimeout - s.timeSpentInChunk
	if remTime <= 0 {
		s.stalled = true
		s.delay.increase()
		if s.metrics != nil {
			s.metrics.recordStallDuration(s.ctx, currentTimeout, "ReadObject", "http", s.target)
		}
		s.mu.Unlock()
		s.closeBody()
		return 0, context.DeadlineExceeded
	}

	// Cap p so a single Read call never reads past the current 2 MiB block boundary.
	remBytes := defaultStallBlockSize - s.bytesInChunk
	if int64(len(p)) > remBytes {
		p = p[:remBytes]
	}

	s.readID++
	myReadID := s.readID
	s.inRead = true
	timer := time.AfterFunc(remTime, func() {
		s.mu.Lock()
		if !s.inRead || s.readID != myReadID || s.closed {
			s.mu.Unlock()
			return
		}
		s.timer = nil
		if s.ctx != nil && s.ctx.Err() != nil {
			s.mu.Unlock()
			s.closeBody()
			return
		}
		s.stalled = true
		s.delay.increase()
		if s.metrics != nil {
			s.metrics.recordStallDuration(s.ctx, currentTimeout, "ReadObject", "http", s.target)
		}
		s.mu.Unlock()
		s.closeBody()
	})
	s.timer = timer
	s.mu.Unlock()

	readStart := time.Now()
	n, err := s.body.Read(p)
	readDuration := time.Since(readStart)
	timer.Stop()

	s.mu.Lock()
	s.inRead = false
	if s.timer == timer {
		s.timer = nil
	}
	if s.stalled {
		if err == io.EOF {
			// The final bytes arrived concurrently with the stall timer firing.
			// Clear stalled and mark eof so subsequent Read calls consistently
			// return (0, io.EOF) rather than context.DeadlineExceeded, and drop
			// partial chunk counters so Close does not record a timed-out block.
			s.stalled = false
			s.eof = true
			s.bytesInChunk = 0
			s.timeSpentInChunk = 0
			s.mu.Unlock()
			return n, io.EOF
		}
		s.mu.Unlock()
		return n, context.DeadlineExceeded
	}
	if s.closed {
		s.mu.Unlock()
		return n, err
	}
	if s.ctx != nil {
		if ctxErr := s.ctx.Err(); ctxErr != nil && err != io.EOF {
			s.hasErr = true
			s.mu.Unlock()
			return n, ctxErr
		}
	}

	if n > 0 && (err == nil || err == io.EOF) {
		s.bytesInChunk += int64(n)
		s.timeSpentInChunk += readDuration

		if s.bytesInChunk >= defaultStallBlockSize {
			s.delay.update(s.timeSpentInChunk)
			s.bytesInChunk = 0
			s.timeSpentInChunk = 0
		}
	}
	if err == io.EOF {
		s.eof = true
		s.flushPartialLocked()
	} else if err != nil {
		s.hasErr = true
	}
	s.mu.Unlock()

	return n, err
}

// drainEOF attempts to consume the trailing EOF (e.g. HTTP/2 END_STREAM or an
// HTTP/1.1 final zero-length chunk) within a bounded timeout after all expected
// payload bytes have been read. This prevents healthy completed downloads from
// triggering RST_STREAM(CANCEL) or losing keep-alive connection reuse, while
// ensuring a stalled END_STREAM still completes promptly without penalizing
// dynamic delay or recording a stall metric.
func (s *stallReader) drainEOF() {
	s.mu.Lock()
	if s.closed || s.eof {
		s.mu.Unlock()
		return
	}
	if s.stalled {
		s.eof = true
		s.mu.Unlock()
		s.closeBody()
		return
	}
	// Flush completed sub-block progress before draining trailing framing so
	// even a stalled END_STREAM still credits the bytes already delivered.
	s.flushPartialLocked()
	if s.ctx != nil && s.ctx.Err() != nil {
		s.eof = true
		s.mu.Unlock()
		s.closeBody()
		return
	}
	timeout := defaultStreamDrainTimeout
	if s.delay != nil {
		if currentTimeout := s.delay.getValue(); currentTimeout < timeout {
			timeout = currentTimeout
		}
	}

	s.readID++
	myReadID := s.readID
	s.inRead = true
	timer := time.AfterFunc(timeout, func() {
		s.mu.Lock()
		if !s.inRead || s.readID != myReadID || s.closed {
			s.mu.Unlock()
			return
		}
		s.timer = nil
		s.mu.Unlock()
		s.closeBody()
	})
	s.timer = timer
	s.mu.Unlock()

	var buf [1]byte
	for range 16 {
		n, err := s.body.Read(buf[:])
		if err != nil || n > 0 {
			break
		}
	}
	timer.Stop()

	s.mu.Lock()
	s.inRead = false
	if s.timer == timer {
		s.timer = nil
	}
	s.eof = true
	s.mu.Unlock()
	s.closeBody()
}

func (s *stallReader) Close() error {
	s.mu.Lock()
	wasInRead := s.inRead
	s.closed = true
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	if !wasInRead {
		s.flushPartialLocked()
	}
	s.mu.Unlock()
	return s.closeBody()
}

// flushPartialLocked contributes completed sub-block read progress to the
// bucket's partial-block accumulator so workloads reading objects smaller than
// 2 MiB can still decay dynamic timeouts after transient stalls.
// Caller must hold s.mu.
func (s *stallReader) flushPartialLocked() {
	if s.delay == nil || s.stalled || s.hasErr || s.bytesInChunk <= 0 {
		return
	}
	if s.ctx != nil && s.ctx.Err() != nil {
		return
	}
	bytes := s.bytesInChunk
	dur := s.timeSpentInChunk
	s.bytesInChunk = 0
	s.timeSpentInChunk = 0
	s.delay.recordPartialBlock(bytes, dur, defaultStallBlockSize)
}

func (s *stallReader) closeBody() error {
	s.closeOnce.Do(func() {
		s.closeErr = s.body.Close()
	})
	return s.closeErr
}
