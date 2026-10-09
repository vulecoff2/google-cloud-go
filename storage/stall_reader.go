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
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// defaultStallBlockSize is the block size boundary for stall detection (2 MiB).
const defaultStallBlockSize int64 = 2 * 1024 * 1024

// stallReader wraps an io.ReadCloser to detect mid-stream stalls during read operations.
// Every defaultStallBlockSize (2 MiB) must be read within the delay dynamically computed
// by bucketDelayManager. If reading stalls for longer than that deadline, stallReader aborts
// the underlying body and returns context.DeadlineExceeded so the outer reader can reopen
// from the last confirmed offset.
type stallReader struct {
	body         io.ReadCloser
	bucket       string
	delayManager *bucketDelayManager
	metrics      *clientMetrics
	target       string
	ctx          context.Context
	blockSize    int64

	mu               sync.Mutex
	inRead           bool
	readID           uint64
	bytesInChunk     int64
	timeSpentInChunk time.Duration
	stalled          atomic.Bool
	closed           atomic.Bool
}

func newStallReader(body io.ReadCloser, bucket string, delayManager *bucketDelayManager, metrics *clientMetrics, target string, ctx context.Context) *stallReader {
	return &stallReader{
		body:         body,
		bucket:       bucket,
		delayManager: delayManager,
		metrics:      metrics,
		target:       target,
		ctx:          ctx,
		blockSize:    defaultStallBlockSize,
	}
}

func (s *stallReader) Read(p []byte) (int, error) {
	if s.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	if s.stalled.Load() {
		return 0, context.DeadlineExceeded
	}

	s.mu.Lock()
	currentTimeout := s.delayManager.getValue(s.bucket)
	remTime := currentTimeout - s.timeSpentInChunk
	if remTime <= 0 {
		s.stalled.Store(true)
		s.delayManager.increase(s.bucket)
		if s.metrics != nil {
			s.metrics.recordStallDuration(s.ctx, currentTimeout, "ReadObject", "http", s.target)
		}
		s.mu.Unlock()
		s.body.Close()
		return 0, context.DeadlineExceeded
	}

	// Cap p so a single Read call never reads past the current 2 MiB block boundary.
	remBytes := s.blockSize - s.bytesInChunk
	if int64(len(p)) > remBytes {
		p = p[:remBytes]
	}
	s.readID++
	myReadID := s.readID
	s.inRead = true
	s.mu.Unlock()

	timer := time.AfterFunc(remTime, func() {
		s.mu.Lock()
		if !s.inRead || s.readID != myReadID || s.closed.Load() {
			s.mu.Unlock()
			return
		}
		s.stalled.Store(true)
		s.delayManager.increase(s.bucket)
		if s.metrics != nil {
			s.metrics.recordStallDuration(s.ctx, currentTimeout, "ReadObject", "http", s.target)
		}
		s.mu.Unlock()
		log.Printf("storage: read stall detected after %v on bucket %s, closing connection", currentTimeout, s.bucket)
		s.body.Close()
	})

	readStart := time.Now()
	n, err := s.body.Read(p)
	readDuration := time.Since(readStart)
	timer.Stop()

	s.mu.Lock()
	s.inRead = false
	if s.stalled.Load() {
		s.mu.Unlock()
		return n, context.DeadlineExceeded
	}

	if n > 0 {
		s.bytesInChunk += int64(n)
		s.timeSpentInChunk += readDuration

		if s.bytesInChunk >= s.blockSize {
			s.delayManager.update(s.bucket, s.timeSpentInChunk)
			s.bytesInChunk = 0
			s.timeSpentInChunk = 0
		}
	}
	s.mu.Unlock()

	return n, err
}

func (s *stallReader) Close() error {
	s.closed.Store(true)
	return s.body.Close()
}
