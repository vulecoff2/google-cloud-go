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
)

type blockingReader struct {
	mu      sync.Mutex
	closed  atomic.Bool
	unblock chan struct{}
}

func newBlockingReader() *blockingReader {
	return &blockingReader{
		unblock: make(chan struct{}),
	}
}

func (b *blockingReader) Read(p []byte) (int, error) {
	if b.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	<-b.unblock
	if b.closed.Load() {
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

func TestStallReader_NormalRead(t *testing.T) {
	bdm, err := newBucketDelayManager(0.99, 1.0, 500*time.Millisecond, 100*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	data := bytes.Repeat([]byte("a"), int(defaultStallBlockSize*2+512))
	body := io.NopCloser(bytes.NewReader(data))

	sr := newStallReader(body, "test-bucket", bdm, nil, "localhost", context.Background())
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
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if totalRead != int64(len(data)) {
		t.Errorf("got %d bytes, want %d bytes", totalRead, len(data))
	}
}

func TestStallReader_StallDetected(t *testing.T) {
	// Min delay: 30ms, initial: 30ms.
	bdm, err := newBucketDelayManager(0.99, 1.0, 30*time.Millisecond, 30*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	initialDelay := bdm.getValue("test-bucket")
	br := newBlockingReader()
	defer br.Close()

	sr := newStallReader(br, "test-bucket", bdm, nil, "localhost", context.Background())
	defer sr.Close()

	buf := make([]byte, 1024)
	start := time.Now()
	_, err = sr.Read(buf)
	duration := time.Since(start)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got: %v", err)
	}

	if duration < 25*time.Millisecond {
		t.Errorf("read returned too quickly: %v", duration)
	}

	// Verify that dynamic delay was increased.
	newDelay := bdm.getValue("test-bucket")
	if newDelay <= initialDelay {
		t.Errorf("expected delay to increase, got %v <= %v", newDelay, initialDelay)
	}
}

func TestStallReader_ApplicationPauseDoesNotStall(t *testing.T) {
	bdm, err := newBucketDelayManager(0.99, 1.0, 50*time.Millisecond, 50*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	data := []byte("hello world")
	body := io.NopCloser(bytes.NewReader(data))

	sr := newStallReader(body, "test-bucket", bdm, nil, "localhost", context.Background())
	defer sr.Close()

	buf := make([]byte, 5)
	n, err := sr.Read(buf)
	if err != nil || n != 5 {
		t.Fatalf("first read: n=%d, err=%v", n, err)
	}

	// Simulate application pausing for 100ms (longer than the 50ms stall timeout)
	time.Sleep(100 * time.Millisecond)

	// Second read should still succeed because application pause is not network wait time.
	n, err = sr.Read(buf)
	if err != nil || n != 5 {
		t.Fatalf("second read after pause: n=%d, err=%v", n, err)
	}
}
