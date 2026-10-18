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
	"encoding/base64"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestStallRequestBody_WriteDelayStall(t *testing.T) {
	writeDM, err := newBucketDelayManager(0.99, 1.0, 30*time.Millisecond, 30*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}
	respDM, err := newBucketDelayManager(0.99, 1.0, 100*time.Millisecond, 100*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	initialWriteDelay := writeDM.getValue("test-bucket")
	data := bytes.Repeat([]byte("x"), 64*1024)
	srb := newStallRequestBody(
		io.NopCloser(bytes.NewReader(data)),
		"test-bucket",
		writeDM,
		respDM,
		nil,
		"localhost",
		context.Background(),
		defaultStallBlockSize,
		"",
		false,
		time.Time{},
	)
	defer srb.Close()
	defer srb.stopTimer()

	buf := make([]byte, 16*1024)
	n, err := srb.Read(buf)
	if err != nil || n != len(buf) {
		t.Fatalf("first Read: n=%d, err=%v", n, err)
	}

	// Simulate net/http blocking in socket.Write for longer than the 30ms Write Delay budget.
	select {
	case <-srb.stalledCh:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for Write Delay stall")
	}

	if !srb.didStall() {
		t.Fatal("expected didStall() == true")
	}
	if newDelay := writeDM.getValue("test-bucket"); newDelay <= initialWriteDelay {
		t.Errorf("expected writeDelayManager to increase on stall: got %v <= %v", newDelay, initialWriteDelay)
	}
	_, err = srb.Read(buf)
	if !errors.Is(err, errWriteStall) || !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected errWriteStall / DeadlineExceeded on subsequent Read, got %v", err)
	}
}

func TestStallRequestBody_QueueingThresholdSkipsFirst4MiB(t *testing.T) {
	// Start initialDelay at 400ms (above minDelay of 100ms) so any update() with <1ms latency decreases it.
	writeDM, err := newBucketDelayManager(0.99, 1.0, 400*time.Millisecond, 100*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}
	respDM, err := newBucketDelayManager(0.99, 1.0, 400*time.Millisecond, 100*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	blockSize := int64(1024) // Use 1 KiB block size for fast unit test (queue threshold = 2 KiB).
	data := bytes.Repeat([]byte("a"), int(blockSize*3))
	srb := newStallRequestBody(
		io.NopCloser(bytes.NewReader(data)),
		"test-bucket",
		writeDM,
		respDM,
		nil,
		"localhost",
		context.Background(),
		blockSize,
		"",
		false,
		time.Time{},
	)
	defer srb.Close()
	defer srb.stopTimer()

	initialVal := writeDM.getValue("test-bucket")
	buf := make([]byte, blockSize)

	// Read block 1 (1 KiB <= 2 KiB queue threshold)
	if _, err := srb.Read(buf); err != nil {
		t.Fatalf("block 1 Read: %v", err)
	}
	// Read block 2 (credits block 1 -> totalBytes = 1 KiB <= 2 KiB queue threshold: must NOT update writeDM)
	if _, err := srb.Read(buf); err != nil {
		t.Fatalf("block 2 Read: %v", err)
	}
	if got := writeDM.getValue("test-bucket"); got != initialVal {
		t.Fatalf("after block 1 (within queue threshold), writeDM changed: got %v, want %v", got, initialVal)
	}

	// Read block 3 (credits block 2 -> totalBytes = 2 KiB <= 2 KiB queue threshold: must NOT update writeDM)
	if _, err := srb.Read(buf); err != nil {
		t.Fatalf("block 3 Read: %v", err)
	}
	if got := writeDM.getValue("test-bucket"); got != initialVal {
		t.Fatalf("after block 2 (at queue threshold), writeDM changed: got %v, want %v", got, initialVal)
	}

	// Final Read hits EOF (credits block 3 -> totalBytes = 3 KiB > 2 KiB queue threshold: MUST update writeDM)
	n, err := srb.Read(buf)
	if err != io.EOF || n != 0 {
		t.Fatalf("expected (0, io.EOF), got (%d, %v)", n, err)
	}
	if got := writeDM.getValue("test-bucket"); got >= initialVal {
		t.Fatalf("after block 3 (past queue threshold), expected writeDM to decrease: got %v, want < %v", got, initialVal)
	}
}

func TestStallRequestBody_ResponseDelayStall(t *testing.T) {
	writeDM, err := newBucketDelayManager(0.99, 1.0, 200*time.Millisecond, 200*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}
	respDM, err := newBucketDelayManager(0.99, 1.0, 30*time.Millisecond, 30*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("newBucketDelayManager: %v", err)
	}

	initialWriteDelay := writeDM.getValue("test-bucket")
	initialRespDelay := respDM.getValue("test-bucket")

	data := []byte("small payload")
	srb := newStallRequestBody(
		io.NopCloser(bytes.NewReader(data)),
		"test-bucket",
		writeDM,
		respDM,
		nil,
		"localhost",
		context.Background(),
		defaultStallBlockSize,
		"",
		false,
		time.Time{},
	)
	defer srb.Close()
	defer srb.stopTimer()

	_, err = io.ReadAll(srb)
	if err != nil {
		t.Fatalf("io.ReadAll: %v", err)
	}

	// Now srb has returned io.EOF and armed Timer 2 (Response Delay = 30ms).
	select {
	case <-srb.stalledCh:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for Response Delay stall")
	}

	if got := writeDM.getValue("test-bucket"); got != initialWriteDelay {
		t.Errorf("expected writeDelayManager to remain %v, got %v", initialWriteDelay, got)
	}
	if got := respDM.getValue("test-bucket"); got <= initialRespDelay {
		t.Errorf("expected responseDelayManager to increase on stall: got %v <= %v", got, initialRespDelay)
	}
}

func TestStallRequestBody_TrueTimePart2Splicing(t *testing.T) {
	writeDM, _ := newBucketDelayManager(0.99, 1.0, 100*time.Millisecond, 100*time.Millisecond, 2*time.Second)
	respDM, _ := newBucketDelayManager(0.99, 1.0, 150*time.Millisecond, 150*time.Millisecond, 2*time.Second)

	// Build a standard 2-part multipart/related body matching gensupport.CombineBodyMedia.
	var rawMultipart bytes.Buffer
	mpw := multipart.NewWriter(&rawMultipart)
	boundary := mpw.Boundary()

	p0, err := mpw.CreatePart(map[string][]string{"Content-Type": {"application/json"}})
	if err != nil {
		t.Fatalf("CreatePart 0: %v", err)
	}
	_, _ = p0.Write([]byte(`{"name":"test-obj"}`))

	p1, err := mpw.CreatePart(map[string][]string{"Content-Type": {"application/octet-stream"}})
	if err != nil {
		t.Fatalf("CreatePart 1: %v", err)
	}
	payload := bytes.Repeat([]byte("payload-bytes-"), 200)
	_, _ = p1.Write(payload)
	_ = mpw.Close()

	baseRecvTime := time.Now().Add(-50 * time.Millisecond)
	srb := newStallRequestBody(
		io.NopCloser(bytes.NewReader(rawMultipart.Bytes())),
		"test-bucket",
		writeDM,
		respDM,
		nil,
		"localhost",
		context.Background(),
		int64(len(payload)),
		boundary,
		true,
		baseRecvTime,
	)
	defer srb.Close()
	defer srb.stopTimer()

	// Read with a small buffer (17 bytes) to stress-test stageBuf / pending boundary holding.
	var out bytes.Buffer
	smallBuf := make([]byte, 17)
	for {
		n, err := srb.Read(smallBuf)
		if n > 0 {
			out.Write(smallBuf[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
	}
	srb.stopTimer()

	// Parse the resulting stream with standard library mime/multipart.NewReader and verify 3 parts!
	mr := multipart.NewReader(&out, boundary)

	part0, err := mr.NextPart()
	if err != nil {
		t.Fatalf("NextPart 0: %v", err)
	}
	b0, _ := io.ReadAll(part0)
	if string(b0) != `{"name":"test-obj"}` {
		t.Errorf("part 0 body = %q, want %q", string(b0), `{"name":"test-obj"}`)
	}

	part1, err := mr.NextPart()
	if err != nil {
		t.Fatalf("NextPart 1: %v", err)
	}
	b1, _ := io.ReadAll(part1)
	if !bytes.Equal(b1, payload) {
		t.Errorf("part 1 body len = %d, want %d", len(b1), len(payload))
	}

	part2, err := mr.NextPart()
	if err != nil {
		t.Fatalf("NextPart 2 (TrueTime part): %v", err)
	}
	offsetStr := part2.Header.Get(deadlineOffsetHeaderName)
	if offsetStr == "" {
		t.Fatalf("expected %s header in Part 2, got headers: %v", deadlineOffsetHeaderName, part2.Header)
	}
	offsetMs, err := strconv.ParseInt(offsetStr, 10, 64)
	if err != nil || offsetMs < 250 {
		t.Errorf("expected offsetMs >= 250ms (50ms elapsed + 100ms write + 150ms resp), got %q (err=%v)", offsetStr, err)
	}

	if _, err := mr.NextPart(); err != io.EOF {
		t.Errorf("expected io.EOF after 3rd part, got %v", err)
	}
}

type pipeListener struct {
	ch     chan net.Conn
	closed chan struct{}
	once   sync.Once
}

func newPipeListener() *pipeListener {
	return &pipeListener{
		ch:     make(chan net.Conn),
		closed: make(chan struct{}),
	}
}

func (l *pipeListener) Accept() (net.Conn, error) {
	select {
	case c := <-l.ch:
		return c, nil
	case <-l.closed:
		return nil, net.ErrClosed
	}
}

func (l *pipeListener) Close() error {
	l.once.Do(func() { close(l.closed) })
	return nil
}

func (l *pipeListener) Addr() net.Addr {
	return &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 8080}
}

func (l *pipeListener) DialContext(ctx context.Context, _, _ string) (net.Conn, error) {
	c1, c2 := net.Pipe()
	select {
	case l.ch <- c2:
		return c1, nil
	case <-l.closed:
		_ = c1.Close()
		_ = c2.Close()
		return nil, net.ErrClosed
	case <-ctx.Done():
		_ = c1.Close()
		_ = c2.Close()
		return nil, ctx.Err()
	}
}

func TestSimpleWriteStall_EndToEndWriteDelayAndTrueTimeRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Use an unbuffered in-memory net.Pipe() listener between net/http.Transport and net/http.Server
	// so there is zero kernel tcp_wmem buffering: the instant the server stops reading req.Body
	// after 4 KiB on Attempt 1, net/http.Transport blocks mid-stream inside conn.Write() and
	// triggers Timer 1 (Write Delay, responsePhase=false) before reaching io.EOF.
	payload := bytes.Repeat([]byte("ABCD"), 64*1024) // 256 KiB payload (< 16 MiB -> Simple Write)
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	warmupPayload := []byte("warmup")
	warmupCRCVal := crc32.Checksum(warmupPayload, crc32cTable)
	warmupCRCB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(warmupCRCVal >> 24), byte(warmupCRCVal >> 16), byte(warmupCRCVal >> 8), byte(warmupCRCVal),
	})

	const serverBaseTSHex = "1964523abc0"
	var uploadAttempts atomic.Int32
	var recMu sync.Mutex
	var attempt1Token, attempt2Token string
	var attempt2BaseTSHeader string
	var attempt2DeadlineOffset string
	var attempt2Payload []byte
	serverDone := make(chan struct{})

	pipeLis := newPipeListener()
	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set(baseTimestampHeaderName, serverBaseTSHex)

			// Warm-up upload to populate X-Goog-Gcs-Base-Ts naturally via recordResponseBaseTS.
			if req.URL.Query().Get("name") == "warmup-obj" {
				_, _ = io.Copy(io.Discard, req.Body)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = fmt.Fprintf(w, `{"name":"warmup-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, warmupCRCB64, len(warmupPayload))
				return
			}

			attempt := uploadAttempts.Add(1)
			if attempt == 1 {
				recMu.Lock()
				attempt1Token = req.Header.Get("X-Goog-Gcs-Idempotency-Token")
				recMu.Unlock()
				// Read the first 4 KiB of the upload and then stall in the middle of the body!
				buf := make([]byte, 4096)
				_, _ = io.ReadFull(req.Body, buf)
				select {
				case <-req.Context().Done():
				case <-serverDone:
				}
				return
			}

			tok2 := req.Header.Get("X-Goog-Gcs-Idempotency-Token")
			baseTS2 := req.Header.Get(baseTimestampHeaderName)

			_, params, err := mime.ParseMediaType(req.Header.Get("Content-Type"))
			if err != nil {
				t.Errorf("ParseMediaType: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			mr := multipart.NewReader(req.Body, params["boundary"])
			// Part 0: JSON metadata
			if p0, err := mr.NextPart(); err == nil {
				_, _ = io.ReadAll(p0)
			}
			// Part 1: Object media bytes
			var payload2 []byte
			if p1, err := mr.NextPart(); err == nil {
				payload2, _ = io.ReadAll(p1)
			}
			// Part 2: TrueTime deadline offset
			var offset2 string
			if p2, err := mr.NextPart(); err == nil {
				offset2 = p2.Header.Get(deadlineOffsetHeaderName)
				_, _ = io.ReadAll(p2)
			}

			recMu.Lock()
			attempt2Token = tok2
			attempt2BaseTSHeader = baseTS2
			attempt2Payload = payload2
			attempt2DeadlineOffset = offset2
			recMu.Unlock()

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = fmt.Fprintf(w, `{"name":"test-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
		}),
	}
	go func() {
		_ = srv.Serve(pipeLis)
	}()
	defer func() {
		close(serverDone)
		_ = pipeLis.Close()
		_ = srv.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", "http://127.0.0.1:8080")
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:                40 * time.Millisecond,
		MinResponseDelay:   250 * time.Millisecond,
		TargetPercentile:   0.99,
		EnableTrueTimePart: true,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hc := client.tc.(*httpStorageClient)
	hc.writeStallTransport.blockSize = 16 * 1024
	hc.writeStallTransport.base = &http.Transport{
		DialContext: pipeLis.DialContext,
	}

	// Step 1: Perform a warm-up write so the client sniffs X-Goog-Gcs-Base-Ts from a real HTTP response.
	ww := client.Bucket("test-bucket").Object("warmup-obj").NewWriter(ctx)
	if _, err := ww.Write(warmupPayload); err != nil {
		t.Fatalf("warmup Write: %v", err)
	}
	if err := ww.Close(); err != nil {
		t.Fatalf("warmup Close: %v", err)
	}
	if sample := hc.writeStallTransport.latestBaseTS.Load(); sample == nil || sample.hexValue != serverBaseTSHex {
		t.Fatalf("expected latestBaseTS to be populated with %q from warm-up response, got %+v", serverBaseTSHex, sample)
	}

	initialWriteDelay := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket")
	initialRespDelay := hc.writeStallTransport.writeResponseDelayMgr.getValue("test-bucket")

	// Step 2: Perform the 256 KiB Simple Write that stalls mid-body on Attempt 1 and succeeds on Attempt 2.
	w := client.Bucket("test-bucket").Object("test-obj").NewWriter(ctx)
	if _, err := w.Write(payload); err != nil {
		t.Fatalf("w.Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close: %v", err)
	}

	if got := uploadAttempts.Load(); got != 2 {
		t.Fatalf("expected 2 upload attempts, got %d", got)
	}
	// Verify that Timer 1 (Write Delay, responsePhase=false) is what fired on Attempt 1!
	if newWriteDelay := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket"); newWriteDelay <= initialWriteDelay {
		t.Errorf("expected Write Delay manager (Timer 1, responsePhase=false) to increase on mid-body stall: got %v, initial %v", newWriteDelay, initialWriteDelay)
	}
	if newRespDelay := hc.writeStallTransport.writeResponseDelayMgr.getValue("test-bucket"); newRespDelay > initialRespDelay {
		t.Errorf("expected Response Delay manager (Timer 2) NOT to increase on mid-body stall: got %v > initial %v", newRespDelay, initialRespDelay)
	}
	recMu.Lock()
	defer recMu.Unlock()
	if attempt1Token == "" || attempt1Token != attempt2Token {
		t.Errorf("expected non-empty matching idempotency tokens across retries: attempt1=%q, attempt2=%q", attempt1Token, attempt2Token)
	}
	if attempt2BaseTSHeader != serverBaseTSHex {
		t.Errorf("attempt 2 %s header = %q, want %q", baseTimestampHeaderName, attempt2BaseTSHeader, serverBaseTSHex)
	}
	if attempt2DeadlineOffset == "" {
		t.Errorf("expected non-empty %s header in Part 2 of attempt 2", deadlineOffsetHeaderName)
	}
	if !bytes.Equal(attempt2Payload, payload) {
		t.Errorf("attempt 2 payload mismatch: got %d bytes, want %d bytes", len(attempt2Payload), len(payload))
	}
}

func TestSimpleWriteStall_EndToEndResponseDelayRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	payload := []byte("hello response delay stall detection")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var attempts atomic.Int32
	serverDone := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempt := attempts.Add(1)
		// Drain the entire request body so the client reaches io.EOF and starts Timer 2 (Response Delay).
		_, _ = io.Copy(io.Discard, req.Body)

		if attempt == 1 {
			// Stall before sending HTTP 200 OK headers!
			select {
			case <-req.Context().Done():
			case <-serverDone:
			}
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"test-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	defer func() {
		close(serverDone)
		ts.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              500 * time.Millisecond,
		MinResponseDelay: 35 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	w := client.Bucket("test-bucket").Object("test-obj").NewWriter(ctx)
	if _, err := w.Write(payload); err != nil {
		t.Fatalf("w.Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close: %v", err)
	}

	if got := attempts.Load(); got != 2 {
		t.Fatalf("expected 2 attempts after Response Delay stall, got %d", got)
	}
}

func TestSimpleWriteStall_Unconditioned503NotRetried(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempts.Add(1)
		_, _ = io.Copy(io.Discard, req.Body)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":503,"message":"backend unavailable"}}`))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              200 * time.Millisecond,
		MinResponseDelay: 200 * time.Millisecond,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	// Unconditioned write (no IfGenerationMatch): 503 must NOT be retried (Conformance Test ID 3).
	w := client.Bucket("test-bucket").Object("test-obj").NewWriter(ctx)
	_, _ = w.Write([]byte("unconditioned write"))
	err = w.Close()
	if err == nil {
		t.Fatal("expected 503 error on unconditioned write, got nil")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("expected exactly 1 attempt on unconditioned 503 (no retry), got %d", got)
	}
}

func TestSimpleWriteStall_ResumableUploadRetriesUntouched(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Use 256 KiB ChunkSize with a 400 KiB payload so PrepareUpload does NOT hit EOF
	// during the first chunk (singleChunk == false -> Resumable Upload).
	const chunkSize = 256 * 1024
	payload := bytes.Repeat([]byte("R"), 400*1024)
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var chunkPutAttempts atomic.Int32
	var tsURL string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		_, _ = io.Copy(io.Discard, req.Body)
		uploadType := req.URL.Query().Get("uploadType")
		if uploadType == "resumable" && req.URL.Query().Get("upload_id") == "" {
			// Session initiation request
			w.Header().Set("Location", tsURL+"/upload/storage/v1/b/test-bucket/o?uploadType=resumable&upload_id=session-123")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Chunk upload request
		attempt := chunkPutAttempts.Add(1)
		if attempt == 1 {
			// Inject transient 503 on first chunk upload; ResumableUpload must still retry 503!
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		if strings.HasSuffix(req.Header.Get("Content-Range"), "/409600") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = fmt.Fprintf(w, `{"name":"resumable-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
			return
		}
		// Intermediate chunk ack
		w.Header().Set("Range", "bytes=0-262143")
		w.WriteHeader(http.StatusPermanentRedirect) // 308 Resume Incomplete via X-GUploader-No-308=yes uses 200 OK
	}))
	defer ts.Close()
	tsURL = ts.URL

	// Fix intermediate chunk response for X-GUploader-No-308: yes (which expects 200 OK + X-Http-Status-Code-Override: 308)
	ts.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		_, _ = io.Copy(io.Discard, req.Body)
		uploadType := req.URL.Query().Get("uploadType")
		if uploadType == "resumable" && req.URL.Query().Get("upload_id") == "" {
			w.Header().Set("Location", tsURL+"/upload/storage/v1/b/test-bucket/o?uploadType=resumable&upload_id=session-123")
			w.WriteHeader(http.StatusOK)
			return
		}
		attempt := chunkPutAttempts.Add(1)
		if attempt == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		if !strings.Contains(req.Header.Get("Content-Range"), "/*") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = fmt.Fprintf(w, `{"name":"resumable-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
			return
		}
		w.Header().Set("Range", "bytes=0-262143")
		w.Header().Set("X-Http-Status-Code-Override", "308")
		w.WriteHeader(http.StatusOK)
	})

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              200 * time.Millisecond,
		MinResponseDelay: 200 * time.Millisecond,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	// Unconditioned Resumable Upload: chunk 503 retries must still work!
	w := client.Bucket("test-bucket").Object("resumable-obj").NewWriter(ctx)
	w.ChunkSize = chunkSize
	if _, err := w.Write(payload); err != nil {
		t.Fatalf("w.Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close failed on resumable upload (rx.Retry was broken): %v", err)
	}
	if got := chunkPutAttempts.Load(); got != 3 {
		t.Fatalf("expected 3 chunk PUT requests (chunk 1 fail 503 + chunk 1 retry + chunk 2 final), got %d", got)
	}
}

func TestSimpleWriteStall_SlowApplicationWriteDoesNotStall(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	chunk1 := []byte("first application chunk ")
	chunk2 := []byte("second application chunk after pause")
	fullPayload := append(append([]byte(nil), chunk1...), chunk2...)
	crcVal := crc32.Checksum(fullPayload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var attempts atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempts.Add(1)
		_, _ = io.Copy(io.Discard, req.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"slow-app-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(fullPayload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              30 * time.Millisecond,
		MinResponseDelay: 200 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	w := client.Bucket("test-bucket").Object("slow-app-obj").NewWriter(ctx)
	if _, err := w.Write(chunk1); err != nil {
		t.Fatalf("w.Write(chunk1): %v", err)
	}
	// Pause the application between w.Write() calls for >3x the 30ms Write Delay timeout.
	// Because Simple Writes buffer in MediaBuffer until w.Close(), this must NOT trigger a stall!
	time.Sleep(100 * time.Millisecond)
	if _, err := w.Write(chunk2); err != nil {
		t.Fatalf("w.Write(chunk2): %v", err)
	}
	time.Sleep(100 * time.Millisecond)
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close: %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("expected exactly 1 attempt (no false stall during application w.Write pauses), got %d", got)
	}
}

func TestSimpleWriteStall_TrueTimeAdvertisedDeadlineWaitBeforeRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify that when Part 2 (x-goog-gcs-deadline-offset-ms) has been spliced into the stream
	// with an advertised TrueTime deadline (WriteDelay 35ms + ResponseDelay 120ms = 155ms) and
	// Timer 1 (35ms) fires while writing the final block containing Part 2, writeStallTransport
	// waits out the remaining ~120ms advertised deadline before issuing Attempt 2!
	payload := []byte("truetime-deadline-wait-payload")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	const serverBaseTSHex = "1964523abc0"
	var attempts atomic.Int32
	var recMu sync.Mutex
	var attempt1Time, attempt2Time time.Time
	serverDone := make(chan struct{})

	pipeLis := newPipeListener()
	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set(baseTimestampHeaderName, serverBaseTSHex)
			attempt := attempts.Add(1)
			if attempt == 1 {
				recMu.Lock()
				attempt1Time = time.Now()
				recMu.Unlock()
				// Read Part 0 (JSON) and Part 1 (media) so the client reaches Part 1 io.EOF,
				// runs spliceTrueTimePart() (setting advertisedDeadline = now + 35ms + 120ms = +155ms),
				// and then stall before reading Part 2 so net/http blocks writing Part 2 and trips
				// Timer 1 after 35ms with ~120ms of advertisedDeadline still remaining!
				if _, params, err := mime.ParseMediaType(req.Header.Get("Content-Type")); err == nil {
					mr := multipart.NewReader(req.Body, params["boundary"])
					if p0, err := mr.NextPart(); err == nil {
						_, _ = io.ReadAll(p0)
					}
					if p1, err := mr.NextPart(); err == nil {
						_, _ = io.ReadAll(p1)
					}
				}
				select {
				case <-req.Context().Done():
				case <-serverDone:
				}
				return
			}
			recMu.Lock()
			attempt2Time = time.Now()
			recMu.Unlock()
			_, _ = io.Copy(io.Discard, req.Body)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = fmt.Fprintf(w, `{"name":"wait-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
		}),
	}
	go func() {
		_ = srv.Serve(pipeLis)
	}()
	defer func() {
		close(serverDone)
		_ = pipeLis.Close()
		_ = srv.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", "http://127.0.0.1:8080")
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:                35 * time.Millisecond,
		MinResponseDelay:   120 * time.Millisecond,
		TargetPercentile:   0.99,
		EnableTrueTimePart: true,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hc := client.tc.(*httpStorageClient)
	hc.writeStallTransport.blockSize = int64(len(payload))
	hc.writeStallTransport.base = &http.Transport{
		DialContext: pipeLis.DialContext,
	}
	hc.writeStallTransport.latestBaseTS.Store(&baseTimestampSample{
		hexValue: serverBaseTSHex,
		recvTime: time.Now(),
	})

	w := client.Bucket("test-bucket").Object("wait-obj").NewWriter(ctx)
	if _, err := w.Write(payload); err != nil {
		t.Fatalf("w.Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close: %v", err)
	}

	if got := attempts.Load(); got != 2 {
		t.Fatalf("expected 2 attempts, got %d", got)
	}
	recMu.Lock()
	gap := attempt2Time.Sub(attempt1Time)
	recMu.Unlock()
	// Timer 1 fired at ~35ms, but advertisedDeadline was +155ms (35ms + 120ms), so Attempt 2
	// must have waited at least 140ms after Attempt 1 started!
	if gap < 140*time.Millisecond {
		t.Errorf("expected retry to wait until advertised TrueTime deadline (>= 140ms), but Attempt 2 arrived after %v", gap)
	}
}

func TestSimpleWriteStall_PostEOFDrainBudgetPreventsFalseStallWithoutTrueTime(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify Fix #1: When an application writes 2 blocks that fit entirely inside kernel tcp_wmem,
	// stallRequestBody.Read() reaches io.EOF immediately while all 2 blocks are still draining
	// over the wire. Even WITHOUT TrueTime active (hasBaseTS == false), Timer 2 (Response Delay)
	// must include the proportional drain budget (2 * 40ms = 80ms) + MinResponseDelay (40ms) = 120ms,
	// so a 75ms wire drain + server commit succeeds on Attempt 1 without a false stall!
	// Furthermore, RoundTrip must subtract the 80ms drainBudget before updating writeResponseDelayMgr
	// so writeResponseDelayMgr does not inflate on normal wire drain time.
	const blockSize = 16 * 1024
	payload := bytes.Repeat([]byte("D"), 2*blockSize) // 32 KiB = 2 blocks
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var attempts atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempts.Add(1)
		_, _ = io.Copy(io.Discard, req.Body)
		// Simulate 75ms of wire drain (2 blocks * ~25ms) + server commit (25ms).
		// Note: 75ms > MinResponseDelay (40ms), which previously caused a 100% false stall
		// when hasBaseTS == false, but is well within drainBudget (~80ms) + MinResponseDelay (40ms) = 120ms!
		time.Sleep(75 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"drain-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              40 * time.Millisecond,
		MinResponseDelay: 40 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hc := client.tc.(*httpStorageClient)
	hc.writeStallTransport.blockSize = blockSize

	w := client.Bucket("test-bucket").Object("drain-obj").NewWriter(ctx)
	if _, err := w.Write(payload); err != nil {
		t.Fatalf("w.Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close failed due to false Response Delay stall: %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("expected 1 attempt (drainBudget + MinResponseDelay prevents false stall), got %d", got)
	}
	// Verify that writeResponseDelayMgr did not inflate above MinResponseDelay (40ms) because
	// 75ms total post-EOF wait - 80ms drainBudget <= 0 (clamped to 1ns sample).
	if respTimeout := hc.writeStallTransport.writeResponseDelayMgr.getValue("test-bucket"); respTimeout > 45*time.Millisecond {
		t.Errorf("expected writeResponseDelayMgr to remain near MinResponseDelay (40ms) after subtracting drainBudget, got %v", respTimeout)
	}
}

func TestSimpleWriteStall_Default2PartWhenServerSendsBaseTSWithoutOptIn(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify Fix #2: If the GCS server begins emitting X-Goog-Gcs-Base-Ts before 3-part
	// multipart/related parsing is enabled on Scotty/JIB (and rejects 3-part requests with 400),
	// a client with default EnableTrueTimePart: false must continue sending standard 2-part
	// multipart/related requests and succeed!
	payload := []byte("standard-2-part-payload")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Server always emits X-Goog-Gcs-Base-Ts
		w.Header().Set(baseTimestampHeaderName, "1964523abc0")
		_, params, err := mime.ParseMediaType(req.Header.Get("Content-Type"))
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		mr := multipart.NewReader(req.Body, params["boundary"])
		partCount := 0
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			_, _ = io.ReadAll(part)
			partCount++
		}
		// Strictly reject 3-part requests (emulating production Scotty/JIB today!)
		if partCount != 2 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprintf(w, `{"error":{"code":400,"message":"expected 2 multipart parts, got %d"}}`, partCount)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	// EnableTrueTimePart left at default (false)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              50 * time.Millisecond,
		MinResponseDelay: 50 * time.Millisecond,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	// Write 1: Server returns X-Goog-Gcs-Base-Ts
	w1 := client.Bucket("test-bucket").Object("obj-1").NewWriter(ctx)
	if _, err := w1.Write(payload); err != nil {
		t.Fatalf("w1.Write: %v", err)
	}
	if err := w1.Close(); err != nil {
		t.Fatalf("w1.Close: %v", err)
	}

	// Write 2: Client now has latestBaseTS cached, but because EnableTrueTimePart is false,
	// it must still send 2 parts and NOT get a 400 Bad Request!
	w2 := client.Bucket("test-bucket").Object("obj-2").NewWriter(ctx)
	if _, err := w2.Write(payload); err != nil {
		t.Fatalf("w2.Write: %v", err)
	}
	if err := w2.Close(); err != nil {
		t.Fatalf("w2.Close failed (sent 3-part body without EnableTrueTimePart opt-in): %v", err)
	}
}

func TestSimpleWriteStall_PreconditionSafetyGate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify Fix #3:
	// Case A: Conditions{DoesNotExist: true} WITHOUT TrueTime active -> must NOT abort on slow
	// response (preventing false 412 Precondition Failed if Attempt 1 committed on the server).
	// Case B: Conditions{GenerationMatch: 42} WITHOUT TrueTime active -> safe to retry when
	// RetryAlways is configured, so it DOES abort and retry on Attempt 2.
	// Case C: Conditions{DoesNotExist: true} WITH TrueTime active -> TrueTime prevents
	// split-brain commits after deadline, so it DOES abort and retry when RetryAlways is configured.
	payload := []byte("precondition-test-payload")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var caseAAttempts, caseBAttempts, caseCAttempts atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set(baseTimestampHeaderName, "1964523abc0")
		bodyBytes, _ := io.ReadAll(req.Body)
		switch {
		case bytes.Contains(bodyBytes, []byte(`"case-a"`)):
			caseAAttempts.Add(1)
			// Sleep 90ms (> 30ms MinResponseDelay). Since DoesNotExist has no TrueTime,
			// client must NOT abort Attempt 1!
			time.Sleep(90 * time.Millisecond)
		case bytes.Contains(bodyBytes, []byte(`"case-b"`)):
			if caseBAttempts.Add(1) == 1 {
				time.Sleep(150 * time.Millisecond)
				return
			}
		case bytes.Contains(bodyBytes, []byte(`"case-c"`)):
			if caseCAttempts.Add(1) == 1 {
				time.Sleep(200 * time.Millisecond)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"cond-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              30 * time.Millisecond,
		MinResponseDelay: 30 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	// Case A: DoesNotExist without TrueTime -> must complete on Attempt 1 despite 90ms server commit!
	wA := client.Bucket("test-bucket").Object("case-a").If(Conditions{DoesNotExist: true}).NewWriter(ctx)
	_, _ = wA.Write(payload)
	if err := wA.Close(); err != nil {
		t.Fatalf("Case A (DoesNotExist without TrueTime) failed: %v", err)
	}
	if got := caseAAttempts.Load(); got != 1 {
		t.Fatalf("Case A: expected 1 attempt (not aborted), got %d", got)
	}

	// Case B: GenerationMatch: 42 + RetryAlways without TrueTime -> safe precondition, aborts & retries!
	wB := client.Bucket("test-bucket").Object("case-b").If(Conditions{GenerationMatch: 42}).Retryer(WithPolicy(RetryAlways)).NewWriter(ctx)
	_, _ = wB.Write(payload)
	if err := wB.Close(); err != nil {
		t.Fatalf("Case B (GenerationMatch: 42) failed: %v", err)
	}
	if got := caseBAttempts.Load(); got != 2 {
		t.Fatalf("Case B: expected 2 attempts (aborted & retried), got %d", got)
	}

	// Case C: DoesNotExist + RetryAlways WITH TrueTime active -> TrueTime makes it safe, aborts & retries!
	clientTT, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:                30 * time.Millisecond,
		MinResponseDelay:   30 * time.Millisecond,
		TargetPercentile:   0.99,
		EnableTrueTimePart: true,
	}))
	if err != nil {
		t.Fatalf("NewClient TT: %v", err)
	}
	defer clientTT.Close()
	hcTT := clientTT.tc.(*httpStorageClient)
	hcTT.writeStallTransport.latestBaseTS.Store(&baseTimestampSample{
		hexValue: "1964523abc0",
		recvTime: time.Now(),
	})

	wC := clientTT.Bucket("test-bucket").Object("case-c").If(Conditions{DoesNotExist: true}).Retryer(WithPolicy(RetryAlways)).NewWriter(ctx)
	_, _ = wC.Write(payload)
	if err := wC.Close(); err != nil {
		t.Fatalf("Case C (DoesNotExist with TrueTime) failed: %v", err)
	}
	if got := caseCAttempts.Load(); got != 2 {
		t.Fatalf("Case C: expected 2 attempts (TrueTime enabled abort & retry), got %d", got)
	}
}

type rewindMockRoundTripper struct {
	inner     http.RoundTripper
	rewound   atomic.Bool
	stallBody atomic.Bool
}

func (r *rewindMockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// Simulate net/http Transport reading a few bytes on a stale pooled connection,
	// encountering a broken pipe, and calling req.GetBody() to rewind internally!
	if r.rewound.CompareAndSwap(false, true) && req.GetBody != nil {
		buf := make([]byte, 32)
		_, _ = req.Body.Read(buf)
		_ = req.Body.Close()

		newBody, err := req.GetBody()
		if err != nil {
			return nil, err
		}
		req.Body = newBody
	}
	if r.stallBody.CompareAndSwap(false, true) {
		// On the rewound body of Attempt 1, verify that stall detection is STILL active!
		buf := make([]byte, 32)
		_, _ = req.Body.Read(buf)
		<-req.Context().Done()
		return nil, req.Context().Err()
	}
	return r.inner.RoundTrip(req)
}

func TestSimpleWriteStall_InternalTransportGetBodyRewind(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify Fix #4: When net/http internally calls req.GetBody() inside base.RoundTrip
	// (e.g., when retrying a stale keep-alive connection), the rewound body must still be
	// wrapped by stallRequestBody and must reset the previous timer cleanly!
	payload := []byte("rewind-getbody-test-payload")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		_, _ = io.Copy(io.Discard, req.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"rewind-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              35 * time.Millisecond,
		MinResponseDelay: 100 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hc := client.tc.(*httpStorageClient)
	rewindRT := &rewindMockRoundTripper{inner: hc.writeStallTransport.base}
	hc.writeStallTransport.base = rewindRT

	w := client.Bucket("test-bucket").Object("rewind-obj").NewWriter(ctx)
	if _, err := w.Write(payload); err != nil {
		t.Fatalf("w.Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close failed after internal GetBody rewind: %v", err)
	}
	if !rewindRT.rewound.Load() {
		t.Fatalf("expected rewindMockRoundTripper to invoke req.GetBody()")
	}
}

func TestSimpleWriteStall_CallerContextCancellationDoesNotRetryOrInflateDelay(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify that when the caller's own context expires mid-upload, the client returns
	// context.DeadlineExceeded immediately after Attempt 1 (no stall retry) and does NOT
	// record a stall sample in writeBlockDelayMgr or writeResponseDelayMgr.
	payload := []byte("caller-cancel-payload")
	var attempts atomic.Int32
	serverDone := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempts.Add(1)
		_, _ = io.Copy(io.Discard, req.Body)
		select {
		case <-req.Context().Done():
		case <-serverDone:
		}
	}))
	defer func() {
		close(serverDone)
		ts.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              500 * time.Millisecond,
		MinResponseDelay: 500 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hc := client.tc.(*httpStorageClient)
	initialWriteDelay := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket")
	initialRespDelay := hc.writeStallTransport.writeResponseDelayMgr.getValue("test-bucket")

	uploadCtx, uploadCancel := context.WithTimeout(ctx, 40*time.Millisecond)
	defer uploadCancel()

	w := client.Bucket("test-bucket").Object("cancel-obj").NewWriter(uploadCtx)
	_, _ = w.Write(payload)
	err = w.Close()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("expected exactly 1 attempt on caller context timeout, got %d", got)
	}
	if got := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket"); got != initialWriteDelay {
		t.Errorf("writeBlockDelayMgr changed on caller context cancel: got %v, want %v", got, initialWriteDelay)
	}
	if got := hc.writeStallTransport.writeResponseDelayMgr.getValue("test-bucket"); got != initialRespDelay {
		t.Errorf("writeResponseDelayMgr changed on caller context cancel: got %v, want %v", got, initialRespDelay)
	}
}

func TestSimpleWriteStall_RetryNeverDisablesStallAbort(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify that when the user configures WithPolicy(RetryNever), Simple Write Stall Detection
	// does NOT attach stall timeouts or abort a slow upload (since aborting without retrying
	// would guaranteed-fail a slow upload that would otherwise succeed).
	payload := []byte("retry-never-payload")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var attempts atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempts.Add(1)
		_, _ = io.Copy(io.Discard, req.Body)
		time.Sleep(90 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"retry-never-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              25 * time.Millisecond,
		MinResponseDelay: 25 * time.Millisecond,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	w := client.Bucket("test-bucket").Object("retry-never-obj").Retryer(WithPolicy(RetryNever)).NewWriter(ctx)
	_, _ = w.Write(payload)
	if err := w.Close(); err != nil {
		t.Fatalf("expected RetryNever upload to succeed without stall abort, got: %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("expected 1 attempt, got %d", got)
	}
}

func TestSimpleWriteStall_ZeroChunkSizeUnbufferedNotAborted(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify that when w.ChunkSize = 0 (unbuffered one-shot multipart upload where req.GetBody == nil),
	// writeStallTransport does NOT wrap or abort the unreplayable stream, even when the application
	// pauses between w.Write() calls for longer than Min.
	chunk1 := []byte("unbuffered-chunk-1-")
	chunk2 := []byte("unbuffered-chunk-2")
	fullPayload := append(append([]byte(nil), chunk1...), chunk2...)
	crcVal := crc32.Checksum(fullPayload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var attempts atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempts.Add(1)
		_, _ = io.Copy(io.Discard, req.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"unbuffered-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(fullPayload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              25 * time.Millisecond,
		MinResponseDelay: 25 * time.Millisecond,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	w := client.Bucket("test-bucket").Object("unbuffered-obj").NewWriter(ctx)
	w.ChunkSize = 0 // Unbuffered streaming mode (req.GetBody == nil)
	if _, err := w.Write(chunk1); err != nil {
		t.Fatalf("w.Write(chunk1): %v", err)
	}
	// Pause for 80ms (> 25ms Min) while the HTTP request is actively streaming!
	time.Sleep(80 * time.Millisecond)
	if _, err := w.Write(chunk2); err != nil {
		t.Fatalf("w.Write(chunk2): %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close failed on ChunkSize=0 upload: %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("expected 1 attempt for ChunkSize=0 upload, got %d", got)
	}
}

func TestSimpleWriteStall_EmptyZeroByteObjectStallAndRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify that a 0-byte Simple Write (where w.Close() is called immediately without w.Write())
	// still detects a stall on Attempt 1 and retries cleanly on Attempt 2.
	crcVal := crc32.Checksum(nil, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var attempts atomic.Int32
	serverDone := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempt := attempts.Add(1)
		_, _ = io.Copy(io.Discard, req.Body)
		if attempt == 1 {
			select {
			case <-req.Context().Done():
			case <-serverDone:
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"empty-obj","bucket":"test-bucket","crc32c":%q,"size":"0"}`, crcB64)
	}))
	defer func() {
		close(serverDone)
		ts.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              35 * time.Millisecond,
		MinResponseDelay: 35 * time.Millisecond,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	w := client.Bucket("test-bucket").Object("empty-obj").NewWriter(ctx)
	if err := w.Close(); err != nil {
		t.Fatalf("0-byte w.Close failed: %v", err)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("expected 2 attempts for 0-byte object stall & retry, got %d", got)
	}
}

func TestSimpleWriteStall_ConcurrentMultiBucketIsolation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Verify concurrent Simple Writes across "slow-bucket" (stalls on first attempts)
	// and "fast-bucket" (never stalls) under -race:
	// 1. All 20 concurrent uploads succeed.
	// 2. "slow-bucket" delay manager adapts upward while "fast-bucket" remains at Min.
	payload := []byte("concurrent-multi-bucket-payload")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var slowStalls remainingStallCounter
	slowStalls.remaining.Store(5)
	serverDone := make(chan struct{})

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		_, _ = io.Copy(io.Discard, req.Body)
		isSlowBucket := strings.Contains(req.URL.Path, "slow-bucket")
		if isSlowBucket && slowStalls.take() {
			select {
			case <-req.Context().Done():
			case <-serverDone:
			}
			return
		}
		bucketName := "fast-bucket"
		if isSlowBucket {
			bucketName = "slow-bucket"
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"obj","bucket":%q,"crc32c":%q,"size":"%d"}`, bucketName, crcB64, len(payload))
	}))
	defer func() {
		close(serverDone)
		ts.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              35 * time.Millisecond,
		MinResponseDelay: 35 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	var wg sync.WaitGroup
	errCh := make(chan error, 20)
	for i := 0; i < 10; i++ {
		wg.Add(2)
		go func(idx int) {
			defer wg.Done()
			w := client.Bucket("slow-bucket").Object(fmt.Sprintf("slow-%d", idx)).NewWriter(ctx)
			_, _ = w.Write(payload)
			if err := w.Close(); err != nil {
				errCh <- fmt.Errorf("slow-%d: %w", idx, err)
			}
		}(i)
		go func(idx int) {
			defer wg.Done()
			w := client.Bucket("fast-bucket").Object(fmt.Sprintf("fast-%d", idx)).NewWriter(ctx)
			_, _ = w.Write(payload)
			if err := w.Close(); err != nil {
				errCh <- fmt.Errorf("fast-%d: %w", idx, err)
			}
		}(i)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Errorf("concurrent upload error: %v", err)
	}

	hc := client.tc.(*httpStorageClient)
	slowRespDelay := hc.writeStallTransport.writeResponseDelayMgr.getValue("slow-bucket")
	fastRespDelay := hc.writeStallTransport.writeResponseDelayMgr.getValue("fast-bucket")
	if slowRespDelay <= fastRespDelay {
		t.Errorf("expected slow-bucket response delay (%v) > fast-bucket response delay (%v)", slowRespDelay, fastRespDelay)
	}
}

type remainingStallCounter struct {
	remaining atomic.Int32
}

func (c *remainingStallCounter) take() bool {
	for {
		cur := c.remaining.Load()
		if cur <= 0 {
			return false
		}
		if c.remaining.CompareAndSwap(cur, cur-1) {
			return true
		}
	}
}

func TestSimpleWriteStall_HTTP2FlowControlExhaustionAndConnEviction(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify HTTP/2 over TLS:
	// In Go's x/net/http2 server, the connection-level flow-control window (cc.flow) is
	// 1 MiB. When a 1.5 MiB Simple Write stalls on Attempt 1 because the server stops
	// reading DATA frames, Attempt 1 consumes the entire 1 MiB connection flow-control
	// window and blocks mid-body in cs.awaitFlowControl (responsePhase=false).
	// In Go's x/net/http2 client, canceling a stream's context (RST_STREAM) does NOT
	// restore cc.flow and does NOT remove the http2ClientConn from ClientConnPool.
	// Verify that abortStalledConn force-closes the stalled HTTP/2 connection so Attempt 2
	// dials a fresh TLS/HTTP2 connection (different client port) and succeeds!
	payload := bytes.Repeat([]byte("H"), 1536*1024) // 1.5 MiB > 1 MiB HTTP/2 connection window
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var attempts atomic.Int32
	var addrMu sync.Mutex
	var attempt1RemoteAddr, attempt2RemoteAddr string
	serverDone := make(chan struct{})

	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		attempt := attempts.Add(1)
		if attempt == 1 {
			addrMu.Lock()
			attempt1RemoteAddr = req.RemoteAddr
			addrMu.Unlock()
			// Do NOT read req.Body on Attempt 1! The client will send 1 MiB of
			// HTTP/2 DATA frames, exhaust the HTTP/2 connection flow-control window,
			// and block mid-body in cs.awaitFlowControl until Timer 1 fires.
			select {
			case <-req.Context().Done():
			case <-serverDone:
			}
			return
		}
		addrMu.Lock()
		attempt2RemoteAddr = req.RemoteAddr
		addrMu.Unlock()
		_, _ = io.Copy(io.Discard, req.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"h2-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	ts.EnableHTTP2 = true
	ts.StartTLS()
	defer func() {
		close(serverDone)
		ts.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              45 * time.Millisecond,
		MinResponseDelay: 500 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	// Use the TLS-And-HTTP/2-enabled transport from httptest.Server as the base RoundTripper.
	hc := client.tc.(*httpStorageClient)
	hc.writeStallTransport.base = ts.Client().Transport
	initialWriteDelay := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket")

	w := client.Bucket("test-bucket").Object("h2-obj").NewWriter(ctx)
	if _, err := w.Write(payload); err != nil {
		t.Fatalf("w.Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("HTTP/2 w.Close failed: %v", err)
	}

	if got := attempts.Load(); got != 2 {
		t.Fatalf("expected 2 HTTP/2 attempts, got %d", got)
	}
	if got := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket"); got <= initialWriteDelay {
		t.Errorf("expected mid-body HTTP/2 flow-control stall (Timer 1) to increase writeBlockDelayMgr: got %v, initial %v", got, initialWriteDelay)
	}
	addrMu.Lock()
	a1, a2 := attempt1RemoteAddr, attempt2RemoteAddr
	addrMu.Unlock()
	if a1 == "" || a2 == "" || a1 == a2 {
		t.Errorf("expected Attempt 2 to dial a new HTTP/2 TLS connection after stall eviction, got attempt1=%q, attempt2=%q", a1, a2)
	}
}

func TestSimpleWriteStall_PreReadConnectionHandshakeStallAndSmallObjectRecovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify two critical properties:
	// 1. If TCP connection establishment / TLS handshake (DialContext) stalls on Attempt 1
	//    BEFORE net/http ever calls req.Body.Read(), startInitialTimer() catches the stall,
	//    aborts DialContext, increases writeBlockDelayMgr, and retries on Attempt 2.
	// 2. When subsequent small uploads (<= 4 MiB) succeed within budget,
	//    shouldDecreaseWriteDelayOnCompletion() decays writeBlockDelayMgr downward so small-object
	//    workloads recover after a transient stall.
	payload := []byte("pre-read-handshake-and-recovery-payload")
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var dialAttempts atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		_, _ = io.Copy(io.Discard, req.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":"dial-obj","bucket":"test-bucket","crc32c":%q,"size":"%d"}`, crcB64, len(payload))
	}))
	defer ts.Close()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              35 * time.Millisecond,
		MinResponseDelay: 100 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hc := client.tc.(*httpStorageClient)
	var baseDialer net.Dialer
	hc.writeStallTransport.base = &http.Transport{
		DialContext: func(dialCtx context.Context, network, addr string) (net.Conn, error) {
			if dialAttempts.Add(1) == 1 {
				// Simulate a blackholed TCP SYN on Attempt 1 (blocks before req.Body.Read is ever called!)
				<-dialCtx.Done()
				return nil, dialCtx.Err()
			}
			return baseDialer.DialContext(dialCtx, network, addr)
		},
	}

	initialWriteDelay := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket")

	w := client.Bucket("test-bucket").Object("dial-obj").NewWriter(ctx)
	_, _ = w.Write(payload)
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close failed after DialContext stall: %v", err)
	}
	if got := dialAttempts.Load(); got != 2 {
		t.Fatalf("expected 2 DialContext attempts (Attempt 1 stalled in TCP connect, Attempt 2 succeeded), got %d", got)
	}

	afterStallWriteDelay := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket")
	if afterStallWriteDelay <= initialWriteDelay {
		t.Fatalf("expected writeBlockDelayMgr to increase after DialContext stall: got %v, initial %v", afterStallWriteDelay, initialWriteDelay)
	}

	// Perform 5 fast small uploads (<= 4 MiB) and verify writeBlockDelayMgr decays downward!
	for i := 0; i < 5; i++ {
		wf := client.Bucket("test-bucket").Object("dial-obj").NewWriter(ctx)
		_, _ = wf.Write(payload)
		if err := wf.Close(); err != nil {
			t.Fatalf("recovery upload %d failed: %v", i, err)
		}
	}
	afterRecoveryWriteDelay := hc.writeStallTransport.writeBlockDelayMgr.getValue("test-bucket")
	if afterRecoveryWriteDelay >= afterStallWriteDelay {
		t.Errorf("expected writeBlockDelayMgr to decay after small-object completions: got %v >= %v", afterRecoveryWriteDelay, afterStallWriteDelay)
	}
}

func TestSimpleWriteStall_HTTP2MultiplexedConcurrentStreamsNoCollateralFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Verify that when multiple concurrent Simple Writes are multiplexed onto a single
	// HTTP/2 TLS connection and ONE stream stalls while sibling streams are actively
	// transferring on that same connection:
	// 1. The stalled stream cancels only its own HTTP/2 stream (RST_STREAM) without
	//    force-closing the shared *tls.Conn out from under the healthy sibling streams.
	// 2. All healthy sibling streams succeed on Attempt 1 (no "http2: client conn is closed"),
	//    and the stalled stream rewinds and succeeds on Attempt 2.
	payload := bytes.Repeat([]byte("M"), 256*1024) // 256 KiB
	crcVal := crc32.Checksum(payload, crc32cTable)
	crcB64 := base64.StdEncoding.EncodeToString([]byte{
		byte(crcVal >> 24), byte(crcVal >> 16), byte(crcVal >> 8), byte(crcVal),
	})

	var stalledOnce atomic.Bool
	var stream0Attempts atomic.Int32
	var siblingStarted sync.WaitGroup
	siblingStarted.Add(3)
	unblockSiblings := make(chan struct{})
	serverDone := make(chan struct{})

	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		objName := req.URL.Query().Get("name")
		if objName == "stream-0" {
			_, _ = io.Copy(io.Discard, req.Body)
			n := stream0Attempts.Add(1)
			if n == 1 && stalledOnce.CompareAndSwap(false, true) {
				// Wait until Stream 0's Response Delay timer fires while Streams 1..3
				// are still active on the same HTTP/2 connection.
				select {
				case <-req.Context().Done():
					close(unblockSiblings)
				case <-serverDone:
				}
				return
			}
		} else if strings.HasPrefix(objName, "stream-") {
			siblingStarted.Done()
			// Hold Streams 1..3 active on the shared HTTP/2 connection until Stream 0 stalls.
			select {
			case <-unblockSiblings:
			case <-req.Context().Done():
				return
			case <-serverDone:
				return
			}
			_, _ = io.Copy(io.Discard, req.Body)
		} else {
			_, _ = io.Copy(io.Discard, req.Body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{"name":%q,"bucket":"test-bucket","crc32c":%q,"size":"%d"}`, objName, crcB64, len(payload))
	}))
	ts.EnableHTTP2 = true
	ts.StartTLS()
	defer func() {
		close(serverDone)
		ts.Close()
	}()

	t.Setenv("STORAGE_EMULATOR_HOST", ts.URL)
	client, err := NewClient(ctx, WithWriteStallTimeout(&WriteStallTimeoutConfig{
		Min:              200 * time.Millisecond,
		MinResponseDelay: 40 * time.Millisecond,
		TargetPercentile: 0.99,
	}))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hc := client.tc.(*httpStorageClient)
	hc.writeStallTransport.base = ts.Client().Transport
	// Give sibling-bucket a generous response delay budget so Streams 1..3 remain
	// healthy and in-flight on the shared HTTP/2 connection while Stream 0 stalls at 40ms.
	for i := 0; i < 60; i++ {
		hc.writeStallTransport.writeResponseDelayMgr.increase("sibling-bucket")
	}

	// Warm up the single shared HTTP/2 TLS connection before launching the 4 concurrent streams.
	wWarm := client.Bucket("sibling-bucket").Object("warmup").NewWriter(ctx)
	_, _ = wWarm.Write(payload)
	if err := wWarm.Close(); err != nil {
		t.Fatalf("warmup Close: %v", err)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 4)
	for i := 1; i <= 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			w := client.Bucket("sibling-bucket").Object(fmt.Sprintf("stream-%d", id)).NewWriter(ctx)
			_, _ = w.Write(payload)
			if err := w.Close(); err != nil {
				errCh <- fmt.Errorf("stream-%d failed: %w", id, err)
			}
		}(i)
	}
	// Wait until all 3 sibling streams are active on the shared HTTP/2 connection,
	// then launch Stream 0 which will stall and trigger cancellation while Streams 1..3 are in flight.
	siblingStarted.Wait()
	wg.Add(1)
	go func() {
		defer wg.Done()
		w := client.Bucket("test-bucket").Object("stream-0").NewWriter(ctx)
		_, _ = w.Write(payload)
		if err := w.Close(); err != nil {
			errCh <- fmt.Errorf("stream-0 failed: %w", err)
		}
	}()

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	if got := stream0Attempts.Load(); got != 2 {
		t.Fatalf("expected stream-0 to take 2 attempts, got %d", got)
	}
}
