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
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net"
	"net/http"
	"net/http/httptrace"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// baseTimestampHeaderName is the HTTP header used by GCS to exchange the
	// TrueTime base timestamp (encoded as hex Unix milliseconds).
	baseTimestampHeaderName = "X-Goog-Gcs-Base-Ts"

	// deadlineOffsetHeaderName is the MIME part header in the 3rd multipart part
	// carrying the client's millisecond offset from X-Goog-Gcs-Base-Ts.
	deadlineOffsetHeaderName = "x-goog-gcs-deadline-offset-ms"

	// baseTimestampTTL bounds maximum local monotonic clock drift (at 100 ppm,
	// 5 minutes of idle time is < 30ms of drift).
	baseTimestampTTL = 5 * time.Minute

	// tcpSendBufferQueueThreshold is the initial byte threshold (4 MiB = 2 blocks)
	// absorbed locally by the OS kernel TCP send buffer (tcp_wmem) before
	// dynamicDelay samples reflect real network throughput.
	tcpSendBufferQueueThreshold = 2 * defaultStallBlockSize
)

// errWriteStall is returned by writeStallTransport when a simple (multipart) write
// exceeds either its per-2MiB Write Delay budget or its post-EOF Response Delay budget.
// It implements Temporary() so gensupport.shouldRetry and storage.ShouldRetry treat it
// as retryable, while unwrapping to context.DeadlineExceeded.
var errWriteStall error = writeStallError{}

type writeStallError struct{}

func (writeStallError) Error() string   { return "storage: simple write stalled" }
func (writeStallError) Temporary() bool { return true }
func (writeStallError) Unwrap() error   { return context.DeadlineExceeded }

// eofTrackingReader wraps an io.Reader (such as the io.PipeReader in OpenWriter)
// and records whether Read ever returned io.EOF. Because ObjectsInsertCall.Media
// synchronously reads the first chunk (up to ChunkSize) before returning, sawEOF
// is true after Media() returns if and only if the upload fits in a single chunk
// (a Simple Write using uploadType=multipart).
type eofTrackingReader struct {
	r      io.Reader
	sawEOF atomic.Bool
}

func (e *eofTrackingReader) Read(p []byte) (int, error) {
	n, err := e.r.Read(p)
	if err == io.EOF {
		e.sawEOF.Store(true)
	}
	return n, err
}

type simpleWriteStallCtxKey struct{}

type simpleWriteStallMeta struct {
	bucket string
}

func withSimpleWriteStallMeta(ctx context.Context, bucket string) context.Context {
	return context.WithValue(ctx, simpleWriteStallCtxKey{}, simpleWriteStallMeta{bucket: bucket})
}

type baseTimestampSample struct {
	hexValue string
	recvTime time.Time
}

// writeStallTransport wraps an http.RoundTripper to:
//  1. Sniff X-Goog-Gcs-Base-Ts from all GCS HTTP responses.
//  2. Detect per-2MiB Write Delay stalls and post-EOF Response Delay stalls on
//     retryable simple (multipart) writes.
//  3. Inject X-Goog-Gcs-Base-Ts and splice the 3rd multipart part carrying
//     x-goog-gcs-deadline-offset-ms at Part 1 io.EOF when a fresh base timestamp
//     is available.
// isSafeCondsForWriteStall reports whether the object preconditions are safe for
// simple write stall cancellation and retry.
// Without server-side TrueTime deadline enforcement, preconditions that can transition
// from a 412 failure to a match after a state change (such as DoesNotExist /
// ifGenerationMatch=0, GenerationNotMatch, or MetagenerationMatch without GenerationMatch>0)
// are unsafe to cancel during Response Delay. When TrueTime is active, waiting out the
// advertised deadline before retrying makes all preconditions safe.
func isSafeCondsForWriteStall(conds *Conditions, trueTimeActive bool) bool {
	if trueTimeActive {
		return true
	}
	if conds == nil {
		return true
	}
	return conds.GenerationMatch > 0
}

type writeStallTransport struct {
	base                  http.RoundTripper
	writeBlockDelayMgr    *bucketDelayManager
	writeResponseDelayMgr *bucketDelayManager
	metrics               *clientMetrics
	blockSize             int64
	enableTrueTimePart    bool
	latestBaseTS          atomic.Pointer[baseTimestampSample]

	activeConnsMu sync.Mutex
	activeConns   map[net.Conn]int
}

func newWriteStallTransport(base http.RoundTripper, writeBlockDM, writeRespDM *bucketDelayManager, metrics *clientMetrics, enableTrueTimePart bool) *writeStallTransport {
	return &writeStallTransport{
		base:                  base,
		writeBlockDelayMgr:    writeBlockDM,
		writeResponseDelayMgr: writeRespDM,
		metrics:               metrics,
		blockSize:             defaultStallBlockSize,
		enableTrueTimePart:    enableTrueTimePart,
		activeConns:           make(map[net.Conn]int),
	}
}

func (t *writeStallTransport) acquireConn(conn net.Conn) {
	if conn == nil {
		return
	}
	t.activeConnsMu.Lock()
	t.activeConns[conn]++
	t.activeConnsMu.Unlock()
}

// releaseConn decrements the active unstalled stream count for conn and reports
// whether no other active unstalled streams remain on conn.
func (t *writeStallTransport) releaseConn(conn net.Conn) bool {
	if conn == nil {
		return true
	}
	t.activeConnsMu.Lock()
	defer t.activeConnsMu.Unlock()
	rem := t.activeConns[conn] - 1
	if rem <= 0 {
		delete(t.activeConns, conn)
		return true
	}
	t.activeConns[conn] = rem
	return false
}

func (t *writeStallTransport) activeStreamCount(conn net.Conn) int {
	if conn == nil {
		return 1
	}
	t.activeConnsMu.Lock()
	n := t.activeConns[conn]
	t.activeConnsMu.Unlock()
	if n < 1 {
		return 1
	}
	return n
}

func (t *writeStallTransport) hasTrueTimeActive() bool {
	if !t.enableTrueTimePart {
		return false
	}
	_, _, ok := t.getFreshBaseTS()
	return ok
}

func (t *writeStallTransport) recordResponseBaseTS(resp *http.Response) {
	if resp == nil || resp.Header == nil {
		return
	}
	ts := resp.Header.Get(baseTimestampHeaderName)
	if ts == "" {
		return
	}
	if _, err := strconv.ParseInt(ts, 16, 64); err != nil {
		return
	}
	t.latestBaseTS.Store(&baseTimestampSample{
		hexValue: ts,
		recvTime: time.Now(),
	})
}

func (t *writeStallTransport) getFreshBaseTS() (string, time.Time, bool) {
	sample := t.latestBaseTS.Load()
	if sample == nil {
		return "", time.Time{}, false
	}
	if time.Since(sample.recvTime) > baseTimestampTTL {
		return "", time.Time{}, false
	}
	return sample.hexValue, sample.recvTime, true
}

func (t *writeStallTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	baseRT := t.base
	if baseRT == nil {
		baseRT = http.DefaultTransport
	}

	meta, hasMeta := req.Context().Value(simpleWriteStallCtxKey{}).(simpleWriteStallMeta)
	if !hasMeta || t.writeBlockDelayMgr == nil || t.writeResponseDelayMgr == nil ||
		req.URL == nil || req.URL.Query().Get("uploadType") != "multipart" ||
		req.GetBody == nil || req.Body == nil {
		resp, err := baseRT.RoundTrip(req)
		t.recordResponseBaseTS(resp)
		return resp, err
	}

	parentCtx := req.Context()
	if err := parentCtx.Err(); err != nil {
		return nil, err
	}

	attemptCtx, cancelAttempt := context.WithCancel(parentCtx)
	var connMu sync.Mutex
	var attemptConn net.Conn
	var tracker *stallRequestBody
	attemptCtx = httptrace.WithClientTrace(attemptCtx, &httptrace.ClientTrace{
		GetConn: func(_ string) {
			if tracker != nil {
				tracker.startInitialTimer()
			}
		},
		GotConn: func(info httptrace.GotConnInfo) {
			connMu.Lock()
			prevConn := attemptConn
			attemptConn = info.Conn
			connMu.Unlock()
			if prevConn != nil && prevConn != info.Conn {
				t.releaseConn(prevConn)
			}
			if prevConn != info.Conn {
				t.acquireConn(info.Conn)
			}
		},
	})
	defer func() {
		connMu.Lock()
		c := attemptConn
		attemptConn = nil
		connMu.Unlock()
		if c != nil {
			t.releaseConn(c)
		}
	}()
	// Clone req so that attemptReq.Header is a private copy for this attempt,
	// preventing concurrent map read/write panics when gensupport.sendAndRetry
	// mutates req.Header on retry while an abandoned HTTP/2 stream winds down.
	attemptReq := req.Clone(attemptCtx)

	var boundary string
	if _, params, err := mime.ParseMediaType(attemptReq.Header.Get("Content-Type")); err == nil {
		boundary = params["boundary"]
	}

	var baseTSHex string
	var baseRecvTime time.Time
	var hasBaseTS bool
	if t.enableTrueTimePart {
		baseTSHex, baseRecvTime, hasBaseTS = t.getFreshBaseTS()
		if hasBaseTS && boundary != "" && attemptReq.ContentLength <= 0 {
			attemptReq.Header.Set(baseTimestampHeaderName, baseTSHex)
		} else {
			hasBaseTS = false
		}
	}

	target := ""
	if attemptReq.URL != nil {
		target = stripPort(attemptReq.URL.Host)
	}

	blockSize := t.blockSize
	if blockSize <= 0 {
		blockSize = defaultStallBlockSize
	}

	tracker = newStallRequestBody(
		attemptReq.Body,
		meta.bucket,
		t.writeBlockDelayMgr,
		t.writeResponseDelayMgr,
		t.metrics,
		target,
		parentCtx,
		blockSize,
		boundary,
		hasBaseTS,
		baseRecvTime,
	)
	tracker.activeStreams = func() int {
		connMu.Lock()
		c := attemptConn
		connMu.Unlock()
		return t.activeStreamCount(c)
	}
	attemptReq.Body = tracker
	origGetBody := attemptReq.GetBody
	attemptReq.GetBody = func() (io.ReadCloser, error) {
		rawBody, err := origGetBody()
		if err != nil {
			return nil, err
		}
		tracker.resetWithBody(rawBody)
		return tracker, nil
	}

	type roundTripResult struct {
		resp *http.Response
		err  error
	}
	doneCh := make(chan roundTripResult, 1)
	go func() {
		resp, err := baseRT.RoundTrip(attemptReq)
		doneCh <- roundTripResult{resp: resp, err: err}
	}()

	select {
	case <-tracker.stalledCh:
		tracker.stopTimer()
		cancelAttempt()
		_ = tracker.Close()
		connMu.Lock()
		conn := attemptConn
		attemptConn = nil
		connMu.Unlock()
		noOtherActiveStreams := t.releaseConn(conn)
		if noOtherActiveStreams {
			abortStalledConn(conn)
		}
		// Wait briefly for baseRT.RoundTrip to observe context cancellation (or the
		// closed connection when no sibling streams share conn).
		cleanupTimer := time.NewTimer(20 * time.Millisecond)
		select {
		case <-doneCh:
			cleanupTimer.Stop()
		case <-cleanupTimer.C:
		}
		if noOtherActiveStreams {
			t.CloseIdleConnections()
		}
		if err := parentCtx.Err(); err != nil {
			return nil, err
		}
		// If a Response Delay stall occurred on a TrueTime-protected attempt and
		// any fraction of the advertised deadline remains, wait until it passes
		// before issuing the retry.
		if waitDur := tracker.remainingAdvertisedDeadline(); waitDur > 0 {
			timer := time.NewTimer(waitDur)
			select {
			case <-parentCtx.Done():
				timer.Stop()
				return nil, parentCtx.Err()
			case <-timer.C:
			}
		}
		return nil, errWriteStall

	case res := <-doneCh:
		tracker.stopTimer()
		t.recordResponseBaseTS(res.resp)
		if res.err != nil {
			cancelAttempt()
			_ = tracker.Close()
			if parentCtx.Err() == nil {
				if tracker.didStall() {
					return nil, errWriteStall
				}
				// If a multiplexed HTTP/2 connection was closed mid-body (before io.EOF /
				// closing multipart boundary was reached, so GCS cannot have committed),
				// surface errWriteStall so onlyStallRetry rewinds and retries cleanly.
				if tracker.getEOFTime().IsZero() && (errors.Is(res.err, net.ErrClosed) || strings.Contains(res.err.Error(), "http2: client conn is closed")) {
					return nil, errWriteStall
				}
			}
			return res.resp, res.err
		}
		if tracker.shouldDecreaseWriteDelayOnCompletion() {
			t.writeBlockDelayMgr.decrease(meta.bucket)
		}
		if eofTime := tracker.getEOFTime(); !eofTime.IsZero() {
			commitLat := time.Since(eofTime) - tracker.getDrainBudget()
			if commitLat < 0 {
				commitLat = 0
			}
			t.writeResponseDelayMgr.update(meta.bucket, commitLat)
		}
		if res.resp != nil && res.resp.Body != nil {
			res.resp.Body = &cancelOnCloseReadCloser{
				ReadCloser: res.resp.Body,
				cancel:     cancelAttempt,
			}
		} else {
			cancelAttempt()
		}
		return res.resp, nil
	}
}

// abortStalledConn force-closes the underlying network connection of a stalled
// attempt so that HTTP/2's ClientConnPool immediately evicts the stalled
// http2ClientConn (whose connection-level flow-control window or TCP socket may
// be wedged) instead of routing the retry onto the same stalled connection.
func abortStalledConn(conn net.Conn) {
	if conn == nil {
		return
	}
	_ = conn.SetDeadline(time.Unix(1, 0))
	if tc, ok := conn.(*tls.Conn); ok {
		if nc := tc.NetConn(); nc != nil {
			_ = nc.SetDeadline(time.Unix(1, 0))
			_ = nc.Close()
		}
	}
	_ = conn.Close()
}

func (t *writeStallTransport) CloseIdleConnections() {
	type closeIdler interface {
		CloseIdleConnections()
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	if tr, ok := base.(closeIdler); ok {
		tr.CloseIdleConnections()
	}
}

type cancelOnCloseReadCloser struct {
	io.ReadCloser
	once   sync.Once
	cancel context.CancelFunc
}

func (c *cancelOnCloseReadCloser) Close() error {
	err := c.ReadCloser.Close()
	c.once.Do(c.cancel)
	return err
}

// stallRequestBody wraps the multipart request body for a single simple-write attempt.
// Because net/http calls socket.Write between req.Body.Read calls, the Write Delay timer
// runs between Read calls (while net/http writes the previously returned bytes to the
// socket) and is paused while pulling bytes from the underlying multipartReader.
type stallRequestBody struct {
	body                  io.ReadCloser
	bucket                string
	writeDelayManager     *bucketDelayManager
	responseDelayManager  *bucketDelayManager
	metrics               *clientMetrics
	target                string
	ctx                   context.Context
	blockSize             int64
	boundary              string
	closingBoundary       []byte
	hasBaseTS             bool
	baseRecvTime          time.Time
	queueThresholdBytes   int64
	activeStreams         func() int

	stalledCh   chan struct{}
	stalledOnce sync.Once
	stalled     atomic.Bool
	closed      atomic.Bool

	mu                 sync.Mutex
	timer              *time.Timer
	timerGen           uint64
	inSocketWrite      bool
	inResponseWait     bool
	prevBytes          int64
	prevWriteStart     time.Time
	bytesInChunk       int64
	timeSpentInChunk   time.Duration
	totalBytes         int64
	eofTime            time.Time
	drainBudget        time.Duration
	responseBudget     time.Duration
	advertisedDeadline time.Time

	// Fields used for holding back the closing boundary to splice Part 2 at Part 1 io.EOF.
	stageBuf []byte
	pending  []byte
	bodyEOF  bool
}

func newStallRequestBody(
	body io.ReadCloser,
	bucket string,
	writeDM *bucketDelayManager,
	respDM *bucketDelayManager,
	metrics *clientMetrics,
	target string,
	ctx context.Context,
	blockSize int64,
	boundary string,
	hasBaseTS bool,
	baseRecvTime time.Time,
) *stallRequestBody {
	var closing []byte
	if hasBaseTS && boundary != "" {
		closing = []byte("\r\n--" + boundary + "--\r\n")
	}
	return &stallRequestBody{
		body:                 body,
		bucket:               bucket,
		writeDelayManager:    writeDM,
		responseDelayManager: respDM,
		metrics:              metrics,
		target:               target,
		ctx:                  ctx,
		blockSize:            blockSize,
		boundary:             boundary,
		closingBoundary:      closing,
		hasBaseTS:            hasBaseTS,
		baseRecvTime:         baseRecvTime,
		queueThresholdBytes:  2 * blockSize,
		stalledCh:            make(chan struct{}),
	}
}

func (s *stallRequestBody) getActiveStreams() int {
	if s.activeStreams == nil {
		return 1
	}
	n := s.activeStreams()
	if n < 1 {
		return 1
	}
	return n
}

func (s *stallRequestBody) Read(p []byte) (int, error) {
	if s.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	if s.stalled.Load() {
		return 0, errWriteStall
	}

	streams := s.getActiveStreams()

	s.mu.Lock()
	// 1. Credit the previous Read's bytes now that net/http has finished writing them to the socket.
	if s.prevBytes > 0 {
		if s.timer != nil {
			s.timer.Stop()
		}
		s.timerGen++
		s.inSocketWrite = false

		writeDur := time.Since(s.prevWriteStart)
		s.timeSpentInChunk += writeDur
		s.bytesInChunk += s.prevBytes
		s.totalBytes += s.prevBytes
		s.prevBytes = 0

		if s.bytesInChunk >= s.blockSize {
			// Skip DynamicDelay.update on the first 4 MiB (2 blocks) of socket buffer
			// (queueThresholdBytes / streams per multiplexed HTTP/2 stream) because those
			// bytes are absorbed at RAM speed by the local OS TCP send buffer (tcp_wmem).
			// Normalize the measured block duration by the number of active streams sharing
			// the underlying net.Conn so writeDelayManager tracks true per-2MiB connection
			// throughput regardless of HTTP/2 stream multiplexing.
			if s.totalBytes > s.queueThresholdBytes/int64(streams) {
				s.writeDelayManager.update(s.bucket, s.timeSpentInChunk/time.Duration(streams))
			}
			s.bytesInChunk = 0
			s.timeSpentInChunk = 0
		}
	} else if s.inSocketWrite {
		// Disarm the initial pre-Read connection/header setup timer once net/http
		// begins reading the request body.
		if s.timer != nil {
			s.timer.Stop()
		}
		s.timerGen++
		s.inSocketWrite = false
	}

	if len(p) == 0 {
		s.mu.Unlock()
		return 0, nil
	}

	// 2. Cap p so a single Read+Write cycle never crosses a 2 MiB block boundary.
	remBytes := s.blockSize - s.bytesInChunk
	if int64(len(p)) > remBytes {
		p = p[:remBytes]
	}
	s.mu.Unlock()

	// 3. Read from the underlying multipart buffer/pipe with the Write Delay timer paused.
	n, err := s.readFromUnderlying(p)

	s.mu.Lock()
	if s.stalled.Load() {
		s.mu.Unlock()
		return 0, errWriteStall
	}

	if n > 0 && err == nil {
		// 4A. Arm Timer 1 (Write Delay) for the socket.Write that net/http will perform next.
		// Scale the per-stream 2 MiB block timeout by the number of active write streams
		// currently multiplexed on the same net.Conn (which transmits streams * 2 MiB across
		// the socket over one per-stream block wave).
		s.prevBytes = int64(n)
		s.prevWriteStart = time.Now()
		s.inSocketWrite = true
		blockTimeout := s.writeDelayManager.getValue(s.bucket) * time.Duration(streams)
		budget := blockTimeout - s.timeSpentInChunk
		if budget <= 0 {
			s.triggerStallLocked(false, blockTimeout)
			s.mu.Unlock()
			_ = s.body.Close()
			return 0, errWriteStall
		}
		s.armTimerLocked(budget, blockTimeout, false)
	} else if err == io.EOF {
		// 4B. All request body bytes have been handed to the socket; switch to Timer 2 (Response Delay).
		// Include the proportional drain budget for up to queueThresholdBytes (4 MiB in tcp_wmem)
		// that may still be queued in the local kernel TCP send buffer at io.EOF.
		s.inSocketWrite = false
		s.inResponseWait = true
		s.eofTime = time.Now()
		if s.responseBudget <= 0 {
			s.drainBudget, s.responseBudget = s.computeCommitBudgetLocked(s.totalBytes)
		}
		s.armTimerLocked(s.responseBudget, s.responseBudget, true)
	}
	s.mu.Unlock()

	return n, err
}

// readFromUnderlying reads from s.body. When TrueTime (s.hasBaseTS) is active, it holds
// back the final len(s.closingBoundary) bytes ("\r\n--<boundary>--\r\n") until s.body
// returns io.EOF, then splices Part 2 (x-goog-gcs-deadline-offset-ms) right before the
// closing boundary. It always returns (n > 0, nil) before returning (0, io.EOF) on the
// subsequent call so that the final socket.Write is timed under Timer 1 before Timer 2 starts.
func (s *stallRequestBody) readFromUnderlying(p []byte) (int, error) {
	if len(s.pending) > 0 {
		n := copy(p, s.pending)
		s.pending = s.pending[n:]
		return n, nil
	}
	if s.bodyEOF {
		return 0, io.EOF
	}

	if !s.hasBaseTS || len(s.closingBoundary) == 0 {
		n, err := s.body.Read(p)
		if err == io.EOF {
			s.bodyEOF = true
			if n > 0 {
				return n, nil
			}
			return 0, io.EOF
		}
		return n, err
	}

	keep := len(s.closingBoundary)
	for {
		if len(s.stageBuf) > keep {
			releaseLen := len(s.stageBuf) - keep
			n := copy(p, s.stageBuf[:releaseLen])
			s.stageBuf = append(s.stageBuf[:0], s.stageBuf[n:]...)
			return n, nil
		}
		rn, rerr := s.body.Read(p)
		if rn > 0 {
			s.stageBuf = append(s.stageBuf, p[:rn]...)
		}
		if rerr == io.EOF {
			s.bodyEOF = true
			if bytes.HasSuffix(s.stageBuf, s.closingBoundary) {
				s.spliceTrueTimePart()
			}
			n := copy(p, s.stageBuf)
			s.pending = s.stageBuf[n:]
			s.stageBuf = nil
			if n > 0 {
				return n, nil
			}
			return 0, io.EOF
		}
		if rerr != nil {
			return 0, rerr
		}
	}
}

func (s *stallRequestBody) computeCommitBudgetLocked(streamBytes int64) (drainBudget, commitBudget time.Duration) {
	drainBytes := streamBytes
	if drainBytes > s.queueThresholdBytes {
		drainBytes = s.queueThresholdBytes
	}
	if drainBytes < 0 {
		drainBytes = 0
	}
	writeTimeout := s.writeDelayManager.getValue(s.bucket)
	respTimeout := s.responseDelayManager.getValue(s.bucket)
	if s.blockSize > 0 && drainBytes > 0 {
		drainBudget = time.Duration(float64(writeTimeout) * float64(drainBytes) / float64(s.blockSize))
	}
	commitBudget = drainBudget + respTimeout
	return drainBudget, commitBudget
}

// spliceTrueTimePart is called at the exact instant Part 1 hits io.EOF. It computes the
// TrueTime deadline offset (elapsed time since baseRecvTime + local buffer drain budget
// + ResponseDelayTimeout) and splices Part 2 before the closing multipart boundary.
func (s *stallRequestBody) spliceTrueTimePart() {
	s.mu.Lock()
	totalStreamBytes := s.totalBytes + int64(len(s.stageBuf))
	drainBudget, commitBudget := s.computeCommitBudgetLocked(totalStreamBytes)
	offsetMs := time.Since(s.baseRecvTime).Milliseconds() + commitBudget.Milliseconds()
	if offsetMs < 1 {
		offsetMs = 1
	}
	s.drainBudget = drainBudget
	s.responseBudget = commitBudget
	s.advertisedDeadline = s.baseRecvTime.Add(time.Duration(offsetMs) * time.Millisecond)
	s.mu.Unlock()

	part2AndClose := fmt.Sprintf("\r\n--%s\r\n%s: %d\r\n\r\n\r\n--%s--\r\n",
		s.boundary, deadlineOffsetHeaderName, offsetMs, s.boundary)
	prefixLen := len(s.stageBuf) - len(s.closingBoundary)
	s.stageBuf = append(s.stageBuf[:prefixLen], part2AndClose...)
}

func (s *stallRequestBody) armTimerLocked(budget, stallDuration time.Duration, isResponsePhase bool) {
	if s.timer != nil {
		s.timer.Stop()
	}
	s.timerGen++
	myGen := s.timerGen
	s.timer = time.AfterFunc(budget, func() {
		s.mu.Lock()
		if s.timerGen != myGen {
			s.mu.Unlock()
			return
		}
		if !isResponsePhase && (s.closed.Load() || !s.inSocketWrite) {
			s.mu.Unlock()
			return
		}
		if isResponsePhase && !s.inResponseWait {
			s.mu.Unlock()
			return
		}
		s.triggerStallLocked(isResponsePhase, stallDuration)
		s.mu.Unlock()
		log.Printf("storage: write stall detected (responsePhase=%v) after %v on bucket %s, aborting attempt", isResponsePhase, stallDuration, s.bucket)
		_ = s.body.Close()
	})
}

func (s *stallRequestBody) triggerStallLocked(isResponsePhase bool, budget time.Duration) {
	s.stalled.Store(true)
	if isResponsePhase {
		s.responseDelayManager.increase(s.bucket)
	} else {
		s.writeDelayManager.increase(s.bucket)
	}
	if s.metrics != nil {
		s.metrics.recordStallDuration(s.ctx, budget, "WriteObject", "http", s.target)
	}
	s.stalledOnce.Do(func() {
		close(s.stalledCh)
	})
}

func (s *stallRequestBody) stopTimer() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.timer != nil {
		s.timer.Stop()
	}
	s.timerGen++
	s.inSocketWrite = false
	s.inResponseWait = false
}

func (s *stallRequestBody) didStall() bool {
	return s.stalled.Load()
}

func (s *stallRequestBody) getEOFTime() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eofTime
}

func (s *stallRequestBody) getDrainBudget() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.drainBudget
}

// shouldDecreaseWriteDelayOnCompletion reports whether RoundTrip should call
// writeBlockDelayMgr.decrease(bucket) when the upload succeeds. Because
// per-block updates during Read() are skipped for the first queueThresholdBytes
// (4 MiB in tcp_wmem), uploads <= 4 MiB otherwise never decay writeBlockDelayMgr
// back toward Min after a transient stall. Once RoundTrip receives the HTTP
// response headers, all buffered bytes have drained over the wire within budget.
func (s *stallRequestBody) shouldDecreaseWriteDelayOnCompletion() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalBytes <= s.queueThresholdBytes
}

// startInitialTimer arms Timer 1 (Write Delay) when http.Transport enters GetConn
// (after OAuth2 token acquisition has completed) so that stalls during TCP/TLS
// connection establishment, HTTP/2 SETTINGS exchange, or sending HTTP request headers
// (before net/http makes its first req.Body.Read call) are also detected and retried.
func (s *stallRequestBody) startInitialTimer() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed.Load() || s.stalled.Load() || s.inSocketWrite || s.inResponseWait {
		return
	}
	s.inSocketWrite = true
	s.prevWriteStart = time.Now()
	timeout := s.writeDelayManager.getValue(s.bucket)
	s.armTimerLocked(timeout, timeout, false)
}

// resetWithBody resets the tracker when net/http internally calls req.GetBody()
// during a single RoundTrip (for example, when retrying after a stale pooled
// connection reset).
func (s *stallRequestBody) resetWithBody(newBody io.ReadCloser) {
	s.mu.Lock()
	if s.timer != nil {
		s.timer.Stop()
	}
	s.timerGen++
	oldBody := s.body
	s.body = newBody
	s.inSocketWrite = true
	s.inResponseWait = false
	s.prevBytes = 0
	s.prevWriteStart = time.Now()
	s.bytesInChunk = 0
	s.timeSpentInChunk = 0
	s.totalBytes = 0
	s.eofTime = time.Time{}
	s.drainBudget = 0
	s.responseBudget = 0
	s.advertisedDeadline = time.Time{}
	s.stageBuf = nil
	s.pending = nil
	s.bodyEOF = false
	s.closed.Store(false)
	timeout := s.writeDelayManager.getValue(s.bucket)
	s.armTimerLocked(timeout, timeout, false)
	s.mu.Unlock()
	if oldBody != nil {
		_ = oldBody.Close()
	}
}

func (s *stallRequestBody) remainingAdvertisedDeadline() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.advertisedDeadline.IsZero() {
		return 0
	}
	rem := time.Until(s.advertisedDeadline)
	if rem < 0 {
		return 0
	}
	return rem
}

func (s *stallRequestBody) Close() error {
	s.closed.Store(true)
	s.mu.Lock()
	inRespWait := s.inResponseWait
	s.mu.Unlock()
	// When net/http finishes sending the request body (at io.EOF), it immediately calls
	// req.Body.Close() before waiting for the HTTP response headers. Do not disarm Timer 2
	// (Response Delay) when Close() is called during the response wait phase; RoundTrip
	// disarms Timer 2 via stopTimer() once the response headers arrive or a stall fires.
	if !inRespWait {
		s.stopTimer()
	}
	return s.body.Close()
}
