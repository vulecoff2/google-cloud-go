// Copyright 2024 Google LLC
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
	"hash/crc32"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"cloud.google.com/go/storage/experimental"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/googleapis/gax-go/v2"
	"github.com/googleapis/gax-go/v2/callctx"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	"google.golang.org/api/option"
)

func TestSetHeadersFromContext(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	xGoogKey := "X-Goog-Api-Client"

	for _, test := range []struct {
		desc            string
		originalHeaders http.Header
		headersOnCtx    []string // keyval pairs
		wantHeaders     http.Header
	}{
		{
			desc: "all empty values",
		},
		{
			desc: "regular headers",
			originalHeaders: http.Header{
				"Headerkey-A": {"value1", "value2"},
				"Headerkey-B": {"v1", "v2"},
			},
			headersOnCtx: []string{"key-c", "val1", "headerkey-a", "value3"},
			wantHeaders: http.Header{
				"Headerkey-A": {"value3"},
				"Headerkey-B": {"v1", "v2"},
				"Key-C":       {"val1"},
			},
		},
		{
			desc: "x-goog-api-client merging",
			originalHeaders: http.Header{
				"Headerkey-A": {"value1", "value2"},
			},
			headersOnCtx: []string{"key-c", "val1", xGoogKey, "k1/v1 k2/v2", xGoogKey, "k3/v3"},
			wantHeaders: http.Header{
				"Headerkey-A": {"value1", "value2"},
				"Key-C":       {"val1"},
				xGoogKey:      {"k1/v1 k2/v2 k3/v3"},
			},
		},
		{
			desc: "x-goog-api-client merging with values already set",
			originalHeaders: http.Header{
				"Headerkey-A": {"value1", "value2"},
				xGoogKey:      {"k4/v4 k5/v5"},
			},
			headersOnCtx: []string{"key-c", "val1", xGoogKey, "k1/v1 k2/v2", xGoogKey, "k3/v3"},
			wantHeaders: http.Header{
				"Headerkey-A": {"value1", "value2"},
				"Key-C":       {"val1"},
				xGoogKey:      {"k1/v1 k2/v2 k3/v3 k4/v4 k5/v5"},
			},
		},
	} {
		t.Run(test.desc, func(t *testing.T) {
			ctx := callctx.SetHeaders(ctx, test.headersOnCtx...)
			got := test.originalHeaders.Clone()
			setHeadersFromCtx(ctx, got)

			if len(got) != len(test.wantHeaders) {
				t.Errorf("Headers not set correctly: got: %+v, want: %+v\n", got, test.wantHeaders)
			}

			for k, wantVals := range test.wantHeaders {
				if diff := cmp.Diff(got[k], wantVals, cmpopts.SortSlices(func(a, b string) bool { return len(a) < len(b) })); diff != "" {
					t.Errorf("Header %q not set correctly: got(-),want(+):\n%s", k, diff)
				}
			}
		})
	}
}

// TestXMLReaderRetryInvocationHeaders verifies that retried XML read requests
// carry only the invocation headers for the current attempt. The underlying
// *http.Request is reused across attempts, so headers from earlier attempts
// must not leak into later ones.
func TestXMLReaderRetryInvocationHeaders(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	const failures = 2
	var (
		mu         sync.Mutex
		gotHeaders []string
	)
	hc, closeServer := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotHeaders = append(gotHeaders, strings.Join(r.Header.Values(xGoogHeaderKey), " "))
		n := len(gotHeaders)
		mu.Unlock()
		if n <= failures {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(readData)))
		w.Write([]byte(readData))
	})
	defer closeServer()

	client, err := NewClient(ctx, option.WithHTTPClient(hc))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("b").Object("o").Retryer(
		WithBackoff(gax.Backoff{Initial: time.Millisecond, Max: time.Millisecond}),
		WithPolicy(RetryAlways),
	).NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()
	if _, err := io.ReadAll(r); err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(gotHeaders) != failures+1 {
		t.Fatalf("got %d requests, want %d", len(gotHeaders), failures+1)
	}
	for i, h := range gotHeaders {
		attempt := i + 1
		if got := strings.Count(h, "gccl-attempt-count/"); got != 1 {
			t.Errorf("attempt %d: %q has %d gccl-attempt-count tokens, want 1", attempt, h, got)
		}
		if want := fmt.Sprintf("gccl-attempt-count/%d", attempt); !strings.Contains(h, want) {
			t.Errorf("attempt %d: %q does not contain %q", attempt, h, want)
		}
		if got := strings.Count(h, "gccl-invocation-id/"); got != 1 {
			t.Errorf("attempt %d: %q has %d gccl-invocation-id tokens, want 1", attempt, h, got)
		}
	}
}

func TestAppendableWriteUnsupported(t *testing.T) {
	ctx := context.Background()
	c, err := newHTTPStorageClient(ctx)
	if err != nil {
		t.Fatalf("failed to create HTTP client: %v", err)
	}

	bucket := "a-bucket"
	object := "an-object"
	attrs := ObjectAttrs{
		Bucket:     bucket,
		Name:       object,
		Generation: defaultGen,
	}

	params := &openWriterParams{
		attrs:    &attrs,
		bucket:   bucket,
		append:   true,
		ctx:      ctx,
		donec:    make(chan struct{}),
		setError: func(_ error) {},        // no-op
		progress: func(_ int64) {},        // no-op
		setObj:   func(_ *ObjectAttrs) {}, // no-op
	}
	_, err = c.OpenWriter(params)
	if err == nil {
		t.Errorf("OpenWriter: got ok; want error")
	}
}

func TestValidateChecksumFromServer(t *testing.T) {
	correctChecksum := 1
	tests := []struct {
		name          string
		wrongChecksum bool
		wantErr       error
	}{
		{
			name:          "correct checksum",
			wrongChecksum: false,
		},
		{
			name:          "wrong checksum",
			wrongChecksum: true,
			wantErr:       fmt.Errorf("storage: checksum mismatch: computed %q, server %q; the bucket may contain corrupted object", encodeUint32(2), encodeUint32(1)),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			serverChecksumChan := make(chan uint32, 1)
			checksum := correctChecksum
			if test.wrongChecksum {
				checksum = 2
			}
			hiw := &httpInternalWriter{
				serverChecksumChan: serverChecksumChan,
				fullObjectChecksum: uint32(checksum),
			}
			go func() {
				serverChecksumChan <- uint32(correctChecksum)
			}()
			err := hiw.validateChecksumFromServer()
			if test.wantErr == nil {
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				return
			}
			if err.Error() != test.wantErr.Error() {
				t.Errorf("Want err: %v, got err: %v", err, test.wantErr)
			}
		})
	}
}

func TestTrackingTransport(t *testing.T) {
	mock := &mockTransport{}
	mock.addResult(&http.Response{Status: "200 OK"}, nil)
	tt := &trackingTransport{
		base:     mock,
		features: uint32(1 << featurePCU),
	}

	ctx := context.Background()
	ctx = addFeatureAttributes(ctx, featureMultistreamInMRD)
	req, _ := http.NewRequestWithContext(ctx, "GET", "http://example.com", nil)

	_, err := tt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	gotHeader := mock.gotReq.Header.Get(featureTrackerHeaderName)
	wantFeatures := uint32(1<<featurePCU) | uint32(1<<featureMultistreamInMRD)
	wantHeader := encodeFeatureMask(wantFeatures)

	if gotHeader != wantHeader {
		t.Errorf("Header %s = %q; want %q", featureTrackerHeaderName, gotHeader, wantHeader)
	}

	// Verify original request was not modified.
	if req.Header.Get(featureTrackerHeaderName) != "" {
		t.Errorf("Original request header was modified")
	}
}

// TestTrackingTransport_OwnsHeader verifies that the transport always rewrites
// the feature header from its own bitmask: foreign or malformed values present
// on the request (e.g. set via callctx for a different client) are either
// replaced by the merged mask or dropped when there is nothing to report.
func TestTrackingTransport_OwnsHeader(t *testing.T) {
	foreign := strings.Repeat("Zm9v", 130) // ~390 bytes of non-mask payload.
	tests := []struct {
		desc        string
		features    uint32
		ctxFeatures []trackedFeature
		reqHeader   []string
		wantHeader  []string
	}{
		{
			desc:       "no features, no header: untouched",
			wantHeader: nil,
		},
		{
			desc:       "no features, zero mask header: dropped",
			reqHeader:  []string{encodeFeatureMask(0)},
			wantHeader: nil,
		},
		{
			desc:       "no features, foreign value: dropped",
			reqHeader:  []string{foreign},
			wantHeader: nil,
		},
		{
			desc:       "features, foreign value: replaced by mask",
			features:   uint32(1 << featurePCU),
			reqHeader:  []string{foreign},
			wantHeader: []string{encodeFeatureMask(uint32(1 << featurePCU))},
		},
		{
			desc:        "legacy standard-alphabet value is merged and re-encoded",
			ctxFeatures: []trackedFeature{featureMultistreamInMRD},
			reqHeader:   []string{"AAAA+A=="}, // 0xF8 in standard base64 (bits 0 and 1 unset).
			wantHeader:  []string{encodeFeatureMask(0xF8 | uint32(1<<featureMultistreamInMRD))},
		},
		{
			desc:       "multiple values collapse to one",
			reqHeader:  []string{encodeFeatureMask(1), encodeFeatureMask(2), foreign},
			wantHeader: []string{encodeFeatureMask(3)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			mock := &mockTransport{}
			mock.addResult(&http.Response{Status: "200 OK"}, nil)
			tt := &trackingTransport{base: mock, features: tc.features}

			ctx := context.Background()
			if len(tc.ctxFeatures) > 0 {
				ctx = addFeatureAttributes(ctx, tc.ctxFeatures...)
			}
			req, _ := http.NewRequestWithContext(ctx, "GET", "http://example.com", nil)
			for _, v := range tc.reqHeader {
				req.Header.Add(featureTrackerHeaderName, v)
			}

			if _, err := tt.RoundTrip(req); err != nil {
				t.Fatalf("RoundTrip failed: %v", err)
			}
			got := mock.gotReq.Header.Values(featureTrackerHeaderName)
			if diff := cmp.Diff(tc.wantHeader, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("header %s mismatch (-want +got):\n%s", featureTrackerHeaderName, diff)
			}
			if diff := cmp.Diff(tc.reqHeader, req.Header.Values(featureTrackerHeaderName), cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("caller request header %s modified (-want +got):\n%s", featureTrackerHeaderName, diff)
			}
		})
	}
}

type mockCloseIdler struct {
	mockTransport
	closedIdle bool
}

func (m *mockCloseIdler) CloseIdleConnections() {
	m.closedIdle = true
}

func TestTrackingTransport_CloseIdleConnections(t *testing.T) {
	mock := &mockCloseIdler{}
	tt := &trackingTransport{
		base: mock,
	}

	tt.CloseIdleConnections()
	if !mock.closedIdle {
		t.Errorf("CloseIdleConnections was not called on base transport")
	}
}

func TestXMLReaderMidStreamStallRetry(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	const chunkSize = 1024 * 1024 // 1 MiB per attempt; 3 MiB total across two stalls.
	var (
		mu        sync.Mutex
		gotRanges []string
	)
	crcVal := crc32.Checksum(randomBytes3MiB, crc32cTable)
	crcHeader := "crc32c=" + encodeUint32(crcVal)

	hc, closeServer := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		rangeHdr := r.Header.Get("Range")
		mu.Lock()
		gotRanges = append(gotRanges, rangeHdr)
		mu.Unlock()

		w.Header().Set("X-Goog-Generation", "12345")
		w.Header().Set("X-Goog-Hash", crcHeader)

		switch rangeHdr {
		case "":
			w.Header().Set("Content-Length", strconv.Itoa(len(randomBytes3MiB)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(randomBytes3MiB[:chunkSize])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
		case fmt.Sprintf("bytes=%d-", chunkSize):
			remLen := len(randomBytes3MiB) - chunkSize
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", chunkSize, len(randomBytes3MiB)-1, len(randomBytes3MiB)))
			w.Header().Set("Content-Length", strconv.Itoa(remLen))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(randomBytes3MiB[chunkSize : 2*chunkSize])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
		case fmt.Sprintf("bytes=%d-", 2*chunkSize):
			start := 2 * chunkSize
			rem := randomBytes3MiB[start:]
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(randomBytes3MiB)-1, len(randomBytes3MiB)))
			w.Header().Set("Content-Length", strconv.Itoa(len(rem)))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(rem)
		default:
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
		}
	})
	defer closeServer()

	client, err := NewClient(ctx,
		option.WithHTTPClient(hc),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              250 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("storage.NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll: %v", err)
	}

	wantRanges := []string{
		"",
		fmt.Sprintf("bytes=%d-", chunkSize),
		fmt.Sprintf("bytes=%d-", 2*chunkSize),
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff(wantRanges, gotRanges); diff != "" {
		t.Errorf("Range headers mismatch (-want +got):\n%s", diff)
	}
	if !bytes.Equal(got, randomBytes3MiB) {
		t.Errorf("io.ReadAll() len = %d, want %d (matching randomBytes3MiB)", len(got), len(randomBytes3MiB))
	}

	// Verify TTFB and mid-stream body stall timeouts are tracked independently.
	hcClient := client.tc.(*httpStorageClient)
	if bodyDelay := hcClient.dynamicReadBodyStallTimeout.getValue("test-bucket"); bodyDelay <= 250*time.Millisecond {
		t.Errorf("dynamicReadBodyStallTimeout.getValue(\"test-bucket\") = %v, want > 250ms", bodyDelay)
	}
	if reqDelay := hcClient.dynamicReadReqStallTimeout.getValue("test-bucket"); reqDelay != 250*time.Millisecond {
		t.Errorf("dynamicReadReqStallTimeout.getValue(\"test-bucket\") = %v, want 250ms", reqDelay)
	}
}

func TestJSONReaderSkipsMidStreamStallDetection(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	payload := []byte("hello-json-midstream-pause")

	hc, closeServer := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload[:10])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Pause longer than ReadStallTimeoutConfig.Min; JSON path should not abort.
		time.Sleep(60 * time.Millisecond)
		_, _ = w.Write(payload[10:])
	})
	defer closeServer()

	client, err := NewClient(ctx,
		option.WithHTTPClient(hc),
		WithJSONReads(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              15 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("storage.NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll: %v", err)
	}
	if gotAttempts := attempts.Load(); gotAttempts != 1 {
		t.Errorf("JSON reader request attempts = %d, want 1", gotAttempts)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("io.ReadAll() = %q, want %q", got, payload)
	}
}

func TestXMLReaderMidStreamStallHonorsRetryNever(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	hc, closeServer := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Length", strconv.Itoa(len(randomBytes3MiB)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(randomBytes3MiB[:64*1024])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	})
	defer closeServer()

	client, err := NewClient(ctx,
		option.WithHTTPClient(hc),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              200 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("storage.NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").Retryer(WithPolicy(RetryNever)).NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	_, err = io.ReadAll(r)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("io.ReadAll() error = %v, want %v", err, context.DeadlineExceeded)
	}
	if gotAttempts := attempts.Load(); gotAttempts != 1 {
		t.Errorf("request attempts with RetryNever = %d, want 1", gotAttempts)
	}
}

func newHTTP2TestServer(t *testing.T, handler http.HandlerFunc) *http.Client {
	t.Helper()
	srv := httptest.NewUnstartedServer(handler)
	srv.EnableHTTP2 = true
	srv.StartTLS()
	tr := &http.Transport{
		TLSClientConfig:   &tls.Config{InsecureSkipVerify: true},
		ForceAttemptHTTP2: true,
	}
	t.Cleanup(func() {
		tr.CloseIdleConnections()
		srv.Close()
	})
	host := srv.Listener.Addr().String()
	return &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			clone := req.Clone(req.Context())
			clone.URL.Scheme = "https"
			clone.URL.Host = host
			return tr.RoundTrip(clone)
		}),
	}
}

func TestXMLReaderHTTP2EOFStallFullRead(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	payload := bytes.Repeat([]byte("z"), 64*1024)
	crcHeader := "crc32c=" + encodeUint32(crc32.Checksum(payload, crc32cTable))
	var (
		mu         sync.Mutex
		gotRanges  []string
		protoMajor atomic.Int32
	)

	hc := newHTTP2TestServer(t, func(w http.ResponseWriter, r *http.Request) {
		protoMajor.Store(int32(r.ProtoMajor))
		rangeHdr := r.Header.Get("Range")
		mu.Lock()
		gotRanges = append(gotRanges, rangeHdr)
		mu.Unlock()

		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("X-Goog-Hash", crcHeader)
		if rangeHdr == "" {
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(payload)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			// Stall before returning so HTTP/2 END_STREAM is never sent.
			<-r.Context().Done()
			return
		}
		http.ServeContent(w, r, "obj", time.Now(), bytes.NewReader(payload))
	})

	client, err := NewClient(ctx,
		option.WithHTTPClient(hc),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              200 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll() error = %v, want nil", err)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("io.ReadAll() len = %d, want %d", len(got), len(payload))
	}
	if gotProto := protoMajor.Load(); gotProto != 2 {
		t.Errorf("request ProtoMajor = %d, want 2", gotProto)
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]string{""}, gotRanges); diff != "" {
		t.Errorf("Range headers mismatch (-want +got):\n%s", diff)
	}
}

func TestXMLReaderHTTP2EOFStallRangeRead(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	fullObj := bytes.Repeat([]byte("r"), 1000)
	const (
		offset = int64(100)
		length = int64(500)
	)
	want := fullObj[offset : offset+length]
	var (
		mu        sync.Mutex
		gotRanges []string
	)

	hc := newHTTP2TestServer(t, func(w http.ResponseWriter, r *http.Request) {
		rangeHdr := r.Header.Get("Range")
		mu.Lock()
		gotRanges = append(gotRanges, rangeHdr)
		mu.Unlock()

		w.Header().Set("X-Goog-Generation", "1")
		if rangeHdr == "bytes=100-599" {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes 100-599/%d", len(fullObj)))
			w.Header().Set("Content-Length", strconv.Itoa(len(want)))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(want)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			// Stall before HTTP/2 END_STREAM; reader must not reopen with bytes=600-599.
			<-r.Context().Done()
			return
		}
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
	})

	client, err := NewClient(ctx,
		option.WithHTTPClient(hc),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              200 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").NewRangeReader(ctx, offset, length)
	if err != nil {
		t.Fatalf("NewRangeReader: %v", err)
	}
	defer r.Close()

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll() error = %v, want nil", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("io.ReadAll() len = %d, want %d", len(got), len(want))
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]string{"bytes=100-599"}, gotRanges); diff != "" {
		t.Errorf("Range headers mismatch (-want +got):\n%s", diff)
	}
}

func TestXMLReaderHealthyBudgetExhaustedAtEOF(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	payload := bytes.Repeat([]byte("a"), 64*1024)
	var (
		mu        sync.Mutex
		gotRanges []string
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rangeHdr := r.Header.Get("Range")
		mu.Lock()
		gotRanges = append(gotRanges, rangeHdr)
		mu.Unlock()

		w.Header().Set("X-Goog-Generation", "1")
		if rangeHdr == "" {
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(payload[:32*1024])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(35 * time.Millisecond)
			_, _ = w.Write(payload[32*1024:])
			return
		}
		http.ServeContent(w, r, "obj", time.Now(), bytes.NewReader(payload))
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.5,
			Min:              20 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	// Raise bucket body delay comfortably above 35ms initially, then shrink it
	// back to Min (20ms) right after all payload bytes are read so
	// timeSpentInChunk >= currentTimeout when the caller reads for io.EOF.
	hcClient := client.tc.(*httpStorageClient)
	for hcClient.dynamicReadBodyStallTimeout.getValue("test-bucket") < 200*time.Millisecond {
		hcClient.dynamicReadBodyStallTimeout.increase("test-bucket")
	}

	r, err := client.Bucket("test-bucket").Object("test-object").NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	buf := make([]byte, len(payload))
	if _, err := io.ReadFull(r, buf); err != nil {
		t.Fatalf("io.ReadFull() error = %v, want nil", err)
	}
	for hcClient.dynamicReadBodyStallTimeout.getValue("test-bucket") > 20*time.Millisecond {
		hcClient.dynamicReadBodyStallTimeout.decrease("test-bucket")
	}

	// Next Read must observe io.EOF immediately without reopening for 416.
	n, err := r.Read(make([]byte, 16))
	if n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("r.Read() after full payload = (%d, %v), want (0, %v)", n, err, io.EOF)
	}
	if !bytes.Equal(buf, payload) {
		t.Errorf("read payload mismatch")
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]string{""}, gotRanges); diff != "" {
		t.Errorf("Range headers mismatch (-want +got):\n%s", diff)
	}
}

func TestXMLReaderMidStreamStallHonorsMaxAttempts(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")

	tests := []struct {
		name           string
		maxAttempts    int
		firstHeader503 bool
		wantAttempts   int32
	}{
		{
			name:         "max_attempts_1",
			maxAttempts:  1,
			wantAttempts: 1,
		},
		{
			name:         "max_attempts_2_zero_progress",
			maxAttempts:  2,
			wantAttempts: 2,
		},
		{
			name:           "max_attempts_2_header_503_then_body_stall",
			maxAttempts:    2,
			firstHeader503: true,
			wantAttempts:   2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()

			var attempts atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempt := attempts.Add(1)
				if tc.firstHeader503 && attempt == 1 {
					w.WriteHeader(http.StatusServiceUnavailable)
					return
				}
				w.Header().Set("X-Goog-Generation", "1")
				w.Header().Set("Content-Length", "65536")
				w.WriteHeader(http.StatusOK)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				<-r.Context().Done()
			}))
			defer srv.Close()

			client, err := NewClient(ctx,
				option.WithEndpoint(srv.URL+"/storage/v1/"),
				option.WithoutAuthentication(),
				experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
					TargetPercentile: 0.99,
					Min:              60 * time.Millisecond,
				}),
			)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer client.Close()

			obj := client.Bucket("test-bucket").Object("test-object").Retryer(
				WithMaxAttempts(tc.maxAttempts),
				WithBackoff(gax.Backoff{Initial: 10 * time.Millisecond, Max: 20 * time.Millisecond, Multiplier: 2}),
			)
			r, err := obj.NewReader(ctx)
			if err != nil {
				t.Fatalf("NewReader: %v", err)
			}
			defer r.Close()

			_, err = io.ReadAll(r)
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("io.ReadAll() error = %v, want %v", err, context.DeadlineExceeded)
			}
			if got := attempts.Load(); got != tc.wantAttempts {
				t.Errorf("HTTP attempts with WithMaxAttempts(%d) = %d, want %d", tc.maxAttempts, got, tc.wantAttempts)
			}
		})
	}
}

func TestXMLReaderMidStreamStallHonorsMaxRetryDuration(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("Content-Length", "65536")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              50 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	obj := client.Bucket("test-bucket").Object("test-object").Retryer(
		WithMaxRetryDuration(120*time.Millisecond),
		WithBackoff(gax.Backoff{Initial: 10 * time.Millisecond, Max: 20 * time.Millisecond, Multiplier: 2}),
	)
	r, err := obj.NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	start := time.Now()
	_, err = io.ReadAll(r)
	elapsed := time.Since(start)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("io.ReadAll() error = %v, want %v", err, context.DeadlineExceeded)
	}
	if ctx.Err() != nil {
		t.Fatalf("caller ctx expired before WithMaxRetryDuration took effect: %v", ctx.Err())
	}
	if elapsed >= 2*time.Second {
		t.Errorf("io.ReadAll() duration = %v, want prompt stop near 120ms maxRetryDuration", elapsed)
	}
	if got := attempts.Load(); got < 1 {
		t.Errorf("HTTP attempts = %d, want >= 1", got)
	}
}

func TestXMLReaderMidStreamStallHonorsErrorFunc(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("Content-Length", "65536")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, 1024))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              60 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	var predicateCalled atomic.Bool
	obj := client.Bucket("test-bucket").Object("test-object").Retryer(
		WithErrorFunc(func(err error) bool {
			predicateCalled.Store(true)
			return false
		}),
	)
	r, err := obj.NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	_, err = io.ReadAll(r)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("io.ReadAll() error = %v, want %v", err, context.DeadlineExceeded)
	}
	if !predicateCalled.Load() {
		t.Errorf("custom WithErrorFunc predicate was not called on mid-stream stall")
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("HTTP attempts when WithErrorFunc returns false = %d, want 1", got)
	}
}

func TestXMLReaderDecompressiveTranscodingReopenStall(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	payload := bytes.Repeat([]byte("g"), 64*1024)
	var attempts atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := attempts.Add(1)
		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("X-Goog-Stored-Content-Encoding", "gzip")
		w.WriteHeader(http.StatusOK)
		switch attempt {
		case 1:
			// Deliver 16 KiB and stall mid-stream.
			_, _ = w.Write(payload[:16*1024])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
		case 2:
			// On reopen, stall halfway through discarding the first 16 KiB.
			_, _ = w.Write(payload[:8*1024])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
		default:
			_, _ = w.Write(payload)
		}
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              60 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	obj := client.Bucket("test-bucket").Object("test-object").Retryer(
		WithBackoff(gax.Backoff{Initial: 10 * time.Millisecond, Max: 20 * time.Millisecond, Multiplier: 2}),
	)
	r, err := obj.NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll() error = %v, want nil", err)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("io.ReadAll() len = %d, want %d", len(got), len(payload))
	}
	if gotAttempts := attempts.Load(); gotAttempts != 3 {
		t.Errorf("HTTP attempts = %d, want 3", gotAttempts)
	}
}

func TestXMLReaderNilRequestFallbackTargetMetric(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	mr := sdkmetric.NewManualReader()
	provider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(mr))
	defer provider.Shutdown(context.Background())

	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: 65536,
			Header:        http.Header{"X-Goog-Generation": {"1"}},
			Body:          newBlockingReader(),
			// Deliberately leave Request nil to verify c.xmlHost fallback.
		}, nil
	})

	client, err := NewClient(ctx,
		option.WithHTTPClient(&http.Client{Transport: rt}),
		option.WithEndpoint("https://custom-gcs.example.com:443/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithOtelMetrics(),
		experimental.WithOtelDebugMetrics(),
		experimental.WithMeterProvider(provider),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              30 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").Retryer(WithPolicy(RetryNever)).NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	if _, err := io.ReadAll(r); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("io.ReadAll() error = %v, want %v", err, context.DeadlineExceeded)
	}

	var rm metricdata.ResourceMetrics
	if err := mr.Collect(ctx, &rm); err != nil {
		t.Fatalf("mr.Collect() error = %v, want nil", err)
	}
	var gotAddr string
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			if m.Name == "gcp.storage.client.stall.duration" {
				hist := m.Data.(metricdata.Histogram[float64])
				if len(hist.DataPoints) > 0 {
					gotAddr = getHistAttr(hist.DataPoints[0], "server.address")
				}
			}
		}
	}
	if want := "custom-gcs.example.com"; gotAddr != want {
		t.Errorf("stall.duration server.address = %q, want %q", gotAddr, want)
	}
}

func TestXMLReaderMidStreamStallDisabledWithoutDevEnvVar(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	payload := []byte("hello-midstream-default-off")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload[:10])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Pause longer than ReadStallTimeoutConfig.Min; without the dev env var,
		// mid-stream body stall detection must remain disabled.
		time.Sleep(70 * time.Millisecond)
		_, _ = w.Write(payload[10:])
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              20 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	hcClient := client.tc.(*httpStorageClient)
	if hcClient.dynamicReadReqStallTimeout == nil {
		t.Errorf("dynamicReadReqStallTimeout = nil, want non-nil")
	}
	if hcClient.dynamicReadBodyStallTimeout != nil {
		t.Errorf("dynamicReadBodyStallTimeout = %v, want nil when %s is unset", hcClient.dynamicReadBodyStallTimeout, storageMidStreamReadStallDevVar)
	}

	r, err := client.Bucket("test-bucket").Object("test-object").NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("io.ReadAll() = %q, want %q", got, payload)
	}
	if gotAttempts := attempts.Load(); gotAttempts != 1 {
		t.Errorf("HTTP attempts = %d, want 1", gotAttempts)
	}
}

func TestXMLReaderHTTP2HealthyNoCancel(t *testing.T) {
	for _, tc := range []struct {
		name        string
		enableStall bool
	}{
		{name: "stall_detection_disabled", enableStall: false},
		{name: "stall_detection_enabled", enableStall: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableStall {
				t.Setenv(storageMidStreamReadStallDevVar, "true")
			} else {
				t.Setenv(storageMidStreamReadStallDevVar, "")
			}
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()

			payload := bytes.Repeat([]byte("h"), 64*1024)
			var (
				canceled   atomic.Int32
				protoMajor atomic.Int32
				mu         sync.Mutex
				doneCh     chan struct{}
			)

			hc := newHTTP2TestServer(t, func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				ch := doneCh
				mu.Unlock()
				defer close(ch)

				protoMajor.Store(int32(r.ProtoMajor))
				w.Header().Set("X-Goog-Generation", "1")
				w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(payload)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				// Delay handler return slightly so END_STREAM arrives in a
				// separate HTTP/2 frame after the final DATA frame.
				timer := time.NewTimer(5 * time.Millisecond)
				defer timer.Stop()
				select {
				case <-r.Context().Done():
					canceled.Add(1)
				case <-timer.C:
					if r.Context().Err() != nil {
						canceled.Add(1)
					}
				}
			})

			opts := []option.ClientOption{option.WithHTTPClient(hc)}
			if tc.enableStall {
				opts = append(opts, experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
					TargetPercentile: 0.99,
					Min:              200 * time.Millisecond,
				}))
			}
			client, err := NewClient(ctx, opts...)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer client.Close()

			const runs = 5
			for i := 0; i < runs; i++ {
				ch := make(chan struct{})
				mu.Lock()
				doneCh = ch
				mu.Unlock()

				r, err := client.Bucket("test-bucket").Object("test-object").NewReader(ctx)
				if err != nil {
					t.Fatalf("run %d NewReader: %v", i, err)
				}
				got, err := io.ReadAll(r)
				r.Close()
				<-ch
				if err != nil {
					t.Fatalf("run %d io.ReadAll: %v", i, err)
				}
				if !bytes.Equal(got, payload) {
					t.Fatalf("run %d payload len = %d, want %d", i, len(got), len(payload))
				}
			}
			if gotProto := protoMajor.Load(); gotProto != 2 {
				t.Errorf("request ProtoMajor = %d, want 2", gotProto)
			}
			if gotCanceled := canceled.Load(); gotCanceled != 0 {
				t.Errorf("server saw RST_STREAM(CANCEL) on %d/%d healthy HTTP/2 downloads, want 0", gotCanceled, runs)
			}
		})
	}
}

func TestXMLReaderHTTP1Chunked206KeepAliveReuse(t *testing.T) {
	for _, tc := range []struct {
		name        string
		enableStall bool
	}{
		{name: "stall_detection_disabled", enableStall: false},
		{name: "stall_detection_enabled", enableStall: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableStall {
				t.Setenv(storageMidStreamReadStallDevVar, "true")
			} else {
				t.Setenv(storageMidStreamReadStallDevVar, "")
			}
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()

			payload := bytes.Repeat([]byte("k"), 100)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("X-Goog-Generation", "1")
				w.Header().Set("Content-Range", "bytes 0-99/1000")
				w.WriteHeader(http.StatusPartialContent)
				_, _ = w.Write(payload)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				// Delay final 0-length chunk briefly so keep-alive requires draining EOF.
				time.Sleep(5 * time.Millisecond)
			}))
			defer srv.Close()

			opts := []option.ClientOption{
				option.WithEndpoint(srv.URL + "/storage/v1/"),
				option.WithoutAuthentication(),
			}
			if tc.enableStall {
				opts = append(opts, experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
					TargetPercentile: 0.99,
					Min:              200 * time.Millisecond,
				}))
			}
			client, err := NewClient(ctx, opts...)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer client.Close()

			var (
				mu     sync.Mutex
				reused []bool
			)
			traceCtx := httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
				GotConn: func(info httptrace.GotConnInfo) {
					mu.Lock()
					reused = append(reused, info.Reused)
					mu.Unlock()
				},
			})

			for i := 0; i < 3; i++ {
				r, err := client.Bucket("test-bucket").Object("test-object").NewRangeReader(traceCtx, 0, 100)
				if err != nil {
					t.Fatalf("iter %d NewRangeReader: %v", i, err)
				}
				got, err := io.ReadAll(r)
				r.Close()
				if err != nil {
					t.Fatalf("iter %d io.ReadAll: %v", i, err)
				}
				if !bytes.Equal(got, payload) {
					t.Fatalf("iter %d payload len = %d, want %d", i, len(got), len(payload))
				}
			}

			mu.Lock()
			defer mu.Unlock()
			want := []bool{false, true, true}
			if diff := cmp.Diff(want, reused); diff != "" {
				t.Errorf("GotConnInfo.Reused mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestXMLReaderMaxRetryDurationExpiredBeforeReopen(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("Content-Length", "65536")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, 1024))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              30 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").Retryer(
		WithMaxRetryDuration(10*time.Millisecond),
		WithErrorFunc(func(err error) bool {
			// Consume more than maxRetryDuration inside the retry predicate.
			time.Sleep(15 * time.Millisecond)
			return true
		}),
	).NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	_, err = io.ReadAll(r)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("io.ReadAll() error = %v, want wrapping %v", err, context.DeadlineExceeded)
	}
	if !strings.Contains(err.Error(), "retry deadline of 10ms reached") {
		t.Errorf("io.ReadAll() error = %q, want substring %q", err.Error(), "retry deadline of 10ms reached")
	}
	if strings.Contains(err.Error(), "request not sent") {
		t.Errorf("io.ReadAll() error = %q, must not report request not sent", err.Error())
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("HTTP attempts = %d, want 1", got)
	}
}

func TestXMLReaderMaxRetryDurationIdleBeforeFirstRead(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	payload := bytes.Repeat([]byte("p"), 16*1024)
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := attempts.Add(1)
		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
		w.WriteHeader(http.StatusOK)
		if attempt == 1 {
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
			return
		}
		_, _ = w.Write(payload)
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              40 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").Retryer(
		WithMaxRetryDuration(250*time.Millisecond),
		WithBackoff(gax.Backoff{Initial: 10 * time.Millisecond, Max: 20 * time.Millisecond, Multiplier: 2}),
	).NewReader(ctx)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer r.Close()

	// Pause longer than maxRetryDuration before the first Read. Because
	// NewReader succeeded cleanly without header retries, this idle pause must
	// not consume the retry duration window for a subsequent body stall.
	time.Sleep(300 * time.Millisecond)

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll() error = %v, want nil", err)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("io.ReadAll() len = %d, want %d", len(got), len(payload))
	}
	if gotAttempts := attempts.Load(); gotAttempts != 2 {
		t.Errorf("HTTP attempts = %d, want 2", gotAttempts)
	}
}

func TestXMLReaderMaxAttemptsAndMaxRetryDurationAttribution(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              500 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	_, err = client.Bucket("test-bucket").Object("test-object").Retryer(
		WithMaxAttempts(2),
		WithMaxRetryDuration(160*time.Millisecond),
		WithBackoff(gax.Backoff{Initial: time.Millisecond, Max: time.Millisecond, Multiplier: 1}),
	).NewReader(ctx)
	if err == nil {
		t.Fatal("NewReader() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "storage: retry failed after 2 attempts") {
		t.Errorf("NewReader() error = %q, want substring %q", err.Error(), "storage: retry failed after 2 attempts")
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("HTTP attempts = %d, want 2", got)
	}
}

func TestXMLReaderSuffixRangeDecompressiveTranscodingClearsStaleRange(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "true")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	payload := bytes.Repeat([]byte("s"), 200)
	var (
		mu        sync.Mutex
		gotRanges []string
		attempts  atomic.Int32
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := attempts.Add(1)
		mu.Lock()
		gotRanges = append(gotRanges, r.Header.Get("Range"))
		mu.Unlock()

		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("X-Goog-Stored-Content-Encoding", "gzip")
		w.WriteHeader(http.StatusOK)
		if attempt == 1 {
			// Deliver exactly -offset (100) bytes so start = -100 + 100 = 0 on reopen.
			_, _ = w.Write(payload[:100])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
			return
		}
		_, _ = w.Write(payload)
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
		experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
			TargetPercentile: 0.99,
			Min:              50 * time.Millisecond,
		}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	r, err := client.Bucket("test-bucket").Object("test-object").NewRangeReader(ctx, -100, -1)
	if err != nil {
		t.Fatalf("NewRangeReader: %v", err)
	}
	defer r.Close()

	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("io.ReadAll() error = %v, want nil", err)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("io.ReadAll() len = %d, want %d", len(got), len(payload))
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]string{"bytes=-100", ""}, gotRanges); diff != "" {
		t.Errorf("Range headers mismatch (-want +got):\n%s", diff)
	}
}

func TestXMLReaderHTTP2MidStreamErrorWithShouldRetry(t *testing.T) {
	for _, tc := range []struct {
		name        string
		useJSON     bool
		enableStall bool
	}{
		{name: "xml_dev_off", useJSON: false, enableStall: false},
		{name: "json_dev_off", useJSON: true, enableStall: false},
		{name: "xml_dev_on", useJSON: false, enableStall: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableStall {
				t.Setenv(storageMidStreamReadStallDevVar, "true")
			} else {
				t.Setenv(storageMidStreamReadStallDevVar, "")
			}
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()

			const firstChunk = 16 * 1024
			payload := bytes.Repeat([]byte("e"), 64*1024)
			var (
				attempts   atomic.Int32
				protoMajor atomic.Int32
			)

			hc := newHTTP2TestServer(t, func(w http.ResponseWriter, r *http.Request) {
				protoMajor.Store(int32(r.ProtoMajor))
				attempt := attempts.Add(1)
				w.Header().Set("X-Goog-Generation", "1")
				if attempt == 1 {
					w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write(payload[:firstChunk])
					if f, ok := w.(http.Flusher); ok {
						f.Flush()
					}
					// Abort the HTTP/2 stream mid-body to emit RST_STREAM(INTERNAL_ERROR).
					panic(http.ErrAbortHandler)
				}
				rem := payload[firstChunk:]
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", firstChunk, len(payload)-1, len(payload)))
				w.Header().Set("Content-Length", strconv.Itoa(len(rem)))
				w.WriteHeader(http.StatusPartialContent)
				_, _ = w.Write(rem)
			})

			opts := []option.ClientOption{option.WithHTTPClient(hc)}
			if tc.useJSON {
				opts = append(opts, WithJSONReads())
			}
			if tc.enableStall {
				opts = append(opts, experimental.WithReadStallTimeout(&experimental.ReadStallTimeoutConfig{
					TargetPercentile: 0.99,
					Min:              200 * time.Millisecond,
				}))
			}
			client, err := NewClient(ctx, opts...)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer client.Close()

			r, err := client.Bucket("test-bucket").Object("test-object").Retryer(
				WithErrorFunc(ShouldRetry),
			).NewReader(ctx)
			if err != nil {
				t.Fatalf("NewReader: %v", err)
			}
			defer r.Close()

			got, err := io.ReadAll(r)
			if err != nil {
				t.Fatalf("io.ReadAll() error = %v, want nil", err)
			}
			if !bytes.Equal(got, payload) {
				t.Errorf("io.ReadAll() len = %d, want %d", len(got), len(payload))
			}
			if gotProto := protoMajor.Load(); gotProto != 2 {
				t.Errorf("request ProtoMajor = %d, want 2", gotProto)
			}
			if gotAttempts := attempts.Load(); gotAttempts != 2 {
				t.Errorf("HTTP attempts = %d, want 2", gotAttempts)
			}
		})
	}
}

func TestXMLReaderDefaultOffShortMaxRetryDurationZeroProgressReopen(t *testing.T) {
	t.Setenv(storageMidStreamReadStallDevVar, "")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	payload := bytes.Repeat([]byte("d"), 4096)
	var attemptInTrial atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := attemptInTrial.Add(1)
		w.Header().Set("X-Goog-Generation", "1")
		w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
		w.WriteHeader(http.StatusOK)
		if attempt == 1 {
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			// Drop connection at 0 bytes to trigger io.ErrUnexpectedEOF mid-stream.
			panic(http.ErrAbortHandler)
		}
		_, _ = w.Write(payload)
	}))
	defer srv.Close()

	client, err := NewClient(ctx,
		option.WithEndpoint(srv.URL+"/storage/v1/"),
		option.WithoutAuthentication(),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	// With GO_STORAGE_DEV_HTTP_MIDSTREAM_READ_STALL unset, default backoff +
	// short WithMaxRetryDuration must preserve upstream's immediate reopen on
	// 0-byte mid-stream drops across repeated trials.
	for i := 0; i < 10; i++ {
		attemptInTrial.Store(0)
		r, err := client.Bucket("test-bucket").Object("test-object").Retryer(
			WithMaxRetryDuration(150 * time.Millisecond),
		).NewReader(ctx)
		if err != nil {
			t.Fatalf("trial %d NewReader: %v", i, err)
		}
		got, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatalf("trial %d io.ReadAll() error = %v, want nil", i, err)
		}
		if !bytes.Equal(got, payload) {
			t.Fatalf("trial %d payload len = %d, want %d", i, len(got), len(payload))
		}
	}
}
