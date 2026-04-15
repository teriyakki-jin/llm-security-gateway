// Package proxy implements the PQC reverse proxy that decrypts incoming
// requests and forwards them plaintext to the internal gateway service.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/llm-security-gateway/pqc-proxy/internal/kem"
	"github.com/llm-security-gateway/pqc-proxy/internal/transport"
)

// PQCReverseProxy handles PQC handshakes and proxies decrypted traffic
// to the downstream gateway.
//
// Session lifecycle:
//  1. Client GETs /pqc/keys → receives server public keys.
//  2. Client POSTs /pqc/handshake → 4-way handshake, session stored in Redis.
//  3. Client POSTs /pqc/finished → Finished MAC verified, session activated.
//  4. Subsequent requests use X-Session-ID to resume without re-handshake.
//  5. Sessions expire after 1 hour (sliding TTL on each use).
type PQCReverseProxy struct {
	target       *url.URL
	proxy        *httputil.ReverseProxy
	// hotSessions caches recently used sessions in memory to avoid Redis RTT
	// on every request. Evicted after 5 minutes of local inactivity.
	hotSessions  sync.Map // sessionID → *transport.SessionState
	sessionStore *transport.SessionStore
	log          *zap.Logger
}

// NewPQCReverseProxy creates a proxy that forwards decrypted requests to targetURL.
// store may be nil — in that case sessions are kept in memory only (no resumption across restarts).
func NewPQCReverseProxy(targetURL string, store *transport.SessionStore, log *zap.Logger) (*PQCReverseProxy, error) {
	target, err := url.Parse(targetURL)
	if err != nil {
		return nil, fmt.Errorf("parse target url: %w", err)
	}

	p := &PQCReverseProxy{
		target:       target,
		sessionStore: store,
		log:          log,
	}

	p.proxy = &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host
			req.Header.Set("X-Forwarded-By", "pqc-proxy")
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Error("proxy upstream error", zap.Error(err), zap.String("path", r.URL.Path))
			http.Error(w, "upstream error", http.StatusBadGateway)
		},
		ModifyResponse: func(resp *http.Response) error {
			resp.Header.Set("X-PQC-Protected", "ML-KEM-768+X25519")
			return nil
		},
	}

	return p, nil
}

// HandshakeHandler handles POST /pqc/handshake requests.
// Clients initiate the 4-way handshake here before sending encrypted requests.
func (p *PQCReverseProxy) HandshakeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var clientHello transport.ClientHello
	if err := json.NewDecoder(r.Body).Decode(&clientHello); err != nil {
		p.log.Warn("decode client hello failed", zap.Error(err))
		http.Error(w, "invalid client hello", http.StatusBadRequest)
		return
	}

	serverHello, session, err := transport.ServerHandshake(&clientHello)
	if err != nil {
		p.log.Warn("server handshake failed", zap.Error(err))
		http.Error(w, "handshake failed", http.StatusBadRequest)
		return
	}

	sessionID := fmt.Sprintf("%x", clientHello.Nonce)

	// Store in hot (in-memory) cache for immediate use.
	p.hotSessions.Store(sessionID, session)

	// Persist to Redis for session resumption across restarts / instances.
	if p.sessionStore != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := p.sessionStore.Save(ctx, sessionID, session); err != nil {
			p.log.Warn("session_store_save_failed", zap.String("session", sessionID), zap.Error(err))
			// Non-fatal: session still works from hot cache.
		}
	}

	// Evict from hot cache after 1 hour regardless of Redis state.
	go func() {
		time.Sleep(time.Hour)
		p.hotSessions.Delete(sessionID)
	}()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Session-ID", sessionID)
	if err := json.NewEncoder(w).Encode(serverHello); err != nil {
		p.log.Error("encode server hello", zap.Error(err))
	}
}

// FinishedHandler handles POST /pqc/finished — the client sends the HMAC-verified Finished message.
func (p *PQCReverseProxy) FinishedHandler(w http.ResponseWriter, r *http.Request) {
	sessionID := r.Header.Get("X-Session-ID")
	if sessionID == "" {
		http.Error(w, "missing X-Session-ID", http.StatusBadRequest)
		return
	}

	raw, ok := p.hotSessions.Load(sessionID)
	if !ok {
		http.Error(w, "session not found — complete handshake first", http.StatusUnauthorized)
		return
	}
	session := raw.(*transport.SessionState)

	var finished transport.Finished
	if err := json.NewDecoder(r.Body).Decode(&finished); err != nil {
		http.Error(w, "invalid finished message", http.StatusBadRequest)
		return
	}

	if err := transport.VerifyFinished(session, &finished); err != nil {
		p.log.Warn("finished verification failed", zap.String("session", sessionID), zap.Error(err))
		p.hotSessions.Delete(sessionID)
		if p.sessionStore != nil {
			p.sessionStore.Delete(context.Background(), sessionID) //nolint:errcheck
		}
		http.Error(w, "handshake verification failed", http.StatusUnauthorized)
		return
	}

	p.log.Info("handshake complete", zap.String("session", sessionID))
	w.WriteHeader(http.StatusOK)
}

// ProxyHandler decrypts an incoming encrypted request and forwards it to the gateway.
// Expects the request body to be AES-256-GCM ciphertext produced by the client SDK.
func (p *PQCReverseProxy) ProxyHandler(w http.ResponseWriter, r *http.Request) {
	sessionID := r.Header.Get("X-Session-ID")
	if sessionID == "" {
		http.Error(w, "missing X-Session-ID", http.StatusUnauthorized)
		return
	}

	session, err := p.loadSession(r.Context(), sessionID)
	if err != nil {
		p.log.Error("session load error", zap.String("session", sessionID), zap.Error(err))
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	if session == nil {
		http.Error(w, "session not found or expired — re-handshake required", http.StatusUnauthorized)
		return
	}

	// Read and decrypt request body.
	encrypted, err := io.ReadAll(io.LimitReader(r.Body, 10<<20)) // 10 MB limit
	if err != nil {
		http.Error(w, "read body failed", http.StatusBadRequest)
		return
	}

	plaintext, err := session.Decrypt(encrypted)
	if err != nil {
		p.log.Warn("decrypt failed", zap.String("session", sessionID), zap.Error(err))
		http.Error(w, "decryption failed", http.StatusBadRequest)
		return
	}

	// Replace request body with plaintext before forwarding.
	r.Body = io.NopCloser(bytes.NewReader(plaintext))
	r.ContentLength = int64(len(plaintext))
	r.Header.Del("X-Session-ID")

	// Capture the response to encrypt it before sending back.
	rec := &responseRecorder{header: make(http.Header)}
	p.proxy.ServeHTTP(rec, r)

	// Encrypt the upstream response.
	encryptedResp, err := session.Encrypt(rec.body)
	if err != nil {
		p.log.Error("encrypt response failed", zap.Error(err))
		http.Error(w, "response encryption failed", http.StatusInternalServerError)
		return
	}

	for k, v := range rec.header {
		w.Header()[k] = v
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(rec.status)
	w.Write(encryptedResp) //nolint:errcheck
}

// KeyServerHandler returns the proxy's current ML-KEM public key for key exchange bootstrapping.
func (p *PQCReverseProxy) KeyServerHandler(serverKP *kem.HybridKeyPair) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := struct {
			X25519Pub []byte `json:"x25519_pub"`
			MLKEMPub  []byte `json:"mlkem_pub"`
		}{
			X25519Pub: serverKP.X25519Public[:],
			MLKEMPub:  serverKP.MLKEM.PublicKey,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp) //nolint:errcheck
	}
}

// loadSession looks up a session: hot cache first, then Redis (session resumption).
// Returns (nil, nil) if not found anywhere.
func (p *PQCReverseProxy) loadSession(ctx context.Context, sessionID string) (*transport.SessionState, error) {
	// 1. Hot cache hit — no Redis RTT.
	if raw, ok := p.hotSessions.Load(sessionID); ok {
		return raw.(*transport.SessionState), nil
	}

	// 2. Redis lookup — session resumption after proxy restart or cross-instance.
	if p.sessionStore == nil {
		return nil, nil
	}

	rctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond) // strict SLO
	defer cancel()

	state, err := p.sessionStore.Load(rctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("redis session load: %w", err)
	}
	if state == nil {
		return nil, nil // not found
	}

	// Warm the hot cache so subsequent requests skip Redis.
	p.hotSessions.Store(sessionID, state)
	p.log.Info("session_resumed_from_redis", zap.String("session", sessionID))
	return state, nil
}

// responseRecorder captures an HTTP response for post-processing.
type responseRecorder struct {
	header http.Header
	body   []byte
	status int
}

func (r *responseRecorder) Header() http.Header { return r.header }
func (r *responseRecorder) WriteHeader(code int) { r.status = code }
func (r *responseRecorder) Write(b []byte) (int, error) {
	r.body = append(r.body, b...)
	return len(b), nil
}
