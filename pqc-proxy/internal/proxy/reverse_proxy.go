// Package proxy implements the PQC reverse proxy that decrypts incoming
// requests and forwards them plaintext to the internal gateway service.
package proxy

import (
	"bytes"
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
type PQCReverseProxy struct {
	target   *url.URL
	proxy    *httputil.ReverseProxy
	sessions sync.Map // requestID → *transport.SessionState
	log      *zap.Logger
}

// NewPQCReverseProxy creates a proxy that forwards decrypted requests to targetURL.
func NewPQCReverseProxy(targetURL string, log *zap.Logger) (*PQCReverseProxy, error) {
	target, err := url.Parse(targetURL)
	if err != nil {
		return nil, fmt.Errorf("parse target url: %w", err)
	}

	p := &PQCReverseProxy{
		target: target,
		log:    log,
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

	// Store session keyed by a session ID derived from the client nonce.
	sessionID := fmt.Sprintf("%x", clientHello.Nonce)
	p.sessions.Store(sessionID, session)

	// Schedule session cleanup after 1 hour.
	go func() {
		time.Sleep(time.Hour)
		p.sessions.Delete(sessionID)
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

	raw, ok := p.sessions.Load(sessionID)
	if !ok {
		http.Error(w, "session not found", http.StatusUnauthorized)
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
		p.sessions.Delete(sessionID)
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

	raw, ok := p.sessions.Load(sessionID)
	if !ok {
		http.Error(w, "session not found or expired", http.StatusUnauthorized)
		return
	}
	session := raw.(*transport.SessionState)

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
