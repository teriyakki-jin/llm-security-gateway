// Package transport implements the PQC-secured 4-way handshake protocol.
//
// Protocol flow:
//  1. Client → Server: ClientHello { x25519_pub, mlkem_pub, nonce }
//  2. Server → Client: ServerHello { x25519_pub, mlkem_pub, hybrid_ciphertext, nonce }
//  3. Client: HybridDecapsulate → session_key
//  4. Client → Server: Finished { HMAC-SHA256(session_key, transcript) }
package transport

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/llm-security-gateway/pqc-proxy/internal/kem"
)

const (
	// ProtocolVersion is the current handshake protocol version.
	ProtocolVersion uint8 = 1

	// NonceLen is the length of handshake nonces in bytes.
	NonceLen = 32

	// HandshakeTimeout is the maximum time allowed to complete the handshake.
	HandshakeTimeout = 10 * time.Second
)

// ClientHello is the first message sent by the client.
type ClientHello struct {
	Version      uint8
	X25519Pub    [kem.X25519KeyLen]byte
	MLKEMPub     []byte
	Nonce        [NonceLen]byte
	SentAt       time.Time
}

// ServerHello is the server's response containing encapsulated key material.
type ServerHello struct {
	Version       uint8
	X25519Pub     [kem.X25519KeyLen]byte
	MLKEMPub      []byte
	HybridCT      *kem.HybridCiphertext
	Nonce         [NonceLen]byte
	SentAt        time.Time
}

// Finished is the client's final message proving possession of the session key.
type Finished struct {
	MAC [sha256.Size]byte
}

// HandshakeState holds ephemeral state during handshake processing.
type HandshakeState struct {
	localKeyPair *kem.HybridKeyPair
	sessionKey   []byte
	transcript   []byte
	startedAt    time.Time
}

// NewClientHello creates a ClientHello using the provided key pair.
func NewClientHello(kp *kem.HybridKeyPair) (*ClientHello, error) {
	hello := &ClientHello{
		Version:   ProtocolVersion,
		X25519Pub: kp.X25519Public,
		MLKEMPub:  kp.MLKEM.PublicKey,
		SentAt:    time.Now().UTC(),
	}
	if _, err := io.ReadFull(rand.Reader, hello.Nonce[:]); err != nil {
		return nil, fmt.Errorf("generate client nonce: %w", err)
	}
	return hello, nil
}

// ServerHandshake processes a ClientHello and returns a ServerHello + session state.
// Called by the proxy server.
func ServerHandshake(clientHello *ClientHello) (*ServerHello, *SessionState, error) {
	if err := validateClientHello(clientHello); err != nil {
		return nil, nil, fmt.Errorf("invalid client hello: %w", err)
	}

	// Generate server's own hybrid key pair.
	serverKP, err := kem.GenerateHybridKeyPair()
	if err != nil {
		return nil, nil, fmt.Errorf("server keygen: %w", err)
	}

	// Encapsulate to the client's public keys → produces hybrid ciphertext + session key.
	hybridCT, sessionKey, err := kem.HybridEncapsulate(clientHello.X25519Pub, clientHello.MLKEMPub)
	if err != nil {
		return nil, nil, fmt.Errorf("server encapsulate: %w", err)
	}

	serverHello := &ServerHello{
		Version:   ProtocolVersion,
		X25519Pub: serverKP.X25519Public,
		MLKEMPub:  serverKP.MLKEM.PublicKey,
		HybridCT:  hybridCT,
		SentAt:    time.Now().UTC(),
	}
	if _, err := io.ReadFull(rand.Reader, serverHello.Nonce[:]); err != nil {
		return nil, nil, fmt.Errorf("generate server nonce: %w", err)
	}

	transcript := buildTranscript(clientHello, serverHello)
	session := &SessionState{
		key:       sessionKey,
		createdAt: time.Now().UTC(),
	}
	session.expectedFinishedMAC = computeMAC(sessionKey, transcript)

	return serverHello, session, nil
}

// ClientHandshake processes a ServerHello and derives the session key.
// Called by the client SDK after receiving ServerHello.
func ClientHandshake(clientHello *ClientHello, serverHello *ServerHello, clientKP *kem.HybridKeyPair) (*SessionState, *Finished, error) {
	if err := validateServerHello(serverHello); err != nil {
		return nil, nil, fmt.Errorf("invalid server hello: %w", err)
	}
	if time.Since(clientHello.SentAt) > HandshakeTimeout {
		return nil, nil, errors.New("handshake timeout")
	}

	// Decapsulate the hybrid ciphertext to recover the session key.
	sessionKey, err := kem.HybridDecapsulate(serverHello.HybridCT, clientKP.X25519Private, clientKP.MLKEM.SecretKey)
	if err != nil {
		return nil, nil, fmt.Errorf("client decapsulate: %w", err)
	}

	transcript := buildTranscript(clientHello, serverHello)
	mac := computeMAC(sessionKey, transcript)

	finished := &Finished{}
	copy(finished.MAC[:], mac)

	session := &SessionState{
		key:       sessionKey,
		createdAt: time.Now().UTC(),
	}

	return session, finished, nil
}

// VerifyFinished validates the client's Finished message on the server side.
func VerifyFinished(session *SessionState, finished *Finished) error {
	if !hmac.Equal(finished.MAC[:], session.expectedFinishedMAC) {
		return errors.New("finished MAC mismatch: possible tampering or key mismatch")
	}
	return nil
}

// buildTranscript concatenates handshake messages for MAC binding.
// Binds both nonces to prevent replay across sessions.
func buildTranscript(clientHello *ClientHello, serverHello *ServerHello) []byte {
	t := make([]byte, 0, NonceLen*2+kem.X25519KeyLen*2+len(clientHello.MLKEMPub)+len(serverHello.MLKEMPub)+2)
	t = append(t, clientHello.Version)
	t = append(t, clientHello.Nonce[:]...)
	t = append(t, clientHello.X25519Pub[:]...)
	t = append(t, clientHello.MLKEMPub...)
	t = append(t, serverHello.Version)
	t = append(t, serverHello.Nonce[:]...)
	t = append(t, serverHello.X25519Pub[:]...)
	t = append(t, serverHello.MLKEMPub...)
	return t
}

func computeMAC(key, transcript []byte) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write(transcript)
	return mac.Sum(nil)
}

func validateClientHello(h *ClientHello) error {
	if h.Version != ProtocolVersion {
		return fmt.Errorf("unsupported version %d (expected %d)", h.Version, ProtocolVersion)
	}
	if len(h.MLKEMPub) == 0 {
		return errors.New("mlkem public key is empty")
	}
	if time.Since(h.SentAt) > HandshakeTimeout {
		return errors.New("client hello expired")
	}
	return nil
}

func validateServerHello(h *ServerHello) error {
	if h.Version != ProtocolVersion {
		return fmt.Errorf("unsupported version %d (expected %d)", h.Version, ProtocolVersion)
	}
	if len(h.MLKEMPub) == 0 {
		return errors.New("mlkem public key is empty")
	}
	if h.HybridCT == nil || len(h.HybridCT.MLKEMCiphertext) == 0 {
		return errors.New("hybrid ciphertext is empty")
	}
	return nil
}
