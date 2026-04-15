package transport_test

import (
	"bytes"
	"testing"
	"time"

	"github.com/llm-security-gateway/pqc-proxy/internal/kem"
	"github.com/llm-security-gateway/pqc-proxy/internal/transport"
)

// TestFullHandshake verifies the complete 4-way handshake produces matching session keys.
func TestFullHandshake(t *testing.T) {
	clientKP, err := kem.GenerateHybridKeyPair()
	if err != nil {
		t.Fatalf("client keygen: %v", err)
	}

	clientHello, err := transport.NewClientHello(clientKP)
	if err != nil {
		t.Fatalf("NewClientHello: %v", err)
	}

	serverHello, serverSession, err := transport.ServerHandshake(clientHello)
	if err != nil {
		t.Fatalf("ServerHandshake: %v", err)
	}

	clientSession, finished, err := transport.ClientHandshake(clientHello, serverHello, clientKP)
	if err != nil {
		t.Fatalf("ClientHandshake: %v", err)
	}

	if err := transport.VerifyFinished(serverSession, finished); err != nil {
		t.Fatalf("VerifyFinished: %v", err)
	}

	// Both sides must derive the same session key.
	if !bytes.Equal(clientSession.SessionKey(), serverSession.SessionKey()) {
		t.Error("session keys do not match between client and server")
	}
}

// TestHandshake_TamperedServerHello ensures a modified ServerHello causes verification failure.
func TestHandshake_TamperedServerHello(t *testing.T) {
	clientKP, _ := kem.GenerateHybridKeyPair()
	clientHello, _ := transport.NewClientHello(clientKP)
	serverHello, serverSession, _ := transport.ServerHandshake(clientHello)

	// Tamper: flip one byte in the ML-KEM ciphertext.
	serverHello.HybridCT.MLKEMCiphertext[0] ^= 0xFF

	clientSession, finished, err := transport.ClientHandshake(clientHello, serverHello, clientKP)
	if err != nil {
		// Decapsulation failure is acceptable security behavior.
		t.Logf("ClientHandshake failed after tampering (acceptable): %v", err)
		return
	}

	// If client didn't error, VerifyFinished must catch the mismatch.
	if err := transport.VerifyFinished(serverSession, finished); err == nil {
		// Session keys must not match.
		if bytes.Equal(clientSession.SessionKey(), serverSession.SessionKey()) {
			t.Error("tampered handshake produced matching session keys (security failure!)")
		}
	}
}

// TestHandshake_ReplayAttack verifies that replaying the same ClientHello
// produces a different ServerHello (different nonce) and thus different keys.
func TestHandshake_ReplayAttack(t *testing.T) {
	clientKP, _ := kem.GenerateHybridKeyPair()
	clientHello, _ := transport.NewClientHello(clientKP)

	_, session1, _ := transport.ServerHandshake(clientHello)
	_, session2, _ := transport.ServerHandshake(clientHello)

	// Different server nonces → different session keys even for same ClientHello.
	if bytes.Equal(session1.SessionKey(), session2.SessionKey()) {
		t.Error("two ServerHandshake calls produced the same session key (nonce reuse!)")
	}
}

// TestHandshake_WrongClientKP verifies that using a different key pair to process
// ServerHello fails or produces a non-matching session key.
func TestHandshake_WrongClientKP(t *testing.T) {
	clientKP, _ := kem.GenerateHybridKeyPair()
	wrongKP, _ := kem.GenerateHybridKeyPair()

	clientHello, _ := transport.NewClientHello(clientKP)
	serverHello, serverSession, _ := transport.ServerHandshake(clientHello)

	// Process with wrong key pair.
	_, finished, err := transport.ClientHandshake(clientHello, serverHello, wrongKP)
	if err != nil {
		t.Logf("ClientHandshake with wrong KP failed (acceptable): %v", err)
		return
	}

	// If no error, Finished MAC must not verify.
	if err := transport.VerifyFinished(serverSession, finished); err == nil {
		t.Error("VerifyFinished should have failed with wrong key pair")
	}
}

// TestHandshake_UnsupportedVersion verifies that a mismatched version is rejected.
func TestHandshake_UnsupportedVersion(t *testing.T) {
	clientKP, _ := kem.GenerateHybridKeyPair()
	clientHello, _ := transport.NewClientHello(clientKP)
	clientHello.Version = 99 // Unsupported version.

	_, _, err := transport.ServerHandshake(clientHello)
	if err == nil {
		t.Error("expected error for unsupported protocol version, got nil")
	}
}

// TestHandshake_ExpiredClientHello verifies that a stale ClientHello is rejected.
func TestHandshake_ExpiredClientHello(t *testing.T) {
	clientKP, _ := kem.GenerateHybridKeyPair()
	clientHello, _ := transport.NewClientHello(clientKP)
	clientHello.SentAt = time.Now().Add(-time.Minute) // Expired.

	_, _, err := transport.ServerHandshake(clientHello)
	if err == nil {
		t.Error("expected error for expired ClientHello, got nil")
	}
}

func TestSession_EncryptDecrypt(t *testing.T) {
	clientKP, _ := kem.GenerateHybridKeyPair()
	clientHello, _ := transport.NewClientHello(clientKP)
	serverHello, serverSession, _ := transport.ServerHandshake(clientHello)
	clientSession, finished, _ := transport.ClientHandshake(clientHello, serverHello, clientKP)
	transport.VerifyFinished(serverSession, finished) //nolint:errcheck

	plaintext := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`)

	encrypted, err := clientSession.Encrypt(plaintext)
	if err != nil {
		t.Fatalf("Encrypt: %v", err)
	}
	if bytes.Equal(encrypted, plaintext) {
		t.Error("encrypted output equals plaintext")
	}

	decrypted, err := serverSession.Decrypt(encrypted)
	if err != nil {
		t.Fatalf("Decrypt: %v", err)
	}
	if !bytes.Equal(decrypted, plaintext) {
		t.Errorf("decrypted mismatch: got %q want %q", decrypted, plaintext)
	}
}

func TestSession_ReplayRejected(t *testing.T) {
	clientKP, _ := kem.GenerateHybridKeyPair()
	clientHello, _ := transport.NewClientHello(clientKP)
	serverHello, serverSession, _ := transport.ServerHandshake(clientHello)
	clientSession, finished, _ := transport.ClientHandshake(clientHello, serverHello, clientKP)
	transport.VerifyFinished(serverSession, finished) //nolint:errcheck

	plaintext := []byte("test message")
	encrypted, _ := clientSession.Encrypt(plaintext)

	// First decrypt succeeds.
	if _, err := serverSession.Decrypt(encrypted); err != nil {
		t.Fatalf("first Decrypt: %v", err)
	}

	// Second decrypt of same message must fail (replay attack).
	if _, err := serverSession.Decrypt(encrypted); err == nil {
		t.Error("expected replay to be rejected, but Decrypt succeeded")
	}
}

func BenchmarkFullHandshake(b *testing.B) {
	for i := 0; i < b.N; i++ {
		clientKP, _ := kem.GenerateHybridKeyPair()
		clientHello, _ := transport.NewClientHello(clientKP)
		serverHello, serverSession, _ := transport.ServerHandshake(clientHello)
		_, finished, _ := transport.ClientHandshake(clientHello, serverHello, clientKP)
		transport.VerifyFinished(serverSession, finished) //nolint:errcheck
	}
}
