package kem_test

import (
	"bytes"
	"testing"

	"github.com/llm-security-gateway/pqc-proxy/internal/kem"
)

func TestGenerateKeyPair(t *testing.T) {
	kp, err := kem.GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair() error = %v", err)
	}
	if len(kp.PublicKey) == 0 {
		t.Error("public key is empty")
	}
	if len(kp.SecretKey) == 0 {
		t.Error("secret key is empty")
	}
}

func TestEncapsulateDecapsulate(t *testing.T) {
	kp, err := kem.GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair() error = %v", err)
	}

	ct, ss1, err := kem.Encapsulate(kp.PublicKey)
	if err != nil {
		t.Fatalf("Encapsulate() error = %v", err)
	}
	if len(ct) == 0 {
		t.Error("ciphertext is empty")
	}
	if len(ss1) == 0 {
		t.Error("shared secret is empty")
	}

	ss2, err := kem.Decapsulate(ct, kp.SecretKey)
	if err != nil {
		t.Fatalf("Decapsulate() error = %v", err)
	}

	if !bytes.Equal(ss1, ss2) {
		t.Errorf("shared secrets do not match: encap=%x decap=%x", ss1, ss2)
	}
}

func TestEncapsulateDecapsulate_WrongSecretKey(t *testing.T) {
	kp1, _ := kem.GenerateKeyPair()
	kp2, _ := kem.GenerateKeyPair()

	ct, ss1, err := kem.Encapsulate(kp1.PublicKey)
	if err != nil {
		t.Fatalf("Encapsulate() error = %v", err)
	}

	// Decapsulate with the wrong secret key: should not error but produce different shared secret.
	ss2, err := kem.Decapsulate(ct, kp2.SecretKey)
	if err != nil {
		// Some implementations error on invalid key; both outcomes are acceptable security behavior.
		t.Logf("Decapsulate with wrong key returned error (acceptable): %v", err)
		return
	}

	if bytes.Equal(ss1, ss2) {
		t.Error("decapsulation with wrong key produced same shared secret (security failure!)")
	}
}

func TestEncapsulate_EmptyPublicKey(t *testing.T) {
	_, _, err := kem.Encapsulate(nil)
	if err == nil {
		t.Error("expected error for empty public key, got nil")
	}
}

func TestDecapsulate_EmptyCiphertext(t *testing.T) {
	kp, _ := kem.GenerateKeyPair()
	_, err := kem.Decapsulate(nil, kp.SecretKey)
	if err == nil {
		t.Error("expected error for empty ciphertext, got nil")
	}
}

func TestEncapsulateDecapsulate_Repeated(t *testing.T) {
	kp, _ := kem.GenerateKeyPair()
	seen := make(map[string]bool)

	for i := 0; i < 10; i++ {
		ct, ss, err := kem.Encapsulate(kp.PublicKey)
		if err != nil {
			t.Fatalf("iteration %d: Encapsulate() error = %v", i, err)
		}

		key := string(ss)
		if seen[key] {
			t.Errorf("iteration %d: duplicate shared secret (randomness failure!)", i)
		}
		seen[key] = true

		recovered, err := kem.Decapsulate(ct, kp.SecretKey)
		if err != nil {
			t.Fatalf("iteration %d: Decapsulate() error = %v", i, err)
		}
		if !bytes.Equal(ss, recovered) {
			t.Errorf("iteration %d: shared secret mismatch", i)
		}
	}
}

func BenchmarkGenerateKeyPair(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := kem.GenerateKeyPair(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncapsulate(b *testing.B) {
	kp, _ := kem.GenerateKeyPair()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := kem.Encapsulate(kp.PublicKey); err != nil {
			b.Fatal(err)
		}
	}
}
