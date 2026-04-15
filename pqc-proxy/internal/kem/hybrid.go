package kem

import (
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"io"

	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/hkdf"
)

const (
	// X25519KeyLen is the length of X25519 public/private keys in bytes.
	X25519KeyLen = 32

	// HybridSharedSecretLen is the final session key length after HKDF combining.
	HybridSharedSecretLen = 32

	hkdfInfo = "llm-security-gateway hybrid kem v1"
)

// HybridKeyPair holds both X25519 and ML-KEM-768 key pairs.
type HybridKeyPair struct {
	X25519Private [X25519KeyLen]byte
	X25519Public  [X25519KeyLen]byte
	MLKEM         *KeyPair
}

// HybridCiphertext holds both the X25519 ephemeral public key and the ML-KEM ciphertext.
type HybridCiphertext struct {
	// X25519EphemeralPub is the encapsulator's ephemeral X25519 public key.
	X25519EphemeralPub [X25519KeyLen]byte
	// MLKEMCiphertext is the ML-KEM-768 ciphertext.
	MLKEMCiphertext []byte
}

// GenerateHybridKeyPair generates a new X25519 + ML-KEM-768 hybrid key pair.
func GenerateHybridKeyPair() (*HybridKeyPair, error) {
	kp := &HybridKeyPair{}

	// Generate X25519 key pair.
	if _, err := io.ReadFull(rand.Reader, kp.X25519Private[:]); err != nil {
		return nil, fmt.Errorf("x25519 random: %w", err)
	}
	// Clamp the private key per RFC 7748.
	kp.X25519Private[0] &= 248
	kp.X25519Private[31] &= 127
	kp.X25519Private[31] |= 64

	pub, err := curve25519.X25519(kp.X25519Private[:], curve25519.Basepoint)
	if err != nil {
		return nil, fmt.Errorf("x25519 pubkey: %w", err)
	}
	copy(kp.X25519Public[:], pub)

	// Generate ML-KEM-768 key pair.
	mlkemKP, err := GenerateKeyPair()
	if err != nil {
		return nil, fmt.Errorf("mlkem keygen: %w", err)
	}
	kp.MLKEM = mlkemKP

	return kp, nil
}

// HybridEncapsulate creates a shared secret and ciphertext for the given recipient public keys.
// It combines X25519 ECDH and ML-KEM-768 shared secrets via HKDF-SHA256.
// Returns (ciphertext, sessionKey, error).
func HybridEncapsulate(recipientX25519Pub [X25519KeyLen]byte, recipientMLKEMPub []byte) (*HybridCiphertext, []byte, error) {
	// Generate ephemeral X25519 key pair for this encapsulation.
	var ephPriv [X25519KeyLen]byte
	if _, err := io.ReadFull(rand.Reader, ephPriv[:]); err != nil {
		return nil, nil, fmt.Errorf("ephemeral x25519 random: %w", err)
	}
	ephPriv[0] &= 248
	ephPriv[31] &= 127
	ephPriv[31] |= 64

	ephPub, err := curve25519.X25519(ephPriv[:], curve25519.Basepoint)
	if err != nil {
		return nil, nil, fmt.Errorf("ephemeral x25519 pubkey: %w", err)
	}

	// X25519 ECDH: ephemeral_private * recipient_public.
	x25519SS, err := curve25519.X25519(ephPriv[:], recipientX25519Pub[:])
	if err != nil {
		return nil, nil, fmt.Errorf("x25519 dh: %w", err)
	}

	// ML-KEM-768 encapsulation.
	mlkemCT, mlkemSS, err := Encapsulate(recipientMLKEMPub)
	if err != nil {
		return nil, nil, fmt.Errorf("mlkem encapsulate: %w", err)
	}

	// Combine both shared secrets via HKDF-SHA256.
	sessionKey, err := combineSecrets(x25519SS, mlkemSS)
	if err != nil {
		return nil, nil, fmt.Errorf("combine secrets: %w", err)
	}

	ct := &HybridCiphertext{
		MLKEMCiphertext: mlkemCT,
	}
	copy(ct.X25519EphemeralPub[:], ephPub)

	return ct, sessionKey, nil
}

// HybridDecapsulate recovers the session key from a HybridCiphertext using the recipient's private keys.
func HybridDecapsulate(ct *HybridCiphertext, x25519Priv [X25519KeyLen]byte, mlkemSecretKey []byte) ([]byte, error) {
	// X25519 ECDH: recipient_private * ephemeral_public.
	x25519SS, err := curve25519.X25519(x25519Priv[:], ct.X25519EphemeralPub[:])
	if err != nil {
		return nil, fmt.Errorf("x25519 dh: %w", err)
	}

	// ML-KEM-768 decapsulation.
	mlkemSS, err := Decapsulate(ct.MLKEMCiphertext, mlkemSecretKey)
	if err != nil {
		return nil, fmt.Errorf("mlkem decapsulate: %w", err)
	}

	// Combine both shared secrets via HKDF-SHA256.
	sessionKey, err := combineSecrets(x25519SS, mlkemSS)
	if err != nil {
		return nil, fmt.Errorf("combine secrets: %w", err)
	}

	return sessionKey, nil
}

// combineSecrets derives a single session key from two shared secrets using HKDF-SHA256.
// IKM = x25519_ss || mlkem_ss ensures both must be correct for the key to match.
func combineSecrets(x25519SS, mlkemSS []byte) ([]byte, error) {
	ikm := make([]byte, len(x25519SS)+len(mlkemSS))
	copy(ikm, x25519SS)
	copy(ikm[len(x25519SS):], mlkemSS)

	reader := hkdf.New(sha256.New, ikm, nil, []byte(hkdfInfo))
	sessionKey := make([]byte, HybridSharedSecretLen)
	if _, err := io.ReadFull(reader, sessionKey); err != nil {
		return nil, fmt.Errorf("hkdf expand: %w", err)
	}

	return sessionKey, nil
}
