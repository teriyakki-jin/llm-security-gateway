// Package kem provides ML-KEM-768 (NIST FIPS 203) key encapsulation.
// Requires liboqs C library (via CGO). Build inside Docker for proper setup.
package kem

import (
	"errors"
	"fmt"

	"github.com/open-quantum-safe/liboqs-go/oqs"
)

const (
	// AlgorithmMLKEM768 is the NIST FIPS 203 standard algorithm name.
	AlgorithmMLKEM768 = "ML-KEM-768"

	// SharedSecretLen is the length of the shared secret in bytes.
	SharedSecretLen = 32
)

// KeyPair holds an ML-KEM-768 public/secret key pair.
type KeyPair struct {
	PublicKey []byte
	SecretKey []byte
}

// GenerateKeyPair generates a new ML-KEM-768 key pair.
func GenerateKeyPair() (*KeyPair, error) {
	kem := oqs.KeyEncapsulation{}
	if err := kem.Init(AlgorithmMLKEM768, nil); err != nil {
		return nil, fmt.Errorf("kem init: %w", err)
	}
	defer kem.Clean()

	pub, err := kem.GenerateKeyPair()
	if err != nil {
		return nil, fmt.Errorf("generate key pair: %w", err)
	}

	return &KeyPair{
		PublicKey: pub,
		SecretKey: kem.ExportSecretKey(),
	}, nil
}

// Encapsulate generates a shared secret and ciphertext using the recipient's public key.
// Returns (ciphertext, sharedSecret, error).
func Encapsulate(publicKey []byte) (ciphertext []byte, sharedSecret []byte, err error) {
	if len(publicKey) == 0 {
		return nil, nil, errors.New("public key is empty")
	}

	kem := oqs.KeyEncapsulation{}
	if err := kem.Init(AlgorithmMLKEM768, nil); err != nil {
		return nil, nil, fmt.Errorf("kem init: %w", err)
	}
	defer kem.Clean()

	ct, ss, err := kem.EncapSecret(publicKey)
	if err != nil {
		return nil, nil, fmt.Errorf("encapsulate: %w", err)
	}

	return ct, ss, nil
}

// Decapsulate recovers the shared secret from a ciphertext using the secret key.
func Decapsulate(ciphertext []byte, secretKey []byte) ([]byte, error) {
	if len(ciphertext) == 0 {
		return nil, errors.New("ciphertext is empty")
	}
	if len(secretKey) == 0 {
		return nil, errors.New("secret key is empty")
	}

	kem := oqs.KeyEncapsulation{}
	if err := kem.Init(AlgorithmMLKEM768, secretKey); err != nil {
		return nil, fmt.Errorf("kem init: %w", err)
	}
	defer kem.Clean()

	ss, err := kem.DecapSecret(ciphertext)
	if err != nil {
		return nil, fmt.Errorf("decapsulate: %w", err)
	}

	return ss, nil
}
