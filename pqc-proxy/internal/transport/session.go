package transport

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/crypto/hkdf"
)

const (
	// keyLen is AES-256 key length.
	keyLen = 32

	// nonceLen is AES-GCM nonce length (96-bit).
	nonceLen = 12

	// counterLen is the 8-byte send/recv counter prefix in the nonce.
	counterLen = 8

	// keyRotationInterval triggers re-keying after this many messages.
	keyRotationMessages = 1 << 24 // ~16 million messages

	// keyRotationDuration triggers re-keying after this duration.
	keyRotationDuration = time.Hour

	rotationHKDFInfo = "llm-security-gateway session key rotation v1"
)

// SessionState holds the symmetric session key and counters for an established session.
// All fields are safe for concurrent use.
type SessionState struct {
	key                 []byte
	sendCounter         atomic.Uint64
	recvCounter         atomic.Uint64
	createdAt           time.Time
	mu                  sync.RWMutex
	expectedFinishedMAC []byte
}

// Encrypt encrypts plaintext using AES-256-GCM.
// Nonce = 8-byte counter (big-endian) || 4-byte random suffix.
// The counter prevents nonce reuse; replay attack detection is handled by Decrypt.
func (s *SessionState) Encrypt(plaintext []byte) ([]byte, error) {
	s.mu.RLock()
	key := s.key
	s.mu.RUnlock()

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("aes cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("aes-gcm: %w", err)
	}

	nonce, counter, err := s.buildEncryptNonce()
	if err != nil {
		return nil, fmt.Errorf("build nonce: %w", err)
	}

	// Prepend counter (8 bytes) + nonce (12 bytes) so Decrypt can verify ordering.
	counterBytes := make([]byte, counterLen)
	binary.BigEndian.PutUint64(counterBytes, counter)

	ciphertext := gcm.Seal(nil, nonce, plaintext, counterBytes)

	out := make([]byte, counterLen+nonceLen+len(ciphertext))
	copy(out, counterBytes)
	copy(out[counterLen:], nonce)
	copy(out[counterLen+nonceLen:], ciphertext)

	if err := s.maybeRotateKey(); err != nil {
		return nil, fmt.Errorf("key rotation: %w", err)
	}

	return out, nil
}

// Decrypt decrypts a message produced by Encrypt.
// It enforces monotonically increasing counters to prevent replay attacks.
func (s *SessionState) Decrypt(data []byte) ([]byte, error) {
	if len(data) < counterLen+nonceLen {
		return nil, errors.New("message too short")
	}

	counterBytes := data[:counterLen]
	nonce := data[counterLen : counterLen+nonceLen]
	ciphertext := data[counterLen+nonceLen:]

	incomingCounter := binary.BigEndian.Uint64(counterBytes)

	// Enforce strictly increasing counter to block replays.
	for {
		current := s.recvCounter.Load()
		if incomingCounter <= current {
			return nil, fmt.Errorf("replay attack detected: counter %d <= expected > %d", incomingCounter, current)
		}
		if s.recvCounter.CompareAndSwap(current, incomingCounter) {
			break
		}
	}

	s.mu.RLock()
	key := s.key
	s.mu.RUnlock()

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("aes cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("aes-gcm: %w", err)
	}

	plaintext, err := gcm.Open(nil, nonce, ciphertext, counterBytes)
	if err != nil {
		return nil, fmt.Errorf("aes-gcm decrypt: %w (possible tampering)", err)
	}

	return plaintext, nil
}

// RotateKey derives a new session key from the current one using HKDF-SHA256.
// Called automatically after keyRotationMessages or keyRotationDuration.
func (s *SessionState) RotateKey() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	reader := hkdf.New(sha256.New, s.key, nil, []byte(rotationHKDFInfo))
	newKey := make([]byte, keyLen)
	if _, err := io.ReadFull(reader, newKey); err != nil {
		return fmt.Errorf("hkdf key rotation: %w", err)
	}

	s.key = newKey
	s.createdAt = time.Now().UTC()
	return nil
}

// SessionKey returns a copy of the current session key (for diagnostic use only).
func (s *SessionState) SessionKey() []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cp := make([]byte, len(s.key))
	copy(cp, s.key)
	return cp
}

func (s *SessionState) buildEncryptNonce() (nonce []byte, counter uint64, err error) {
	counter = s.sendCounter.Add(1)
	counterBytes := make([]byte, counterLen)
	binary.BigEndian.PutUint64(counterBytes, counter)

	// 4-byte random suffix for additional nonce entropy.
	var randomSuffix [4]byte
	if _, err := io.ReadFull(rand.Reader, randomSuffix[:]); err != nil {
		return nil, 0, fmt.Errorf("random nonce suffix: %w", err)
	}

	nonce = make([]byte, nonceLen)
	copy(nonce[:counterLen], counterBytes)
	copy(nonce[counterLen:], randomSuffix[:])
	return nonce, counter, nil
}

func (s *SessionState) maybeRotateKey() error {
	sendCount := s.sendCounter.Load()
	if sendCount%keyRotationMessages == 0 || time.Since(s.createdAt) > keyRotationDuration {
		return s.RotateKey()
	}
	return nil
}
