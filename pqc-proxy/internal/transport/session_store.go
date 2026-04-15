// Package transport provides Redis-backed session resumption for PQC sessions.
//
// Design rationale:
//   - Full handshake (ML-KEM-768 keygen + encapsulation) takes ~2-3ms.
//     At 1000 RPM per client this is acceptable, but at scale it adds up.
//   - Session keys are AES-256-GCM encrypted before writing to Redis using
//     a server master key, so the Redis instance never holds plaintext keys.
//   - TTL is 1 hour; clients can resume without re-running the PQC handshake.
//   - On multi-instance deployments, all proxies share the same Redis store,
//     so any instance can resume a session created by another.
package transport

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	// SessionTTL is the duration a resumed session remains valid.
	SessionTTL = time.Hour

	// sessionKeyPrefix is the Redis key prefix for stored sessions.
	sessionKeyPrefix = "pqc:session:"
)

// SessionRecord is the serializable form of a SessionState stored in Redis.
// The key material is encrypted with the server master key before storage.
type SessionRecord struct {
	// EncryptedKey is the AES-256-GCM encrypted session key.
	EncryptedKey []byte `json:"encrypted_key"`
	// Nonce is the AES-GCM nonce used to encrypt EncryptedKey.
	Nonce []byte `json:"nonce"`
	// CreatedAt is when the session was established.
	CreatedAt time.Time `json:"created_at"`
	// SendCounter is persisted to prevent counter reuse after resumption.
	SendCounter uint64 `json:"send_counter"`
	// RecvCounter is the last received counter value.
	RecvCounter uint64 `json:"recv_counter"`
}

// SessionStore manages PQC session persistence in Redis.
type SessionStore struct {
	rdb       *redis.Client
	masterKey [32]byte // AES-256 key for encrypting session keys at rest
}

// NewSessionStore creates a new store backed by the given Redis client.
// masterKey must be 32 bytes (AES-256). Keep it in a secret manager.
func NewSessionStore(rdb *redis.Client, masterKey [32]byte) *SessionStore {
	return &SessionStore{rdb: rdb, masterKey: masterKey}
}

// Save serializes and encrypts a SessionState, then stores it in Redis.
func (s *SessionStore) Save(ctx context.Context, sessionID string, state *SessionState) error {
	encryptedKey, nonce, err := s.encryptKey(state.SessionKey())
	if err != nil {
		return fmt.Errorf("encrypt session key: %w", err)
	}

	record := SessionRecord{
		EncryptedKey: encryptedKey,
		Nonce:        nonce,
		CreatedAt:    state.createdAt,
		SendCounter:  state.sendCounter.Load(),
		RecvCounter:  state.recvCounter.Load(),
	}

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("marshal session record: %w", err)
	}

	key := sessionKeyPrefix + sessionID
	if err := s.rdb.Set(ctx, key, data, SessionTTL).Err(); err != nil {
		return fmt.Errorf("redis set: %w", err)
	}

	return nil
}

// Load retrieves and decrypts a SessionState from Redis.
// Returns (nil, nil) if the session does not exist or has expired.
func (s *SessionStore) Load(ctx context.Context, sessionID string) (*SessionState, error) {
	key := sessionKeyPrefix + sessionID
	data, err := s.rdb.Get(ctx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, nil // session not found / expired
	}
	if err != nil {
		return nil, fmt.Errorf("redis get: %w", err)
	}

	var record SessionRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("unmarshal session record: %w", err)
	}

	sessionKey, err := s.decryptKey(record.EncryptedKey, record.Nonce)
	if err != nil {
		return nil, fmt.Errorf("decrypt session key: %w", err)
	}

	state := &SessionState{
		key:       sessionKey,
		createdAt: record.CreatedAt,
	}
	state.sendCounter.Store(record.SendCounter)
	state.recvCounter.Store(record.RecvCounter)

	// Refresh TTL on successful load (sliding expiry).
	s.rdb.Expire(ctx, key, SessionTTL) //nolint:errcheck

	return state, nil
}

// Delete removes a session from Redis (e.g., on explicit logout or key compromise).
func (s *SessionStore) Delete(ctx context.Context, sessionID string) error {
	return s.rdb.Del(ctx, sessionKeyPrefix+sessionID).Err()
}

// Metrics returns the number of active sessions tracked in Redis.
func (s *SessionStore) ActiveCount(ctx context.Context) (int64, error) {
	keys, err := s.rdb.Keys(ctx, sessionKeyPrefix+"*").Result()
	if err != nil {
		return 0, err
	}
	return int64(len(keys)), nil
}

// encryptKey encrypts a session key using AES-256-GCM with the master key.
func (s *SessionStore) encryptKey(sessionKey []byte) (ciphertext []byte, nonce []byte, err error) {
	block, err := aes.NewCipher(s.masterKey[:])
	if err != nil {
		return nil, nil, fmt.Errorf("aes cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, nil, fmt.Errorf("aes-gcm: %w", err)
	}

	n := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, n); err != nil {
		return nil, nil, fmt.Errorf("random nonce: %w", err)
	}

	ct := gcm.Seal(nil, n, sessionKey, nil)
	return ct, n, nil
}

// decryptKey decrypts a session key using AES-256-GCM with the master key.
func (s *SessionStore) decryptKey(ciphertext []byte, nonce []byte) ([]byte, error) {
	block, err := aes.NewCipher(s.masterKey[:])
	if err != nil {
		return nil, fmt.Errorf("aes cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("aes-gcm: %w", err)
	}

	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("aes-gcm decrypt: %w", err)
	}

	return plaintext, nil
}

// resumedSessionState reconstructs a SessionState with restored counters.
// Used internally by SessionStore.Load — exported only for testing.
func resumedSessionState(key []byte, createdAt time.Time, sendCounter, recvCounter uint64) *SessionState {
	s := &SessionState{key: key, createdAt: createdAt}
	s.sendCounter = atomic.Uint64{}
	s.sendCounter.Store(sendCounter)
	s.recvCounter = atomic.Uint64{}
	s.recvCounter.Store(recvCounter)
	return s
}
