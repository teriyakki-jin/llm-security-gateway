package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"

	"github.com/llm-security-gateway/pqc-proxy/internal/config"
	"github.com/llm-security-gateway/pqc-proxy/internal/kem"
	"github.com/llm-security-gateway/pqc-proxy/internal/proxy"
)

func main() {
	configPath := flag.String("config", "", "Path to YAML config file (optional, uses env vars otherwise)")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load config: %v\n", err)
		os.Exit(1)
	}

	log, err := buildLogger(cfg.Log.Level, cfg.Log.Format)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init logger: %v\n", err)
		os.Exit(1)
	}
	defer log.Sync() //nolint:errcheck

	log.Info("generating server hybrid key pair (ML-KEM-768 + X25519)")
	serverKP, err := kem.GenerateHybridKeyPair()
	if err != nil {
		log.Fatal("key generation failed", zap.Error(err))
	}
	log.Info("server key pair ready",
		zap.Int("mlkem_pub_len", len(serverKP.MLKEM.PublicKey)),
		zap.Int("x25519_pub_len", len(serverKP.X25519Public)),
	)

	pqcProxy, err := proxy.NewPQCReverseProxy(cfg.Gateway.URL, log)
	if err != nil {
		log.Fatal("create proxy", zap.Error(err))
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/pqc/keys", pqcProxy.KeyServerHandler(serverKP))
	mux.HandleFunc("/pqc/handshake", pqcProxy.HandshakeHandler)
	mux.HandleFunc("/pqc/finished", pqcProxy.FinishedHandler)
	mux.HandleFunc("/v1/", pqcProxy.ProxyHandler)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`)) //nolint:errcheck
	})

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeoutSec) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeoutSec) * time.Second,
	}

	// Start server in background.
	go func() {
		log.Info("pqc proxy listening", zap.String("addr", addr))
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal("server error", zap.Error(err))
		}
	}()

	// Graceful shutdown on SIGINT/SIGTERM.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Error("shutdown error", zap.Error(err))
	}
	log.Info("shutdown complete")
}

func buildLogger(level, format string) (*zap.Logger, error) {
	var cfg zap.Config
	if format == "console" {
		cfg = zap.NewDevelopmentConfig()
	} else {
		cfg = zap.NewProductionConfig()
	}

	switch level {
	case "debug":
		cfg.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
	case "warn":
		cfg.Level = zap.NewAtomicLevelAt(zap.WarnLevel)
	case "error":
		cfg.Level = zap.NewAtomicLevelAt(zap.ErrorLevel)
	default:
		cfg.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	}

	return cfg.Build()
}
