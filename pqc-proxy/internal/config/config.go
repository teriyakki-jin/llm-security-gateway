// Package config loads proxy configuration from YAML files and environment variables.
package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

// Config holds all proxy configuration.
type Config struct {
	Server  ServerConfig  `mapstructure:"server"`
	Gateway GatewayConfig `mapstructure:"gateway"`
	Log     LogConfig     `mapstructure:"log"`
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	// Host is the address the proxy listens on.
	Host string `mapstructure:"host"`
	// Port is the TCP port the proxy listens on.
	Port int `mapstructure:"port"`
	// ReadTimeoutSec is the HTTP read timeout in seconds.
	ReadTimeoutSec int `mapstructure:"read_timeout_sec"`
	// WriteTimeoutSec is the HTTP write timeout in seconds.
	WriteTimeoutSec int `mapstructure:"write_timeout_sec"`
}

// GatewayConfig holds the downstream gateway connection settings.
type GatewayConfig struct {
	// URL is the target gateway URL (e.g., http://gateway:8000).
	URL string `mapstructure:"url"`
}

// LogConfig holds logging settings.
type LogConfig struct {
	// Level is the log level: debug, info, warn, error.
	Level string `mapstructure:"level"`
	// Format is the log format: json, console.
	Format string `mapstructure:"format"`
}

// Load reads configuration from the given YAML file and environment variables.
// Environment variables take precedence. Prefix: PQC_PROXY_.
// Example: PQC_PROXY_SERVER_PORT=9090 overrides server.port.
func Load(configPath string) (*Config, error) {
	v := viper.New()

	// Defaults.
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 8443)
	v.SetDefault("server.read_timeout_sec", 30)
	v.SetDefault("server.write_timeout_sec", 60)
	v.SetDefault("gateway.url", "http://gateway:8000")
	v.SetDefault("log.level", "info")
	v.SetDefault("log.format", "json")

	// Config file.
	if configPath != "" {
		v.SetConfigFile(configPath)
		if err := v.ReadInConfig(); err != nil {
			return nil, fmt.Errorf("read config file %q: %w", configPath, err)
		}
	}

	// Environment variable overrides.
	v.SetEnvPrefix("PQC_PROXY")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshal config: %w", err)
	}

	if err := validate(&cfg); err != nil {
		return nil, fmt.Errorf("config validation: %w", err)
	}

	return &cfg, nil
}

func validate(cfg *Config) error {
	if cfg.Server.Port < 1 || cfg.Server.Port > 65535 {
		return fmt.Errorf("server.port %d is out of range [1, 65535]", cfg.Server.Port)
	}
	if cfg.Gateway.URL == "" {
		return fmt.Errorf("gateway.url must not be empty")
	}
	return nil
}
