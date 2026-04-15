"""Prometheus metrics registry for LLM Security Gateway."""

from prometheus_client import Counter, Gauge, Histogram

# ── Detection metrics ──────────────────────────────────────────

detection_requests_total = Counter(
    "gateway_detection_requests_total",
    "Total detection pipeline evaluations",
    ["action"],  # "blocked" | "shadow_blocked" | "passed"
)

detection_latency_seconds = Histogram(
    "gateway_detection_latency_seconds",
    "Detection pipeline latency in seconds",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# ── LLM proxy metrics ─────────────────────────────────────────

llm_requests_total = Counter(
    "gateway_llm_requests_total",
    "Total requests forwarded to LLM providers",
    ["provider", "status"],  # status: "success" | "error"
)

llm_latency_seconds = Histogram(
    "gateway_llm_latency_seconds",
    "Round-trip latency for LLM provider calls in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# ── Admin state gauges ────────────────────────────────────────

shadow_mode_active = Gauge(
    "gateway_shadow_mode_active",
    "1 if shadow mode is currently enabled, 0 otherwise",
)

detection_threshold_current = Gauge(
    "gateway_detection_threshold_current",
    "Current detection risk-score blocking threshold (0.0 – 1.0)",
)
