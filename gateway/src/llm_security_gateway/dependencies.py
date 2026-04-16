"""FastAPI dependency providers for shared singletons."""

from __future__ import annotations

from functools import lru_cache

from llm_security_gateway.config import GatewaySettings, get_settings
from llm_security_gateway.detection.data_leakage.response_filter import ResponseFilter
from llm_security_gateway.detection.engine import DetectionEngine
from llm_security_gateway.detection.jailbreak.pattern_classifier import PatternClassifier
from llm_security_gateway.detection.jailbreak.semantic_classifier import SemanticClassifier
from llm_security_gateway.detection.prompt_injection.heuristic_detector import HeuristicDetector
from llm_security_gateway.detection.prompt_injection.ml_detector import MLDetector
from llm_security_gateway.detection.prompt_injection.rule_detector import RuleDetector
from llm_security_gateway.llm_clients.base import BaseLLMClient
from llm_security_gateway.llm_clients.factory import create_client


@lru_cache(maxsize=1)
def get_detection_engine() -> DetectionEngine:
    """
    Build and cache the detection pipeline.

    Pipeline order (cheapest → most expensive):
      1. RuleDetector      — regex, ~0.1ms
      2. PatternClassifier — jailbreak signatures, ~0.2ms
      3. HeuristicDetector — statistics, ~0.5ms
      4. MLDetector        — ONNX DeBERTa, ~30ms (skipped on early exit)
      5. SemanticClassifier — embeddings, ~5ms
    """
    settings: GatewaySettings = get_settings()
    detectors = [
        RuleDetector(),
        PatternClassifier(),
        HeuristicDetector(),
        MLDetector(),
        SemanticClassifier(),
    ]
    return DetectionEngine(
        detectors=detectors,
        threshold=settings.detection_threshold,
        shadow_mode=settings.detection_shadow_mode,
    )


@lru_cache(maxsize=1)
def get_response_filter() -> ResponseFilter:
    return ResponseFilter()


@lru_cache(maxsize=1)
def get_llm_client() -> BaseLLMClient:
    """Return a singleton LLM client (connection pool is reused across requests)."""
    settings: GatewaySettings = get_settings()
    return create_client(settings.default_provider, settings)
