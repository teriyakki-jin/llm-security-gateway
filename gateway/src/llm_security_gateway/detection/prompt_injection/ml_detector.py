"""
ML-based prompt injection detector using ONNX Runtime + DeBERTa-v3-base.

Model pipeline:
  1. Tokenize with AutoTokenizer (max 512 tokens, truncation)
  2. ONNX Runtime InferenceSession (CPU or CUDA)
  3. Softmax → [benign, injection, jailbreak] probabilities
  4. Return DetectorOutput with score = max(injection, jailbreak)

Model source:
  Base: microsoft/deberta-v3-base
  Fine-tuned on: deepset/deberta-v3-base-injection + custom dataset
  Exported: export_onnx.py → INT8 dynamic quantization

If the ONNX model is not found, the detector returns score=0.0 and logs a warning.
This allows the gateway to start without the ML model (rule + heuristic still active).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from llm_security_gateway.detection.base import BaseDetector
from llm_security_gateway.detection.result import DetectorOutput

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent.parent.parent / "ml" / "models" / "detector.onnx"
_DEFAULT_TOKENIZER = "deepset/deberta-v3-base-injection"
_MAX_LENGTH = 512


class MLDetector(BaseDetector):
    """
    ONNX-based DeBERTa prompt injection classifier.

    Lazy-loads the model on first call. Falls back gracefully if the model
    file is not present (useful during development before training is done).
    """

    detector_type = "ml"

    def __init__(
        self,
        model_path: Path = _DEFAULT_MODEL_PATH,
        tokenizer_name: str = _DEFAULT_TOKENIZER,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path
        self._tokenizer_name = tokenizer_name
        self._device = device
        self._session = None
        self._tokenizer = None
        self._available = False
        self._try_load()

    def detect(self, text: str) -> DetectorOutput:
        if not self._available:
            return DetectorOutput(
                detector_name="ml",
                risk_score=0.0,
                labels=(),
                detail="ML model not loaded — rule+heuristic only",
            )

        if not text.strip():
            return DetectorOutput(detector_name="ml", risk_score=0.0, labels=())

        try:
            probs = self._infer(text)
            # probs: [benign, injection, jailbreak]
            injection_score = float(probs[1])
            jailbreak_score = float(probs[2])
            risk_score = max(injection_score, jailbreak_score)

            labels: list[str] = []
            if injection_score >= 0.5:
                labels.append("ml_injection_detected")
            if jailbreak_score >= 0.5:
                labels.append("ml_jailbreak_detected")

            return DetectorOutput(
                detector_name="ml",
                risk_score=round(risk_score, 4),
                labels=tuple(labels),
                confidence=risk_score,
                detail=f"injection={injection_score:.3f} jailbreak={jailbreak_score:.3f}",
            )
        except Exception as exc:
            logger.warning("ml_detector_inference_failed: %s", exc)
            return DetectorOutput(
                detector_name="ml",
                risk_score=0.0,
                labels=(),
                detail=f"inference error: {exc}",
            )

    def _infer(self, text: str) -> np.ndarray:
        assert self._session is not None
        assert self._tokenizer is not None

        inputs = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=_MAX_LENGTH,
            padding="max_length",
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        logits = self._session.run(None, ort_inputs)[0][0]
        # Softmax
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def _try_load(self) -> None:
        """Attempt to load the ONNX model. Non-fatal if not found."""
        if not self._model_path.exists():
            logger.warning(
                "ML model not found at %s — ML detection disabled. "
                "Run ml/scripts/train.py to train and export the model.",
                self._model_path,
            )
            return

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(str(self._model_path), providers=providers)
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            self._available = True
            logger.info("ML detector loaded from %s", self._model_path)
        except Exception as exc:
            logger.warning("Failed to load ML model: %s", exc)
