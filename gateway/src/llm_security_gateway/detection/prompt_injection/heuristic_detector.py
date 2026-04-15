"""Heuristic detector — statistical anomalies that suggest injection attempts."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from llm_security_gateway.detection.base import BaseDetector
from llm_security_gateway.detection.result import DetectorOutput

# Normal English prose Shannon entropy sits around 4.0–4.5 bits/char.
# Encoded/obfuscated payloads typically exceed 5.5.
_HIGH_ENTROPY_THRESHOLD = 4.8

# Structural anomaly thresholds
_MAX_SPECIAL_CHAR_RATIO = 0.25  # > 25% special chars is suspicious
_MAX_CONSECUTIVE_SPECIAL = 8    # 8+ consecutive special chars

# Length anomaly: prompts > 10x the rolling average are suspicious
_LENGTH_SPIKE_FACTOR = 10.0
_INITIAL_BASELINE_CHARS = 500   # assumed baseline if no history


class HeuristicDetector(BaseDetector):
    """
    Statistical heuristic detector. Catches obfuscation and structural attacks
    that regex patterns might miss.

    Signals:
      - Shannon entropy (high → possible encoding)
      - Language switching (sudden script change mid-prompt)
      - Special character density (delimiter flooding)
      - Length spike vs. rolling average
    """

    detector_type = "heuristic"

    def __init__(self) -> None:
        # Exponential moving average of request lengths for spike detection.
        self._avg_length: float = _INITIAL_BASELINE_CHARS
        self._alpha: float = 0.1  # EMA smoothing factor

    def detect(self, text: str) -> DetectorOutput:
        if not text.strip():
            return DetectorOutput(detector_name="heuristic", risk_score=0.0, labels=())

        scores: dict[str, float] = {}
        labels: list[str] = []

        # 1. Shannon entropy
        entropy = self._shannon_entropy(text)
        if entropy > _HIGH_ENTROPY_THRESHOLD:
            score = min((entropy - _HIGH_ENTROPY_THRESHOLD) / 2.0, 1.0)
            scores["high_entropy"] = score
            labels.append("encoding_evasion_suspected")

        # 2. Language/script switching
        if self._detects_script_switch(text):
            scores["script_switch"] = 0.65
            labels.append("multilingual_injection_suspected")

        # 3. Special character density
        special_ratio = self._special_char_ratio(text)
        if special_ratio > _MAX_SPECIAL_CHAR_RATIO:
            scores["special_chars"] = min(special_ratio * 2.0, 0.80)
            labels.append("delimiter_flooding")

        consecutive = self._max_consecutive_special(text)
        if consecutive >= _MAX_CONSECUTIVE_SPECIAL:
            scores["consecutive_special"] = min(consecutive / 20.0, 0.75)
            labels.append("delimiter_abuse")

        # 4. Length spike
        length = len(text)
        spike_ratio = length / max(self._avg_length, 1)
        if spike_ratio > _LENGTH_SPIKE_FACTOR:
            scores["length_spike"] = min((spike_ratio / _LENGTH_SPIKE_FACTOR - 1) * 0.3, 0.70)
            labels.append("abnormal_length")

        # Update rolling average.
        self._avg_length = self._alpha * length + (1 - self._alpha) * self._avg_length

        if not scores:
            return DetectorOutput(detector_name="heuristic", risk_score=0.0, labels=())

        final_score = max(scores.values())
        return DetectorOutput(
            detector_name="heuristic",
            risk_score=round(final_score, 4),
            labels=tuple(labels),
            confidence=final_score,
            detail=f"signals={list(scores.keys())}",
        )

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        """Calculate Shannon entropy in bits per character."""
        if not text:
            return 0.0
        freq = Counter(text)
        length = len(text)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in freq.values()
        )

    @staticmethod
    def _detects_script_switch(text: str) -> bool:
        """
        Detect abrupt script changes within a single message.
        E.g., English prompt suddenly switching to Arabic/Chinese mid-sentence.
        Legitimate multilingual use rarely switches script multiple times.
        """
        scripts: list[str] = []
        for char in text:
            if char.isalpha():
                try:
                    name = unicodedata.name(char, "")
                    if "LATIN" in name:
                        scripts.append("latin")
                    elif "CJK" in name or "HANGUL" in name or "HIRAGANA" in name or "KATAKANA" in name:
                        scripts.append("cjk")
                    elif "ARABIC" in name:
                        scripts.append("arabic")
                    elif "CYRILLIC" in name:
                        scripts.append("cyrillic")
                except Exception:
                    continue

        if len(scripts) < 20:
            return False

        # Count script transitions.
        transitions = sum(1 for i in range(1, len(scripts)) if scripts[i] != scripts[i - 1])
        # More than 3 transitions per 100 chars is suspicious.
        return (transitions / len(scripts)) > 0.03

    @staticmethod
    def _special_char_ratio(text: str) -> float:
        special = sum(
            1 for c in text
            if not c.isalnum() and not c.isspace() and c not in ".,!?;:'\"-"
        )
        return special / max(len(text), 1)

    @staticmethod
    def _max_consecutive_special(text: str) -> int:
        """Find the longest run of consecutive special characters."""
        max_run = 0
        current_run = 0
        for c in text:
            if not c.isalnum() and not c.isspace():
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
