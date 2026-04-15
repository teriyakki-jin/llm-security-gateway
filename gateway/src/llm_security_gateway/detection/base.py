"""Abstract base class for all detectors."""

from abc import ABC, abstractmethod

from llm_security_gateway.detection.result import DetectorOutput


class BaseDetector(ABC):
    """All detectors implement this interface so the engine can run them uniformly."""

    # Must be one of: "rule", "heuristic", "ml"
    detector_type: str = "base"

    @abstractmethod
    def detect(self, text: str) -> DetectorOutput:
        """Analyze text and return a DetectorOutput."""
