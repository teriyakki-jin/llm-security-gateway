"""Jailbreak type taxonomy."""

from enum import Enum


class JailbreakType(str, Enum):
    DAN = "dan"                    # "Do Anything Now" family
    AIM = "aim"                    # "Always Intelligent and Machiavellian"
    ROLE_PLAY = "role_play"        # Character-based bypass
    HYPOTHETICAL = "hypothetical"  # "In a hypothetical scenario..."
    ENCODING = "encoding"          # Base64, ROT13, unicode obfuscation
    MULTILINGUAL = "multilingual"  # Language-switch bypass
    CRESCENDO = "crescendo"        # Gradual escalation
    PAIR = "pair"                  # Automated adversarial prompts
    UNKNOWN = "unknown"
