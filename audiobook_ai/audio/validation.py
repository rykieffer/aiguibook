"""WhisperValidator - Validates TTS output using faster-whisper speech recognition."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating TTS output against expected text."""
    wer: float  # Word Error Rate (percentage)
    passed: bool  # Whether validation passed
    transcribed_text: str
    error_msg: str = ""

    def to_dict(self) -> dict:
        return {
            "wer": self.wer,
            "passed": self.passed,
            "transcribed_text": self.transcribed_text,
            "error_msg": self.error_msg,
        }


# French whisper model mapping
FRENCH_WHISPER_MODELS = {
    "distil-small.en":  # fallback - use multilingual
    "distil-whisper/distil-large-v3",
}


class WhisperValidator:
    """Validates TTS output quality using faster-whisper speech recognition."""

    def __init__(
        self,
        model: str = "distil-small.en",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        """
        Args:
            model: Whisper model name to use
            device: "cuda" or "cpu"
            compute_type: Compute type for faster-whisper
        """
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._validation_results: List[ValidationResult] = []

        # Use a multilingual model for French support
        # distil-small.en is English-only; switch to multilingual if needed
        self._is_multilingual = "distil-large" in model or "large" in model or "medium" in model

    def _ensure_model(self, force_multilingual: bool = False):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper required. Install with: pip install faster-whisper"
            )

        model_name = self.model_name

        # Use multilingual model for better French recognition
        if force_multilingual and self._is_english_only(model_name):
            model_name = "medium"
            logger.info(f"Using multilingual model '{model_name}' for validation")

        logger.info(f"Loading Whisper model: {model_name} on {self.device}")
        try:
            self._model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, trying medium: {e}")
            self._model = WhisperModel(
                "medium",
                device=self.device,
                compute_type=self.compute_type,
            )

    def _is_english_only(self, model_name: str) -> bool:
        """Check if a model is English-only (indicated by .en suffix)."""
        return model_name.endswith(".en")

    def _is_multilingual_multilingual(self, model_name: str) -> bool:
        """Check if model name suggests multilingual support."""
        return (
            "distil-large" in model_name
            or "large" in model_name
            or "medium" == model_name
            or "small" == model_name
            or "tiny" == model_name
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for WER comparison."""
        if not text:
            return ""
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def validate(
        self,
        audio_path: str,
        expected_text: str,
        language: str = "french",
        max_wer: float = 15.0,
    ) -> ValidationResult:
        """Validate TTS output by transcribing and comparing with expected text.

        Args:
            audio_path: Path to the audio file to validate
            expected_text: The text that was originally used to generate the audio
            language: Language of the audio ("french", "english", etc.)
            max_wer: Maximum acceptable Word Error Rate percentage

        Returns:
            ValidationResult with WER and pass/fail status
        """
        if not os.path.exists(audio_path):
            return ValidationResult(
                wer=100.0,
                passed=False,
                transcribed_text="",
                error_msg=f"Audio file not found: {audio_path}",
            )

        try:
            self._ensure_model(force_multilingual=(language.lower() not in ("en", "english")))
        except Exception as e:
            return ValidationResult(
                wer=100.0,
                passed=False,
                transcribed_text="",
                error_msg=f"Failed to load Whisper model: {e}",
            )

        try:
            # Detect language for Whisper
            is_french = language.lower() in ("french", "fr")

            # Transcribe using faster-whisper
            if is_french:
                segments, info = self._model.transcribe(
                    audio_path,
                    language="fr",
                    beam_size=5,
                    vad_filter=True,
                )
            else:
                segments, info = self._model.transcribe(
                    audio_path,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                )

            # Collect full transcription
            transcribed_parts = []
            for segment in segments:
                transcribed_parts.append(segment.text.strip())

            transcribed_text = " ".join(transcribed_parts)

            # Calculate WER
            wer_value = self._calculate_wer(
                self._normalize_text(expected_text),
                self._normalize_text(transcribed_text),
            )

            passed = wer_value <= max_wer

            result = ValidationResult(
                wer=wer_value,
                passed=passed,
                transcribed_text=transcribed_text,
            )
            self._validation_results.append(result)

            if not passed:
                logger.warning(
                    f"Validation FAILED: WER={wer_value:.1f}% (max {max_wer}%). "
                    f"Expected: '{expected_text[:50]}...', Got: '{transcribed_text[:50]}...'"
                )
            else:
                logger.debug(f"Validation PASSED: WER={wer_value:.1f}%")

            return result

        except Exception as e:
            logger.error(f"Validation error for {audio_path}: {e}")
            return ValidationResult(
                wer=100.0,
                passed=False,
                transcribed_text="",
                error_msg=str(e),
            )

    def _calculate_wer(self, expected: str, actual: str) -> float:
        """Calculate Word Error Rate between two texts.

        Uses jiwer library if available, fallback to simple calculation.

        Args:
            expected: Expected text (normalized)
            actual: Actual transcribed text (normalized)

        Returns:
            WER as percentage
        """
        if not expected and not actual:
            return 0.0
        if not expected or not actual:
            return 100.0

        # Try using jiwer
        try:
            import jiwer
            return jiwer.wer(expected, actual) * 100.0
        except ImportError:
            pass

        # Fallback: simple word-level Levenshtein
        expected_words = expected.split()
        actual_words = actual.split()

        if not expected_words:
            return 100.0

        # Build distance matrix
        m, n = len(expected_words), len(actual_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if expected_words[i - 1] == actual_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],     # deletion
                        dp[i][j - 1],     # insertion
                        dp[i - 1][j - 1], # substitution
                    )

        return (dp[m][n] / m) * 100.0

    def get_validation_summary(self) -> Dict[str, float]:
        """Get summary statistics for all validations performed.

        Returns:
            Dict with total, passed, failed, avg_wer
        """
        total = len(self._validation_results)
        if total == 0:
            return {"total": 0, "passed": 0, "failed": 0, "avg_wer": 0.0}

        passed = sum(1 for r in self._validation_results if r.passed)
        failed = total - passed
        avg_wer = sum(r.wer for r in self._validation_results) / total

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "avg_wer": round(avg_wer, 2),
        }

    def reset(self):
        """Clear validation history."""
        self._validation_results.clear()

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return f"WhisperValidator(model='{self.model_name}', {loaded})"
