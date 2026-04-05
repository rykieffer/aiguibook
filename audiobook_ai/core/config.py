"""AudiobookConfig - Manages application configuration in YAML."""

from __future__ import annotations

import logging
import os
import copy
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "tts": {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "backend_local": True,
        "device": "cuda",
        "dtype": "bfloat16",
        "batch_size": 4,
    },
    "analysis": {
        "llm_backend": "openrouter",
        "openrouter_api_key": "",
        "openrouter_model": "anthropic/claude-sonnet-4-20250514",
        "ollama_model": "qwen3:32b",
        "ollama_base_url": "http://localhost:11434",
        "lmstudio_base_url": "http://localhost:1234/v1",
        "lmstudio_model": "",
    },
    "voices": {
        "narrator_ref": "",
        "character_refs": {},
    },
    "output": {
        "format": "m4b",
        "bitrate": "128k",
        "sample_rate": 24000,
        "chapter_markers": True,
        "normalize_audio": True,
        "crossfade_duration": 0.5,
    },
    "validation": {
        "enabled": True,
        "whisper_model": "distil-small.en",
        "max_wer": 15,
        "max_retries": 2,
    },
    "general": {
        "language": "french",
        "language_fallback": "english",
        "max_segments": 99999,
        "preview_mode": False,
    },
}


class AudiobookConfig:
    """Manages configuration saved as YAML."""

    CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".aiguibook")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Optional path to config file. Defaults to ~/.aiguibook/config.yaml
        """
        self.config_path = config_path or self.CONFIG_FILE
        self._config: Dict[str, Any] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load the default configuration."""
        self._config = copy.deepcopy(DEFAULT_CONFIG)

    def load(self, path: Optional[str] = None) -> "AudiobookConfig":
        """Load configuration from a YAML file.

        Args:
            path: Optional path override

        Returns:
            Self for chaining
        """
        config_path = path or self.config_path

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                self._merge_config(self._config, user_config)
                logger.info(f"Configuration loaded from: {config_path}")
            except yaml.YAMLError as e:
                logger.warning(f"Error parsing config YAML: {e}. Using defaults.")
            except IOError as e:
                logger.warning(f"Could not read config file: {e}. Using defaults.")
        else:
            logger.info(f"No config file at {config_path}, using defaults")

        # Apply ENV fallbacks
        self._apply_env_fallbacks()
        self._config["config_path"] = config_path
        return self

    def _apply_env_fallbacks(self):
        """Read API keys and settings from environment variables as fallback."""
        # OpenRouter API key
        if not self.get("analysis", "openrouter_api_key"):
            env_key = os.environ.get("OPENROUTER_API_KEY", "")
            if env_key:
                self.set("analysis", "openrouter_api_key", env_key)

        # Ollama base URL
        env_ollama = os.environ.get("OLLAMA_BASE_URL", "")
        if env_ollama and not self.get("analysis", "ollama_base_url"):
            self.set("analysis", "ollama_base_url", env_ollama)

        # Override TTS device
        env_device = os.environ.get("AIGUIBOOK_TTS_DEVICE", "")
        if env_device:
            self.set("tts", "device", env_device)

    def save(self, path: Optional[str] = None):
        """Save configuration to a YAML file.

        Args:
            path: Optional path override
        """
        config_path = path or self.config_path
        config_dir = os.path.dirname(config_path)
        os.makedirs(config_dir, exist_ok=True)

        # Remove internal keys
        config_copy = {k: v for k, v in self._config.items() if not k.startswith("_")}

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Configuration saved to: {config_path}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            section: Config section name
            key: Key within section
            default: Default value if not found

        Returns:
            Configuration value
        """
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any):
        """Set a configuration value.

        Args:
            section: Config section name
            key: Key within section
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def get_section(self, section: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """Get an entire config section.

        Args:
            section: Section name
            default: Default dict if section missing

        Returns:
            Section dictionary
        """
        return self._config.get(section, default or {})

    def validate(self) -> list:
        """Validate configuration and return list of warnings.

        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []

        # TTS dtype validation
        dtype = self.get("tts", "dtype", "bfloat16")
        valid_dtypes = ("float16", "bfloat16", "float32")
        if dtype not in valid_dtypes:
            warnings.append(
                f"Invalid TTS dtype '{dtype}', must be one of {valid_dtypes}"
            )

        # Bitrate format
        bitrate = self.get("output", "bitrate", "128k")
        if not str(bitrate).endswith("k") and not str(bitrate).endswith("m"):
            warnings.append(
                f"Bitrate '{bitrate}' should end with 'k' or 'm' (e.g., '128k')"
            )

        # Crossfade must be non-negative
        crossfade = self.get("output", "crossfade_duration", 0.5)
        if crossfade < 0:
            warnings.append("crossfade_duration must be non-negative")

        # WER threshold must be reasonable
        max_wer = self.get("validation", "max_wer", 15)
        if max_wer < 0 or max_wer > 100:
            warnings.append("max_wer must be between 0 and 100")

        # Sample rate should be positive
        sample_rate = self.get("output", "sample_rate", 24000)
        if sample_rate <= 0:
            warnings.append("sample_rate must be positive")

        # Language validation
        lang = self.get("general", "language", "french")
        valid_langs = ("french", "english", "spanish", "german", "japanese",
                       "korean", "russian", "portuguese", "italian", "chinese")
        if lang not in valid_langs:
            warnings.append(
                f"Language '{lang}' not in known supported languages: {valid_langs}"
            )

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return copy.deepcopy(self._config)

    @staticmethod
    def _merge_config(base: dict, override: dict):
        """Recursively merge override dict into base dict.

        Args:
            base: Base configuration dict (modified in place)
            override: User configuration dict
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                AudiobookConfig._merge_config(base[key], value)
            else:
                base[key] = value

    def __repr__(self) -> str:
        model = self.get("tts", "model", "unknown")
        backend = self.get("tts", "device", "unknown")
        return f"AudiobookConfig(model='{model}', device='{backend}')"
