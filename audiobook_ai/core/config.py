"""AudiobookConfig - Manages application configuration in YAML."""

import os
import copy
import logging
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

# Default configuration values
# batch_size=5 and max_tokens=1500 are optimized for 16GB VRAM
DEFAULT_CONFIG = {
    "tts": {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "backend_local": True,
        "device": "cuda",
        "dtype": "bfloat16",
        "batch_size": 5,  # Reduced from 4/20 to avoid OOM on 16GB
    },
    "analysis": {
        "llm_backend": "lmstudio",
        "lmstudio_base_url": "http://localhost:1234",
        "lmstudio_model": "google/gemma-4-26b-a4b",
        "openrouter_api_key": "",
        "openrouter_model": "qwen/qwen3.6-plus:free",
        "ollama_model": "qwen3:32b",
        "ollama_base_url": "http://localhost:11434",
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
        "whisper_model": "distil-medium.en",
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
    CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".aiguibook")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.CONFIG_FILE
        self._config: Dict[str, Any] = {}
        self._load_defaults()

    def _load_defaults(self):
        self._config = copy.deepcopy(DEFAULT_CONFIG)

    def load(self, path: Optional[str] = None) -> "AudiobookConfig":
        config_path = path or self.config_path
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f)
                if isinstance(user_config, dict):
                    # Deep merge
                    for k, v in user_config.items():
                        if k in self._config and isinstance(v, dict):
                            self._config[k].update(v)
                        else:
                            self._config[k] = v
                logger.info(f"Configuration loaded from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        else:
            logger.info(f"No config file at {config_path}, using defaults")
        
        # Environment variable overrides
        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key:
            self._config["analysis"]["openrouter_api_key"] = env_key
        return self

    def save(self, path: Optional[str] = None) -> "AudiobookConfig":
        config_path = path or self.config_path
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return self

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> "AudiobookConfig":
        self._config.setdefault(section, {})[key] = value
        return self

    def get_section(self, section: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        return self._config.get(section, default or {}).copy()

    def validate(self) -> bool:
        # Basic validation
        return []
