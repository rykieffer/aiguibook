     1|"""AudiobookConfig - Manages application configuration in YAML."""
     2|
     3|from __future__ import annotations
     4|
     5|import logging
     6|import os
     7|import copy
     8|from typing import Any, Dict, Optional
     9|
    10|import yaml
    11|
    12|logger = logging.getLogger(__name__)
    13|
    14|# Default configuration
    15|DEFAULT_CONFIG = {
    16|    "tts": {
    17|        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    18|        "backend_local": True,
    19|        "device": "cuda",
    20|        "dtype": "bfloat16",
    21|        "batch_size": 5,
    22|    },
    23|    "analysis": {
    24|        "llm_backend": "lmstudio",
    25|        "lmstudio_base_url": "http://localhost:1234",
    26|        "lmstudio_model": "gemma-4-26b-a4b",
    27|        "openrouter_api_key": "",
    28|        "openrouter_model": "qwen/qwen3.6-plus:free",
    29|        "ollama_model": "qwen3:32b",
    30|        "ollama_base_url": "http://localhost:11434",
    31|    },
    32|    "voices": {
    33|        "narrator_ref": "",
    34|        "character_refs": {},
    35|    },
    36|    "output": {
    37|        "format": "m4b",
    38|        "bitrate": "128k",
    39|        "sample_rate": 24000,
    40|        "chapter_markers": True,
    41|        "normalize_audio": True,
    42|        "crossfade_duration": 0.5,
    43|    },
    44|    "validation": {
    45|        "enabled": True,
    46|        "whisper_model": "distil-small.en",
    47|        "max_wer": 15,
    48|        "max_retries": 2,
    49|    },
    50|    "general": {
    51|        "language": "french",
    52|        "language_fallback": "english",
    53|        "max_segments": 99999,
    54|        "preview_mode": False,
    55|    },
    56|}
    57|
    58|
    59|class AudiobookConfig:
    60|    """Manages configuration saved as YAML."""
    61|
    62|    CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".aiguibook")
    63|    CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")
    64|
    65|    def __init__(self, config_path: Optional[str] = None):
    66|        """
    67|        Args:
    68|            config_path: Optional path to config file. Defaults to ~/.aiguibook/config.yaml
    69|        """
    70|        self.config_path = config_path or self.CONFIG_FILE
    71|        self._config: Dict[str, Any] = {}
    72|        self._load_defaults()
    73|
    74|    def _load_defaults(self):
    75|        """Load the default configuration."""
    76|        self._config = copy.deepcopy(DEFAULT_CONFIG)
    77|
    78|    def load(self, path: Optional[str] = None) -> "AudiobookConfig":
    79|        """Load configuration from a YAML file.
    80|
    81|        Args:
    82|            path: Optional path override
    83|
    84|        Returns:
    85|            Self for chaining
    86|        """
    87|        config_path = path or self.config_path
    88|
    89|        if os.path.exists(config_path):
    90|            try:
    91|                with open(config_path, "r", encoding="utf-8") as f:
    92|                    user_config = yaml.safe_load(f) or {}
    93|                self._merge_config(self._config, user_config)
    94|                logger.info(f"Configuration loaded from: {config_path}")
    95|            except yaml.YAMLError as e:
    96|                logger.warning(f"Error parsing config YAML: {e}. Using defaults.")
    97|            except IOError as e:
    98|                logger.warning(f"Could not read config file: {e}. Using defaults.")
    99|        else:
   100|            logger.info(f"No config file at {config_path}, using defaults")
   101|
   102|        # Apply ENV fallbacks
   103|        self._apply_env_fallbacks()
   104|        self._config["config_path"] = config_path
   105|        return self
   106|
   107|    def _apply_env_fallbacks(self):
   108|        """Read API keys and settings from environment variables as fallback."""
   109|        # OpenRouter API key
   110|        if not self.get("analysis", "openrouter_api_key"):
   111|            env_key = os.environ.get("OPENROUTER_API_KEY", "")
   112|            if env_key:
   113|                self.set("analysis", "openrouter_api_key", env_key)
   114|
   115|        # Ollama base URL
   116|        env_ollama = os.environ.get("OLLAMA_BASE_URL", "")
   117|        if env_ollama and not self.get("analysis", "ollama_base_url"):
   118|            self.set("analysis", "ollama_base_url", env_ollama)
   119|
   120|        # Override TTS device
   121|        env_device = os.environ.get("AIGUIBOOK_TTS_DEVICE", "")
   122|        if env_device:
   123|            self.set("tts", "device", env_device)
   124|
   125|    def save(self, path: Optional[str] = None):
   126|        """Save configuration to a YAML file.
   127|
   128|        Args:
   129|            path: Optional path override
   130|        """
   131|        config_path = path or self.config_path
   132|        config_dir = os.path.dirname(config_path)
   133|        os.makedirs(config_dir, exist_ok=True)
   134|
   135|        # Remove internal keys
   136|        config_copy = {k: v for k, v in self._config.items() if not k.startswith("_")}
   137|
   138|        with open(config_path, "w", encoding="utf-8") as f:
   139|            yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
   140|        logger.info(f"Configuration saved to: {config_path}")
   141|
   142|    def get(self, section: str, key: str, default: Any = None) -> Any:
   143|        """Get a configuration value.
   144|
   145|        Args:
   146|            section: Config section name
   147|            key: Key within section
   148|            default: Default value if not found
   149|
   150|        Returns:
   151|            Configuration value
   152|        """
   153|        return self._config.get(section, {}).get(key, default)
   154|
   155|    def set(self, section: str, key: str, value: Any):
   156|        """Set a configuration value.
   157|
   158|        Args:
   159|            section: Config section name
   160|            key: Key within section
   161|            value: Value to set
   162|        """
   163|        if section not in self._config:
   164|            self._config[section] = {}
   165|        self._config[section][key] = value
   166|
   167|    def get_section(self, section: str, default: Optional[Dict] = None) -> Dict[str, Any]:
   168|        """Get an entire config section.
   169|
   170|        Args:
   171|            section: Section name
   172|            default: Default dict if section missing
   173|
   174|        Returns:
   175|            Section dictionary
   176|        """
   177|        return self._config.get(section, default or {})
   178|
   179|    def validate(self) -> list:
   180|        """Validate configuration and return list of warnings.
   181|
   182|        Returns:
   183|            List of warning messages (empty if all valid)
   184|        """
   185|        warnings = []
   186|
   187|        # TTS dtype validation
   188|        dtype = self.get("tts", "dtype", "bfloat16")
   189|        valid_dtypes = ("float16", "bfloat16", "float32")
   190|        if dtype not in valid_dtypes:
   191|            warnings.append(
   192|                f"Invalid TTS dtype '{dtype}', must be one of {valid_dtypes}"
   193|            )
   194|
   195|        # Bitrate format
   196|        bitrate = self.get("output", "bitrate", "128k")
   197|        if not str(bitrate).endswith("k") and not str(bitrate).endswith("m"):
   198|            warnings.append(
   199|                f"Bitrate '{bitrate}' should end with 'k' or 'm' (e.g., '128k')"
   200|            )
   201|
   202|        # Crossfade must be non-negative
   203|        crossfade = self.get("output", "crossfade_duration", 0.5)
   204|        if crossfade < 0:
   205|            warnings.append("crossfade_duration must be non-negative")
   206|
   207|        # WER threshold must be reasonable
   208|        max_wer = self.get("validation", "max_wer", 15)
   209|        if max_wer < 0 or max_wer > 100:
   210|            warnings.append("max_wer must be between 0 and 100")
   211|
   212|        # Sample rate should be positive
   213|        sample_rate = self.get("output", "sample_rate", 24000)
   214|        if sample_rate <= 0:
   215|            warnings.append("sample_rate must be positive")
   216|
   217|        # Language validation
   218|        lang = self.get("general", "language", "french")
   219|        valid_langs = ("french", "english", "spanish", "german", "japanese",
   220|                       "korean", "russian", "portuguese", "italian", "chinese")
   221|        if lang not in valid_langs:
   222|            warnings.append(
   223|                f"Language '{lang}' not in known supported languages: {valid_langs}"
   224|            )
   225|
   226|        return warnings
   227|
   228|    def to_dict(self) -> Dict[str, Any]:
   229|        """Return full config as dictionary."""
   230|        return copy.deepcopy(self._config)
   231|
   232|    @staticmethod
   233|    def _merge_config(base: dict, override: dict):
   234|        """Recursively merge override dict into base dict.
   235|
   236|        Args:
   237|            base: Base configuration dict (modified in place)
   238|            override: User configuration dict
   239|        """
   240|        for key, value in override.items():
   241|            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
   242|                AudiobookConfig._merge_config(base[key], value)
   243|            else:
   244|                base[key] = value
   245|
   246|    def __repr__(self) -> str:
   247|        model = self.get("tts", "model", "unknown")
   248|        backend = self.get("tts", "device", "unknown")
   249|        return f"AudiobookConfig(model='{model}', device='{backend}')"
   250|