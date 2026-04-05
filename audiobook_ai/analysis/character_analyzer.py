"""Character Analyzer - Uses LLM to detect characters, emotions, and assign voice IDs."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_EMOTIONS = [
    "calm", "excited", "angry", "sad", "whisper",
    "tense", "urgent", "amused", "contemptuous", "surprised", "neutral",
]

EMOTION_INSTRUCTIONS_FR = {
    "calm": "Parlez avec un ton calme et pose, voix douce et reguliere",
    "excited": "Parlez avec excitation et enthousiasme, voix energique et vive",
    "angry": "Parlez avec colere et tension, voix ferme et intense",
    "sad": "Parlez d'une voix triste et melancholique, ton doux et lent",
    "whisper": "Chuchotez d'une voix mysterieuse, ton bas et intime",
    "tense": "Parlez avec un ton tendu et nerveux, voix serree et rapide",
    "urgent": "Parlez avec urgence, voix rapide et pressante",
    "amused": "Parlez avec amusement, voix legere et joyeuse",
    "contemptuous": "Parlez avec mepris, voix froide et distante",
    "surprised": "Parlez avec surprise, voix etonnee et expressive",
    "neutral": "Parlez d'un ton neutre et naturel, sans emotion particuliere",
}

EMOTION_INSTRUCTIONS_EN = {
    "calm": "Speak in a calm and composed tone, soft and steady voice",
    "excited": "Speak with excitement and enthusiasm, energetic and lively voice",
    "angry": "Speak with anger and tension, firm and intense voice",
    "sad": "Speak with a sad and melancholic tone, soft and slow voice",
    "whisper": "Whisper in a mysterious tone, low and intimate voice",
    "tense": "Speak with a tense and nervous tone, tight and rapid voice",
    "urgent": "Speak with urgency, fast and pressing voice",
    "amused": "Speak with amusement, light and cheerful voice",
    "contemptuous": "Speak with contempt, cold and distant voice",
    "surprised": "Speak with surprise, astonished and expressive voice",
    "neutral": "Speak in a neutral, natural tone without particular emotion",
}

ANALYSIS_PROMPT_TEMPLATE = """You are an expert literary analyst for audiobook production.
Your task is to analyze text segments from a book and return structured JSON with:
1. Who is speaking (narrator vs character dialogue)
2. Which character is speaking (if dialogue)
3. The emotion/tone of the speech
4. A brief character description
5. An appropriate voice profile ID suggestion

You are analyzing French/English bilingual text.

Available Voice Profiles:
- narrator_male: Deep warm male, mature, authoritative
- narrator_female: Soft warm female, clear and elegant
- young_male: Young energetic male, bright
- young_female: Young cheerful female, animated
- elder_male: Older deep male, grave and wise
- elder_female: Older compassionate female, gentle
- robotic: Mechanical synthetic voice for sci-fi
- custom: Placeholder for custom user voices

SEGMENTS TO ANALYZE:
{segments_json}

LANGUAGE: {language}

RULES:
- speaker_type: "narrator" or "dialogue"
- character_name: The actual character name (proper noun), or null for narrator
- emotion: One of: calm, excited, angry, sad, whisper, tense, urgent, amused, contemptuous, surprised, neutral
- character_description: 2-5 words describing the speaker (e.g. "Young energetic male")
- suggested_voice_id: Best match from the Available Voice Profiles list
- voice_id: Same as suggested_voice_id
- emotion_instruction: French instruction for TTS (e.g. "Parlez avec un ton calme et pose")

RESPOND WITH JSON ARRAY ONLY. No markdown, no explanation, no text before or after the JSON.
The response MUST be a valid JSON array starting with [ and ending with ].

Example response format:
[
  {{"segment_id": "seg_001", "speaker_type": "narrator", "character_name": null, "emotion": "calm", "character_description": "Narrator voice", "suggested_voice_id": "narrator_male", "voice_id": "narrator_male", "emotion_instruction": "Parlez avec un ton calme et pose"}},
  {{"segment_id": "seg_002", "speaker_type": "dialogue", "character_name": "Marcus", "emotion": "angry", "character_description": "Young angry male", "suggested_voice_id": "young_male", "voice_id": "young_male", "emotion_instruction": "Parlez avec colere et tension"}}
]
"""


@dataclass
class SpeechTag:
    """Result of character analysis for a single segment."""
    segment_id: str
    speaker_type: str
    character_name: Optional[str]
    emotion: str
    voice_id: str
    emotion_instruction: str
    character_description: str = ""
    suggested_voice_id: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "speaker_type": self.speaker_type,
            "character_name": self.character_name,
            "emotion": self.emotion,
            "voice_id": self.voice_id,
            "emotion_instruction": self.emotion_instruction,
            "character_description": self.character_description,
            "suggested_voice_id": self.suggested_voice_id,
        }


def get_llm_models_from_backend(
    backend: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> Tuple[bool, List[str], str]:
    """Auto-detect available models from an LLM backend.

    Args:
        backend: One of "lmstudio", "ollama", "openrouter"
        base_url: Base URL for local backends (LM Studio, Ollama)
        api_key: API key for remote backends
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, model_list, error_message)
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        return False, [], "urllib not available"

    if backend == "lmstudio":
        url = (base_url or "http://localhost:1234/v1") + "/models"
        try:
            req = urllib.request.Request(url)
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            model_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            if model_ids:
                return True, model_ids, ""
            return False, [], "No models returned"
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                Exception) as e:
            return False, [], str(e)

    elif backend == "ollama":
        url = (base_url or "http://localhost:11434") + "/api/tags"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            model_ids = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            # Ollama models aren't OpenAI API models by default, but we can still list them
            if model_ids:
                return True, model_ids, ""
            return False, [], "No models returned"
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                Exception) as e:
            return False, [], str(e)

    elif backend == "openrouter":
        key = api_key or ""
        if not key:
            return False, [], "OpenRouter API key not provided"
        url = "https://openrouter.ai/api/v1/models"
        try:
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            model_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            if model_ids:
                return True, model_ids, ""
            return False, [], "No models returned"
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                Exception) as e:
            return False, [], str(e)

    return False, [], f"Unknown backend: {backend}"


def test_llm_connection(
    backend: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[bool, str]:
    """Test if an LLM backend is reachable and can process a simple request.

    Args:
        backend: One of "lmstudio", "ollama", "openrouter"
        base_url: Base URL for local backends
        model: Model ID to test with
        api_key: API key for remote backends
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    try:
        from openai import OpenAI
    except ImportError:
        return False, "openai package not installed"

    if backend == "lmstudio":
        base = base_url or "http://localhost:1234/v1"
        client = OpenAI(base_url=base, api_key="unused")
        test_model = model or ""
    elif backend == "ollama":
        base = base_url or "http://localhost:11434/v1"
        client = OpenAI(base_url=base, api_key="ollama")
        test_model = model or "qwen3:32b"
    elif backend == "openrouter":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key or "")
        test_model = model or "openai/gpt-4o-mini"
    else:
        return False, f"Unknown backend: {backend}"

    try:
        response = client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
            timeout=timeout,
        )
        content = response.choices[0].message.content.strip()
        return True, f"Connected. Model replied: \"{content}\""
    except Exception as e:
        return False, f"Connection failed: {e}"


class CharacterAnalyzer:
    """Analyzes text segments to detect characters, emotions, and suggest voices.

    Supports LM Studio, OpenRouter API, and Ollama backends.
    Auto-detects the model from LM Studio/Ollama when configured.
    """

    def __init__(self, config: dict, session=None):
        """
        Args:
            config: Configuration dict for the analysis section
            session: Optional pre-created OpenAI client
        """
        self.config = config
        self._backend = config.get("llm_backend", "lmstudio")
        self._max_retries = config.get("max_retries", 3)
        self._batch_size = config.get("batch_size", 5)

        self._cache: Dict[str, Dict] = {}
        self._characters: Dict[str, set] = {}
        self._model: str = ""
        self._session = session

        if self._session is None:
            self._session, self._model = self._create_client()

    def _create_client(self) -> Tuple[Any, str]:
        """Create an OpenAI-compatible client for the configured backend.

        For LM Studio: auto-detect models if not set.
        For Ollama: use configured model name.
        For OpenRouter: use configured model name + API key.

        Returns:
            Tuple of (OpenAI client, model_name)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for CharacterAnalyzer")

        if self._backend == "lmstudio":
            base_url = self.config.get("lmstudio_base_url", "http://localhost:1234/v1")
            # Ensure base_url ends with /v1 for OpenAI compatibility
            if not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"

            model = self.config.get("lmstudio_model", "")

            # Auto-detect model if not configured or empty
            if not model:
                logger.info("No LM Studio model configured, auto-detecting...")
                ok, models, err = get_llm_models_from_backend(
                    "lmstudio", base_url=base_url
                )
                if ok and models:
                    model = models[0]
                    logger.info(f"Auto-detected LM Studio model: {model}")
                    # Save it back to config for next time
                    self.config["lmstudio_model"] = model
                else:
                    logger.warning(
                        f"Could not auto-detect LM Studio models: {err}. "
                        "Make sure LM Studio has a model loaded and is running."
                    )

            if not model:
                raise ValueError(
                    "No LM Studio model found. Load a model in LM Studio first, "
                    "or set lmstudio_model in config."
                )

            client = OpenAI(base_url=base_url, api_key="unused")
            logger.info(f"CharacterAnalyzer -> LM Studio: {base_url} / {model}")
            return client, model

        elif self._backend == "openrouter":
            api_key = self.config.get("openrouter_api_key", "")
            model = self.config.get("openrouter_model", "openai/gpt-4o-mini")
            if not api_key:
                raise ValueError("OpenRouter API key not set.")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            logger.info(f"CharacterAnalyzer -> OpenRouter: {model}")
            return client, model

        elif self._backend == "ollama":
            base_url = self.config.get("ollama_base_url", "http://localhost:11434")
            model = self.config.get("ollama_model", "qwen3:32b")
            client = OpenAI(
                base_url=f"{base_url.rstrip('/')}/v1",
                api_key="ollama",
            )
            logger.info(f"CharacterAnalyzer -> Ollama: {model}")
            return client, model

        else:
            raise ValueError(f"Unknown LLM backend: {self._backend}")

    @staticmethod
    def discover_models(
        backend: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[bool, List[str], str]:
        """Static helper: discover available models from a backend.

        Delegates to the module-level function.
        """
        return get_llm_models_from_backend(
            backend=backend,
            base_url=base_url,
            api_key=api_key,
        )

    def analyze_segments(
        self,
        segments_list: list,
        language: str = "french",
    ) -> Tuple[Dict[str, SpeechTag], List[str]]:
        """Blocking version: analyzes all segments and returns final result.

        Args:
            segments_list: List of TextSegment objects or dicts
            language: Language of the text

        Returns:
            Tuple of ({segment_id: SpeechTag}, [character_names])
        """
        result = None
        for item in self.analyze_segments_iter(segments_list, language):
            if item.get("status") == "finished":
                result = item["result"]
        return result or ({}, [])

    def analyze_segments_iter(
        self,
        segments_list: list,
        language: str = "french",
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator version: yields progress updates during analysis.

        Yields dicts with keys:
        - status: "init", "batch_start", "batch_done", "finished"
        - msg: Human-readable status text
        - result: (tags_dict, char_list) when status is "finished"
        """
        all_tags: Dict[str, SpeechTag] = {}
        all_chars: List[str] = []
        batch: List[dict] = []
        total = len(segments_list)
        batch_num = 0

        if total > 0:
            yield {"status": "init", "msg": f"Initialized. Total {total} segments to analyze."}

        for segment in segments_list:
            seg_id = segment.id if hasattr(segment, "id") else segment.get("id", "")
            seg_text = segment.text if hasattr(segment, "text") else segment.get("text", "")
            batch.append({"segment_id": seg_id, "text": seg_text})

            if len(batch) >= self._batch_size:
                batch_num += 1
                yield {"status": "batch_start", "msg": f"Analyzing Batch {batch_num}..."}

                tags, chars = self._analyze_batch(batch, language)
                all_tags.update(tags)
                all_chars.extend(chars)

                for char_name in chars:
                    if char_name not in self._characters:
                        self._characters[char_name] = set()
                for tag in all_tags.values():
                    if tag.character_name:
                        self._characters[tag.character_name].add(tag.segment_id)

                batch.clear()
                yield {
                    "status": "batch_done",
                    "msg": f"Batch {batch_num} complete. Found {len(all_chars)} characters so far.",
                }

        # Process remaining segments
        if batch:
            batch_num += 1
            yield {"status": "batch_start", "msg": f"Analyzing Final Batch {batch_num}..."}
            tags, chars = self._analyze_batch(batch, language)
            all_tags.update(tags)
            all_chars.extend(chars)
            for char_name in chars:
                if char_name not in self._characters:
                    self._characters[char_name] = set()
            for tag in all_tags.values():
                if tag.character_name:
                    self._characters[tag.character_name].add(tag.segment_id)
            yield {
                "status": "batch_done",
                "msg": f"All batches complete. Found {len(all_chars)} characters.",
            }

        unique_chars = list(dict.fromkeys(all_chars))
        logger.info(f"Analysis done: {len(all_tags)} segments, {len(unique_chars)} characters")
        yield {
            "status": "finished",
            "msg": "Analysis complete!",
            "result": (all_tags, unique_chars),
        }

    def _analyze_batch(
        self,
        batch: List[dict],
        language: str,
    ) -> Tuple[Dict[str, SpeechTag], List[str]]:
        """Analyze a single batch of segments via LLM."""
        if not batch:
            return {}, []

        cache_key = json.dumps(batch, sort_keys=True)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return self._dict_to_tags(cached), cached.get("characters", [])

        segments_json = json.dumps(batch, ensure_ascii=False, indent=2)
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            segments_json=segments_json,
            language=language,
        )

        for attempt in range(self._max_retries):
            try:
                response = self._session.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert literary analyst. "
                                "Respond ONLY with a valid JSON array. "
                                "No text, no explanation, no markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    timeout=300000
                )

                content = response.choices[0].message.content.strip()
                parsed = self._extract_json(content)

                if parsed is None:
                    logger.warning(
                        f"LLM returned non-JSON response (attempt {attempt + 1})"
                    )
                    continue

                # Unwrap nested dicts (some models wrap the array)
                if isinstance(parsed, dict):
                    for key in ("analysis", "result", "data", "tags"):
                        if key in parsed and isinstance(parsed[key], list):
                            parsed = parsed[key]
                            break

                if not isinstance(parsed, list):
                    logger.warning(
                        f"Expected JSON array, got {type(parsed).__name__} "
                        f"(attempt {attempt + 1})"
                    )
                    if attempt < self._max_retries - 1:
                        continue
                    return {}, []

                tags_dict: Dict[str, SpeechTag] = {}
                characters: List[str] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    seg_id = item.get("segment_id", "")
                    if not seg_id:
                        continue
                    char_name = item.get("character_name")
                    if char_name and isinstance(char_name, str):
                        characters.append(char_name)
                    elif char_name is not None:
                        char_name = None

                    tag = SpeechTag(
                        segment_id=seg_id,
                        speaker_type=item.get("speaker_type", "narrator"),
                        character_name=char_name,
                        emotion=item.get("emotion", "neutral"),
                        voice_id=item.get("voice_id", "narrator_male"),
                        emotion_instruction=item.get("emotion_instruction", ""),
                        character_description=item.get("character_description", ""),
                        suggested_voice_id=item.get("suggested_voice_id", "narrator_male"),
                    )
                    tags_dict[seg_id] = tag

                if tags_dict:
                    self._cache[cache_key] = {
                        "analysis": parsed,
                        "characters": characters,
                    }
                    for cn in characters:
                        if cn not in self._characters:
                            self._characters[cn] = set()
                    return tags_dict, characters

            except Exception as e:
                logger.warning(
                    f"LLM analysis error (attempt {attempt + 1}/{self._max_retries}): {e}"
                )

            if attempt < self._max_retries - 1:
                wait_time = 1.5 * (2 ** attempt)
                logger.info(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        # Fallback: create narrator-only tags
        logger.warning(
            f"Analysis failed after {self._max_retries} attempts, using fallback."
        )
        fallback_tags: Dict[str, SpeechTag] = {}
        for seg in batch:
            tid = seg.get("segment_id", "")
            fallback_tags[tid] = SpeechTag(
                segment_id=tid,
                speaker_type="narrator",
                character_name=None,
                emotion="neutral",
                voice_id="narrator_male",
                emotion_instruction=EMOTION_INSTRUCTIONS_FR["neutral"],
                character_description="Narrator",
                suggested_voice_id="narrator_male",
            )
        return fallback_tags, []

    def _dict_to_tags(self, cached: dict) -> Dict[str, SpeechTag]:
        """Convert cached dict back to SpeechTag dict."""
        analysis = cached.get("analysis", [])
        if not isinstance(analysis, list):
            return {}
        tags: Dict[str, SpeechTag] = {}
        for item in analysis:
            if not isinstance(item, dict):
                continue
            seg_id = item.get("segment_id", "")
            if not seg_id:
                continue
            tag = SpeechTag(
                segment_id=seg_id,
                speaker_type=item.get("speaker_type", "narrator"),
                character_name=item.get("character_name"),
                emotion=item.get("emotion", "neutral"),
                voice_id=item.get("voice_id", "narrator_male"),
                emotion_instruction=item.get("emotion_instruction", ""),
                character_description=item.get("character_description", ""),
                suggested_voice_id=item.get("suggested_voice_id", "narrator_male"),
            )
            tags[seg_id] = tag
        return tags

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract JSON from text, handling markdown and conversational text."""
        text = text.strip()
        if not text:
            return None

        # Remove markdown code blocks
        md_match = re.findall(r'```(?:json)?\s*\n(.*?)\n\s*```', text, re.DOTALL)
        if md_match:
            text = md_match[0].strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            try:
                return json.loads(text[start_obj:end_obj + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not find valid JSON. Preview: {text[:200]}")
        return None

    def get_discovered_characters(self) -> List[str]:
        """Get sorted list of discovered character names."""
        return sorted(self._characters.keys())

    def get_character_segments(self, character_name: str) -> List[str]:
        """Get segment IDs for a specific character."""
        return sorted(self._characters.get(character_name, set()))
