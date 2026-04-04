"""Character Analyzer - Uses LLM to detect characters, emotions, and assign voice IDs.

Handles batched analysis with progress reporting via generator pattern.
Supports LM Studio, OpenRouter API, and Ollama backends.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


class CharacterAnalyzer:
    """Analyzes text segments to detect characters, emotions, and suggest voices.

    Uses an OpenAI-compatible API (LM Studio, OpenRouter, or Ollama).
    Provides both synchronous and generator-based analysis methods.
    """

    def __init__(self, config: dict, session=None):
        """
        Args:
            config: Configuration dict for the analysis section
            session: Optional OpenAI-compatible session
        """
        self.config = config
        self._backend = config.get("llm_backend", "lmstudio")
        self._max_retries = config.get("max_retries", 3)
        self._batch_size = config.get("batch_size", 5)

        self._cache: Dict[str, Dict] = {}
        self._characters: Dict[str, set] = {}

        self._session = session
        if self._session is None:
            self._session = self._create_client()

    def _create_client(self):
        """Create an OpenAI-compatible client for the configured backend."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for CharacterAnalyzer")

        if self._backend == "openrouter":
            api_key = self.config.get("openrouter_api_key", "")
            model = self.config.get("openrouter_model", "qwen/qwen3.6-plus:free")
            if not api_key:
                raise ValueError("OpenRouter API key not set.")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self._model = model

        elif self._backend == "ollama":
            base_url = self.config.get("ollama_base_url", "http://localhost:11434")
            model = self.config.get("ollama_model", "qwen3:32b")
            client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama",
            )
            self._model = model

        elif self._backend == "lmstudio":
            base_url = self.config.get("lmstudio_base_url", "http://localhost:1234")
            model = self.config.get("lmstudio_model", "")
            client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="lm-studio",
            )
            self._model = model if model else ""

        else:
            raise ValueError(f"Unknown LLM backend: {self._backend}")

        logger.info(f"CharacterAnalyzer initialized with {self._backend}/{self._model}")
        return client

    def analyze_segments(self, segments_list, language="french"):
        """Blocking version: analyzes all segments and returns final result."""
        result = None
        for item in self.analyze_segments_iter(segments_list, language):
            if item.get("status") == "finished":
                result = item["result"]
        return result

    def analyze_segments_iter(self, segments_list, language="french"):
        """Generator version: yields progress updates during analysis."""
        all_tags = {}
        all_chars = []
        batch = []
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
                for tag in tags.values():
                    if tag.character_name:
                        self._characters[tag.character_name].add(tag.segment_id)

                batch = []
                yield {
                    "status": "batch_done",
                    "msg": f"Batch {batch_num} complete. Found {len(all_chars)} characters so far.",
                }

        # Process remaining
        if batch:
            batch_num += 1
            yield {"status": "batch_start", "msg": f"Analyzing Final Batch {batch_num}..."}
            tags, chars = self._analyze_batch(batch, language)
            all_tags.update(tags)
            all_chars.extend(chars)
            for char_name in chars:
                if char_name not in self._characters:
                    self._characters[char_name] = set()
            for tag in tags.values():
                if tag.character_name:
                    self._characters[tag.character_name].add(tag.segment_id)
            yield {
                "status": "batch_done",
                "msg": f"All batches complete. Found {len(all_chars)} characters.",
            }

        unique_chars = list(dict.fromkeys(all_chars))
        logger.info(f"Analysis done: {len(all_tags)} segments, {len(unique_chars)} characters")
        yield {"status": "finished", "msg": "Analysis complete!", "result": (all_tags, unique_chars)}

    def _analyze_batch(self, batch, language):
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
                            "content": "You are an expert literary analyst. Respond ONLY with a valid JSON array. No text, no explanation, no markdown."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    timeout=300.0,
                )

                content = response.choices[0].message.content.strip()
                parsed = self._extract_json(content)

                if parsed is None:
                    logger.warning(f"LLM returned non-JSON response (attempt {attempt + 1})")
                    continue

                if isinstance(parsed, dict) and "analysis" in parsed:
                    parsed = parsed["analysis"]
                if isinstance(parsed, dict) and "result" in parsed:
                    parsed = parsed["result"]

                if not isinstance(parsed, list):
                    logger.warning(f"Expected JSON array, got {type(parsed).__name__} (attempt {attempt + 1})")
                    if attempt < self._max_retries - 1:
                        continue
                    return {}, []

                tags_dict = {}
                characters = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    seg_id = item.get("segment_id", "")
                    if not seg_id:
                        continue
                    char_name = item.get("character_name")
                    if char_name:
                        characters.append(char_name)
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
                logger.warning(f"LLM analysis error (attempt {attempt + 1}/{self._max_retries}): {e}")

            if attempt < self._max_retries - 1:
                wait_time = 1.5 * (2 ** attempt)
                logger.info(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        # Fallback: create narrator-only tags
        logger.warning(f"Analysis failed after {self._max_retries} attempts, using fallback.")
        fallback_tags = {}
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

    def _get_emotion_instruction(self, emotion, language):
        """Get emotion-specific TTS instruction text."""
        lang = "french"
        instructions = EMOTION_INSTRUCTIONS_FR
        if "english" in language.lower() or "en" in language.lower():
            instructions = EMOTION_INSTRUCTIONS_EN
        elif "french" in language.lower() or "fr" in language.lower():
            instructions = EMOTION_INSTRUCTIONS_FR
        return instructions.get(emotion, instructions.get("neutral", ""))

    def _parse_analysis_response(self, analysis, batch):
        """Parse the LLM JSON array response into speech tags."""
        if not isinstance(analysis, list):
            return {}, []
        tags = {}
        characters = []
        for item in analysis:
            if not isinstance(item, dict):
                continue
            seg_id = item.get("segment_id", "")
            if not seg_id:
                continue
            char_name = item.get("character_name")
            if char_name:
                characters.append(char_name)
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
            tags[seg_id] = tag
        return tags, characters

    def _dict_to_tags(self, cached):
        """Convert cached dict back to SpeechTag dict."""
        analysis = cached.get("analysis", [])
        if not isinstance(analysis, list):
            return {}
        tags = {}
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
    def _extract_json(text):
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

    def get_discovered_characters(self):
        """Get sorted list of discovered character names."""
        return sorted(self._characters.keys())

    def get_character_segments(self, character_name):
        """Get segment IDs for a specific character."""
        return sorted(self._characters.get(character_name, set()))
