"""Character Analyzer - Uses LLM to detect characters, emotions, and assign voice IDs."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid emotion types
VALID_EMOTIONS = [
    "calm", "excited", "angry", "sad", "whisper",
    "tense", "urgent", "amused", "contemptuous", "surprised", "neutral"
]

# Emotion instructions for Qwen3-TTS in French
EMOTION_INSTRUCTIONS_FR = {
    "calm": "Parlez avec un ton calme et posé, voix douce et régulière",
    "excited": "Parlez avec excitation et enthousiasme, voix énergique et vive",
    "angry": "Parlez avec colère et tension, voix ferme et intense",
    "sad": "Parlez d'une voix triste et mélancolique, ton doux et lent",
    "whisper": "Chuchotez d'une voix mystérieuse, ton bas et intime",
    "tense": "Parlez avec un ton tendu et nerveux, voix serrée et rapide",
    "urgent": "Parlez avec urgence, voix rapide et pressante",
    "amused": "Parlez avec amusement, voix légère et joyeuse",
    "contemptuous": "Parlez avec mépris, voix froide et distante",
    "surprised": "Parlez avec surprise, voix étonnée et expressive",
    "neutral": "Parlez d'un ton neutre et naturel, sans émotion particulière",
}

# Emotion instructions for Qwen3-TTS in English
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


@dataclass
class SpeechTag:
    """Result of character analysis for a single segment."""
    segment_id: str
    speaker_type: str  # "narrator" or "dialogue"
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
            "reasoning": self.reasoning,
        }


# LLM prompt template for character/emotion analysis
ANALYSIS_PROMPT_TEMPLATE = """You are an expert literary analyst for audiobook production.
Your task is to analyze text segments from a book and identify:
1. Who is speaking (narrator vs character dialogue)
2. Which character is speaking (if dialogue)
3. The emotion/tone of the speech
4. An appropriate voice profile ID

You are analyzing segments from a French/English bilingual book. Pay close attention to French dialogue markers like « guillemets » and French dialogue conventions.

SEGMENTS TO ANALYZE:
{segments_json}

LANGUAGE: {language}

INSTRUCTIONS:
- Identify if each segment is "narrator" (story narration) or "dialogue" (character speaking)
- For dialogue, extract the character name from context. Use proper names (capitalize them).
- If the speaker is unclear, use "dialogue" but set character_name to null and voice_id to "narrator"
- Detect emotion from context. Choose from: calm, excited, angry, sad, whisper, tense, urgent, amused, contemptuous, surprised, neutral
- For character_description: Provide 2-5 word description (e.g. "Young male", "Old female", "Robot")
- For suggested_voice_id: Choose from [narrator_male, narrator_female, young_male, young_female, elder_male, elder_female, robotic, custom], no spaces (e.g., "marcus_dupont")
- For narrator voice_id: use "narrator"
- emotion_instruction: Provide a natural French instruction for how the TTS should speak
  - For French segments, write the instruction in French (e.g., "Parlez avec un ton calme et posé, voix douce")
  - For English segments, you may write in English or French, but French is preferred
  - Be specific about tone, pacing, and vocal quality
- reasoning: Brief explanation (one sentence) of your choices

OUTPUT FORMAT:
Return ONLY valid JSON matching this exact schema - no markdown, no explanation, no extra text:
[
  {{
    "segment_id": "the original segment_id",
    "speaker_type": "narrator" or "dialogue",
    "character_name": "CharacterName" or null,
    "emotion": "one of the valid emotion values",
    "voice_id": "voice_profile_id",
    "emotion_instruction": "French instruction for TTS emotion/style",
    "reasoning": "Brief explanation"
  }}
]

Be consistent in character naming across segments. If a character appears multiple times, use the exact same name and voice_id.
"""


class CharacterAnalyzer:
    """Analyzes text to identify characters, emotions, and assign voice profiles.

    Uses either OpenRouter API or local Ollama for LLM-based analysis.
    """

    def __init__(self, config: dict, session=None):
        """
        Args:
            config: Configuration dict for the analysis section
            session: Optional OpenAI-compatible session (OpenRouter or Ollama client)
        """
        self.config = config
        self._backend = config.get("llm_backend", "openrouter")
        self._max_retries = config.get("max_retries", 3)
        self._batch_size = config.get("batch_size", 5)

        # Cache: {hash_of_segments_text: result_dict}
        self._cache: Dict[str, Dict] = {}

        # Discovered characters: {character_name: set of segments}
        self._characters: Dict[str, set] = {}

        # Create OpenAI-compatible client
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
            model = self.config.get("openrouter_model", "anthropic/claude-sonnet-4-20250514")
            if not api_key:
                raise ValueError("OpenRouter API key not set. Set OPENROUTER_API_KEY env var or configure it.")
            
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
                api_key="lm-studio",  # LM Studio uses this as default
            )
            # LM Studio auto-detects loaded model; model param is often ignored
            self._model = model if model else ""

        else:
            raise ValueError(f"Unknown LLM backend: {self._backend}")

        logger.info(f"CharacterAnalyzer initialized with {self._backend}/{self._model}")
        return client


    def analyze_segments_iter(self, segments_list, language="french"):
        """Generator version of analyze_segments. Yields progress updates."""
        all_tags = {}
        all_chars = []
        batch = []
        total = len(segments_list)
        batch_num = 0
        
        # Yield initial status
        yield {"status": "init", "msg": f"Initialized. Segmenting {total} segments."}

        for segment in segments_list:
            seg_id = segment.id if hasattr(segment, 'id') else segment.get('id', '')
            seg_text = segment.text if hasattr(segment, 'text') else segment.get('text', '')
            batch.append({"segment_id": seg_id, "text": seg_text})

            if len(batch) >= self._batch_size:
                batch_num += 1
                yield {"status": "batch_start", "msg": f"Analyzing Batch {batch_num}..."}
                
                # Call internal batch analysis
                tags, chars = self._analyze_batch(batch, language)
                all_tags.update(tags)
                all_chars.extend(chars)
                
                # Update internal tracking
                for char_name in chars:
                    if char_name not in self._characters:
                        self._characters[char_name] = set()
                
                batch = []
                yield {"status": "batch_done", "msg": f"Batch {batch_num} complete. Found {len(all_chars)} characters so far."}

        # Process remaining
        if batch:
            batch_num += 1
            tags, chars = self._analyze_batch(batch, language)
            all_tags.update(tags)
            all_chars.extend(chars)
            for char_name in chars:
                if char_name not in self._characters:
                    self._characters[char_name] = set()
            yield {"status": "batch_done", "msg": f"Final Batch complete. Found {len(all_chars)} characters."}

        unique_chars = list(dict.fromkeys(all_chars))
        yield {"status": "finished", "msg": "Analysis complete!", "result": (all_tags, unique_chars)}

    def analyze_segments(
        self,
        segments_list: list,
        language: str = "french",
    ) -> Tuple[Dict[str, SpeechTag], List[str]]:
        """Analyze a list of text segments to identify speakers and emotions.

        Args:
            segments_list: List of TextSegment objects or dicts with id/text
            language: Primary language of the text ("french" or "english")

        Returns:
            Tuple of:
            - dict mapping segment_id to SpeechTag
            - list of discovered character names
        """
        if not segments_list:
            return {}, []

        all_tags: Dict[str, SpeechTag] = {}
        all_chars: List[str] = []

        # Process in batches
        batch = []
        for segment in segments_list:
            seg_id = segment.id if hasattr(segment, 'id') else segment.get('id', '')
            seg_text = segment.text if hasattr(segment, 'text') else segment.get('text', '')
            batch.append({"segment_id": seg_id, "text": seg_text})

            if len(batch) >= self._batch_size:
                tags, chars = self._analyze_batch(batch, language)
                all_tags.update(tags)
                all_chars.extend(chars)
                batch = []

        # Process remaining
        if batch:
            tags, chars = self._analyze_batch(batch, language)
            all_tags.update(tags)
            all_chars.extend(chars)

        # Deduplicate characters
        unique_chars = list(dict.fromkeys(all_chars))
        logger.info(f"Analyzed {len(all_tags)} segments, found {len(unique_chars)} characters")
        return all_tags, unique_chars

    def _analyze_batch(
        self,
        batch: List[dict],
        language: str,
    ) -> Tuple[Dict[str, SpeechTag], List[str]]:
        """Analyze a single batch of segments via LLM.

        Args:
            batch: List of {"segment_id": str, "text": str}
            language: Language code

        Returns:
            Tuple of (tags_dict, character_names)
        """
        if not batch:
            return {}, []

        # Check cache
        cache_key = json.dumps(batch, sort_keys=True)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return self._dict_to_tags(cached), cached.get("characters", [])

        # Build prompt
        segments_json = json.dumps(batch, ensure_ascii=False, indent=2)
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            segments_json=segments_json,
            language=language,
        )

        tags_dict = {}
        characters = []

        for attempt in range(self._max_retries):
            try:
                response = self._session.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert literary analyst. You respond ONLY with valid JSON. No markdown, no explanation."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    timeout=300.0,               )

                content = response.choices[0].message.content.strip()

                # Try to parse JSON - handle markdown code blocks
                content = self._extract_json(content)
                
                if content is None:
                    logger.warning(f"LLM returned non-JSON response (attempt {attempt + 1})")
                    continue

                # Convert to tags
                tags_dict, characters = self._parse_analysis_response(content, batch)
                if tags_dict:
                    # Cache the result
                    self._cache[cache_key] = {
                        "analysis": content,
                        "characters": characters,
                    }
                    
                    # Track characters
                    for char_name in characters:
                        if char_name not in self._characters:
                            self._characters[char_name] = set()
                    for tag in tags_dict.values():
                        if tag.character_name:
                            self._characters[tag.character_name].add(tag.segment_id)

                    return tags_dict, characters

            except Exception as e:
                wait_time = (2 ** attempt) * 1.5
                logger.warning(
                    f"LLM analysis error (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM analysis failed after {self._max_retries} attempts")
                    break

        # If all retries failed, create fallback tags (all narrator, calm)
        logger.warning("Using fallback analysis for all segments")
        fallback_lang = language
        for item in batch:
            seg_id = item["segment_id"]
            emotion = "neutral"
            instr = self._get_emotion_instruction(emotion, fallback_lang)
            tag = SpeechTag(
                segment_id=seg_id,
                speaker_type="narrator",
                character_name=None,
                emotion=emotion,
                voice_id="narrator",
                emotion_instruction=instr,
                reasoning="Fallback: analysis failed",
            )
            tags_dict[seg_id] = tag

        return tags_dict, []

    def _get_emotion_instruction(self, emotion: str, language: str) -> str:
        """Get emotion instruction text for Qwen3-TTS.

        Args:
            emotion: Emotion type
            language: "french" or "english"

        Returns:
            Instruction string
        """
        if language.lower() in ("french", "fr"):
            return EMOTION_INSTRUCTIONS_FR.get(emotion, EMOTION_INSTRUCTIONS_FR["neutral"])
        else:
            return EMOTION_INSTRUCTIONS_EN.get(emotion, EMOTION_INSTRUCTIONS_EN["neutral"])

    def _parse_analysis_response(
        self,
        json_data: Any,
        batch: List[dict],
    ) -> Tuple[Dict[str, SpeechTag], List[str]]:
        """Parse LLM JSON response into SpeechTag objects.

        Args:
            json_data: Parsed JSON data (list of dicts)
            batch: Original batch of segments (for fallback matching)

        Returns:
            Tuple of (tags_dict, character_names)
        """
        tags = {}
        characters_set = set()

        if not isinstance(json_data, list):
            logger.warning("LLM response is not a list")
            return tags, []

        for item in json_data:
            if not isinstance(item, dict):
                continue

            seg_id = item.get("segment_id", "")
            if not seg_id:
                continue

            speaker_type = item.get("speaker_type", "narrator")
            if speaker_type not in ("narrator", "dialogue"):
                speaker_type = "narrator"

            character_name = item.get("character_name")
            if character_name and isinstance(character_name, str):
                character_name = character_name.strip()
                if not character_name or character_name.lower() in ("none", "null", ""):
                    character_name = None
            else:
                character_name = None

            emotion = item.get("emotion", "neutral")
            # Validate emotion
            emotions_list = [e.lower() for e in VALID_EMOTIONS]
            if emotion.lower() not in emotions_list:
                emotion = "neutral"
            else:
                # Normalize to canonical form
                idx = emotions_list.index(emotion.lower())
                emotion = VALID_EMOTIONS[idx]

            voice_id = item.get("voice_id", "narrator")
            if not voice_id or speaker_type == "narrator":
                voice_id = "narrator"
            else:
                voice_id = voice_id.lower().replace(" ", "_")

            # Build emotion instruction
            emotion_instr = item.get("emotion_instruction", "")
            if not emotion_instr:
                emotion_instr = self._get_emotion_instruction(emotion, "french")

            reasoning = item.get("reasoning", "")

            tag = SpeechTag(
                segment_id=seg_id,
                speaker_type=speaker_type,
                character_name=character_name,
                emotion=emotion,
                voice_id=voice_id,
                emotion_instruction=emotion_instr,
                reasoning=reasoning,
            )
            tags[seg_id] = tag

            if character_name:
                characters_set.add(character_name)

        return tags, list(characters_set)

    def _dict_to_tags(self, cached: dict) -> Dict[str, SpeechTag]:
        """Convert cached analysis back to SpeechTag dict."""
        # Re-parse the stored JSON
        analysis = cached.get("analysis", [])
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except json.JSONDecodeError:
                return {}
        tags = {}
        for item in analysis:
            if not isinstance(item, dict):
                continue
            seg_id = item.get("segment_id", "")
            if not seg_id:
                continue
            tag = SpeechTag(
                segment_id=item.get("segment_id", ""),
                speaker_type=item.get("speaker_type", "narrator"),
                character_name=item.get("character_name"),
                emotion=item.get("emotion", "neutral"),
                voice_id=item.get("voice_id", "narrator"),
                emotion_instruction=item.get("emotion_instruction", ""),
                reasoning=item.get("reasoning", ""),
            )
            tags[seg_id] = tag
        return tags

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract JSON from text, handling markdown code blocks and extra text.

        Args:
            text: Raw LLM response text

        Returns:
            Parsed JSON object, or None
        """
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in text
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

        logger.warning("Could not find valid JSON in LLM response")
        return None

    def get_discovered_characters(self) -> List[str]:
        """Get list of all discovered character names.

        Returns:
            Sorted list of unique character names
        """
        return sorted(self._characters.keys())

    def get_character_segments(self, character_name: str) -> List[str]:
        """Get list of segment IDs for a specific character.

        Args:
            character_name: Character name

        Returns:
            List of segment IDs
        """
        return list(self._characters.get(character_name, set()))

    def clear_cache(self):
        """Clear analysis cache."""
        self._cache.clear()
        self._characters.clear()
