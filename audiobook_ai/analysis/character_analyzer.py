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

# Simple per-segment prompt — very short, local model friendly
SEGMENT_PROMPT = '''Analyze this text segment for audiobook production.

TEXT: "{text}"

Return ONLY a JSON object with these fields:
- speaker_type: "narrator" if narration, or "dialogue" if a character speaks
- character_name: The character's name if dialogue, or null for narrator
- emotion: one of: calm, excited, angry, sad, whisper, tense, urgent, amused, contemptuous, surprised, neutral

Example: {{"speaker_type":"narrator","character_name":null,"emotion":"calm"}}
Example: {{"speaker_type":"dialogue","character_name":"Jean","emotion":"angry"}}'''


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
    timeout: float = 5.0,
) -> Tuple[bool, List[str], str]:
    """Auto-detect available models from an LLM backend."""
    import urllib.request
    import urllib.error

    if backend == "lmstudio":
        url = (base_url or "http://localhost:1234/v1") + "/models"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            model_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            if model_ids:
                return True, model_ids, ""
            return False, [], "No models returned"
        except Exception as e:
            return False, [], str(e)

    elif backend == "ollama":
        url = (base_url or "http://localhost:11434") + "/api/tags"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            model_ids = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            if model_ids:
                return True, model_ids, ""
            return False, [], "No models returned"
        except Exception as e:
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
        except Exception as e:
            return False, [], str(e)

    return False, [], f"Unknown backend: {backend}"


def test_llm_connection(
    backend: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[bool, str]:
    """Test if an LLM backend is reachable."""
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
    """Analyzes text segments one-by-one to detect characters, emotions, and voices.
    
    Uses single-segment prompts for reliability with local models.
    """

    def __init__(self, config: dict, session=None):
        self.config = config
        self._backend = config.get("llm_backend", "lmstudio")
        self._max_retries = 2  # Lower retries since each call is fast
        # Process one segment at a time for reliability
        self._batch_size = 1

        self._cache: Dict[str, Dict] = {}
        self._characters: Dict[str, set] = {}
        self._model: str = ""
        self._session = session

        if self._session is None:
            self._session, self._model = self._create_client()

    def _create_client(self) -> Tuple[Any, str]:
        """Create an OpenAI-compatible client and determine model name."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for CharacterAnalyzer")

        if self._backend == "lmstudio":
            base_url = self.config.get("lmstudio_base_url", "http://localhost:1234/v1")
            if not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"

            model = self.config.get("lmstudio_model", "")

            if not model:
                logger.info("No LM Studio model configured, auto-detecting...")
                ok, models, err = get_llm_models_from_backend("lmstudio", base_url=base_url)
                if ok and models:
                    model = models[0]
                    logger.info(f"Auto-detected LM Studio model: {model}")
                    self.config["lmstudio_model"] = model
                else:
                    raise ValueError(
                        f"No LM Studio model found. Load a model first. Error: {err}"
                    )

            client = OpenAI(base_url=base_url, api_key="unused")
            logger.info(f"CharacterAnalyzer -> LM Studio: {base_url} / {model}")
            return client, model

        elif self._backend == "openrouter":
            api_key = self.config.get("openrouter_api_key", "")
            model = self.config.get("openrouter_model", "openai/gpt-4o-mini")
            if not api_key:
                raise ValueError("OpenRouter API key not set.")
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            logger.info(f"CharacterAnalyzer -> OpenRouter: {model}")
            return client, model

        elif self._backend == "ollama":
            base_url = self.config.get("ollama_base_url", "http://localhost:11434")
            model = self.config.get("ollama_model", "qwen3:32b")
            client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="ollama")
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
        """Static helper: discover available models from a backend."""
        return get_llm_models_from_backend(
            backend=backend, base_url=base_url, api_key=api_key,
        )

    def deduplicate_characters(self, char_list: List[str]) -> Dict[str, str]:
        """Ask LLM to merge character name variants.

        Sends the character list to the LLM and asks which names refer to the
        same person. Returns a mapping of variant -> canonical name.

        Example: {'Bobbie': 'Bobbie', 'Draper': 'Bobbie',
                  'Roberta Draper': 'Bobbie'}

        Args:
            char_list: List of character names found during analysis

        Returns:
            Dict mapping each variant to its canonical name
        """
        if len(char_list) <= 1:
            return {c: c for c in char_list}

        names_str = "\n".join(f"- {c}" for c in char_list)
        prompt = (
            f"Here is a list of character names detected in a French translation "
            f"of a book. Some names refer to the same character (nicknames, "
            f"titles, full names, partial names, etc.).\n\n"
            f"NAMES:\n{names_str}\n\n"
            f"Group names that refer to the SAME character. Return ONLY a JSON "
            f"object where keys are the original names and values are the chosen "
            f"canonical name for that group.\n\n"
            f"Example:\n"
            f'{{"Bobbie":"Bobbie","Draper":"Bobbie","Roberta Draper":"Bobbie",'
            f'"Chrisjen Avasarala":"Avasarala","Avasarala":"Avasarala"}}\n\n'
            f"Return ONLY the JSON object. No explanation."
        )

        for attempt in range(3):
            try:
                response = self._session.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                    timeout=60.0,
                )
                raw = response.choices[0].message.content or ""
                content = raw.strip()

                if not content:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    break

                parsed = self._extract_json(content)
                if isinstance(parsed, dict):
                    # Validate: all char_list names should be keys
                    result = {}
                    for c in char_list:
                        result[c] = parsed.get(c, c)  # default to self
                    print(f"\n[DEDUP] Merged {len(char_list)} names -> "
                          f"{len(set(result.values()))} unique characters")
                    for canonical in sorted(set(result.values())):
                        variants = [k for k, v in result.items() if v == canonical]
                        if len(variants) > 1:
                            print(f"  {canonical} <- {', '.join(variants)}")
                    return result
                else:
                    print(f"[DEDUP] Unexpected response type: {type(parsed).__name__}")
                    if attempt < 2:
                        continue
                    break

            except Exception as e:
                print(f"[DEDUP] Error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(1)

        print("[DEDUP] Deduplication failed, using names as-is")
        return {c: c for c in char_list}

    def build_voice_descriptions(self) -> Dict[str, dict]:
        """Generate ElevenLabs-style voice descriptions for each character.
        
        Returns dict mapping character name to voice description info."""
        descriptions = {}
        for char_name, segments in self._characters.items():
            char_lower = char_name.lower()
            # Gender inference
            is_female = any(w in char_lower for w in ["madame", "mademoiselle", "mme", "elle", "miss", "lady", "woman", "femme", "mei", "elise", "naomi", "giora", "tannen", "demanda"])
            gender_desc = "female" if is_female else "male"
            
            # Voice prompt generation
            is_military = any(w in char_lower for w in ["sergent", "capitaine", "draper", "bobbie", "roberta"])
            is_political = any(w in char_lower for w in ["avasarala", "chrisjen", "errin", "genera", "walter"])
            
            if is_military:
                voice_prompt = f"A {gender_desc} military voice, French accent. Firm, disciplined, authoritative tone."
            elif is_political:
                voice_prompt = f"A sophisticated {gender_desc} political voice, French accent. Measured and diplomatic."
            else:
                voice_prompt = f"A natural {gender_desc} voice, French accent. Clear and expressive audiobook voice."
            
            descriptions[char_name] = {
                "elevenlabs_prompt": voice_prompt,
                "french_description": voice_prompt,
                "voice_type": "custom",
                "segment_count": len(segments)
            }
        return descriptions


    def save_analysis(self, filepath: str,
                      segment_tags: Dict[str, SpeechTag],
                      char_list: List[str],
                      dedup_map: Optional[Dict[str, str]] = None):
        """Save character analysis to a JSON file for reuse."""
        data = {
            "segment_tags": {sid: tag.to_dict() for sid, tag in segment_tags.items()},
            "characters": char_list,
            "dedup_map": dedup_map or {},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved analysis to {filepath}")
        logger.info(f"Saved analysis to {filepath}")

    @staticmethod
    def load_analysis(filepath: str):
        """Load character analysis from a JSON file.

        Returns:
            (segment_tags_dict, char_list, dedup_map)
            segment_tags_dict: {segment_id: SpeechTag}
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        segment_tags = {}
        for sid, d in data.get("segment_tags", {}).items():
            segment_tags[sid] = SpeechTag(
                segment_id=d.get("segment_id", sid),
                speaker_type=d.get("speaker_type", "narrator"),
                character_name=d.get("character_name"),
                emotion=d.get("emotion", "neutral"),
                voice_id=d.get("voice_id", "narrator_male"),
                emotion_instruction=d.get("emotion_instruction", ""),
                character_description=d.get("character_description", ""),
                suggested_voice_id=d.get("suggested_voice_id", "narrator_male"),
            )

        char_list = data.get("characters", [])
        dedup_map = data.get("dedup_map", {})

        print(f"Loaded analysis: {len(segment_tags)} segments, "
              f"{len(char_list)} characters from {filepath}")
        logger.info(f"Loaded analysis: {len(segment_tags)} segments, "
                     f"{len(char_list)} characters from {filepath}")
        return segment_tags, char_list, dedup_map

    def analyze_segments(
        self,
        segments_list: list,
        language: str = "french",
    ) -> Tuple[Dict[str, SpeechTag], List[str], Dict[str, str]]:
        """Analyze all segments, returning (tags_dict, character_list, dedup_map)."""
        result = None
        for item in self.analyze_segments_iter(segments_list, language):
            if item.get("status") == "finished":
                result = item["result"]
        return result or ({}, [], {})

    def analyze_segments_iter(
        self,
        segments_list: list,
        language: str = "french",
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator: yields progress updates during analysis."""
        all_tags: Dict[str, SpeechTag] = {}
        all_chars: List[str] = []
        total = len(segments_list)
        done = 0
        start_time = time.time()

        if total > 0:
            yield {"status": "init", "msg": f"Initialized. {total} segments to analyze."}

        for i, segment in enumerate(segments_list):
            seg_id = segment.id if hasattr(segment, "id") else segment.get("id", "")
            seg_text = segment.text if hasattr(segment, "text") else segment.get("text", "")
            
            done = i + 1
            
            if not seg_text.strip():
                tag = SpeechTag(
                    segment_id=seg_id,
                    speaker_type="narrator",
                    character_name=None,
                    emotion="neutral",
                    voice_id="narrator_male",
                    emotion_instruction=EMOTION_INSTRUCTIONS_FR["neutral"],
                )
                all_tags[seg_id] = tag
            else:
                tag = self._analyze_single_segment(seg_id, seg_text, language)
                all_tags[seg_id] = tag

            if tag.character_name:
                if tag.character_name not in self._characters:
                    self._characters[tag.character_name] = set()
                self._characters[tag.character_name].add(seg_id)
                if tag.character_name not in all_chars:
                    all_chars.append(tag.character_name)

            # Print progress update every 10 segments + final
            if done % 10 == 0 or done == total:
                elapsed = time.time() - start_time
                avg = elapsed / done
                remaining = avg * (total - done)
                eta_m = int(remaining / 60)
                eta_s = int(remaining % 60)
                pct = done / total * 100
                
                chars_str = ", ".join(all_chars[:8])
                if len(all_chars) > 8:
                    chars_str += f" +{len(all_chars)-8} more"
                elif not chars_str:
                    chars_str = "none yet"

                progress_msg = (
                    f"[{done}/{total}] {pct:5.1f}%  "
                    f"ETA {eta_m:02d}:{eta_s:02d}  "
                    f"{len(all_chars)} chars: {chars_str}"
                )
                print(progress_msg)
    
                yield {
                    "status": "progress",
                    "msg": progress_msg,
                }

        total_time = time.time() - start_time
        unique_chars = list(dict.fromkeys(all_chars))
        print(f"\nAnalysis complete! {len(unique_chars)} characters in {total_time:.0f}s")
        for cn in unique_chars:
            print(f"  - {cn}: {len(self._characters[cn])} segments")

        # Deduplicate character name variants - merge similar names
        # Uses a simple heuristic: if one name is a substring of another, merge them
        deduped = self.deduplicate_characters(unique_chars)
        unique_merged = list(deduped.keys())
        print(f"\n[DEDUP] Merged {len(unique_chars)} characters -> {len(unique_merged)} unique characters:")
        for canonical, variants in deduped.items():
            if len(variants) > 1:
                print(f"  {canonical} <- {', '.join(variants)}")

        logger.info(
            f"Analysis complete: {len(all_tags)} segments, "
            f"{len(unique_chars)} chars -> {len(unique_merged)} deduped in {total_time:.0f}s"
        )

        # Recalculate segment counts with deduped names
        merged_chars = {}
        for cn in unique_merged:
            merged_chars[cn] = set()
        for seg_id, tag in all_tags.items():
            if tag.character_name:
                # Find which canonical group this variant belongs to
                for canonical, variants in deduped.items():
                    if tag.character_name in variants:
                        merged_chars[canonical].add(seg_id)
                        break

        self._characters = merged_chars

        yield {"status": "finished", "msg": "Analysis complete!", "result": (all_tags, unique_merged)}
        logger.info(
            f"Analysis complete: {len(all_tags)} segments, "
            f"{len(unique_merged)} unique characters in {total_time:.0f}s"
        )
        yield {"status": "finished", "msg": "Analysis complete!", "result": (all_tags, unique_merged, dedup_map)}

    # Dialogue marker characters (French + English quotes, dashes, etc.)
    _DIALOGUE_MARKERS = {"\"", "\u201c", "\u201d", "\u00ab", "\u00bb", "\u2014", "\u2013", "\u002d"}

    def _analyze_single_segment(
        self,
        seg_id: str,
        text: str,
        language: str,
    ) -> SpeechTag:
        """Analyze a single text segment via LLM, with pre-filter for narration."""
        # --- FAST PRE-FILTER: skip LLM for pure narration ---
        # If no dialogue markers exist, it's 99%+ certainly narration
        has_dialogue = any(ch in self._DIALOGUE_MARKERS for ch in text)
        if not has_dialogue:
            # Quick check: look for colon before quote pattern (French: Il dit : « ... »)
            # or simple quote-dash pattern
            import re
            if not re.search(r'[\u00ab\u201c"]|\u2014|\u2013\s+\w+', text):
                return SpeechTag(
                    segment_id=seg_id,
                    speaker_type="narrator",
                    character_name=None,
                    emotion="neutral",
                    voice_id="narrator_male",
                    emotion_instruction=EMOTION_INSTRUCTIONS_FR["neutral"],
                )

        # Check cache
        cache_key = json.dumps({"id": seg_id, "text": text[:200]}, sort_keys=True)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return self._tag_from_dict(cached)

        prompt = SEGMENT_PROMPT.format(text=text[:500])  # Truncate very long text

        for attempt in range(self._max_retries):
            try:
                response = self._session.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=120.0,
                )

                raw = response.choices[0].message.content or ""
                content = raw.strip()

                # Debug output
                print(f"\n[{seg_id}] LLM response (attempt {attempt+1}): {content[:200]}")

                if not content:
                    if attempt < self._max_retries - 1:
                        logger.warning(f"Empty response for {seg_id}, retrying...")
                        time.sleep(0.5)
                        continue
                    break

                parsed = self._extract_json(content)
                if parsed is None:
                    if attempt < self._max_retries - 1:
                        logger.warning(f"No JSON for {seg_id}, retrying...")
                        time.sleep(0.5)
                        continue
                    break

                # If parser returns a list, take first element
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else None

                if parsed and isinstance(parsed, dict):
                    tag = self._tag_from_dict({
                        "segment_id": seg_id,
                        "speaker_type": parsed.get("speaker_type", "narrator"),
                        "character_name": parsed.get("character_name"),
                        "emotion": parsed.get("emotion", "neutral"),
                    })
                    self._cache[cache_key] = {"tag": parsed}
                    return tag

            except Exception as e:
                logger.warning(f"LLM error for {seg_id} (attempt {attempt+1}): {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(1)

        # Fallback: all narrator
        logger.debug(f"Using fallback for {seg_id}")
        return SpeechTag(
            segment_id=seg_id,
            speaker_type="narrator",
            character_name=None,
            emotion="neutral",
            voice_id="narrator_male",
            emotion_instruction=EMOTION_INSTRUCTIONS_FR["neutral"],
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Extract JSON from text with balanced bracket tracking."""
        text = text.strip()
        if not text:
            return None

        # Remove markdown
        md = re.findall(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if md:
            text = md[0].strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try balanced bracket extraction
        for start in range(len(text)):
            if text[start] not in ('{', '['):
                continue
            open_ch = text[start]
            close_ch = '}' if open_ch == '{' else ']'
            depth = 0
            in_str = False
            escaped = False
            for i in range(start, len(text)):
                ch = text[i]
                if escaped:
                    escaped = False
                    continue
                if ch == '\\':
                    escaped = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start:i + 1])
                            except json.JSONDecodeError:
                                pass
                            break

        return None

    @staticmethod
    def _tag_from_dict(d: dict) -> SpeechTag:
        """Convert a dict to a SpeechTag."""
        speaker_type = d.get("speaker_type", "narrator")
        if speaker_type not in ("narrator", "dialogue"):
            speaker_type = "narrator"

        char_name = d.get("character_name")
        if char_name and isinstance(char_name, str):
            char_name = char_name.strip()
            if not char_name or char_name.lower() in ("null", "none", ""):
                char_name = None
        else:
            char_name = None

        emotion = d.get("emotion", "neutral")
        # LLM sometimes returns a list instead of string — take first element
        if isinstance(emotion, list):
            emotion = emotion[0] if emotion else "neutral"
        if not isinstance(emotion, str):
            emotion = "neutral"
        if emotion.lower() not in [e.lower() for e in VALID_EMOTIONS]:
            emotion = "neutral"

        voice_id = "narrator_male" if speaker_type == "narrator" else (
            char_name.lower().replace(" ", "_") if char_name else "narrator_male"
        )

        instr_dict = EMOTION_INSTRUCTIONS_FR
        return SpeechTag(
            segment_id=d.get("segment_id", ""),
            speaker_type=speaker_type,
            character_name=char_name,
            emotion=emotion,
            voice_id=voice_id,
            emotion_instruction=instr_dict.get(emotion, instr_dict["neutral"]),
        )

    def get_discovered_characters(self) -> List[str]:
        """Get sorted list of discovered character names."""
        return sorted(self._characters.keys())

    def get_character_segments(self, character_name: str) -> List[str]:
        """Get segment IDs for a specific character."""
        return sorted(self._characters.get(character_name, set()))
