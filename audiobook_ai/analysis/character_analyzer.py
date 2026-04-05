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
    backend, base_url=None, api_key=None, timeout=5.0,
):
    """Auto-detect available models from an LLM backend."""
    import urllib.request, urllib.error

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
            req.add_header("Authorization", "Bearer " + key)
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            model_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            if model_ids:
                return True, model_ids, ""
            return False, [], "No models returned"
        except Exception as e:
            return False, [], str(e)

    return False, [], "Unknown backend: " + backend


def test_llm_connection(backend, base_url=None, model=None, api_key=None, timeout=30.0):
    """Test if an LLM backend is reachable."""
    try:
        from openai import OpenAI
    except ImportError:
        return False, "openai package not installed"

    if backend == "lmstudio":
        client = OpenAI(base_url=base_url or "http://localhost:1234/v1", api_key="unused")
        test_model = model or ""
    elif backend == "ollama":
        client = OpenAI(base_url=(base_url or "http://localhost:11434") + "/v1", api_key="ollama")
        test_model = model or "qwen3:32b"
    elif backend == "openrouter":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key or "")
        test_model = model or "openai/gpt-4o-mini"
    else:
        return False, "Unknown backend"

    try:
        response = client.chat.completions.create(
            model=test_model, messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10, timeout=timeout,
        )
        content = response.choices[0].message.content.strip()
        return True, 'Connected. Model replied: "%s"' % content
    except Exception as e:
        return False, "Connection failed: %s" % e


class CharacterAnalyzer:
    """Analyzes text segments one-by-one to detect characters and emotions."""

    def __init__(self, config, session=None):
        self.config = config
        self._backend = config.get("llm_backend", "lmstudio")
        self._max_retries = 2
        self._batch_size = 1
        self._cache = {}
        self._characters = {}
        self._model = ""
        self._session = session

        if self._session is None:
            self._session, self._model = self._create_client()

    def _create_client(self):
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
                    logger.info("Auto-detected LM Studio model: %s" % model)
                    self.config["lmstudio_model"] = model
                else:
                    raise ValueError("No LM Studio model found. Load a model first. Error: %s" % err)
            client = OpenAI(base_url=base_url, api_key="unused")
            logger.info("CharacterAnalyzer -> LM Studio: %s / %s" % (base_url, model))
            return client, model

        elif self._backend == "openrouter":
            api_key = self.config.get("openrouter_api_key", "")
            model = self.config.get("openrouter_model", "openai/gpt-4o-mini")
            if not api_key:
                raise ValueError("OpenRouter API key not set.")
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            logger.info("CharacterAnalyzer -> OpenRouter: %s" % model)
            return client, model

        elif self._backend == "ollama":
            base_url = self.config.get("ollama_base_url", "http://localhost:11434")
            model = self.config.get("ollama_model", "qwen3:32b")
            client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="ollama")
            logger.info("CharacterAnalyzer -> Ollama: %s" % model)
            return client, model

        else:
            raise ValueError("Unknown LLM backend: %s" % self._backend)

    @staticmethod
    def discover_models(backend, base_url=None, api_key=None):
        """Static helper: discover available models from a backend."""
        return get_llm_models_from_backend(backend=backend, base_url=base_url, api_key=api_key)

    def deduplicate_characters(self, char_list):
        """Merge character name variants using rule-based heuristics + LLM fallback.

        Merges names that refer to the same character based on substring matching,
        known aliases, and common prefix/suffix patterns. Falls back to LLM if enabled.
        """
        if len(char_list) <= 1:
            return {c: c for c in char_list}

        # Rule-based deduplication
        mapping = {c: c for c in char_list}

        # Step 1: Substring matching - merge shorter names into longer ones
        for i, name_a in enumerate(char_list):
            for j, name_b in enumerate(char_list):
                if i == j:
                    continue
                name_a_lower = name_a.lower().strip()
                name_b_lower = name_b.lower().strip()
                # If name_a is a substring of name_b (and name_b has more words), merge
                if name_a_lower in name_b_lower and len(name_a_lower) < len(name_b_lower):
                    mapping[name_a] = name_b

        # Step 2: Normalize and merge known patterns
        canonical_map = {}
        for name, mapped in mapping.items():
            # Remove parenthetical info for matching
            clean = re.sub(r'\s*\(.*?\)\s*', '', mapped).strip()
            if clean not in canonical_map:
                canonical_map[clean] = mapped
            else:
                # Keep the one with more segments (prefer full name)
                existing = canonical_map[clean]
                if len(mapped) > len(existing):
                    canonical_map[clean] = mapped

        # Build reverse map: variant -> canonical
        result = {}
        for name in char_list:
            mapped = mapping[name]
            clean = re.sub(r'\s*\(.*?\)\s*', '', mapped).strip()
            canonical = canonical_map.get(clean, mapped)
            result[name] = canonical

        # Count merged
        unique = set(result.values())
        print("\n[DEDUP] %d names -> %d unique characters:" % (len(char_list), len(unique)))
        for canonical in sorted(unique):
            variants = [k for k, v in result.items() if v == canonical]
            if len(variants) > 1:
                print("  %s <- %s" % (canonical, ", ".join(variants)))

        return result

    def build_voice_descriptions(self):
        """Generate ElevenLabs-style voice descriptions for each character."""
        descriptions = {}
        for char_name, segments in self._characters.items():
            char_lower = char_name.lower()
            is_female = any(w in char_lower for w in [
                "madame", "mademoiselle", "mme", "miss", "lady", "woman",
                "mei", "naomi", "giora", "tannen", "demanda", "glo",
                "jeune femme", "la femme", "elise", "mere",
            ])
            gender_desc = "female" if is_female else "male"

            is_military = any(w in char_lower for w in [
                "draper", "bobbie", "roberta", "sergent", "capitaine",
                "ashford", "cotyar", "wendell", "larson", "tseng",
            ])
            is_political = any(w in char_lower for w in [
                "avasarala", "chrisjen", "errinw", "mao", "walter", "philips",
                "nettleford", "genera",
            ])
            is_scientist = any(w in char_lower for w in [
                "prax", "nicola", "basia",
            ])

            if is_military:
                voice_prompt = "A %s military voice, French accent. Firm, disciplined, authoritative tone." % gender_desc
            elif is_political:
                voice_prompt = "A sophisticated %s political voice, French accent. Measured and diplomatic." % gender_desc
            elif is_scientist:
                voice_prompt = "A %s academic voice, French accent. Thoughtful and precise tone." % gender_desc
            else:
                voice_prompt = "A natural %s voice with French accent. Clear and expressive for audiobook narration." % gender_desc

            descriptions[char_name] = {
                "elevenlabs_prompt": voice_prompt,
                "french_description": voice_prompt,
                "voice_type": "custom",
                "segment_count": len(segments),
            }
        return descriptions

    def save_analysis(self, filepath, segment_tags, char_list, dedup_map=None):
        """Save character analysis to a JSON file for reuse."""
        data = {
            "segment_tags": {sid: tag.to_dict() for sid, tag in segment_tags.items()},
            "characters": char_list,
            "dedup_map": dedup_map or {},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Saved analysis to %s" % filepath)
        logger.info("Saved analysis to %s" % filepath)

    @staticmethod
    def load_analysis(filepath):
        """Load character analysis from a JSON file."""
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

        print("Loaded analysis: %d segments, %d characters from %s" % (
            len(segment_tags), len(char_list), filepath))
        logger.info("Loaded analysis: %d segments, %d characters from %s" % (
            len(segment_tags), len(char_list), filepath))
        return segment_tags, char_list, dedup_map

    # ---- Dialogue detection markers ----
    _DIALOGUE_MARKERS = {"\"", "\u201c", "\u201d", "\u00ab", "\u00bb", "\u2014", "\u2013", "\u002d"}

    def _has_dialogue(self, text):
        """Check if text contains dialogue markers (fast pre-filter)."""
        if not any(ch in self._DIALOGUE_MARKERS for ch in text):
            return False
        # Also check for French dash dialogue patterns
        if re.search(r'[\u00ab\u201c"]|\u2014|\u2013\s+\w+', text):
            return True
        return False

    def analyze_segments(self, segments_list, language="french"):
        """Analyze all segments, returning (tags_dict, character_list, dedup_map)."""
        result = None
        for item in self.analyze_segments_iter(segments_list, language):
            if item.get("status") == "finished":
                result = item["result"]
        return result or ({}, [], {})

    def analyze_segments_iter(self, segments_list, language="french"):
        """Generator: yields progress updates during analysis."""
        all_tags = {}
        all_chars = []
        total = len(segments_list)
        done = 0
        start_time = time.time()

        if total > 0:
            yield {"status": "init", "msg": "Initialized. %d segments to analyze." % total}

        for i, segment in enumerate(segments_list):
            seg_id = segment.id if hasattr(segment, "id") else segment.get("id", "")
            seg_text = segment.text if hasattr(segment, "text") else segment.get("text", "")
            done = i + 1

            if not seg_text.strip():
                tag = SpeechTag(
                    segment_id=seg_id, speaker_type="narrator", character_name=None,
                    emotion="neutral", voice_id="narrator_male",
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

            if done % 10 == 0 or done == total:
                elapsed = time.time() - start_time
                avg = elapsed / done
                remaining = avg * (total - done)
                eta_m = int(remaining / 60)
                eta_s = int(remaining % 60)
                pct = done / total * 100
                chars_str = ", ".join(all_chars[:8])
                if len(all_chars) > 8:
                    chars_str += " +%d more" % (len(all_chars) - 8)
                elif not chars_str:
                    chars_str = "none yet"

                progress_msg = "[%d/%d] %5.1f%%  ETA %02d:%02d  %d chars: %s" % (
                    done, total, pct, eta_m, eta_s, len(all_chars), chars_str,
                )
                print(progress_msg)
                yield {"status": "progress", "msg": progress_msg}

        total_time = time.time() - start_time
        unique_chars = list(dict.fromkeys(all_chars))
        print("\nAnalysis complete! %d characters in %.0fs" % (len(unique_chars), total_time))
        for cn in unique_chars:
            print("  - %s: %d segments" % (cn, len(self._characters[cn])))

        # Rule-based deduplication (fast, no LLM needed)
        print("\n[Deduplication] Merging character name variants...")
        deduped = self.deduplicate_characters(unique_chars)
        unique_merged = sorted(set(deduped.values()))
        print("[Deduplication] %d -> %d unique characters\n" % (len(unique_chars), len(unique_merged)))

        # Recalculate segment counts
        merged_chars = {}
        for cn in unique_merged:
            merged_chars[cn] = set()
        for seg_id, tag in all_tags.items():
            if tag.character_name:
                canonical = deduped.get(tag.character_name, tag.character_name)
                merged_chars.setdefault(canonical, set()).add(seg_id)
        self._characters = {k: v for k, v in merged_chars.items() if v}

        logger.info("Analysis complete: %d segments, %d chars -> %d deduped in %.0fs" % (
            len(all_tags), len(unique_chars), len(unique_merged), total_time,
        ))
        yield {
            "status": "finished",
            "msg": "Analysis complete!",
            "result": (all_tags, unique_merged, deduped),
        }

    def _analyze_single_segment(self, seg_id, text, language):
        """Analyze a single text segment via LLM, with pre-filter for narration."""
        # --- FAST PRE-FILTER: skip LLM for pure narration ---
        if not self._has_dialogue(text):
            return SpeechTag(
                segment_id=seg_id, speaker_type="narrator", character_name=None,
                emotion="neutral", voice_id="narrator_male",
                emotion_instruction=EMOTION_INSTRUCTIONS_FR["neutral"],
            )

        # Check cache
        cache_key = json.dumps({"id": seg_id, "text": text[:200]}, sort_keys=True)
        if cache_key in self._cache:
            return self._tag_from_dict(self._cache[cache_key])

        prompt = SEGMENT_PROMPT.format(text=text[:500])

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

                if not content or len(content) < 3:
                    if attempt < self._max_retries - 1:
                        logger.warning("Empty response for %s, retrying..." % seg_id)
                        time.sleep(0.5)
                    continue

                print("\n[%s] LLM response (attempt %d): %s" % (seg_id, attempt + 1, content[:200]))

                parsed = self._extract_json(content)
                if parsed is None:
                    if attempt < self._max_retries - 1:
                        logger.warning("No JSON for %s, retrying..." % seg_id)
                        time.sleep(0.5)
                    continue

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
                logger.warning("LLM error for %s (attempt %d): %s" % (seg_id, attempt + 1, e))
                if attempt < self._max_retries - 1:
                    time.sleep(1)

        # Fallback
        return SpeechTag(
            segment_id=seg_id, speaker_type="narrator", character_name=None,
            emotion="neutral", voice_id="narrator_male",
            emotion_instruction=EMOTION_INSTRUCTIONS_FR["neutral"],
        )

    @staticmethod
    def _extract_json(text):
        """Extract JSON from text with balanced bracket tracking."""
        text = text.strip()
        if not text:
            return None

        # Remove markdown
        md = re.findall(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if md:
            text = md[0].strip()
            
        # FIX: Remove LLM comments (e.g., '// inferred narrator') that cause JSON errors
        text = re.sub(r'//.*?(?=\s*[,}\]])', '', text)

        # Fix 1: Remove LLM comments (e.g., "null // inferred narrator")
        # Replace double-slash comments that appear outside of strings
        # Simple regex to remove // ... until end of line or closing brace
        import re
        text = re.sub(r'//.*?(?=,|\}|\]|\n)', '', text)

        # Fix 2: Handle multiline values if LLM messes up newlines in JSON
        # (This is handled by the bracket extractor mostly, but good cleanup)

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
    def _tag_from_dict(d):
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
        # LLM sometimes returns a list instead of string
        if isinstance(emotion, list):
            emotion = emotion[0] if emotion else "neutral"
        if not isinstance(emotion, str):
            emotion = "neutral"
        valid_lower = [e.lower() for e in VALID_EMOTIONS]
        if emotion.lower() not in valid_lower:
            emotion = "neutral"

        voice_id = "narrator_male" if speaker_type == "narrator" else (
            char_name.lower().replace(" ", "_") if char_name else "narrator_male"
        )

        return SpeechTag(
            segment_id=d.get("segment_id", ""),
            speaker_type=speaker_type,
            character_name=char_name,
            emotion=emotion,
            voice_id=voice_id,
            emotion_instruction=EMOTION_INSTRUCTIONS_FR.get(emotion, EMOTION_INSTRUCTIONS_FR["neutral"]),
        )

    def get_discovered_characters(self):
        """Get sorted list of discovered character names."""
        return sorted(self._characters.keys())

    def get_character_segments(self, character_name):
        """Get segment IDs for a specific character."""
        return sorted(self._characters.get(character_name, set()))
