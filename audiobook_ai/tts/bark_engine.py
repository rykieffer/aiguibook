"""BarkEngine - TTS engine using Suno's Bark model."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Bark voice presets with French descriptions
# Bark uses speaker IDs like 'v2/fr_speaker_0' to 'v2/fr_speaker_9'
BARK_VOICES = {
    # Narrator voices
    "narrator_male": {
        "id": "v2/fr_speaker_2",
        "desc": "Voix grave et autoritaire",
        "use": "Narrateur masculin principal",
    },
    "narrator_female": {
        "id": "v2/fr_speaker_9",
        "desc": "Voix féminine douce et claire",
        "use": "Narratrice féminine",
    },
    # Character voices
    "young_male": {
        "id": "v2/fr_speaker_3",
        "desc": "Voix jeune et energique",
        "use": "Personnage masculin jeune",
    },
    "young_female": {
        "id": "v2/fr_speaker_6",
        "desc": "Voix féminine jeunes et chaleureuse",
        "use": "Personnage feminin jeune",
    },
    "elder_male": {
        "id": "v2/fr_speaker_1",
        "desc": "Voix masculine chaude et mature",
        "use": "Personnage masculin âge",
    },
    "elder_female": {
        "id": "v2/fr_speaker_8",
        "desc": "Voix feminine grave",
        "use": "Personnage feminin âgée",
    },
    "angry_male": {
        "id": "v2/fr_speaker_4",
        "desc": "Voix masculine douce mais ferme",
        "use": "Personnage en colère ou autoritaire",
    },
    "soft_female": {
        "id": "v2/fr_speaker_7",
        "desc": "Voix féminine claire",
        "use": "Personnage doux ou craintif",
    },
    "robotic": {
        "id": "v2/fr_speaker_5",
        "desc": "Voix jeune neutre",
        "use": "Robot, IA, ou voix mécanique",
    },
}

# Mapping emotions to Bark text modifications
# Bark responds well to punctuation and capitalization for tone
EMOTION_MODIFIER = {
    "calm": "",
    "excited": "!",
    "angry": "!",
    "sad": "...",
    "whisper": "",
    "tense": "...",
    "urgent": "!",
    "amused": "",
    "contemptuous": "...",
    "surprised": "?",
    "neutral": ".",
}


class BarkEngine:
    """TTS engine using Suno's Bark.

    Bark generates expressive, multi-lingual audio from text prompts.
    No reference audio needed for voice cloning — voices are pre-built.
    Supports 14+ languages including French.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_small: bool = False,
        temperature: float = 0.7,
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            use_small: Use small models (faster, lower quality)
            temperature: Generation creativity (0.0-1.0)
        """
        self.device = device
        self.use_small = use_small
        self.temperature = temperature
        self._model_loaded = False
        self._voice_history = {}  # Cache extracted voice prompts

    def initialize(self):
        """Load Bark models to GPU/CPU."""
        if self._model_loaded:
            return

        logger.info(f"Loading Bark models (device={self.device}, small={self.use_small})...")
        start = time.time()

        try:
            from bark import preload_models
            preload_models(
                text_use_gpu=(self.device == "cuda"),
                text_use_small=self.use_small,
                coarse_use_gpu=(self.device == "cuda"),
                coarse_use_small=self.use_small,
                fine_use_gpu=(self.device == "cuda"),
                fine_use_small=self.use_small,
                force_reload=False,
            )
            self._model_loaded = True
            elapsed = time.time() - start
            logger.info(f"Bark models loaded in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load Bark models: {e}")
            raise RuntimeError(
                f"Failed to load Bark. Install:\n"
                f"pip install git+https://github.com/suno-ai/bark.git\n"
                f"Error: {e}"
            )

    @staticmethod
    def get_available_voices() -> Dict[str, dict]:
        """Get all available Bark voice presets."""
        return BARK_VOICES

    @staticmethod
    def suggest_voice_for_character(
        character_name: str,
        emotions: List[str],
        segment_count: int,
    ) -> str:
        """Suggest the best Bark voice for a character based on detected traits.

        Args:
            character_name: Character name
            emotions: List of emotions detected for this character
            segment_count: Number of segments for this character

        Returns:
            Bark voice ID (e.g., 'v2/fr_speaker_3')
        """
        # If it's the narrator, use main narrator voice
        if "narrat" in character_name.lower() or "narrateur" in character_name.lower():
            return BARK_VOICES["narrator_male"]["id"]

        # Analyze emotions to pick the best voice match
        em_set = set(e.lower() for e in emotions)

        if {"angry", "contemptuous", "urgent"} & em_set:
            return BARK_VOICES["angry_male"]["id"]
        elif {"sad", "whisper", "tense"} & em_set:
            return BARK_VOICES["soft_female"]["id"]
        elif {"excited", "amused", "surprised"} & em_set:
            if segment_count > 15:
                return BARK_VOICES["young_male"]["id"]
            return BARK_VOICES["young_female"]["id"]
        elif segment_count > 20:
            # Major character
            return BARK_VOICES["narrator_male"]["id"]
        elif segment_count > 10:
            return BARK_VOICES["elder_male"]["id"]
        else:
            return BARK_VOICES["narrator_female"]["id"]

    def _get_voice_id(
        self,
        voice_id: Optional[str] = None,
        character_name: Optional[str] = None,
    ) -> str:
        """Get the Bark voice ID to use for generation.

        Args:
            voice_id: Explicit voice ID or Bark preset name
            character_name: Character name for auto-detection

        Returns:
            Bark voice preset ID (e.g., 'v2/fr_speaker_2')
        """
        if voice_id and voice_id in BARK_VOICES:
            return BARK_VOICES[voice_id]["id"]
        elif voice_id and voice_id.startswith("v2/"):
            return voice_id  # Already a Bark voice ID
        elif character_name and character_name in BARK_VOICES:
            return BARK_VOICES[character_name]["id"]
        else:
            return BARK_VOICES["narrator_male"]["id"]  # Fallback

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        character_name: Optional[str] = None,
        emotion: Optional[str] = None,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[str, float]:
        """Generate speech from text using Bark.

        Args:
            text: Text to synthesize
            voice: Voice ID or preset name (e.g., 'narrator_male' or 'v2/fr_speaker_2')
            character_name: Character name for auto voice suggestion
            emotion: Emotion to apply (affects punctuation for Bark)
            output_path: Output WAV file path
            progress_callback: Optional callback(percent, status_text)

        Returns:
            Tuple of (audio_path, duration_seconds)
        """
        from bark import generate_audio

        if progress_callback:
            progress_callback(0.1, "Generating speech...")

        # Get voice ID
        voice_id = self._get_voice_id(voice, character_name)

        # Apply emotion modifier to text (Bark responds to punctuation)
        text_to_speak = text.strip()
        if emotion and emotion in EMOTION_MODIFIER:
            modifier = EMOTION_MODIFIER[emotion]
            if modifier and not text_to_speak.endswith(modifier):
                # Replace ending punctuation with emotion marker
                text_to_speak = text_to_speak.rstrip(".!?;") + modifier

        # Bark has a ~15-second limit per generation, roughly 200 chars
        # For longer text, split into sentences and concatenate
        max_chars = 200
        if len(text_to_speak) > max_chars:
            logger.debug(
                f"Text too long ({len(text_to_speak)} chars) for single Bark call. "
                f"Will handle in chunks."
            )

        try:
            if len(text_to_speak) <= max_chars:
                audio_array = generate_audio(
                    text_to_speak,
                    history_prompt=voice_id,
                    temperature=self.temperature,
                    silent=True,
                )
            else:
                # Split into sentences for chunk generation
                import re
                sentences = re.split(r'(?<=[.!?;])\s+', text_to_speak)
                chunks = []
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
                    else:
                        current_chunk = (current_chunk + " " + sent).strip()
                if current_chunk:
                    chunks.append(current_chunk)

                # Generate each chunk and concatenate
                all_audio = []
                for i, chunk in enumerate(chunks):
                    if progress_callback:
                        pct = 0.1 + (i / len(chunks)) * 0.8
                        progress_callback(pct, f"Chunk {i+1}/{len(chunks)}...")
                    chunk_audio = generate_audio(
                        chunk,
                        history_prompt=voice_id,
                        temperature=self.temperature,
                        silent=True,
                    )
                    all_audio.append(chunk_audio)
                    # Small gap between sentences for natural flow
                    gap = np.zeros(12000, dtype=np.float32)  # 0.5s gap at 24kHz
                    all_audio.append(gap)

                audio_array = np.concatenate(all_audio)

            # Determine output path
            if output_path is None:
                output_path = os.path.join(
                    tempfile.gettempdir(),
                    f"bark_tts_{int(time.time())}_{hash(text_to_speak) % 10000}.wav",
                )
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Save audio at 24kHz (Bark's sample rate)
            sf.write(output_path, audio_array, 24000, subtype="PCM_16")

            duration = len(audio_array) / 24000

            if progress_callback:
                progress_callback(1.0, "Generation complete")

            logger.info(
                f"Bark generated {len(text_to_speak)} chars in {duration:.2f}s -> {output_path}"
            )
            return output_path, duration

        except Exception as e:
            logger.error(f"Bark generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Bark generation failed: {e}") from e

    def batch_generate(
        self,
        segments: List[dict],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Tuple[str, float]]:
        """Generate speech for multiple segments.

        Args:
            segments: List of dicts with keys:
                - text (str): Text to speak
                - voice_id (str, optional): Bark voice ID
                - character_name (str, optional): Character name
                - emotion (str, optional): Emotion
                - output_path (str): Output WAV path
            progress_callback: Optional callback

        Returns:
            List of (audio_path, duration_seconds) tuples
        """
        results = []
        total = len(segments)

        for i, seg in enumerate(segments):
            try:
                path, dur = self.generate(
                    text=seg.get("text", ""),
                    voice=seg.get("voice_id"),
                    character_name=seg.get("character_name"),
                    emotion=seg.get("emotion"),
                    output_path=seg.get("output_path"),
                )
                results.append((path, dur))
                if progress_callback:
                    progress_callback(
                        (i + 1) / total,
                        f"Generated {i+1}/{total}"
                    )
            except Exception as e:
                logger.error(f"Bark batch generation failed for segment {i}: {e}")
                results.append((None, 0.0))

        return results

    def __repr__(self) -> str:
        status = "loaded" if self._model_loaded else "not loaded"
        return f"BarkEngine(device={self.device}, {status})"
