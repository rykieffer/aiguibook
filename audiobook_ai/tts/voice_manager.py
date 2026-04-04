"""VoiceManager - Manages voice profiles for multi-speaker audiobooks."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

# Default voice descriptions for French-friendly audiobooks
DEFAULT_VOICE_DESCRIPTIONS = {
    "narrator_male": {
        "description": "Deep warm male voice, mature, French accent, authoritative yet gentle, perfect for narration",
        "example_text": "Il était une fois, dans un pays lointain, un aventurier nommé Julien qui partit explorer le monde inconnu.",
    },
    "narrator_female": {
        "description": "Soft warm female voice, clear and elegant, French accent, soothing and expressive",
        "example_text": "Elle regarda par la fenêtre et observa le soleil se coucher sur la ville endormie.",
    },
    "young_male": {
        "description": "Young male voice, energetic, bright, French accent, clear and vibrant",
        "example_text": "Attends-moi, je viens avec toi, on va faire ça ensemble c'est sûr!",
    },
    "young_female": {
        "description": "Young female voice, light and cheerful, French accent, expressive and animated",
        "example_text": "C'est magnifique, je n'ai jamais rien vu d'aussi beau de toute ma vie!",
    },
    "elder_male": {
        "description": "Older deep male voice, grave, wise, French accent, powerful and resonant",
        "example_text": "J'ai vécu bien longtemps et j'ai appris que la patience est la plus grande des vertus.",
    },
    "elder_female": {
        "description": "Older female voice, warm and compassionate, French accent, gentle and wise",
        "example_text": "Mon enfant, écoute bien ce que je vais te dire, c'est important pour ton avenir.",
    },
    "robotic": {
        "description": "Slightly mechanical synthetic voice for sci-fi, precise rhythm, French accent, otherworldly",
        "example_text": "Analyse en cours. Systèmes opérationnels. La mission peut commencer.",
    },
}


class VoiceManager:
    """Manages voice profiles for multi-speaker audiobooks."""

    # Voice profile metadata file
    PROFILE_FILE = "voice_profiles.json"

    def __init__(self, voices_dir: str):
        """
        Args:
            voices_dir: Directory to store voice profiles
        """
        self.voices_dir = os.path.abspath(voices_dir)
        self.profiles_file = os.path.join(self.voices_dir, self.PROFILE_FILE)
        self._voices: Dict[str, dict] = {}  # {name: {ref_audio, ref_text, ...}}

        os.makedirs(self.voices_dir, exist_ok=True)
        self._load_profiles()

    def _load_profiles(self):
        """Load voice profiles from disk."""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, "r", encoding="utf-8") as f:
                    self._voices = json.load(f)
                logger.info(f"Loaded {len(self._voices)} voice profiles")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load voice profiles: {e}")
                self._voices = {}

    def _save_profiles(self):
        """Save voice profiles to disk."""
        with open(self.profiles_file, "w", encoding="utf-8") as f:
            json.dump(self._voices, f, indent=2, ensure_ascii=False)

    def register_speaker(
        self,
        name: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
    ) -> bool:
        """Register a voice from reference audio.

        Args:
            name: Unique voice name
            ref_audio_path: Path to reference audio file (WAV preferred)
            ref_text: Transcript of reference audio

        Returns:
            True if successful
        """
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        # Copy reference to voices directory
        ext = os.path.splitext(ref_audio_path)[1] or ".wav"
        dest_path = os.path.join(self.voices_dir, f"{name}{ext}")
        shutil.copy2(ref_audio_path, dest_path)

        self._voices[name] = {
            "ref_audio": dest_path,
            "ref_text": ref_text or "",
            "type": "cloned",
            "created_at": None,  # Could add timestamp
        }
        self._save_profiles()

        logger.info(f"Voice registered: {name} from {ref_audio_path}")
        return True

    def create_voice_with_design(
        self,
        name: str,
        description: str,
        example_text: str,
        tts_model=None,
    ) -> str:
        """Create a unique voice from a text description using VoiceDesign.

        Args:
            name: Unique voice name
            description: Text description of the voice
                e.g., "Deep male voice, mature, French accent, warm and authoritative"
            example_text: Text to speak when generating the voice sample
            tts_model: Optional TTSEngine instance for model access

        Returns:
            Path to the generated voice sample audio file

        Raises:
            RuntimeError if voice design fails
        """
        output_path = os.path.join(self.voices_dir, f"{name}.wav")

        try:
            if tts_model and tts_model._model is not None:
                # Use the VoiceDesign model via the loaded TTS model
                logger.info(f"Creating voice '{name}' via VoiceDesign: {description}")

                # Try generate_voice_design
                try:
                    result = tts_model._model.generate_voice_design(
                        text=example_text,
                        language="French",
                        instruct=description,
                        output_path=output_path,
                    )
                except AttributeError:
                    # If method doesn't exist, fall back to standard generation
                    # with the instruction incorporated
                    result = tts_model._model.generate(
                        text=example_text,
                        language="French",
                        output_path=output_path,
                    )

                if isinstance(result, dict) and "wav" in result:
                    wav = result["wav"]
                    sr = result.get("sample_rate", 24000)
                    if hasattr(wav, "detach"):
                        import torch
                        wav = wav.cpu().detach()
                    if hasattr(wav, "numpy"):
                        wav = wav.numpy()
                    if isinstance(wav, np.ndarray):
                        sf.write(output_path, wav, sr, subtype="PCM_16")

            else:
                logger.warning(
                    "No TTS model loaded. Creating placeholder voice for design. "
                    "Load a VoiceDesign model first for real voice creation."
                )
                # Create a short silent placeholder
                sr = 24000
                dur = 3.0
                silence = np.zeros(int(sr * dur), dtype=np.float32)
                sf.write(output_path, silence, sr, subtype="PCM_16")

        except Exception as e:
            logger.error(f"Voice design failed for '{name}': {e}")
            # Create a placeholder so the system doesn't break
            sr = 24000
            silence = np.zeros(int(sr * 1.0), dtype=np.float32)
            sf.write(output_path, silence, sr, subtype="PCM_16")

        # Register the voice
        self._voices[name] = {
            "ref_audio": output_path,
            "ref_text": example_text,
            "type": "designed",
            "description": description,
        }
        self._save_profiles()

        logger.info(f"Voice designed and saved: {name}")
        return output_path

    def get_voice(self, speaker_name: str) -> Tuple[str, str]:
        """Get voice reference data for a speaker.

        Args:
            speaker_name: Voice name

        Returns:
            Tuple of (ref_audio_path, ref_text)
        """
        if speaker_name in self._voices:
            profile = self._voices[speaker_name]
            return profile.get("ref_audio", ""), profile.get("ref_text", "")

        # Fallback: check if there's a default voice for this name pattern
        name_lower = speaker_name.lower()
        for vname, vdata in self._voices.items():
            if vname.lower() == name_lower:
                return vdata.get("ref_audio", ""), vdata.get("ref_text", "")

        return "", ""

    def create_default_voices(self, tts_model=None) -> List[str]:
        """Create a set of default French-friendly voices.

        Creates placeholder entries that can be used as targets for VoiceDesign
        or voice cloning later.

        Args:
            tts_model: Optional TTSEngine for actual voice generation

        Returns:
            List of created voice names
        """
        created = []
        for name, info in DEFAULT_VOICE_DESCRIPTIONS.items():
            voice_file = os.path.join(self.voices_dir, f"{name}.wav")

            # Check if voice already exists
            if name in self._voices and os.path.exists(self._voices[name].get("ref_audio", "")):
                created.append(name)
                continue

            # Generate voice using design
            try:
                voice_path = self.create_voice_with_design(
                    name=name,
                    description=info["description"],
                    example_text=info["example_text"],
                    tts_model=tts_model,
                )
                created.append(name)
            except Exception as e:
                logger.warning(f"Could not create default voice '{name}': {e}")
                # Still register with placeholder
                if not os.path.exists(voice_file):
                    sr = 24000
                    silence = np.zeros(int(sr * 3.0), dtype=np.float32)
                    sf.write(voice_file, silence, sr, subtype="PCM_16")
                self._voices[name] = {
                    "ref_audio": voice_file,
                    "ref_text": info["example_text"],
                    "type": "placeholder",
                    "description": info["description"],
                }
                self._save_profiles()
                created.append(name)

        logger.info(f"Created {len(created)} default voices")
        return created

    def list_voices(self) -> Dict[str, dict]:
        """List all available voice profiles.

        Returns:
            Dict mapping voice name to profile info
        """
        result = {}
        for name, profile in self._voices.items():
            audio_path = profile.get("ref_audio", "")
            exists = os.path.exists(audio_path) if audio_path else False
            result[name] = {
                "ref_audio": audio_path,
                "ref_text": profile.get("ref_text", ""),
                "type": profile.get("type", "unknown"),
                "exists": exists,
                "description": profile.get("description", ""),
            }
        return result

    def delete_voice(self, name: str) -> bool:
        """Delete a voice profile.

        Args:
            name: Voice name to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self._voices:
            return False

        profile = self._voices[name]
        audio_path = profile.get("ref_audio", "")

        # Remove audio file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        del self._voices[name]
        self._save_profiles()

        logger.info(f"Voice deleted: {name}")
        return True

    def import_voice(self, name: str, ref_audio_path: str, ref_text: str = "") -> bool:
        """Import a voice from an external audio file.

        Args:
            name: Voice name
            ref_audio_path: Path to audio file
            ref_text: Optional reference transcript

        Returns:
            True if successful
        """
        return self.register_speaker(name, ref_audio_path, ref_text)

    def __repr__(self) -> str:
        return f"VoiceManager({len(self._voices)} voices)"
    def suggest_voice_for_character(self, character_name, analysis_results):
        """Analyze LLM results to suggest the best voice for a character.
        Returns: dict with suggested_voice, confidence, description"""
        voice_votes = {}
        descriptions = []
        total = 0

        for res in analysis_results:
            if isinstance(res, dict):
                char = res.get('character_name', '')
                desc = res.get('character_description', '')
                suggested = res.get('suggested_voice_id', '')
            else:
                char = getattr(res, 'character_name', '')
                desc = getattr(res, 'character_description', '')
                suggested = getattr(res, 'suggested_voice_id', '')

            if char and char.lower() == character_name.lower():
                if desc:
                    descriptions.append(desc)
                if suggested and suggested != 'narrator':
                    voice_votes[suggested] = voice_votes.get(suggested, 0) + 1
                    total += 1

        if not voice_votes:
            return {"suggested_voice": "narrator_male", "confidence": 0.0, "description": ""}

        best = max(voice_votes, key=voice_votes.get)
        conf = round((voice_votes[best] / total) * 100, 1) if total > 0 else 0
        unique = list(set(descriptions))[:3]
        desc_str = ", ".join(unique) if unique else ""
        
        return {"suggested_voice": best, "confidence": conf, "description": desc_str}
