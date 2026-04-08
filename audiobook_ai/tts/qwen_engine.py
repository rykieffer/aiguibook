"""
TTSEngine - Wraps faster-qwen3-tts for high-performance GPU inference.

Uses CUDA graph capture for fast generation. No flash-attn required.
Supports VoiceDesign (creating voices) and VoiceClone (generating with acting).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import traceback
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("AIGUIBook.Engine")

# Model variant paths
MODEL_VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


class TTSEngine:
    """Engine manager for faster-qwen3-tts.
    
    Loads specific model variants on demand:
    - VoiceDesign model: for creating reference voices from text descriptions
    - Base model: for voice cloning + emotion acting during audiobook generation
    """

    def __init__(self):
        self.model = None
        self.model_name = ""
        self.device = "cuda"

    def load_model(self, model_name: str, device: str = "cuda"):
        """Load a specific Qwen3-TTS model variant.
        
        Args:
            model_name: HuggingFace model ID (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base)
            device: Device string (default: "cuda")
        """
        if self.model is not None and self.model_name == model_name:
            logger.info(f"Model {model_name} already loaded, reusing.")
            return

        # Unload previous model first
        if self.model is not None:
            self.unload_model()

        logger.info(f"Loading FasterQwen3TTS: {model_name} on {device}")

        try:
            from faster_qwen3_tts import FasterQwen3TTS
            import torch

            self.model = FasterQwen3TTS.from_pretrained(
                model_name=model_name,
                device=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            self.model_name = model_name
            self.device = device
            logger.info(f"Model loaded successfully: {model_name}")

        except ImportError:
            raise RuntimeError(
                "faster-qwen3-tts not installed. Run: pip install faster-qwen3-tts"
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.model = None
            raise

    def unload_model(self):
        """Clear VRAM by deleting the model."""
        if self.model is not None:
            import torch

            logger.info("Unloading model to free VRAM...")
            del self.model
            self.model = None
            self.model_name = ""
            torch.cuda.empty_cache()
            logger.info("VRAM cleared.")

    def design_voice(
        self,
        text: str,
        instruction: str,
        language: str,
        output_path: str,
    ) -> Optional[str]:
        """Generate a brand new voice from a text description.
        
        Uses the VoiceDesign model variant. The generated audio serves as
        a reference for voice cloning later.
        
        Args:
            text: Text to speak (becomes the ref_text for cloning later)
            instruction: Voice description (e.g. "Warm male voice, French accent")
            language: Language code (e.g. "french", "english")
            output_path: Where to save the WAV file
            
        Returns:
            Path to the generated WAV, or None on failure
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            logger.info(f"Designing voice: [{instruction[:80]}...]")

            audio_list, sr = self.model.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language,
            )

            # audio_list is a list of numpy arrays
            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list

            # Normalize to prevent clipping
            if isinstance(audio_data, np.ndarray):
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    audio_data = audio_data / max_val

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sf.write(output_path, audio_data, sr)
            logger.info(f"Voice designed and saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Voice design failed: {e}\n{traceback.format_exc()}")
            return None

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
        emotion_instruction: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Clone a reference voice and optionally apply emotion acting.
        
        Uses the Base model variant. The instruct parameter is used for
        emotion/style instructions (e.g. "Parlez avec colere et tension").
        
        Args:
            text: The text to synthesize
            ref_audio_path: Path to the reference WAV file (from voice design)
            ref_text: The text spoken in the reference audio
            language: Language code
            emotion_instruction: Optional emotion instruction in French/English
            output_path: Where to save the output WAV
            
        Returns:
            Path to the generated WAV, or None on failure
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not text.strip():
            logger.warning("Empty text provided, skipping generation.")
            return None

        out_path = output_path or os.path.join(
            tempfile.gettempdir(), f"tts_{int(time.time())}.wav"
        )
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        try:
            logger.info(f"Generating [{len(text)} chars] with emotion: {emotion_instruction}")

            # Use the instruct parameter for emotion acting
            # This is the correct way per faster-qwen3-tts API
            audio_list, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                instruct=emotion_instruction,  # Emotion goes here!
                non_streaming_mode=True,
            )

            # audio_list is a list of numpy arrays
            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list

            # Normalize to prevent clipping
            if isinstance(audio_data, np.ndarray):
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    audio_data = audio_data / max_val

            sf.write(out_path, audio_data, sr)
            logger.info(f"Audio generated: {out_path}")
            return out_path

        except Exception as e:
            logger.error(f"Voice clone generation failed: {e}\n{traceback.format_exc()}")
            return None

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate speech using a built-in CustomVoice speaker.
        
        Uses the CustomVoice model variant with predefined speakers.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID (e.g. "ryan", "aiden")
            language: Language code
            output_path: Where to save the output WAV
            
        Returns:
            Path to the generated WAV, or None on failure
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        out_path = output_path or os.path.join(
            tempfile.gettempdir(), f"tts_custom_{int(time.time())}.wav"
        )
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        try:
            audio_list, sr = self.model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
            )

            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list

            if isinstance(audio_data, np.ndarray):
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    audio_data = audio_data / max_val

            sf.write(out_path, audio_data, sr)
            logger.info(f"Custom voice generated: {out_path}")
            return out_path

        except Exception as e:
            logger.error(f"Custom voice generation failed: {e}\n{traceback.format_exc()}")
            return None

    def list_speakers(self) -> List[str]:
        """List available built-in speakers for the CustomVoice model."""
        if not self.model:
            return []
        try:
            return self.model.get_supported_speakers()
        except Exception:
            return []

    def __del__(self):
        self.unload_model()
