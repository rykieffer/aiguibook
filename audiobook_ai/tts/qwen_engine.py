"""TTSEngine - Wraps Qwen3-TTS model for local GPU inference."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional, Tuple

import soundfile as sf

logger = logging.getLogger(__name__)

class TTSEngine:
    """Engine manager for Qwen3-TTS.
    
    Handles loading of VoiceDesign (for creating voices) and Base (for generation).
    """

    def __init__(self):
        self.model = None
        self.model_name = ""
        self.device = "cuda"
        self.dtype = "bfloat16"  # or torch.bfloat16 logic

    def load_model(self, model_path: str, device: str = "cuda"):
        """Load the specified Qwen3 model variant."""
        try:
            from qwen_tts import Qwen3TTSModel
            import torch

            if self.model is not None and self.model_name == model_path:
                return

            logger.info(f"Loading Qwen3 model: {model_path}")
            
            # Determine torch dtype
            dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

            # Use ONLY model_path. 
            # Wrapper models often crash if given 'device_map' or 'dtype' which they don't handle.
            self.model = Qwen3TTSModel.from_pretrained(model_path)
            self.model_name = model_path
            logger.info(f"Model loaded: {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise

    def unload_model(self):
        """Clear VRAM."""
        if self.model is not None:
            import torch
            logger.info("Unloading model to free VRAM")
            del self.model
            self.model = None
            self.model_name = ""
            torch.cuda.empty_cache()

    def design_voice(
        self, 
        text: str, 
        instruction: str, 
        language: str,
        output_path: str
    ) -> Optional[str]:
        """Create a voice reference WAV from text description.
        Uses the VoiceDesign model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Run load_model() first.")

        try:
            logger.info(f"Designing voice: [{instruction}]")
            
            # Syntax: model.generate_voice_design(text, instruct, language) -> (audio_list, sr)
            audio_list, sample_rate = self.model.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language,
            )
            
            # audio_list is usually a list of numpy arrays
            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_data, sample_rate)
            logger.info(f"Voice designed and saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Voice design failed: {e}")
            return None

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str, # The text spoken in ref_audio_path
        language: str,
        emotion_instruction: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate speech by cloning a voice and adding emotion.
        Uses the Base model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Run load_model() first.")

        # Apply emotion instruction to text
        if emotion_instruction:
            full_prompt = f"{emotion_instruction}. {text}"
        else:
            full_prompt = text

        # Fallback output path
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"tts_{time.time()}.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            logger.info(f"Cloning voice for: [{full_prompt[:30]}...]")
            
            # Syntax: model.generate_voice_clone(...)
            # Note: The API might take file paths or tensors. Based on typical Qwen wrappers:
            # We assume it takes paths. If it errors, we might need sf.read().
            
            result = self.model.generate_voice_clone(
                text=full_prompt,
                language=language,
                ref_audio=ref_audio_path, # Check parameter name: ref_audio or ref_wavs?
                ref_text=ref_text,
            )
            
            # Handle return: could be (audio_list, sr) or just audio_data
            if isinstance(result, tuple):
                audio_data, sample_rate = result
            elif hasattr(result, 'audio'): # HuggingFace style
                audio_data = result.audio
                sample_rate = result.sample_rate
            elif isinstance(result, list):
                audio_data = result[0]
                sample_rate = 24000 # Default fallback
            else:
                logger.warning(f"Unknown return type from model. Output might be empty.")
                return None

            audio_data = audio_data[0] if isinstance(audio_data, list) else audio_data
            
            # Normalize if needed (prevent clipping)
            import numpy as np
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            sf.write(output_path, audio_data, sample_rate)
            return output_path

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Fallback attempt with different parameter names if needed
            # e.g. ref_wavs instead of ref_audio
            try:
                logger.info("Retrying with parameter 'ref_wavs'...")
                audio_list, sample_rate = self.model.generate_voice_clone(
                    text=full_prompt,
                    language=language,
                    ref_wavs=[sf.read(ref_audio_path)], # Pass as tensor/array
                    ref_text=ref_text,
                )
                audio_data = audio_list[0]
                sf.write(output_path, audio_data, sample_rate)
                return output_path
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                return None
