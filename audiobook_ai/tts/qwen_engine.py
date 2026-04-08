"""
QwenEngine - Fast TTS generation using faster-qwen3-tts.
Optimized for local GPU inference without brittle dependencies like flash-attn.
"""

import logging
import os
import tempfile
import time
from typing import Optional, List, Tuple
import soundfile as sf
import numpy as np

logger = logging.getLogger("AIGUIBook.Engine")


class TTSEngine:
    def __init__(self):
        self.model = None
        self.model_name = ""
        self.device = "cuda"

    def load_model(self, model_path: str, device: str = "cuda"):
        if self.model is not None and self.model_name == model_path:
            return

        logger.info(f"Loading FasterQwen3TTS: {model_path} on {device}")
        
        try:
            from faster_qwen3_tts import FasterQwen3TTS
            
            # Load the optimized model
            self.model = FasterQwen3TTS.from_pretrained(
                pretrained_model=model_path,
                device=device
            )
            
            self.model_name = model_path
            self.device = device
            logger.info("Model loaded successfully.")
            
        except ImportError:
            raise RuntimeError("Please install the optimized engine: pip install faster-qwen3-tts")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            self.model = None
            raise

    def unload_model(self):
        if self.model:
            import torch
            del self.model
            self.model = None
            self.model_name = ""
            torch.cuda.empty_cache()
            logger.info("VRAM cleared.")

    def design_voice(self, text: str, instruction: str, language: str, output_path: str) -> Optional[str]:
        """Generate a brand new voice from a text description."""
        if not self.model:
            raise RuntimeError("Model not loaded.")

        try:
            logger.info(f"Designing voice: {instruction}")
            
            # The faster-qwen3-tts library supports voice design on the VoiceDesign model variant
            audio_list, sr = self.model.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language
            )
            
            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_data, sr)
            return output_path
            
        except Exception as e:
            logger.error(f"Voice design failed: {e}")
            return None

    def generate_voice_clone(self, text: str, ref_audio_path: str, ref_text: str, language: str, emotion_instruction: Optional[str] = None, output_path: Optional[str] = None) -> Optional[str]:
        """Clone an existing voice and apply emotion acting to the text."""
        if not self.model:
            raise RuntimeError("Model not loaded.")

        full_prompt = f"{emotion_instruction}. {text}" if emotion_instruction else text
        out_path = output_path or os.path.join(tempfile.gettempdir(), f"tts_{time.time()}.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            logger.info(f"Generating audio [{len(text)} chars]")
            
            audio_list, sr = self.model.generate_voice_clone(
                text=full_prompt,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )
            
            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list
            sf.write(out_path, audio_data, sr)
            return out_path
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
