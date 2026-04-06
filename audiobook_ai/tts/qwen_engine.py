"""TTSEngine - Wraps Qwen3-TTS model for local GPU inference."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    """Wraps Qwen3-TTS model for local GPU inference with voice cloning support."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 4,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._initialized = False
        self._lock = threading.Lock()
        self._dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

    def initialize(self):
        """Load the model onto the GPU with proper dtype."""
        with self._lock:
            if self._initialized:
                return

            logger.info(f"Loading Qwen3-TTS model: {self.model_path}")
            
            # FIX: Use standard Transformers API instead of Qwen3TTSModel wrapper
            # The wrapper hides the .generate() method and causes the AttributeError.
            try:
                from transformers import AutoModel, AutoProcessor
                
                torch_dtype = self._dtype_map.get(self.dtype_str, torch.bfloat16)
                
                logger.info("Loading Processor...")
                self._processor = AutoProcessor.from_pretrained(self.model_path)
                
                logger.info("Loading Model...")
                self._model = AutoModel.from_pretrained(
                    self.model_path,
                    device_map=f"{self.device}" if self.device == "cuda" else self.device,
                    torch_dtype=torch_dtype,
                )
                self._model.eval()
                
            except ImportError:
                raise ImportError("transformers package required. Install with: pip install transformers")

            self._initialized = True
            logger.info("Qwen3-TTS model initialized successfully (Transformers API)")

    def generate(
        self,
        text: str,
        language: str = "French",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        emotion_instruction: Optional[str] = None,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[str, float]:
        """Generate speech from text."""
        if not self._initialized:
            raise RuntimeError("TTSEngine not initialized. Call initialize() first.")

        if not text.strip():
            raise ValueError("Empty text provided for generation")

        full_text = text.strip()
        if emotion_instruction:
            # Add instruction to text if using the custom voice capability
            full_text = f"{emotion_instruction}. {full_text}"

        # Determine output path
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"aiguibook_tts_{int(time.time())}_{hash(full_text) % 10000}.wav",
            )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        start_time = time.time()

        try:
            # FIX: Use Processor for text-to-tokens
            inputs = self._processor(text=full_text, language=language, return_tensors="pt")
            
            # Move inputs to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            logger.debug(f"Input IDs shape: {inputs.get('input_ids').shape if hasattr(inputs, 'get') else inputs.input_ids.shape}")

            with torch.no_grad():
                # Generate
                output = self._model.generate(**inputs, max_new_tokens=4096)
            
            # Decode audio
            audio_np = output.cpu().numpy().squeeze()
            
            # Save to WAV
            # Assuming 24kHz for Qwen models
            sf.write(output_path, audio_np, 24000, subtype="PCM_16")

            duration = time.time() - start_time
            
            logger.info(f"TTS generated {len(text)} chars in {duration:.2f}s -> {output_path}")
            return output_path, duration

        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            raise RuntimeError(f"TTS generation failed: {e}") from e

    def batch_generate(self, segments_list: List[dict], progress_callback=None) -> List[Tuple[str, float]]:
        """Generate speech for multiple segments."""
        results = []
        for i, seg in enumerate(segments_list):
            try:
                path, dur = self.generate(
                    text=seg.get("text", ""),
                    language=seg.get("language", "French"),
                    emotion_instruction=seg.get("emotion_instruction"),
                    output_path=seg.get("output_path"),
                )
                results.append((path, dur))
            except Exception as e:
                logger.error(f"Batch generation failed for segment {i}: {e}")
                results.append((None, 0.0))
        return results

    def __del__(self):
        if self._model is not None:
            try:
                del self._model
                del self._processor
                self._initialized = False
                torch.cuda.empty_cache()
            except:
                pass
