"""TTSEngine - Wraps Qwen3-TTS model for local GPU inference."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        """
        Args:
            model_path: Hugging Face model path
            device: PyTorch device ("cuda" or "cpu")
            dtype: Torch dtype string ("bfloat16", "float16", "float32")
            batch_size: Batch size for batch_generate
        """
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype
        self.batch_size = batch_size
        self._model = None
        self._initialized = False
        self._lock = threading.Lock()
        self._dtype_map = {
            "bfloat16": "torch.bfloat16",
            "float16": "torch.float16",
            "float32": "torch.float32",
        }

    def initialize(self):
        """Load the model onto the GPU with proper dtype.

        Must be called before generate(). Thread-safe.
        """
        with self._lock:
            if self._initialized:
                return

            logger.info(f"Loading Qwen3-TTS model: {self.model_path}")
            logger.info(f"Device: {self.device}, dtype: {self.dtype_str}")

            try:
                import torch
            except ImportError:
                raise ImportError("torch is required for TTSEngine")

            # Map dtype string to torch dtype
            dtype_str = self._dtype_map.get(self.dtype_str, "torch.bfloat16")

            try:
                from qwen_tts import Qwen3TTSModel
            except ImportError:
                raise ImportError(
                    "qwen-tts package required. Install with: pip install qwen-tts"
                )

            # Try loading with flash_attention_2 first
            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    self.model_path,
                    device_map=f"{self.device}:0" if self.device == "cuda" else self.device,
                    torch_dtype=eval(dtype_str),
                    attn_implementation="flash_attention_2",
                )
                logger.info("Model loaded with flash_attention_2")
            except Exception as e:
                logger.warning(f"flash_attention_2 not available, falling back: {e}")
                try:
                    self._model = Qwen3TTSModel.from_pretrained(
                        self.model_path,
                        device_map=f"{self.device}:0" if self.device == "cuda" else self.device,
                        torch_dtype=eval(dtype_str),
                    )
                    logger.info("Model loaded without flash_attention_2")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load Qwen3-TTS model: {e2}") from e2

            # Handle wrappers safely (RTX 5080 / cu130 updates often wrap the model)
            if hasattr(self._model, 'eval'):
                self._model.eval()
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'eval'):
                self._model.model.eval()
            self._initialized = True
            logger.info("Qwen3-TTS model initialized successfully")

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
        """Generate speech from text.

        Args:
            text: Text to synthesize
            language: Target language ("French", "English", etc.)
            ref_audio: Path to reference audio for voice cloning
            ref_text: Transcript of reference audio (required for voice cloning)
            emotion_instruction: Natural language instruction for emotional delivery
            output_path: Where to save the output audio file
            progress_callback: Optional callback(current_percent, status_text)

        Returns:
            Tuple of (audio_path, duration_seconds)
        """
        if not self._initialized:
            raise RuntimeError("TTSEngine not initialized. Call initialize() first.")

        if progress_callback:
            progress_callback(0.1, "Preparing generation...")

        text_to_generate = text.strip()
        if not text_to_generate:
            raise ValueError("Empty text provided for generation")

        # Prepend emotion instruction if provided
        if emotion_instruction:
            full_text = f"{emotion_instruction}. {text_to_generate}"
        else:
            full_text = text_to_generate

        # Determine output path
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"aiguibook_tts_{int(time.time())}_{hash(full_text) % 10000}.wav",
            )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        duration = 0.0

        try:
            if ref_audio and os.path.exists(ref_audio):
                # Voice cloning mode
                logger.debug(
                    f"Voice clone: lang={language}, ref={ref_audio}, ref_text_len={len(ref_text or '')}"
                )
                if progress_callback:
                    progress_callback(0.3, "Voice cloning...")

                start_time = time.time()
                
                # Build keyword arguments
                gen_kwargs = {
                    "text": full_text,
                    "language": language,
                    "ref_audio": ref_audio,
                    "output_path": output_path,
                }
                if ref_text:
                    gen_kwargs["ref_text"] = ref_text

                result = self._model.generate_voice_clone(**gen_kwargs)
                duration = time.time() - start_time

                if isinstance(result, dict) and "duration" in result:
                    duration = result["duration"]
                    
            else:
                # Standard generation (built-in voice)
                logger.debug(f"Standard generation: lang={language}")
                if progress_callback:
                    progress_callback(0.3, "Generating speech...")

                start_time = time.time()
                result = self._model.generate(
                    text=full_text,
                    language=language,
                    output_path=output_path,
                )
                duration = time.time() - start_time

                if isinstance(result, dict) and "duration" in result:
                    duration = result["duration"]

            if progress_callback:
                progress_callback(1.0, "Generation complete")

            # Verify output was created
            if not os.path.exists(output_path):
                # Fallback: try to generate into a bytes buffer
                logger.warning(
                    f"Output file not created at {output_path}, trying alternative"
                )
                import torch
                import torchaudio

                # Try direct tensor output
                with torch.no_grad():
                    gen_kwargs = {
                        "text": full_text,
                        "language": language,
                    }
                    if ref_audio and os.path.exists(ref_audio):
                        gen_kwargs["ref_audio"] = ref_audio
                        if ref_text:
                            gen_kwargs["ref_text"] = ref_text
                    
                    audio_tensor = self._model(**gen_kwargs)
                    if isinstance(audio_tensor, dict):
                        audio_tensor = audio_tensor.get("wav", audio_tensor.get("audio"))
                    
                    if isinstance(audio_tensor, dict):
                        audio_tensor = audio_tensor.get("wav", audio_tensor.get("audio"))
                    
                    if audio_tensor is not None:
                        if hasattr(audio_tensor, "numpy"):
                            audio_tensor = audio_tensor.numpy()
                        
                        import numpy as np
                        import soundfile as sf
                        
                        if isinstance(audio_tensor, np.ndarray):
                            # Normalize
                            if np.abs(audio_tensor).max() > 0:
                                audio_tensor = audio_tensor / np.abs(audio_tensor).max()
                            sf.write(
                                output_path,
                                audio_tensor,
                                24000,
                                subtype="PCM_16",
                            )
                            duration = len(audio_tensor) / 24000

            logger.info(
                f"TTS generated {len(text_to_generate)} chars in {duration:.2f}s -> {output_path}"
            )
            return output_path, duration

        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            raise RuntimeError(f"TTS generation failed: {e}") from e

    def batch_generate(
        self,
        segments_list: List[dict],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Tuple[str, float]]:
        """Generate speech for multiple segments.

        Args:
            segments_list: List of dicts with keys:
                - text (str): Text to speak
                - language (str): Target language
                - ref_audio (str, optional): Reference audio path
                - ref_text (str, optional): Reference audio transcript
                - emotion_instruction (str, optional): Emotion/style instruction
                - output_path (str): Output file path
            progress_callback: Optional callback

        Returns:
            List of (audio_path, duration_seconds) tuples
        """
        results = []
        total = len(segments_list)

        for i, seg in enumerate(segments_list):
            try:
                text = seg.get("text", "")
                language = seg.get("language", "French")
                ref_audio = seg.get("ref_audio")
                ref_text = seg.get("ref_text")
                emotion_instr = seg.get("emotion_instruction")
                output_path = seg.get("output_path")

                def seg_callback(pct, status):
                    if progress_callback:
                        overall = (i + pct) / total
                        progress_callback(overall, f"{status} ({i+1}/{total})")

                path, dur = self.generate(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    emotion_instruction=emotion_instr,
                    output_path=output_path,
                    progress_callback=seg_callback,
                )
                results.append((path, dur))
            except Exception as e:
                logger.error(f"Batch generation failed for segment {i}: {e}")
                results.append((None, 0.0))

        return results


    def generate_voice_design(
        self,
        text: str,
        language: str = "French",
        instruct: str = "",
        output_path: Optional[str] = None,
        model_variant: str = "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
    ) -> Tuple[str, float]:
        """Generate a unique voice from text description using VoiceDesign model.

        Args:
            text: Example text to speak with the designed voice
            language: Target language
            instruct: Natural language voice description 
                      (e.g., "Deep warm male voice, French accent")
            output_path: Where to save output
            model_variant: Which VoiceDesign model to use

        Returns:
            (audio_path, duration_seconds)
        """
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"aiguibook_vd_{int(time.time())}.wav")

        logger.info(f"Generating voice design: {instruct[:60]}...")
        logger.info(f"Using VoiceDesign model: {model_variant}")

        # Save current state
        old_model = self._model
        old_initialized = self._initialized
        old_path = self.model_path

        try:
            # Temporarily switch to VoiceDesign model
            self._model = None
            self._initialized = False
            self.model_path = model_variant
            self.initialize()

            import soundfile as sf
            import torch

            wavs, sr = self._model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )

            # Convert to numpy and save
            if isinstance(wavs, torch.Tensor):
                audio_np = wavs.detach().cpu().numpy()
            else:
                audio_np = wavs

            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)

            sf.write(output_path, audio_np.T if audio_np.shape[0] == 1 else audio_np, sr)

            duration = len(audio_np) / sr
            logger.info(f"Voice design audio saved: {output_path} ({duration:.2f}s)")
            return output_path, duration

        except Exception as e:
            logger.error(f"Voice design generation failed: {e}")
            raise
        finally:
            # Restore original model
            try:
                if old_model is not None:
                    del self._model
            except Exception:
                pass
            self._model = old_model
            self._initialized = old_initialized
            self.model_path = old_path


    def cleanup(self):
        """Release model resources."""
        if self._model is not None:
            try:
                import torch
                del self._model
                self._model = None
                self._initialized = False
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                logger.info("TTS model resources released")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        self.cleanup()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"TTSEngine(model='{self.model_path}', {status})"
