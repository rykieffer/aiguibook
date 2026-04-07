"""TTSEngine - Qwen3-TTS wrapper with VoiceDesign and VoiceClone support.
Prioritizes faster-qwen3-tts for high-performance inference."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import threading
import queue
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Emotion instructions for French
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

# ============================================================
# Library Detection
# ============================================================
try:
    from faster_qwen3_tts import FasterQwen3TTS
    USES_FASTER = True
    logger.info("faster-qwen3-tts is installed and ready")
except ImportError:
    USES_FASTER = False
    try:
        from qwen_tts import Qwen3TTSModel
        logger.info("faster-qwen3-tts not found, using standard qwen-tts")
    except ImportError:
        logger.warning("Neither faster-qwen3-tts nor qwen-tts is installed!")

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class GenerationTask:
    """A single segment to generate."""
    segment_id: str
    text: str
    language: str
    ref_audio: str
    ref_text: str
    emotion: str
    emotion_instruction: str
    output_path: str


@dataclass
class GenerationTypeResult:
    """Result of a single generation task."""
    segment_id: str
    success: bool
    output_path: Optional[str]
    duration: float
    error: Optional[str] = None


# ============================================================
# TTSEngine
# ============================================================

class TTSEngine:
    """Engine wrapper that prioritizes faster-qwen3-tts if available."""

    def __init__(self, model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        self.model_path = model_path
        self._model = None
        self._loaded = False
        self._lock = threading.Lock()
        self._sample_rate = 24000

    def load_model(self, model_path: str = None, device: str = "cuda:0"):
        """Load the TTS model. Prioritizes FasterQwen3TTS."""
        if model_path:
            self.model_path = model_path
        self.load(device=device)

    def load(self, device: str = "cuda:0"):
        """Load the model."""
        if "cuda" not in str(device).lower():
            device = "cuda:0"

        with self._lock:
            if self._loaded:
                return
            try:
                if USES_FASTER:
                    logger.info(f"Loading FasterQwen3TTS: {self.model_path} on {device}")
                    # FasterQwen3TTS uses device="cuda" not device_map
                    self._model = FasterQwen3TTS.from_pretrained(self.model_path, device=device)
                else:
                    # Fallback to standard
                    from qwen_tts import Qwen3TTSModel
                    import torch
                    logger.info(f"Loading Qwen3TTSModel: {self.model_path} on {device}")
                    self._model = Qwen3TTSModel.from_pretrained(
                        self.model_path, device_map=device, torch_dtype=torch.bfloat16
                    )
                    # Safe eval for standard model
                    if hasattr(self._model, 'model') and hasattr(self._model.model, 'eval'):
                        self._model.model.eval()
                    elif hasattr(self._model, 'eval'):
                        self._model.eval()
                    self._sample_rate = 24000
                
                self._loaded = True
                logger.info(f"Model loaded successfully: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def unload(self):
        """Free VRAM."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._loaded = False
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def design_voice(self, text: str, instruction: str, language: str, output_path: str) -> Optional[str]:
        """Generate reference voice WAV using VoiceDesign."""
        try:
            # For Voice Design, we might not need the speed of graphs as much as raw capacity,
            # but faster-qwen3-tts exposes it too.
            model_to_use = self._model
            if not model_to_use:
                # Load a temporary model for design if none exists
                if USES_FASTER:
                             model_to_use = FasterQwen3TTS.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device="cuda"
                    )
                else:
                    from qwen_tts import Qwen3TTSModel
                    import torch
                    model_to_use = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        device_map="cuda",
                        torch_dtype=torch.bfloat16
                    )
                is_temp = True
            else:
                is_temp = False

            logger.info(f"Designing voice: [{instruction}]")
            
            # Call generate_voice_design
            # API: text, instruct, language
            if hasattr(model_to_use, 'generate_voice_design'):
                result = model_to_use.generate_voice_design(
                    text=text,
                    instruct=instruction,
                    language=language.lower()
                )
                if isinstance(result, tuple):
                    audio_list, sample_rate = result
                else:
                    audio_list, sample_rate = result, 24000
            elif hasattr(model_to_use, 'generate_voice_clone'):
                # Some versions might use a different name
                 raise NotImplementedError("generate_voice_design not found on this model")

            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_data, sample_rate)
            
            if is_temp:
                del model_to_use
                torch.cuda.empty_cache()
            return output_path
        except Exception as e:
            logger.error(f"Voice design failed: {e}")
            return None

    def generate_voice_clone(
        self, text: str, ref_audio: str, ref_text: str,
        language: str = "french", emotion_instruction: Optional[str] = None,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> GenerationTypeResult:
        """Generate speech using voice cloning."""
        if not self._loaded:
            self.load()

        if not text.strip():
            return GenerationTypeResult(segment_id="?", success=False, output_path=None, duration=0, error="Empty text")

        full_prompt = text.strip()
        if emotion_instruction:
            full_prompt = f"{emotion_instruction}. {text}"

        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"aiguibook_tts_{int(time.time())}_{hash(text) % 10000}.wav")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        start_time = time.time()
        audio = None
        sample_rate = self._sample_rate

        try:
            if progress_callback:
                progress_callback(0.2, "Generating speech...")
            
            # Use standard API compatible with both standard and faster-qwen3-tts
            # API: text, language, ref_audio, ref_text
            if hasattr(self._model, 'generate_voice_clone'):
                res = self._model.generate_voice_clone(
                    text=full_prompt,
                    language=language.lower(),
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                
                if isinstance(res, tuple):
                    audio, sample_rate = res
                elif isinstance(res, list):
                    audio = np.concatenate(res, axis=0)
                else:
                    audio = res
                    sample_rate = self._sample_rate # Default
            else:
                raise NotImplementedError("Model does not support generate_voice_clone")

            if audio is None:
                raise RuntimeError("Model returned None. Generation failed.")

            # Ensure numpy array
            if isinstance(audio, list):
                audio = np.concatenate(audio, axis=0)
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio)

            # Normalize and save
            max_val = np.max(np.abs(audio))
            if max_val > 0 and max_val > 1.0:
                audio = audio / max_val
            sf.write(output_path, audio, sample_rate, subtype="PCM_16")
            
            return GenerationTypeResult(
                segment_id="?",
                success=True,
                output_path=output_path,
                duration=len(audio) / sample_rate,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return GenerationTypeResult(
                segment_id="?",
                success=False,
                output_path=None,
                duration=0,
                error=str(e),
            )


# ============================================================
# TTSEnginePool (Worker Pool)
# ============================================================

class TTSEnginePool:
    """Pool of TTS engines. Supports concurrent generation."""

    def __init__(self, model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", pool_size: int = 1):
        self.model_path = model_path
        self.pool_size = pool_size
        self.workers: List[TTSEngine] = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._threads: List[threading.Thread] = []
        self._running = False

    def load_all(self):
        """Load all engines in the pool."""
        logger.info(f"Initializing pool with {self.pool_size} engines...")
        for i in range(self.pool_size):
            engine = TTSEngine(model_path=self.model_path)
            engine.load(device="cuda")
            self.workers.append(engine)
            logger.info(f"Engine {i+1}/{self.pool_size} loaded")
        self._running = True

    def unload_all(self):
        """Shutdown pool."""
        self._running = False
        for engine in self.workers:
            engine.unload()
        self.workers = []
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        logger.info("Pool shut down.")

    def _worker_loop(self, worker_id: int):
        engine = self.workers[worker_id]
        logger.info(f"Worker {worker_id+1} started.")
        while self._running:
            try:
                task = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break

            try:
                emo = task.emotion_instruction or EMOTION_INSTRUCTIONS_FR.get(task.emotion, "")
                result = engine.generate_voice_clone(
                    text=task.text,
                    ref_audio=task.ref_audio,
                    ref_text=task.ref_text,
                    language=task.language,
                    emotion_instruction=emo,
                    output_path=task.output_path,
                )
                result.segment_id = task.segment_id
                self.result_queue.put(result)
            except Exception as e:
                self.result_queue.put(
                    GenerationTypeResult(
                        segment_id=task.segment_id,
                        success=False,
                        output_path=None,
                        duration=0,
                        error=str(e),
                    )
                )
            finally:
                self.task_queue.task_done()
        logger.info(f"Worker {worker_id+1} stopped.")

    def start_generation(self, tasks: List[GenerationTask], progress_callback=None) -> List[GenerationTypeResult]:
        """Start processing tasks."""
        if not self.workers:
            raise RuntimeError("Pool not loaded.")
        
        self._running = True
        self.result_queue = queue.Queue()
        total = len(tasks)

        if progress_callback:
            progress_callback(0, total, "Starting...")

        self._threads = []
        for i in range(self.pool_size):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)

        for task in tasks:
            self.task_queue.put(task)
        for _ in range(self.pool_size):
            self.task_queue.put(None)

        results = []
        completed = 0
        start_time = time.time()

        while completed < total:
            try:
                result = self.result_queue.get(timeout=2.0)
                results.append(result)
                completed += 1
                elapsed = time.time() - start_time
                eta = 0
                if completed > 0:
                    eta = (elapsed / completed) * (total - completed)

                msg = f"[{completed}/{total}] ETA {int(eta/60):02d}:{int(eta%60):02d}"
                if progress_callback:
                    progress_callback(completed, total, msg)
            except queue.Empty:
                continue

        self.task_queue.join()
        results.sort(key=lambda r: r.segment_id)
        return results
