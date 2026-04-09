"""
TTSEngine - Wraps faster-qwen3-tts for optimized local GPU inference.
Uses FasterQwen3TTS with CUDA graph capture for 10x faster generation.
"""

import logging
import os
import tempfile
import time
import subprocess
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("AIGUIBook.Engine")


class TTSEngine:
    """Manages Qwen3-TTS models: VoiceDesign for creating voices, Base for generation."""

    def __init__(self):
        self.model = None
        self.model_name = ""
        self.device = "cuda"

    def load_model(self, model_name: str, device: str = "cuda"):
        """Load a Qwen3-TTS model variant via faster-qwen3-tts."""
        if self.model is not None and self.model_name == model_name:
            return

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
            logger.info(f"Model loaded: {model_name}")

        except ImportError:
            raise RuntimeError(
                "faster-qwen3-tts not installed. Run: pip install faster-qwen3-tts"
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.model = None
            raise

    def unload_model(self):
        """Clear VRAM."""
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            self.model_name = ""
            torch.cuda.empty_cache()
            logger.info("VRAM cleared.")

    # ── Voice Design ──────────────────────────────────────────

    def design_voice(
        self,
        text: str,
        instruction: str,
        language: str,
        output_path: str,
    ) -> Optional[str]:
        """Create a brand new voice from a text description.
        Uses the VoiceDesign model variant.

        Args:
            text: Sample text to speak
            instruction: Voice description (e.g. "Deep male voice, French accent")
            language: Target language (e.g. "french")
            output_path: Where to save the WAV

        Returns:
            Path to saved WAV, or None on failure
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")

        try:
            logger.info(f"Designing voice: [{instruction}]")

            # faster-qwen3-tts API: generate_voice_design(text, instruct, language)
            audio_list, sr = self.model.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language,
            )

            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sf.write(output_path, audio_data, sr)
            logger.info(f"Voice designed: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Voice design failed: {e}")
            return None

    # ── Voice Clone + Emotion Acting ───────────────────────────

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
        emotion_instruction: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Clone a voice and apply emotion acting.

        Uses the Base model variant with generate_voice_clone().
        The `instruct` parameter carries the emotion/acting direction.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference WAV (3+ seconds)
            ref_text: Transcription of the reference audio
            language: Target language
            emotion_instruction: Acting direction (e.g. "Parlez avec colère")
            output_path: Where to save the WAV

        Returns:
            Path to saved WAV, or None on failure
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")

        if not ref_audio_path or not os.path.exists(ref_audio_path):
            logger.error(f"Reference audio not found: {ref_audio_path}")
            return None

        out_path = output_path or os.path.join(
            tempfile.gettempdir(), f"tts_{int(time.time())}.wav"
        )
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        try:
            logger.info(f"Generating [{len(text)} chars] emotion={emotion_instruction or 'none'}")

            # faster-qwen3-tts API: generate_voice_clone(text, language, ref_audio, ref_text, instruct)
            audio_list, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                instruct=emotion_instruction,  # Emotion acting goes here, NOT prepended to text
                xvec_only=False,  # ICL mode for best instruct compliance
            )

            audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list

            # Normalize if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            sf.write(out_path, audio_data, sr)
            logger.info(f"Generated: {out_path}")
            return out_path

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # ── Assembly ───────────────────────────────────────────────

    @staticmethod
    def assemble_wav_files(
        wav_files: List[str],
        output_path: str,
        silence_duration: float = 0.75,
        sample_rate: int = 24000,
        normalize: bool = True,
        book_title: str = "Audiobook",
        chapter_titles: Optional[List[str]] = None,
    ) -> str:
        """Concatenate WAV segments into a final M4A audiobook.

        Uses FFmpeg for concatenation with silence gaps,
        loudness normalization, and AAC encoding.
        """
        if not wav_files:
            raise ValueError("No WAV files provided")

        logger.info(f"Assembling {len(wav_files)} WAV files -> {output_path}")

        # 1. Create silence padding file
        tmp_dir = tempfile.mkdtemp(prefix="aiguibook_asm_")
        silence_path = os.path.join(tmp_dir, "silence.wav")
        silence_samples = int(silence_duration * sample_rate)
        silence_data = np.zeros(silence_samples, dtype=np.float32)
        sf.write(silence_path, silence_data, sample_rate)

        # 2. Build concat list for FFmpeg
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for wav in wav_files:
                f.write(f"file '{wav}'\n")
                f.write(f"file '{silence_path}'\n")

        # 3. Concatenate with FFmpeg
        raw_output = os.path.join(tmp_dir, "raw.wav")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-ar", str(sample_rate),
            raw_output,
        ]
        logger.info("Concatenating segments...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        # 4. Normalize and encode to M4A
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        norm_cmd = [
            "ffmpeg", "-y",
            "-i", raw_output,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "128k",
            "-ar", "44100",
            "-movflags", "+faststart",
        ]

        # Add chapter metadata if provided
        if chapter_titles:
            metadata_file = os.path.join(tmp_dir, "chapters.txt")
            # Simple chapter markers based on segment count
            with open(metadata_file, "w") as f:
                f.write(";FFMETADATA1\n")
                # We can't easily know durations without reading each WAV,
                # so we skip detailed chapter markers for now
            # TODO: Add proper chapter markers based on WAV durations

        norm_cmd.append(output_path)
        logger.info("Normalizing and encoding M4A...")
        result = subprocess.run(norm_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback: copy raw as wav
            logger.warning(f"M4A encoding failed: {result.stderr}. Falling back to WAV.")
            import shutil
            wav_output = output_path.rsplit(".", 1)[0] + ".wav"
            shutil.copy2(raw_output, wav_output)
            output_path = wav_output

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(f"Audiobook saved: {output_path}")
        return output_path
