"""
QwenEngine - Fast TTS generation using faster-qwen3-tts.
Correct API: from_pretrained(model_name=...), generate_voice_design(instruct=...), generate_voice_clone(ref_audio=...)
"""

import logging
import os
import tempfile
import time
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger("AIGUIBook.Engine")


class TTSEngine:
    def __init__(self):
        self.model = None
        self.model_name = ""
        self.device = "cuda"

    def load_model(self, model_path: str, device: str = "cuda"):
        """Load a faster-qwen3-tts model.
        
        Valid models:
          - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign  (generate voices from text descriptions)
          - Qwen/Qwen3-TTS-12Hz-1.7B-Base          (voice cloning from reference audio)
          - Qwen/Qwen3-TTS-12Hz-0.6B-Base          (lightweight voice cloning)
        """
        if self.model is not None and self.model_name == model_path:
            return

        logger.info(f"Loading FasterQwen3TTS: {model_path}")
        
        try:
            from faster_qwen3_tts import FasterQwen3TTS
            
            self.model = FasterQwen3TTS.from_pretrained(
                model_name=model_path,
                device=device,
            )
            
            self.model_name = model_path
            self.device = device
            logger.info("Model loaded successfully.")
            
        except ImportError:
            raise RuntimeError(
                "faster-qwen3-tts not installed. Run: pip install faster-qwen3-tts"
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            self.model = None
            raise

    def unload_model(self):
        """Release GPU memory."""
        if self.model:
            import torch
            del self.model
            self.model = None
            self.model_name = ""
            torch.cuda.empty_cache()
            logger.info("VRAM cleared.")

    def design_voice(self, text: str, instruction: str, language: str, output_path: str) -> Optional[str]:
        """Generate a brand new voice from a text description.
        
        Uses the VoiceDesign model variant.
        
        Args:
            text: Sample text for the voice to speak (used to generate the reference audio)
            instruction: Voice description, e.g. "A warm, deep male voice, French accent, authoritative yet gentle"
            language: Language for synthesis (e.g. "french", "english")
            output_path: Where to save the generated WAV
            
        Returns:
            Path to the generated WAV, or None on failure
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Load VoiceDesign model first.")

        try:
            logger.info(f"Designing voice: {instruction}")
            
            # generate_voice_design returns (audio_array, sample_rate)
            audio, sr = self.model.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language,
            )
            
            # Handle list return
            if isinstance(audio, list):
                audio = audio[0]
            
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            import soundfile as sf
            sf.write(output_path, audio, sr)
            logger.info(f"Voice designed and saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Voice design failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_voice_clone(self, text: str, ref_audio_path: str, ref_text: str, language: str, emotion_instruction: Optional[str] = None, output_path: Optional[str] = None) -> Optional[str]:
        """Clone an existing voice and optionally apply emotion acting.
        
        Uses the Base model variant with a reference audio file.
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference WAV file (the voice to clone)
            ref_text: Transcript of the reference audio
            language: Language for synthesis
            emotion_instruction: Optional emotion instruction (e.g. "Parlez avec un ton calme et posé")
            output_path: Where to save the generated WAV
            
        Returns:
            Path to the generated WAV, or None on failure
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Load Base model first.")

        out_path = output_path or os.path.join(tempfile.gettempdir(), f"tts_{time.time()}.wav")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        try:
            logger.info(f"Generating audio [{len(text)} chars]")
            
            kwargs = {
                "text": text,
                "language": language,
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
            }
            
            # Add emotion/style instruction if provided
            if emotion_instruction and emotion_instruction.strip():
                kwargs["instruct"] = emotion_instruction
            
            audio, sr = self.model.generate_voice_clone(**kwargs)
            
            if isinstance(audio, list):
                audio = audio[0]
            
            # Normalize if clipping
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            import soundfile as sf
            sf.write(out_path, audio, sr)
            return out_path
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

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
        """Assemble WAV files into a single M4A with silence gaps and chapter markers.
        
        Args:
            wav_files: Ordered list of WAV file paths
            output_path: Output M4A path
            silence_duration: Seconds of silence between segments
            sample_rate: Target sample rate
            normalize: Apply peak normalization
            book_title: Title for metadata
            chapter_titles: Optional chapter boundary titles
            
        Returns:
            Path to the assembled M4A file
        """
        import soundfile as sf
        import subprocess

        if not wav_files:
            raise ValueError("No WAV files to assemble")
        
        logger.info(f"Assembling {len(wav_files)} WAV files into {output_path}")
        logger.info(f"Silence between segments: {silence_duration}s")
        
        silence_samples = int(silence_duration * sample_rate)
        silence_buf = np.zeros(silence_samples, dtype=np.float32)
        
        all_audio = []
        chapter_starts_ms = []
        cumulative_ms = 0
        
        for i, wav_path in enumerate(wav_files):
            if not os.path.exists(wav_path):
                logger.warning(f"Skipping missing file: {wav_path}")
                continue
            
            try:
                data, sr = sf.read(wav_path, dtype='float32')
                
                # Resample if needed via ffmpeg
                if sr != sample_rate:
                    resampled = wav_path + ".resampled.wav"
                    subprocess.run([
                        "ffmpeg", "-y", "-i", wav_path,
                        "-ar", str(sample_rate), "-ac", "1",
                        resampled
                    ], capture_output=True, timeout=30)
                    data, sr = sf.read(resampled, dtype='float32')
                    try: os.unlink(resampled)
                    except: pass
                
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                chapter_starts_ms.append(int(cumulative_ms))
                all_audio.append(data)
                cumulative_ms += int(len(data) / sample_rate * 1000)
                
                # Add silence between segments (not after last)
                if i < len(wav_files) - 1:
                    all_audio.append(silence_buf)
                    cumulative_ms += int(silence_duration * 1000)
                    
            except Exception as e:
                logger.warning(f"Error reading {wav_path}: {e}")
                continue
        
        if not all_audio:
            raise ValueError("No valid audio data to assemble")
        
        full_audio = np.concatenate(all_audio)
        
        # Peak normalization
        if normalize:
            peak = np.max(np.abs(full_audio))
            if peak > 0:
                target = 0.95
                full_audio = full_audio * (target / peak)
                logger.info(f"Normalized: peak {peak:.3f} -> {target}")
        
        # Write temp WAV
        temp_wav = output_path + ".temp.wav"
        sf.write(temp_wav, full_audio, sample_rate)
        
        # Build FFmpeg metadata
        meta = ";FFMETADATA1\n"
        meta += f"title={book_title}\ngenre=Audiobook\n"
        
        for i, start_ms in enumerate(chapter_starts_ms):
            end_ms = chapter_starts_ms[i + 1] if i + 1 < len(chapter_starts_ms) else int(len(full_audio) / sample_rate * 1000)
            title = f"Part {i+1}"
            if chapter_titles and i < len(chapter_titles):
                title = chapter_titles[i]
            meta += f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\nEND={end_ms}\ntitle={title}\n"
        
        meta_path = output_path + ".meta.txt"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(meta)
        
        # Encode to M4A
        cmd = [
            "ffmpeg", "-y", "-i", temp_wav, "-i", meta_path,
            "-map_metadata", "1", "-c:a", "aac", "-b:a", "128k",
            "-ar", "44100", "-movflags", "+faststart", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr[:500]}")
            # Fallback: save as WAV
            wav_out = output_path.replace(".m4a", ".wav").replace(".m4b", ".wav")
            import shutil
            shutil.copy2(temp_wav, wav_out)
            logger.info(f"Fallback: saved as WAV: {wav_out}")
            output_path = wav_out
        
        for tmp in [temp_wav, meta_path]:
            try: os.unlink(tmp)
            except: pass
        
        logger.info(f"M4A assembled: {output_path}")
        return output_path
