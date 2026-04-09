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
            
            # Normalize if needed (prevent clipping)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            sf.write(out_path, audio_data, sr)
            return out_path
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def generate_silence(duration_sec: float, sample_rate: int = 24000) -> Tuple[np.ndarray, int]:
        """Generate a silence audio array."""
        num_samples = int(duration_sec * sample_rate)
        silence = np.zeros(num_samples, dtype=np.float32)
        return silence, sample_rate

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
        """Assemble a list of WAV files into a single M4A with silence between segments.
        
        Args:
            wav_files: Ordered list of WAV file paths
            output_path: Output M4A/M4B file path
            silence_duration: Seconds of silence between segments (default 0.75)
            sample_rate: Target sample rate
            normalize: Apply loudness normalization
            book_title: Title for metadata
            chapter_titles: Optional list of chapter boundary titles
            
        Returns:
            Path to the assembled M4A file
        """
        if not wav_files:
            raise ValueError("No WAV files to assemble")
        
        logger.info(f"Assembling {len(wav_files)} WAV files into {output_path}")
        logger.info(f"Silence between segments: {silence_duration}s")
        
        # Generate silence buffer
        silence_samples = int(silence_duration * sample_rate)
        silence_buf = np.zeros(silence_samples, dtype=np.float32)
        
        # Concatenate all audio with silence
        all_audio = []
        chapter_starts_ms = []
        cumulative_ms = 0
        
        for i, wav_path in enumerate(wav_files):
            if not os.path.exists(wav_path):
                logger.warning(f"Skipping missing file: {wav_path}")
                continue
            
            try:
                data, sr = sf.read(wav_path, dtype='float32')
                
                # Resample if needed
                if sr != sample_rate:
                    import subprocess
                    # Use ffmpeg for resampling
                    resampled_path = wav_path + ".resampled.wav"
                    subprocess.run([
                        "ffmpeg", "-y", "-i", wav_path,
                        "-ar", str(sample_rate),
                        "-ac", "1",
                        resampled_path
                    ], capture_output=True, timeout=30)
                    data, sr = sf.read(resampled_path, dtype='float32')
                    try: os.unlink(resampled_path)
                    except: pass
                
                # Convert stereo to mono if needed
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                # Track chapter start position
                chapter_starts_ms.append(int(cumulative_ms))
                
                all_audio.append(data)
                cumulative_ms += int(len(data) / sample_rate * 1000)
                
                # Add silence between segments (not after the last one)
                if i < len(wav_files) - 1:
                    all_audio.append(silence_buf)
                    cumulative_ms += int(silence_duration * 1000)
                    
            except Exception as e:
                logger.warning(f"Error reading {wav_path}: {e}")
                continue
        
        if not all_audio:
            raise ValueError("No valid audio data to assemble")
        
        # Concatenate all buffers
        full_audio = np.concatenate(all_audio)
        
        # Normalize loudness (simple peak normalization)
        if normalize:
            peak = np.max(np.abs(full_audio))
            if peak > 0:
                target_peak = 0.95
                full_audio = full_audio * (target_peak / peak)
                logger.info(f"Normalized audio: peak {peak:.3f} -> {target_peak}")
        
        # Write intermediate WAV
        temp_wav = output_path + ".temp.wav"
        sf.write(temp_wav, full_audio, sample_rate)
        
        # Build chapter metadata for FFmpeg
        metadata_content = ";FFMETADATA1\n"
        metadata_content += f"title={book_title}\n"
        metadata_content += "genre=Audiobook\n"
        
        for i, start_ms in enumerate(chapter_starts_ms):
            # Calculate end time (next chapter start, or end of audio)
            if i + 1 < len(chapter_starts_ms):
                end_ms = chapter_starts_ms[i + 1]
            else:
                end_ms = int(len(full_audio) / sample_rate * 1000)
            
            title = "Chapter"
            if chapter_titles and i < len(chapter_titles):
                title = chapter_titles[i]
            else:
                title = f"Part {i+1}"
            
            metadata_content += "\n[CHAPTER]\n"
            metadata_content += "TIMEBASE=1/1000\n"
            metadata_content += f"START={start_ms}\n"
            metadata_content += f"END={end_ms}\n"
            metadata_content += f"title={title}\n"
        
        # Write metadata file
        meta_path = output_path + ".meta.txt"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(metadata_content)
        
        # Encode to M4A with FFmpeg
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_wav,
            "-i", meta_path,
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg encoding failed: {result.stderr[:500]}")
            # Fallback: just copy the WAV as output
            import shutil
            wav_output = output_path.replace(".m4a", ".wav").replace(".m4b", ".wav")
            shutil.copy2(temp_wav, wav_output)
            logger.info(f"Fallback: saved as WAV: {wav_output}")
            output_path = wav_output
        else:
            logger.info(f"M4A encoded successfully: {output_path}")
        
        # Cleanup temp files
        for tmp in [temp_wav, meta_path]:
            try: os.unlink(tmp)
            except: pass
        
        return output_path
