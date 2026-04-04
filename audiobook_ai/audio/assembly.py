"""AudioAssembly - Assembles segment audio into chapters and final M4B."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AudioAssembly:
    """Assembles individual segment audio files into complete chapters and M4B."""

    def __init__(self, project, config):
        """
        Args:
            project: BookProject instance
            config: AudiobookConfig instance
        """
        self.project = project
        self.config = config
        self._ffmpeg_path = self._find_ffmpeg()

    @staticmethod
    def _find_ffmpeg() -> str:
        """Find ffmpeg binary."""
        # Check common locations
        common_paths = [
            "ffmpeg",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/snap/bin/ffmpeg",
        ]
        for p in common_paths:
            try:
                result = subprocess.run(
                    [p, "-version"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return p
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        logger.warning("FFmpeg not found! Audio assembly will fail.")
        return "ffmpeg"  # Return anyway, will error at runtime

    def _run_ffmpeg(self, args: List[str], description: str = "FFmpeg operation") -> subprocess.CompletedProcess:
        """Run ffmpeg with given arguments.

        Args:
            args: FFmpeg command arguments (after 'ffmpeg')
            description: Description for logging

        Returns:
            CompletedProcess result
        """
        cmd = [self._ffmpeg_path] + args
        logger.debug(f"Running FFmpeg: {' '.join(cmd[:10])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max
            )
            if result.returncode != 0:
                logger.error(
                    f"{description} failed: {result.stderr[:500]}"
                )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"{description} timed out")
            raise RuntimeError(f"{description} timed out")
        except Exception as e:
            logger.error(f"{description} error: {e}")
            raise

    def normalize_audio(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Normalize audio to -16 LUFS (audiobook standard).

        Args:
            input_path: Input audio file
            output_path: Output audio file (default: overwrite input)

        Returns:
            Path to normalized audio
        """
        if output_path is None:
            output_path = input_path + "_normalized.wav"
            is_inplace = True
        else:
            is_inplace = False

        # Two-pass loudnorm for accurate normalization
        # Pass 1: measure loudness
        measure_cmd = [
            "-i", input_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f", "null",
            "/dev/null",
        ]
        result = self._run_ffmpeg(measure_cmd, "Loudness measurement")

        # Parse measured values
        target_loudness = "-16.0"
        target_tp = "-1.5"
        target_lra = "11.0"
        measured_i = "-16.0"
        measured_tp = "-1.5"
        measured_lra = "11.0"
        measured_thresh = "-26.0"
        measured_offset = "0.0"

        if result.stderr:
            json_match = re.search(
                r'\{\s*"input_i"\s*:.*?"output_offset"\s*:\s*"[^"]*"\s*\}',
                result.stderr,
                re.DOTALL,
            )
            if json_match:
                try:
                    j = json.loads(json_match.group())
                    measured_i = j.get("input_i", "-16.0")
                    measured_tp = j.get("input_tp", "-1.5")
                    measured_lra = j.get("input_lra", "11.0")
                    measured_thresh = j.get("input_thresh", "-26.0")
                    measured_offset = j.get("output_offset", "0.0")
                except json.JSONDecodeError:
                    logger.warning("Could not parse loudnorm measurements")

        # Pass 2: apply normalization
        loudnorm_filter = (
            f"loudnorm=I={target_loudness}:TP={target_tp}:LRA={target_lra}"
            f":measured_I={measured_i}:measured_TP={measured_tp}"
            f":measured_LRA={measured_lra}:measured_thresh={measured_thresh}"
            f":offset={measured_offset}:linear=true"
        )

        normalize_cmd = [
            "-i", input_path,
            "-af", loudnorm_filter,
            "-ar", str(self.config.get("output", "sample_rate", 24000)),
            "-c:a", "pcm_s16le",
            "-y",
            output_path,
        ]
        self._run_ffmpeg(normalize_cmd, "Audio normalization")

        if is_inplace:
            os.replace(output_path, input_path)
            return input_path

        return output_path

    def concatenate_audio(
        self,
        audio_files: List[str],
        output_path: str,
        crossfade: float = 0.5,
    ) -> str:
        """Concatenate audio files with crossfade.

        Args:
            audio_files: List of audio file paths
            output_path: Output audio file path
            crossfade: Crossfade duration in seconds

        Returns:
            Path to concatenated audio
        """
        if not audio_files:
            raise ValueError("No audio files to concatenate")

        if len(audio_files) == 1:
            # Single file: just copy or normalize
            input_path = audio_files[0]
            if input_path != output_path:
                self._run_ffmpeg([
                    "-i", input_path,
                    "-c:a", "pcm_s16le",
                    "-y",
                    output_path,
                ], "Copy single audio")
            return output_path

        # Build concat complex filter with crossfade
        inputs = []
        filters = []
        crossfade_ms = int(crossfade * 1000)

        for i, fpath in enumerate(audio_files):
            inputs.extend(["-i", fpath])

        # For first file, start with [0:a]
        # Chain with crossfade: [0:a][1:a]acrossfade=d=0.5:c1=tri:c2=tri, ...
        base_label = "0:a"

        for i in range(1, len(audio_files)):
            out_label = f"x{i}"
            if i == len(audio_files) - 1:
                out_label = "out"

            if i == 1:
                filter_str = f"[{base_label}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[{out_label}]"
            else:
                prev_label = f"x{i-1}"
                filter_str = f"[{prev_label}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[{out_label}]"

            filters.append(filter_str)
            base_label = out_label

        filter_complex = ";".join(filters)

        concat_cmd = inputs + [
            "-filter_complex", filter_complex,
            "-ar", str(self.config.get("output", "sample_rate", 24000)),
            "-c:a", "pcm_s16le",
            "-y",
            output_path,
        ]
        self._run_ffmpeg(concat_cmd, f"Concatenate {len(audio_files)} files with crossfade")
        return output_path

    def assemble_chapter(
        self,
        chapter_idx: int,
        segments_list: List[dict],
    ) -> str:
        """Assemble a chapter from segment audio files.

        Args:
            chapter_idx: Chapter index
            segments_list: List of dicts with 'audio_path' keys, in order

        Returns:
            Path to chapter audio file
        """
        audio_files = []
        for seg in segments_list:
            path = seg.get("audio_path")
            if path and os.path.exists(path):
                audio_files.append(path)
            else:
                logger.warning(f"Segment audio missing: {path}")

        if not audio_files:
            raise ValueError(f"No audio files for chapter {chapter_idx}")

        chapter_path = self.project.get_chapter_audio_path(chapter_idx)
        crossfade = self.config.get("output", "crossfade_duration", 0.5)

        # Concatenate with crossfade
        self.concatenate_audio(audio_files, chapter_path, crossfade=crossfade)

        # Normalize if configured
        if self.config.get("output", "normalize_audio", True):
            self.normalize_audio(chapter_path)

        logger.info(f"Chapter {chapter_idx} assembled: {chapter_path}")
        return chapter_path

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in milliseconds using ffprobe.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in milliseconds
        """
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            duration = float(result.stdout.strip())
            return int(duration * 1000)  # Convert to ms
        except (ValueError, TypeError):
            logger.warning(f"Could not determine duration of {audio_path}")
            return 0

    def add_chapter_metadata(
        self,
        input_path: str,
        chapters_metadata: List[Dict[str, Any]],
        output_path: str,
        book_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Add chapter markers/ metadata to an audio file.

        Args:
            input_path: Input audio file (merged)
            chapters_metadata: List of dicts with 'title', 'start_ms', 'end_ms'
            output_path: Output M4B/M4A path
            book_metadata: Optional book metadata dict

        Returns:
            Path to file with metadata
        """
        # Build metadata string for FFmpeg
        metadata_content = ";FFMETADATA1\n"

        if book_metadata:
            if book_metadata.get("title"):
                metadata_content += f"title={book_metadata['title']}\n"
            if book_metadata.get("author"):
                metadata_content += f"artist={book_metadata['author']}\n"
                metadata_content += f"album_artist={book_metadata['author']}\n"
            if book_metadata.get("publisher"):
                metadata_content += f"publisher={book_metadata['publisher']}\n"
            if book_metadata.get("language"):
                metadata_content += f"language={book_metadata['language']}\n"
            if book_metadata.get("description"):
                metadata_content += f"comment={book_metadata['description']}\n"
            metadata_content += f"genre=Audiobook\n"

        # Add chapter markers
        for i, ch in enumerate(chapters_metadata):
            start_ms = ch.get("start_ms", 0)
            end_ms = ch.get("end_ms", 0)
            title = ch.get("title", f"Chapter {i+1}")

            metadata_content += "\n"
            metadata_content += "[CHAPTER]\n"
            metadata_content += "TIMEBASE=1/1000\n"
            metadata_content += f"START={start_ms}\n"
            metadata_content += f"END={end_ms}\n"
            metadata_content += f"title={title}\n"

        # Write metadata to temp file
        meta_fd, meta_path = tempfile.mkstemp(suffix=".txt", prefix="aiguibook_meta_")
        try:
            with os.fdopen(meta_fd, "w", encoding="utf-8") as f:
                f.write(metadata_content)

            cmd = [
                "-i", input_path,
                "-i", meta_path,
                "-map_metadata", "1",
                "-codec:a", "aac",
                "-b:a", str(self.config.get("output", "bitrate", "128k")),
                "-ar", "44100",
                "-movflags", "+faststart",
                "-y",
                output_path,
            ]
            self._run_ffmpeg(cmd, "Add chapter metadata")
        finally:
            os.unlink(meta_path)

        return output_path

    def assemble_full_m4b(
        self,
        chapter_paths: Optional[Dict[int, str]] = None,
        chapter_titles: Optional[Dict[int, str]] = None,
    ) -> str:
        """Assemble all chapters into final M4B with chapter markers.

        Args:
            chapter_paths: Dict mapping chapter_idx to chapter audio path.
                          If None, scans the chapters directory.
            chapter_titles: Dict mapping chapter_idx to chapter title.

        Returns:
            Path to final M4B file
        """
        output_format = self.config.get("output", "format", "m4b")
        output_path = self.project.get_final_output_path(output_format)

        # Collect chapter audio files in order
        if chapter_paths is None:
            chapter_paths = {}
            ch_dir = self.project.chapters_dir
            if os.path.exists(ch_dir):
                for d in sorted(os.listdir(ch_dir), key=lambda x: int(x) if x.isdigit() else -1):
                    ch_dir_full = os.path.join(ch_dir, d)
                    if os.path.isdir(ch_dir_full):
                        for f in os.listdir(ch_dir_full):
                            if f.endswith((".wav", ".mp3", ".flac")):
                                chapter_paths[int(d)] = os.path.join(ch_dir_full, f)
                                break

        audio_files = []
        chapter_info = []
        cumulative_ms = 0

        for idx in sorted(chapter_paths.keys()):
            path = chapter_paths[idx]
            if not os.path.exists(path):
                logger.warning(f"Chapter audio not found: {path}")
                continue

            duration_ms = self._get_audio_duration(path)
            title = chapter_titles.get(idx, f"Chapter {idx+1}") if chapter_titles else f"Chapter {idx+1}"

            audio_files.append(path)
            chapter_info.append({
                "title": title,
                "start_ms": cumulative_ms,
                "end_ms": cumulative_ms + duration_ms,
            })
            cumulative_ms += duration_ms

        if not audio_files:
            raise ValueError("No chapter audio files to assemble")

        # Step 1: Concatenate all chapters
        concat_path = os.path.join(
            self.project.project_dir, "full_concatenated.wav"
        )
        self.concatenate_audio(
            audio_files,
            concat_path,
            crossfade=self.config.get("output", "crossfade_duration", 0.5),
        )

        # Step 2: Normalize full audio
        if self.config.get("output", "normalize_audio", True):
            self.normalize_audio(concat_path)

        # Step 3: Add chapter metadata and encode to M4B
        book_meta = self.project.book_metadata if hasattr(self.project, "book_metadata") else {}
        self.add_chapter_metadata(
            concat_path,
            chapter_info,
            output_path,
            book_metadata=book_meta,
        )

        # Clean up temp concat file
        if os.path.exists(concat_path):
            try:
                os.remove(concat_path)
            except OSError:
                pass

        logger.info(f"Full audiobook assembled: {output_path}")
        return output_path

    def __repr__(self) -> str:
        return f"AudioAssembly(project='{self.project.book_title}')"
