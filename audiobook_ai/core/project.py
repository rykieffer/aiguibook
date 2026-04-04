"""BookProject - Manages project directory structure, state, and file paths."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Segment status values
STATUS_PENDING = "pending"
STATUS_GENERATING = "generating"
STATUS_GENERATED = "generated"
STATUS_VALIDATED = "validated"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"

VALID_STATUSES = {
    STATUS_PENDING,
    STATUS_GENERATING,
    STATUS_GENERATED,
    STATUS_VALIDATED,
    STATUS_FAILED,
    STATUS_ERROR,
}


class BookProject:
    """Manages a project directory with all audio files, state, and metadata."""

    def __init__(self, book_title: str, work_dir: str, output_dir: str):
        """
        Args:
            book_title: Title of the book
            work_dir: Base working directory for intermediate files
            output_dir: Directory for final output files
        """
        self.book_title = book_title
        self.work_dir = os.path.abspath(work_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.project_dir = os.path.join(self.work_dir, self._sanitize_title(book_title))
        self.audio_dir = os.path.join(self.project_dir, "audio")
        self.chapters_dir = os.path.join(self.project_dir, "audio", "chapters")
        self.segments_dir = os.path.join(self.project_dir, "audio", "segments")
        self.metadata_file = os.path.join(self.project_dir, "project_state.json")

        # Track segment statuses: {segment_id: status}
        self.segment_status_map: Dict[str, str] = {}
        # Track additional segment metadata: {segment_id: {chapter_idx, duration, validation_result, ...}}
        self.segment_metadata: Dict[str, Dict[str, Any]] = {}

        # Book metadata
        self.book_metadata: Dict[str, str] = {}
        self.total_chapters = 0
        self.total_segments = 0

    @staticmethod
    def _sanitize_title(title: str) -> str:
        """Create a filesystem-safe directory name from title."""
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        safe = safe.strip("._- ")
        return safe or "untitled_book"

    def create(self):
        """Create project directory structure."""
        dirs = [
            self.project_dir,
            self.audio_dir,
            self.chapters_dir,
            self.segments_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        logger.info(f"Project directory created: {self.project_dir}")

    def save_state(self, state_dict: Optional[Dict[str, Any]] = None):
        """Save current project state to JSON file.

        Args:
            state_dict: If provided, save this dict directly. Otherwise save current state.
        """
        if state_dict is None:
            state_dict = self._build_state_dict()

        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
        logger.debug(f"Project state saved: {self.metadata_file}")

    def load_state(self) -> Dict[str, Any]:
        """Load project state from JSON file.

        Returns:
            State dictionary, or empty dict if no state file found.
        """
        if not os.path.exists(self.metadata_file):
            logger.info(f"No state file found: {self.metadata_file}")
            return {}

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load state: {e}")
            return {}

        # Restore segment statuses
        if "segment_status_map" in state:
            self.segment_status_map = state["segment_status_map"]
        if "segment_metadata" in state:
            self.segment_metadata = state["segment_metadata"]
        if "book_metadata" in state:
            self.book_metadata = state["book_metadata"]
        if "total_chapters" in state:
            self.total_chapters = state["total_chapters"]
        if "total_segments" in state:
            self.total_segments = state["total_segments"]

        logger.info(f"Project state loaded: {len(self.segment_status_map)} segments tracked")
        return state

    def _build_state_dict(self) -> Dict[str, Any]:
        """Build complete state dictionary."""
        return {
            "book_title": self.book_title,
            "work_dir": self.work_dir,
            "output_dir": self.output_dir,
            "project_dir": self.project_dir,
            "book_metadata": self.book_metadata,
            "total_chapters": self.total_chapters,
            "total_segments": self.total_segments,
            "segment_status_map": self.segment_status_map,
            "segment_metadata": self.segment_metadata,
        }

    def set_segment_status(self, segment_id: str, status: str, metadata: Optional[Dict] = None):
        """Set status for a segment.

        Args:
            segment_id: Unique segment identifier
            status: One of VALID_STATUSES
            metadata: Optional additional metadata for this segment
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {VALID_STATUSES}")

        self.segment_status_map[segment_id] = status

        if metadata:
            if segment_id not in self.segment_metadata:
                self.segment_metadata[segment_id] = {}
            self.segment_metadata[segment_id].update(metadata)

    def reset_segment_status(self, segment_id: str = None, status: str = STATUS_PENDING):
        """Reset segment status to pending.

        Args:
            segment_id: If None, reset all segments
            status: Status to reset to (default: pending)
        """
        if segment_id is None:
            for sid in self.segment_status_map:
                self.segment_status_map[sid] = status
        else:
            self.segment_status_map[segment_id] = status

    def get_segment_audio_path(self, chapter_idx: int, segment_id: str, speaker_id: str = "default") -> str:
        """Get file path for a segment's audio file.

        Args:
            chapter_idx: Chapter index
            segment_id: Segment identifier
            speaker_id: Speaker/voice identifier

        Returns:
            Absolute path to the segment audio file
        """
        filename = f"{segment_id}_{speaker_id}.wav"
        return os.path.join(self.segments_dir, str(chapter_idx), filename)

    def get_chapter_audio_path(self, chapter_idx: int) -> str:
        """Get file path for a chapter's assembled audio.

        Args:
            chapter_idx: Chapter index

        Returns:
            Absolute path to the chapter audio file
        """
        os.makedirs(os.path.join(self.chapters_dir, str(chapter_idx)), exist_ok=True)
        return os.path.join(self.chapters_dir, str(chapter_idx), f"chapter_{chapter_idx}.wav")

    def get_final_output_path(self, format: str = "m4b") -> str:
        """Get file path for the final output audiobook.

        Args:
            format: Output format (m4b, m4a, flac)

        Returns:
            Absolute path to the final output file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        safe_title = self._sanitize_title(self.book_title)
        return os.path.join(self.output_dir, f"{safe_title}.{format}")

    def get_pending_segments(self) -> List[str]:
        """List segment IDs that still need generation.

        Returns:
            List of segment IDs with pending or error status
        """
        return [
            sid
            for sid, status in self.segment_status_map.items()
            if status in (STATUS_PENDING, STATUS_ERROR)
        ]

    def get_failed_segments(self) -> List[str]:
        """List segment IDs that failed validation.

        Returns:
            List of segment IDs that need regeneration
        """
        return [
            sid
            for sid, status in self.segment_status_map.items()
            if status == STATUS_FAILED
        ]

    def get_progress(self) -> Tuple[int, int, float]:
        """Get generation progress.

        Returns:
            Tuple of (completed, total, percentage)
        """
        total = len(self.segment_status_map)
        if total == 0:
            return 0, 0, 0.0

        done = sum(
            1
            for s in self.segment_status_map.values()
            if s in (STATUS_GENERATED, STATUS_VALIDATED)
        )
        percent = (done / total) * 100.0 if total > 0 else 0.0
        return done, total, round(percent, 1)

    def get_validation_progress(self) -> Tuple[int, int, float]:
        """Get validation progress.

        Returns:
            Tuple of (validated, total, percentage)
        """
        total = len(self.segment_status_map)
        if total == 0:
            return 0, 0, 0.0

        validated = sum(
            1
            for s in self.segment_status_map.values()
            if s == STATUS_VALIDATED
        )
        percent = (validated / total) * 100.0 if total > 0 else 0.0
        return validated, total, round(percent, 1)

    def set_chapter_segments(self, chapter_idx: int, segment_ids: List[str]):
        """Register all segments for a chapter.

        Args:
            chapter_idx: Chapter index
            segment_ids: List of segment IDs for this chapter
        """
        for sid in segment_ids:
            if sid not in self.segment_status_map:
                self.segment_status_map[sid] = STATUS_PENDING
            if sid in self.segment_metadata:
                self.segment_metadata[sid]["chapter_idx"] = chapter_idx
            else:
                self.segment_metadata[sid] = {"chapter_idx": chapter_idx}
        self.total_segments = len(self.segment_status_map)

    def count_segments_by_status(self) -> Dict[str, int]:
        """Count segments grouped by status.

        Returns:
            Dict mapping status to count
        """
        counts = {s: 0 for s in VALID_STATUSES}
        for status in self.segment_status_map.values():
            if status in counts:
                counts[status] += 1
        return counts

    def cleanup(self, remove_audio=False):
        """Remove project directory.

        Args:
            remove_audio: If False, only remove state files; leave audio intact
        """
        if remove_audio and os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir, ignore_errors=True)
            logger.info(f"Project directory removed: {self.project_dir}")
        elif os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
            logger.info(f"State file removed: {self.metadata_file}")

    def __repr__(self) -> str:
        done, total, pct = self.get_progress()
        return (
            f"BookProject(title='{self.book_title}', "
            f"progress={done}/{total} ({pct}%))"
        )
