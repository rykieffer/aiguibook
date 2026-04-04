"""Text segmenter - splits chapter text into TTS-friendly segments."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentence boundary pattern
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])(?=\s)")
# Dialogue pattern: text surrounded by quotes
_QUOTE_PATTERNS = [
    re.compile(r'["\u201c][^"\u201c\u201d]*["\u201d]'),  # Double quotes
    re.compile(r"[\u00ab][^\u00bb]*[\u00bb]"),  # French guillemets
]


@dataclass
class TextSegment:
    """A single text segment suitable for TTS."""
    id: str  # Unique identifier: "ch{chapter_idx}_s{segment_idx}"
    text: str
    word_count: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "word_count": self.word_count,
        }


class TextSegmenter:
    """Splits chapter text into segments suitable for TTS."""

    def __init__(self, max_words: int = 150, min_words: int = 20):
        """
        Args:
            max_words: Maximum words per segment (default 150)
            min_words: Minimum words per segment before merging with adjacent (default 20)
        """
        self.max_words = max_words
        self.min_words = min_words
        self._quote_pattern_combined = re.compile(
            r'["\u00ab\u201c].*?["\u00bb\u201d]'
        )

    def _count_words(self, text: str) -> int:
        """Count words in text, treating hyphenated words as single."""
        return len(text.split())

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences at sentence boundaries.

        Never splits in the middle of a sentence.
        Respects dialogue paragraphs and quoted text.
        """
        if not text.strip():
            return []

        # Split on paragraph breaks first
        paragraphs = re.split(r"\n\s*\n", text.strip())

        sentences = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if it's a dialogue paragraph (mostly quotes)
            is_dialogue = self._is_dialogue_paragraph(paragraph)

            # Split on sentence boundaries
            parts = _SENTENCE_BOUNDARY.split(paragraph)

            if is_dialogue and self._count_words(paragraph) <= self.max_words + 20:
                # Keep dialogue together if not too long
                sentences.append(paragraph)
            else:
                for part in parts:
                    part = part.strip()
                    if part:
                        sentences.append(part)

        return sentences

    def _is_dialogue_paragraph(self, text: str) -> bool:
        """Check if a paragraph is primarily dialogue (quoted text).

        Args:
            text: The paragraph text

        Returns:
            True if mostly dialogue
        """
        if not text:
            return False
        quotes = self._quote_pattern_combined.findall(text)
        quoted_chars = sum(len(q) for q in quotes)
        return quoted_chars > len(text) * 0.5

    def segment_chapter(
        self,
        chapter_text: str,
        chapter_title: str,
        chapter_idx: int,
    ) -> List[TextSegment]:
        """Split a chapter into TTS-friendly segments.

        Splits on sentence boundaries, never in the middle of a sentence.
        Handles dialogue paragraphs by keeping them together when possible.

        Args:
            chapter_text: The full chapter text
            chapter_title: Chapter title
            chapter_idx: Chapter index (0-based)

        Returns:
            List of TextSegment objects
        """
        sentences = self._split_sentences(chapter_text)

        if not sentences:
            return []

        segments = []
        current_text_parts = []
        current_word_count = 0
        segment_idx = 0

        for sentence in sentences:
            sentence_word_count = self._count_words(sentence)

            # If adding this sentence would exceed max_words, flush current buffer
            if current_word_count + sentence_word_count > self.max_words and current_text_parts:
                # Create segment from current buffer
                segment_text = " ".join(current_text_parts)
                segment = TextSegment(
                    id=f"ch{chapter_idx}_s{segment_idx:03d}",
                    text=segment_text,
                    word_count=len(segment_text.split()),
                )
                segments.append(segment)
                segment_idx += 1
                current_text_parts = []
                current_word_count = 0

            # If this single sentence is huge, we still need to add it
            # (we never split within a sentence)
            current_text_parts.append(sentence)
            current_word_count += sentence_word_count

        # Flush remaining text
        if current_text_parts:
            segment_text = " ".join(current_text_parts)
            # Don't create segments with very small word counts unless it's the only one
            if current_word_count >= self.min_words or segment_idx == 0:
                segment = TextSegment(
                    id=f"ch{chapter_idx}_s{segment_idx:03d}",
                    text=segment_text,
                    word_count=len(segment_text.split()),
                )
                segments.append(segment)
            elif segments:
                # Merge with the last segment
                segments[-1].text += " " + segment_text
                segments[-1].word_count = len(segments[-1].text.split())
            else:
                # Still create it even if small
                segment = TextSegment(
                    id=f"ch{chapter_idx}_s{segment_idx:03d}",
                    text=segment_text,
                    word_count=len(segment_text.split()),
                )
                segments.append(segment)

        logger.debug(
            f"Chapter {chapter_idx} ({chapter_title}): split into {len(segments)} segments"
        )
        return segments

    def segment_full_book(
        self,
        chapters_list: list,
    ) -> Dict[int, List[TextSegment]]:
        """Segment all chapters in a book.

        Args:
            chapters_list: List of Chapter objects from EPUBParser

        Returns:
            Dictionary mapping chapter_idx to list of TextSegment
        """
        result = {}
        for chapter in chapters_list:
            segments = self.segment_chapter(
                chapter.text,
                chapter.title,
                chapter.spine_order,
            )
            if segments:
                result[chapter.spine_order] = segments
            chapter.segments = segments

        total_segments = sum(len(v) for v in result.values())
        logger.info(f"Segmented {len(result)} chapters into {total_segments} total segments")
        return result
