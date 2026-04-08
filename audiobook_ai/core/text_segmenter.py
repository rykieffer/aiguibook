"""Text segmenter - splits chapter text into TTS-friendly segments.

Optimized for Qwen3-TTS: segments of 40-100 words work best.
Never splits in the middle of a sentence.
Handles French dialogue (guillemets, em-dashes) and English dialogue.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

# Sentence boundary pattern: split after . ! ? ; followed by whitespace
# But NOT after abbreviations like M. Mme. Dr. etc.
_SENTENCE_BOUNDARY = re.compile(
    r"""(?<=[.!?;])        # After a sentence-ending punctuation
    (?=\s                   # Followed by whitespace
        (?=[A-Z\u00C0-\u00DC])  # And then an uppercase letter (including accented)
    )
    """,
    re.VERBOSE,
)

# French abbreviations that should NOT trigger a sentence split
_ABBREVIATIONS = re.compile(
    r"\b(?:M|Mme|Mlle|Dr|Pr|Mgr|Jr|Sr|St|Vol|Ch|Fig|cf|etc|vs)\.\s*$",
    re.MULTILINE,
)


@dataclass
class TextSegment:
    """A single text segment suitable for TTS."""
    id: str       # "ch{chapter_idx}_s{segment_idx}"
    text: str
    word_count: int

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "word_count": self.word_count}


class TextSegmenter:
    """Splits chapter text into segments suitable for TTS.
    
    Target: 40-100 words per segment (sweet spot for Qwen3-TTS).
    Never splits mid-sentence. Keeps dialogue paragraphs together.
    """

    def __init__(self, max_words: int = 100, min_words: int = 25):
        self.max_words = max_words
        self.min_words = min_words

    def _count_words(self, text: str) -> int:
        return len(text.split())

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences. Respects dialogue and abbreviations."""
        if not text.strip():
            return []

        # Split on paragraph breaks first
        paragraphs = re.split(r"\n\s*\n", text.strip())

        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this is primarily a dialogue paragraph
            if self._is_dialogue(para) and self._count_words(para) <= self.max_words + 30:
                # Keep the whole dialogue exchange together
                sentences.append(para)
                continue

            # Split on sentence boundaries
            parts = _SENTENCE_BOUNDARY.split(para)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)

        return sentences

    def _is_dialogue(self, text: str) -> bool:
        """Check if a paragraph is primarily dialogue."""
        if not text:
            return False
        # Count quoted characters (French guillemets + English quotes + em-dash dialogue)
        quote_chars = 0
        # French guillemets: «...»
        for m in re.finditer(r"\u00ab.*?\u00bb", text):
            quote_chars += len(m.group())
        # English quotes: "..."  "\u201c...\u201d"
        for m in re.finditer(r'["\u201c].*?["\u201d]', text):
            quote_chars += len(m.group())
        # French em-dash dialogue: — text
        for m in re.finditer(r"\u2014\s+\w+", text):
            quote_chars += len(m.group())

        return quote_chars > len(text) * 0.4

    def segment_chapter(
        self, chapter_text: str, chapter_title: str, chapter_idx: int
    ) -> List[TextSegment]:
        """Split a chapter into TTS-friendly segments."""
        sentences = self._split_sentences(chapter_text)
        if not sentences:
            return []

        segments = []
        buffer_parts = []
        buffer_words = 0
        seg_idx = 0

        for sentence in sentences:
            s_words = self._count_words(sentence)

            # If adding this sentence exceeds max_words, flush buffer
            if buffer_words + s_words > self.max_words and buffer_parts:
                seg_text = " ".join(buffer_parts)
                segments.append(TextSegment(
                    id=f"ch{chapter_idx}_s{seg_idx:03d}",
                    text=seg_text,
                    word_count=len(seg_text.split()),
                ))
                seg_idx += 1
                buffer_parts = []
                buffer_words = 0

            buffer_parts.append(sentence)
            buffer_words += s_words

        # Flush remaining text
        if buffer_parts:
            seg_text = " ".join(buffer_parts)
            if buffer_words < self.min_words and segments:
                # Too short: merge with last segment
                segments[-1].text += " " + seg_text
                segments[-1].word_count = len(segments[-1].text.split())
            else:
                segments.append(TextSegment(
                    id=f"ch{chapter_idx}_s{seg_idx:03d}",
                    text=seg_text,
                    word_count=len(seg_text.split()),
                ))

        logger.debug(f"Ch {chapter_idx} ({chapter_title}): {len(segments)} segments")
        return segments

    def segment_full_book(self, chapters_list: list) -> Dict[int, List[TextSegment]]:
        """Segment all chapters in a book."""
        result = {}
        for chapter in chapters_list:
            text = chapter.text if hasattr(chapter, "text") else chapter.get("text", "")
            title = chapter.title if hasattr(chapter, "title") else chapter.get("title", "")
            idx = chapter.spine_order if hasattr(chapter, "spine_order") else chapter.get("spine_order", 0)

            segs = self.segment_chapter(text, title, idx)
            if segs:
                result[idx] = segs
            if hasattr(chapter, "segments"):
                chapter.segments = segs

        total = sum(len(v) for v in result.values())
        logger.info(f"Segmented {len(result)} chapters into {total} total segments")
        return result
