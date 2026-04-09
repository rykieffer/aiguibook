"""Text segmenter - splits chapter text into TTS-friendly segments.

Key design: NEVER mix dialogue and narration in the same segment.
When a dialogue/narration boundary is detected, force a segment break
even if the current segment is short. Short segments are fine for dialogue.

Handles French dialogue (guillemets «», em-dashes —) and English dialogue.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Sentence boundary: split after . ! ? ; followed by space + uppercase
_SENTENCE_BOUNDARY = re.compile(
    r"""(?<=[.!?;])        # After sentence-ending punctuation
    (?=\s                   # Followed by whitespace
        (?=[A-Z\u00C0-\uDC00])  # And an uppercase letter (including accented)
    )
    """,
    re.VERBOSE,
)

# Dialogue detection patterns
# French guillemets: « ... »
_RE_GUILLEMET = re.compile(r"\u00ab\s*(.*?)\s*\u00bb")
# English double quotes: " ... "
_RE_EN_QUOTE = re.compile(r'["\u201c](.*?)["\u201d]')
# French em-dash dialogue line: — text (at start of paragraph or after \n)
_RE_EMDASH = re.compile(r"(?:^|\n)\s*\u2014\s+")


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
    
    CRITICAL RULE: Never mix dialogue and narration in the same segment.
    When a dialogue/narration boundary is detected, force a hard break.
    Short segments are perfectly fine for dialogue - the TTS engine handles them.
    """

    def __init__(self, max_words: int = 100, min_words: int = 10):
        self.max_words = max_words
        self.min_words = min_words  # Lowered - short dialogue segments are OK

    def _count_words(self, text: str) -> int:
        return len(text.split())

    def _classify_sentence(self, sentence: str) -> str:
        """Classify a sentence as 'dialogue', 'narration', or 'mixed'.
        
        A sentence is 'dialogue' if it contains quoted speech.
        A sentence is 'mixed' if it has both quoted speech and narration.
        A sentence is 'narration' if it has no quoted speech.
        """
        if not sentence.strip():
            return "narration"

        # Check for French guillemets
        has_guillemet = bool(_RE_GUILLEMET.search(sentence))
        # Check for English quotes
        has_en_quote = bool(_RE_EN_QUOTE.search(sentence))
        # Check for em-dash dialogue (at start of line)
        has_emdash = bool(_RE_EMDASH.search(sentence))

        has_quotes = has_guillemet or has_en_quote or has_emdash

        if not has_quotes:
            return "narration"

        # Measure how much of the sentence is inside quotes
        quoted_chars = 0
        for m in _RE_GUILLEMET.finditer(sentence):
            quoted_chars += len(m.group(1))
        for m in _RE_EN_QUOTE.finditer(sentence):
            quoted_chars += len(m.group(1))
        for m in _RE_EMDASH.finditer(sentence):
            # Em-dash dialogue: count from dash to end of that line
            start = m.start()
            end = sentence.find('\n', start)
            if end == -1:
                end = len(sentence)
            quoted_chars += len(sentence[start:end])

        total_chars = len(sentence.strip())
        if total_chars == 0:
            return "narration"

        quote_ratio = quoted_chars / total_chars

        if quote_ratio > 0.5:
            return "dialogue"
        elif quote_ratio > 0.15:
            return "mixed"
        else:
            return "narration"

    def _split_mixed_sentence(self, sentence: str) -> List[Tuple[str, str]]:
        """Split a mixed sentence into (text, type) fragments.
        
        E.g. "Il la regarda. « Bonjour, » dit-il." becomes:
          [ ("Il la regarda.", "narration"), ("« Bonjour, » dit-il.", "dialogue") ]
        
        Returns list of (fragment_text, "dialogue"|"narration") tuples.
        """
        fragments = []

        # Strategy: find dialogue spans and split around them
        # We need to identify contiguous dialogue regions and narration regions
        
        # Find all dialogue spans (guillemets, quotes, em-dash lines)
        dialogue_spans = []
        
        # French guillemets
        for m in _RE_GUILLEMET.finditer(sentence):
            # Include a bit of context around the quotes (dialogue tags like "dit-il")
            start = m.start()
            end = m.end()
            # Look back for dialogue tag (e.g. "dit-il, " before the quote)
            tag_back = sentence.rfind(", ", 0, start)
            if tag_back == -1:
                tag_back = sentence.rfind(". ", 0, start)
            if tag_back != -1 and start - tag_back < 30:
                start = tag_back + 2
            # Look forward for dialogue tag after the quote
            tag_forward = sentence.find(", ", end)
            tag_end = sentence.find(". ", end)
            # Take the closer one if it's a dialogue tag
            for candidate in [tag_forward, tag_end]:
                if candidate != -1 and candidate - end < 25:
                    # Check if it looks like a dialogue tag
                    snippet = sentence[end:candidate].strip()
                    if any(w in snippet.lower() for w in ["dit", "répond", "murmur", "chuchot", "cri", "demand", "ajout", "continu", "s'exclam", "lanc", "soupir", "grogn", "hurl", "whisper", "said", "repli", "ask", "answer"]):
                        end = candidate + 2
                        break
            dialogue_spans.append((start, end))

        # English quotes
        for m in _RE_EN_QUOTE.finditer(sentence):
            start = m.start()
            end = m.end()
            # Look for dialogue tag
            tag_back = sentence.rfind(", ", 0, start)
            if tag_back != -1 and start - tag_back < 30:
                start = tag_back + 2
            dialogue_spans.append((start, end))

        # Em-dash dialogue lines
        for m in _RE_EMDASH.finditer(sentence):
            start = m.start()
            end = sentence.find('\n', start)
            if end == -1:
                end = len(sentence)
            dialogue_spans.append((start, end))

        if not dialogue_spans:
            # No dialogue spans found, return as-is
            cls = self._classify_sentence(sentence)
            return [(sentence, cls)]

        # Merge overlapping spans
        dialogue_spans.sort()
        merged = [dialogue_spans[0]]
        for start, end in dialogue_spans[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Build fragments from gaps and spans
        pos = 0
        for d_start, d_end in merged:
            # Narration before this dialogue
            if pos < d_start:
                narr = sentence[pos:d_start].strip()
                if narr:
                    fragments.append((narr, "narration"))
            # The dialogue itself
            dial = sentence[d_start:d_end].strip()
            if dial:
                fragments.append((dial, "dialogue"))
            pos = d_end

        # Narration after last dialogue
        if pos < len(sentence):
            narr = sentence[pos:].strip()
            if narr:
                fragments.append((narr, "narration"))

        # If we ended up with no fragments, return original
        if not fragments:
            return [(sentence, "mixed")]

        return fragments

    def _split_sentences(self, text: str) -> List[Tuple[str, str]]:
        """Split text into (sentence, type) pairs.
        
        Each sentence is classified as 'dialogue', 'narration', or 'mixed'.
        Mixed sentences are further split into dialogue/narration fragments.
        """
        if not text.strip():
            return []

        # Split on paragraph breaks first
        paragraphs = re.split(r"\n\s*\n", text.strip())

        results = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Handle em-dash dialogue lines: each — line is its own segment
            emdash_lines = _RE_EMDASH.split(para)
            if len(emdash_lines) > 1:
                # Re-split: each em-dash line is a separate unit
                lines = para.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if _RE_EMDASH.search(line):
                        results.append((line, "dialogue"))
                    else:
                        # Could be narration or contain quotes
                        cls = self._classify_sentence(line)
                        if cls == "mixed":
                            results.extend(self._split_mixed_sentence(line))
                        else:
                            results.append((line, cls))
                continue

            # Split on sentence boundaries
            parts = _SENTENCE_BOUNDARY.split(para)
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                cls = self._classify_sentence(part)

                if cls == "mixed":
                    # Split mixed sentences into dialogue/narration fragments
                    results.extend(self._split_mixed_sentence(part))
                else:
                    results.append((part, cls))

        return results

    def segment_chapter(
        self, chapter_text: str, chapter_title: str, chapter_idx: int
    ) -> List[TextSegment]:
        """Split a chapter into TTS-friendly segments.
        
        CRITICAL: Never mix dialogue and narration in the same segment.
        When the type changes (narration->dialogue or dialogue->narration),
        force a segment break even if the current segment is short.
        """
        typed_sentences = self._split_sentences(chapter_text)
        if not typed_sentences:
            return []

        segments = []
        buffer_parts = []
        buffer_words = 0
        buffer_type = None  # "dialogue" or "narration"
        seg_idx = 0

        def flush_buffer():
            nonlocal buffer_parts, buffer_words, buffer_type, seg_idx
            if not buffer_parts:
                return
            seg_text = " ".join(buffer_parts)
            # No min_words check for dialogue - short dialogue is fine
            # For narration, only merge with previous if very short
            if buffer_words < self.min_words and segments and buffer_type == "narration":
                segments[-1].text += " " + seg_text
                segments[-1].word_count = len(segments[-1].text.split())
            else:
                segments.append(TextSegment(
                    id=f"ch{chapter_idx}_s{seg_idx:03d}",
                    text=seg_text,
                    word_count=len(seg_text.split()),
                ))
                seg_idx += 1
            buffer_parts = []
            buffer_words = 0
            buffer_type = None

        for sentence, sent_type in typed_sentences:
            s_words = self._count_words(sentence)

            # TYPE CHANGE: force a break when switching between narration and dialogue
            if buffer_type is not None and sent_type != buffer_type:
                flush_buffer()

            # SIZE: if adding this sentence exceeds max_words, flush first
            if buffer_words + s_words > self.max_words and buffer_parts:
                flush_buffer()

            buffer_parts.append(sentence)
            buffer_words += s_words
            buffer_type = sent_type

        # Flush remaining
        flush_buffer()

        logger.debug(f"Ch {chapter_idx} ({chapter_title}): {len(segments)} segments")
        return segments

    def segment_full_book(self, chapters_list: list) -> Dict[int, List[TextSegment]]:
        """Segment all chapters in a book."""
        result = {}
        for ch_idx, chapter in enumerate(chapters_list):
            text = chapter.text if hasattr(chapter, "text") else chapter.get("text", "")
            title = chapter.title if hasattr(chapter, "title") else chapter.get("title", "")

            segs = self.segment_chapter(text, title, ch_idx)
            if segs:
                result[ch_idx] = segs
            if hasattr(chapter, "segments"):
                chapter.segments = segs

        total = sum(len(v) for v in result.values())
        logger.info(f"Segmented {len(result)} chapters into {total} total segments")
        return result
