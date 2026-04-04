"""Core module exports."""

from audiobook_ai.core.epub_parser import EPUBParser, Chapter, TOCEntry
from audiobook_ai.core.text_segmenter import TextSegmenter, TextSegment
from audiobook_ai.core.project import BookProject
from audiobook_ai.core.config import AudiobookConfig

__all__ = [
    "EPUBParser",
    "Chapter",
    "TOCEntry",
    "TextSegmenter",
    "TextSegment",
    "BookProject",
    "AudiobookConfig",
]
