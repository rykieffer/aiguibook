"""EPUB Parser - Versatile extraction of chapters, metadata, and TOC from any EPUB file.

Handles EPUB 2.0 and 3.0, various HTML/XHTML structures, nested content,
and edge cases found in real-world EPUB files (old ebooks, converted texts, etc.).
"""

from __future__ import annotations

import logging
import os
import re
import html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import ebooklib
from ebooklib import epub

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """Represents a single chapter/section in an EPUB."""
    title: str
    spine_order: int
    content_html: str
    text: str
    href: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "spine_order": self.spine_order,
            "text": self.text,
            "href": self.href,
            "word_count": len(self.text.split()),
        }


@dataclass
class TOCEntry:
    """A table of contents entry."""
    title: str
    href: str
    children: List[TOCEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "href": self.href,
            "children": [c.to_dict() for c in self.children],
        }


class EPUBParser:
    """Versatile EPUB parser that handles diverse EPUB file structures.

    Strategy: Walk the spine (reading order) and extract text from each
    document item. This works better than relying on TOC structure alone,
    which varies wildly between publishers.
    """

    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self._book = None
        self._metadata: Dict[str, str] = {}
        self._chapters: List[Chapter] = []
        self._toc: List[TOCEntry] = []

        # Regexes for HTML-to-text conversion
        self._skip_tag_re = re.compile(r"<style[^>]*>.*?</style>|<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
        self._html_tag_re = re.compile(r"<[^>]+>")
        self._entity_re = re.compile(r"&(#?\w+);")
        self._multi_blank_re = re.compile(r"\n{3,}")
        self._leading_trail_ws = re.compile(r"^[\s]+|[\s]+$", re.MULTILINE)

    def parse(self) -> Dict[str, Any]:
        """Parse the EPUB file and return structured data."""
        logger.info(f"Loading EPUB: {os.path.basename(self.epub_path)}")
        self._book = epub.read_epub(self.epub_path, options={"ignore_ncx": False})

        self._extract_metadata()
        self._extract_toc()
        self._extract_chapters()

        logger.info(f"Parsed {len(self._chapters)} chapters, language={self._metadata.get('language', '?')}")
        return {
            "metadata": dict(self._metadata),
            "toc": self._toc,
            "chapters": list(self._chapters),
        }

    def cleanup(self):
        """Release resources."""
        self._book = None
        self._chapters.clear()
        self._toc.clear()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_metadata(self):
        """Extract DC metadata from the EPUB."""
        if not self._book:
            return

        def dc(name: str, default: str = "") -> str:
            vals = self._book.get_metadata("DC", name)
            return vals[0][0].strip() if vals else default

        self._metadata = {
            "title": dc("title"),
            "author": dc("creator"),
            "language": dc("language", "unknown"),
            "publisher": dc("publisher"),
            "identifier": dc("identifier"),
            "description": dc("description", ""),
            "date": dc("date", ""),
            "rights": dc("rights", ""),
        }

        # Cover image: try multiple strategies
        # Strategy 1: look for cover in guide
        try:
            for ref in self._book.guide:
                if ref.get("type", "").lower() == "cover":
                    href = ref.get("href", "")
                    self._metadata["cover_image"] = href.split("#")[0]
                    break
        except Exception:
            pass

        # Strategy 2: look for items with "cover" in their name
        if "cover_image" not in self._metadata:
            for item in self._book.get_items():
                name = item.get_name().lower()
                if "cover" in name:
                    self._metadata["cover_image"] = item.get_name()
                    break

    # ------------------------------------------------------------------
    # Table of Contents
    # ------------------------------------------------------------------

    def _extract_toc(self):
        """Extract structured TOC from the book's toc attribute."""
        if not self._book:
            return
        try:
            self._toc = self._parse_toc_items(self._book.toc)
        except Exception as e:
            logger.warning(f"TOC extraction failed: {e}")
            self._toc = []

    def _parse_toc_items(self, items, depth: int = 0) -> List[TOCEntry]:
        """Recursively parse TOC items.

        ebooklib TOC items can be:
        - A Link object (from ebooklib.epub.Section or Link)
        - A tuple: (Section, [sub-items])
        - Just a Link object
        """
        entries: List[TOCEntry] = []

        try:
            for item in items:
                if isinstance(item, tuple):
                    # (section_or_link, sub_items)
                    section, sub_items = item[0], item[1]
                    title = getattr(section, "title", str(section)) if hasattr(section, "title") else str(section)
                    href = getattr(section, "href", "")
                    children = self._parse_toc_items(sub_items, depth + 1)
                    entries.append(TOCEntry(title=title, href=href, children=children))

                elif hasattr(item, "href"):
                    # Link or Section object
                    title = getattr(item, "title", "") or item.__class__.__name__
                    href = item.href or ""
                    entries.append(TOCEntry(title=title, href=href, children=[]))

                elif isinstance(item, str):
                    # Bare string reference (some old EPUBs)
                    entries.append(TOCEntry(title=item, href=item, children=[]))

                else:
                    logger.debug(f"Skipping unknown TOC item type: {type(item)}")

        except Exception as e:
            logger.warning(f"Error parsing TOC at depth {depth}: {e}")

        return entries

    # ------------------------------------------------------------------
    # Chapter / Content Extraction
    # ------------------------------------------------------------------

    def _extract_chapters(self):
        """Extract chapters by walking the spine (reading order).

        The spine defines the authoritative reading order.
        We only extract ITEM_DOCUMENT items (type 9 = xhtml/html content).
        """
        if not self._book:
            return

        spine = self._book.spine
        if not spine:
            # Fallback: get all document items
            spine_items = [item for item in self._book.get_items()
                           if item.get_type() == ebooklib.ITEM_DOCUMENT]
        else:
            # spine contains (item_idstring, linear) tuples
            spine_ids = [item_id for item_id, linear in spine if linear]
            spine_items = []
            for item_id in spine_ids:
                try:
                    item = self._book.get_item_with_id(item_id)
                    if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
                        spine_items.append(item)
                except Exception:
                    continue

        # Build a lookup from TOC to help name chapters
        toc_lookup = self._build_toc_lookup()

        chapter_num = 0
        for item in spine_items:
            try:
                content = item.get_content()
                if isinstance(content, bytes):
                    html_content = content.decode("utf-8", errors="replace")
                else:
                    html_content = content

                # Skip very short / empty documents (nav pages, etc.)
                if len(html_content) < 100:
                    continue

                # Extract title
                title = self._extract_title_from_html(html_content)

                # If still empty, try TOC lookup
                if not title:
                    href = item.get_name().lower().replace("\\", "/")
                    for toc_path, toc_title in toc_lookup.items():
                        if toc_path in href or href in toc_path:
                            title = toc_title
                            break

                if not title:
                    chapter_num += 1
                    title = f"Chapter {chapter_num}"

                text = self._html_to_text(html_content)

                # Skip chapters that have no meaningful text after cleaning
                if len(text.strip()) < 50:
                    continue

                logger.debug(f"  Chapter: {title} ({len(text)} chars)")

                self._chapters.append(Chapter(
                    title=title,
                    spine_order=len(self._chapters),
                    content_html=html_content,
                    text=text,
                    href=item.get_name(),
                ))

            except Exception as e:
                logger.warning(f"Failed to extract chapter {item.get_name()}: {e}")
                continue

    def _build_toc_lookup(self) -> Dict[str, str]:
        """Build a flat mapping from href -> title for TOC-based naming."""
        lookup: Dict[str, str] = {}

        def walk(entries):
            for entry in entries:
                if entry.href:
                    clean_href = entry.href.split("#")[0].split("?")[0]
                    lookup[clean_href] = entry.title
                    lookup[entry.href.split("/")[-1]] = entry.title
                if entry.children:
                    walk(entry.children)

        walk(self._toc)
        return lookup

    # ------------------------------------------------------------------
    # HTML to Text
    # ------------------------------------------------------------------

    def _extract_title_from_html(self, html_content: str) -> str:
        """Extract a title from an HTML document."""
        # Try <title> tag
        m = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()

        # Try <h1> through <h3>
        for level in [1, 2, 3]:
            m = re.search(rf"<h{level}[^>]*>(.*?)</h{level}>", html_content, re.IGNORECASE | re.DOTALL)
            if m:
                text = m.group(1).strip()
                text = self._html_tag_re.sub("", text).strip()
                text = html.unescape(text).strip()
                if text and len(text) < 300:
                    return text

        # Try epub:type="title" or epub:type="subtitle"
        m = re.search(r'<(?:h\d|p|div|span)[^>]*epub:type="[^"]*title[^"]*"[^>]*>(.*?)</(?:h\d|p|div|span)>', html_content, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
            text = self._html_tag_re.sub("", text).strip()
            html.unescape(text)
            if text and len(text) < 300:
                return text

        return ""

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML/XHTML content to readable plain text."""
        # Remove style/script blocks
        text = self._skip_tag_re.sub("", html_content)

        # Replace structural tags with newlines
        text = re.sub(r"<!--<br\s*/?>|<br/>", "\n", text)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)

        # Add double newlines after headings and block elements
        for tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            text = re.sub(rf"</{tag}>", "\n\n", text, flags=re.IGNORECASE)

        # Remove opening tags
        text = self._html_tag_re.sub("", text)

        # Decode HTML entities
        text = html.unescape(text)

        # Clean up whitespace
        text = self._multi_blank_re.sub("\n\n", text)
        text = text.strip()

        return text

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_full_text(self) -> str:
        """Return the complete book text with chapter separators."""
        return "\n\n---\n\n".join(
            f"# {ch.title}\n\n{ch.text}" for ch in self._chapters
        )

    def get_chapter_text(self, index: int) -> Optional[str]:
        """Return text for a specific chapter by index."""
        if 0 <= index < len(self._chapters):
            return self._chapters[index].text
        return None

    def __repr__(self) -> str:
        title = self._metadata.get("title", os.path.basename(self.epub_path))
        return f"EPUBParser(title='{title}', chapters={len(self._chapters)})"
