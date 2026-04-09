"""
AudiobookGUI v8 - Project Folder Architecture.
Everything lives under one project folder: analysis.json, voices/, segments/, output.m4a
"""

from __future__ import annotations

import gradio as gr
import logging
import os
import shutil
import json
import time
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AIGUIBook")

DEFAULT_NARRATOR_DESC = "A warm, deep male voice, French accent, authoritative yet gentle."

# Default project root
DEFAULT_PROJECTS_ROOT = os.path.join(os.path.expanduser("~"), "audiobooks")


class AudiobookGUI:
    def __init__(self, config):
        self.config = config
        self.app = None

        # Global State
        self._log_messages = []

        # Project Data
        self.project_dir = ""  # Set by user in Tab 1
        self._epub_parser = None
        self._chapters_list = []
        self._tags = {}
        self._characters = []
        self._dedup_map = {}

        # Voice Data
        self._voice_model_variant_design = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        self._voice_model_variant_base = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self._engine = None
        self.narrator_voice_desc = DEFAULT_NARRATOR_DESC
        self.narrator_wav_path = None
        self.narrator_ref_text = ""
        self.character_voice_paths = {}
        self.character_ref_texts = {}
        self.character_voice_descs = {}
        self.voice_strategy = "single_narrator"

        # Book metadata
        self._book_title = "Audiobook"
        self._book_author = ""

        self._log("AIGUIBook v8 initialized.")
        try:
            import torch
            torch.backends.cudnn.benchmark = True
        except:
            pass

        import gradio as gr
        self.theme = gr.themes.Soft()
        self.css = ""

    # ── Helpers ──────────────────────────────────────────────────

    def _log(self, msg):
        self._log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    def _get_logs(self):
        return "\n".join(self._log_messages[-100:])

    def _get_engine(self):
        if self._engine is None:
            from audiobook_ai.tts.qwen_engine import TTSEngine
            self._engine = TTSEngine()
        return self._engine

    def _ensure_project_dir(self, project_dir: str) -> str:
        """Create project folder structure and return the path."""
        if not project_dir:
            project_dir = os.path.join(DEFAULT_PROJECTS_ROOT, f"book_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "voices"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "segments"), exist_ok=True)
        self.project_dir = project_dir
        self._log(f"Project folder: {project_dir}")
        return project_dir

    @property
    def analysis_json_path(self):
        return os.path.join(self.project_dir, "analysis.json") if self.project_dir else ""

    @property
    def voices_dir(self):
        return os.path.join(self.project_dir, "voices") if self.project_dir else ""

    @property
    def segments_dir(self):
        return os.path.join(self.project_dir, "segments") if self.project_dir else ""

    def _auto_save_analysis(self, state):
        """Auto-save analysis JSON to project folder."""
        if not self.project_dir:
            return
        tags_dict = {}
        raw_tags = state.get("tags", {}) if state else {}
        for sid, tag in raw_tags.items():
            if hasattr(tag, 'emotion'):
                tags_dict[sid] = {
                    'speaker': getattr(tag, 'speaker_type', 'narrator'),
                    'char': getattr(tag, 'character_name', None),
                    'emotion': getattr(tag, 'emotion', 'neutral'),
                    'emotion_instruction': getattr(tag, 'emotion_instruction', ''),
                    'text': getattr(tag, 'text', ''),
                }
            elif isinstance(tag, dict):
                tags_dict[sid] = tag
        data = {
            "book_title": self._book_title,
            "book_author": self._book_author,
            "chars": state.get("chars", []) if state else [],
            "tags": tags_dict,
        }
        path = self.analysis_json_path
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._log(f"Analysis auto-saved to: {path}")

    # ── Build GUI ───────────────────────────────────────────────

    def build(self):
        theme = gr.themes.Soft(primary_hue="violet", secondary_hue="blue")
        css = ".log-box textarea {font-family: monospace; font-size: 12px;}"
        self.theme = theme
        self.css = css

        with gr.Blocks(title="AIGUIBook") as self.app:
            gr.Markdown("# AIGUIBook v8\n### EPUB -> Audiobook (Qwen3-TTS)")

            state = gr.State({"loaded": False, "parsed": False, "analyzed": False})

            with gr.Tabs():

                # ═══════════════════════════════════════
                # TAB 1: ANALYSIS
                # ═══════════════════════════════════════
                with gr.Tab("1. Analysis"):
                    gr.Markdown("### Setup Project & Analyze Book")

                    with gr.Row():
                        with gr.Column():
                            # PROJECT FOLDER - auto-creates on use
                            txt_project_dir = gr.Textbox(
                                label="Project Folder (auto-created on first use)",
                                placeholder=f"e.g. {DEFAULT_PROJECTS_ROOT}/my_book",
                                value="",
                                lines=1,
                            )

                            file_epub = gr.File(label="Upload EPUB (auto-parses on upload)", file_types=[".epub"])
                            book_info = gr.Textbox(label="Metadata", lines=4, interactive=False)

                        with gr.Column():
                            btn_analyze = gr.Button("Analyze Characters (LLM)", variant="primary")
                            status_bar = gr.Textbox(label="Status", lines=2)
                            char_list_df = gr.Dataframe(
                                label="Detected Characters",
                                headers=["Character", "Count", "Emotions"],
                                interactive=False,
                            )
                            # Load existing project
                            gr.Markdown("---\n**Load Existing Project**")
                            btn_load_project = gr.Button("Load Project from Folder", variant="secondary")
                            status_load = gr.Textbox(label="Load Status", interactive=False)

                    # Events - EPUB upload auto-parses and auto-sets project folder
                    file_epub.change(
                        fn=self.parse_epub,
                        inputs=[file_epub, txt_project_dir, state],
                        outputs=[book_info, char_list_df, state],
                    )
                    btn_analyze.click(
                        fn=self.run_analysis,
                        inputs=[file_epub, txt_project_dir, state],
                        outputs=[status_bar, char_list_df, state],
                    )
                    btn_load_project.click(
                        fn=self.load_project,
                        inputs=[txt_project_dir, state],
                        outputs=[status_load, book_info, char_list_df, state],
                    )

                # ═══════════════════════════════════════
                # TAB 2: VOICE DESIGN
                # ═══════════════════════════════════════
                with gr.Tab("2. Voice Design"):
                    gr.Markdown("### Voice Strategy & Design")
                    gr.Markdown("*Voices are saved automatically to your project's `voices/` folder.*")

                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("#### Production Mode")
                                voice_strategy_radio = gr.Radio(
                                    choices=[
                                        ("Single Narrator (Voice Acting)", "single_narrator"),
                                        ("Full Ensemble Cast (Multi-Voice)", "full_ensemble"),
                                    ],
                                    value="single_narrator",
                                    label="Mode",
                                )
                                gr.Markdown("* **Single Narrator**: One voice for all, with emotion acting.")
                                gr.Markdown("* **Full Ensemble**: Different voice per character.")

                                gr.Markdown("#### Narrator Voice")
                                txt_narrator_desc = gr.Textbox(
                                    label="Voice Description",
                                    value=DEFAULT_NARRATOR_DESC,
                                    lines=3,
                                )
                                file_narrator_ref = gr.File(
                                    label="OR Upload Reference WAV",
                                    file_types=[".wav"],
                                    type="filepath",
                                )
                                btn_design_narrator = gr.Button("Design Narrator Voice", variant="primary")
                                status_narrator = gr.Textbox(label="Status", interactive=False)
                                audio_narrator_preview = gr.Audio(label="Preview", interactive=False)

                        with gr.Column(scale=2):
                            with gr.Group():
                                gr.Markdown("#### Character Voices (Ensemble Mode)")
                                md_char_info = gr.Markdown("*Run Analysis in Tab 1 first.*")
                                df_char_voices = gr.Dataframe(
                                    headers=["Character", "Voice", "Status"],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    label="Characters",
                                )
                                txt_char_desc_global = gr.Textbox(
                                    label="Global Character Description",
                                    placeholder="e.g., A young energetic male, French accent",
                                    lines=2,
                                )
                                btn_design_all_chars = gr.Button("Design ALL Character Voices", variant="primary")
                                status_chars = gr.Textbox(label="Status", interactive=False)

                    btn_design_narrator.click(
                        fn=self.design_narrator,
                        inputs=[txt_narrator_desc, file_narrator_ref],
                        outputs=[status_narrator, audio_narrator_preview],
                    )
                    btn_design_all_chars.click(
                        fn=self.design_all_characters,
                        inputs=[txt_char_desc_global, state],
                        outputs=[status_chars, df_char_voices],
                    )
                    voice_strategy_radio.change(
                        fn=lambda v: setattr(self, 'voice_strategy', v) or v,
                        inputs=[voice_strategy_radio],
                        outputs=[],
                    )

                # ═══════════════════════════════════════
                # TAB 3: PRODUCTION
                # ═══════════════════════════════════════
                with gr.Tab("3. Production"):
                    gr.Markdown("### Generate Audiobook")
                    gr.Markdown(f"*Segment WAVs go to `{{project}}/segments/`, final M4A goes to `{{project}}/`*")

                    with gr.Row():
                        with gr.Column():
                            chk_preview = gr.Checkbox(label="Preview Mode (First Chapter)", value=True)
                            silence_slider = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.75, step=0.25,
                                label="Silence Between Segments (seconds)",
                            )
                            btn_start_prod = gr.Button("START GENERATION", variant="primary", size="lg")
                            btn_resume_prod = gr.Button("RESUME (skip existing WAVs)", variant="secondary", size="lg")

                        with gr.Column():
                            progress = gr.Slider(value=0, label="Progress")
                            phase = gr.Textbox(label="Current Phase", value="Ready")
                            logs = gr.Textbox(label="System Log", lines=10, elem_classes=["log-box"])
                            m4a_out_prod = gr.File(label="Final Audiobook (M4A)", interactive=False)

                    btn_start_prod.click(
                        fn=self.start_generation,
                        inputs=[chk_preview, silence_slider, state],
                        outputs=[progress, phase, logs, m4a_out_prod],
                    )
                    btn_resume_prod.click(
                        fn=self.resume_generation,
                        inputs=[silence_slider, state],
                        outputs=[progress, phase, logs, m4a_out_prod],
                    )

        return self.app

    # ── Tab 1 Handlers ──────────────────────────────────────────

    def parse_epub(self, file_epub, project_dir_text, state):
        if not state:
            state = {}

        if not file_epub:
            return "No file selected.", [], state

        # Ensure project dir
        self._ensure_project_dir(project_dir_text.strip() if project_dir_text else "")

        self._log(f"Parsing EPUB: {os.path.basename(file_epub)}")
        try:
            from audiobook_ai.core.epub_parser import EPUBParser
            parser = EPUBParser(file_epub)
            data = parser.parse()
            self._epub_parser = parser
            self._chapters_list = data.get("chapters", [])
            meta = data.get("metadata", {})

            self._book_title = meta.get("title", "Audiobook")
            self._book_author = meta.get("author", "")

            state["parsed"] = True
            state["epub_path"] = file_epub
            state["meta"] = meta
            if "analyzed" not in state:
                state["analyzed"] = False

            info = f"Title: {self._book_title}\nAuthor: {self._book_author}\nChapters: {len(self._chapters_list)}\nProject: {self.project_dir}"
            self._log(f"Parsed: {info}")
            return info, [], state
        except Exception as e:
            self._log(f"Parse Error: {e}")
            return f"Error: {e}", [], state

    def run_analysis(self, file_epub, project_dir_text, state):
        """Run the full character analysis pipeline with live progress."""
        table_data = []

        try:
            if not state:
                state = {"parsed": False, "analyzed": False}

            # Parse if needed
            if not state.get("parsed"):
                if file_epub:
                    info, _, state = self.parse_epub(file_epub, project_dir_text, state)
                    if not state.get("parsed"):
                        yield "Parse failed.", [], state
                        return
                else:
                    yield "Please upload a book first.", [], state
                    return

            # Ensure project dir
            self._ensure_project_dir(project_dir_text.strip() if project_dir_text else "")

            yield "Segmenting text...", [], state

            from audiobook_ai.core.text_segmenter import TextSegmenter
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer

            seg = TextSegmenter()
            all_segs = []
            chapters = self._chapters_list
            if not chapters and self._epub_parser:
                chapters = getattr(self._epub_parser, '_chapters', [])

            yield f"Found {len(chapters)} chapters. Segmenting...", [], state

            for ch in chapters:
                txt = ch.get("text", "") if isinstance(ch, dict) else getattr(ch, 'text', "")
                title = ch.get("title", "") if isinstance(ch, dict) else getattr(ch, 'title', "")
                idx = ch.get("spine_order", 0) if isinstance(ch, dict) else getattr(ch, 'spine_order', 0)
                if txt:
                    all_segs.extend(seg.segment_chapter(txt, title, idx))

            if not all_segs:
                yield "Error: No text found to analyze.", [], state
                return

            yield f"Found {len(all_segs)} segments. Analyzing with LLM...", [], state

            self.analyzer = CharacterAnalyzer(self.config.get_section("analysis"))
            tags, chars, _dedup_map = {}, [], {}

            for item in self.analyzer.analyze_segments_iter(all_segs):
                if item["status"] == "progress":
                    yield item["msg"], [], state
                elif item["status"] == "finished":
                    result = item["result"]
                    tags, chars, _dedup_map = result[0], result[1], result[2]

            self.tags = tags
            self._characters = chars
            self.dedup_map = _dedup_map
            state["analyzed"] = True
            state["tags"] = tags
            state["chars"] = chars
            state["dedup_map"] = _dedup_map

            # AUTO-SAVE to project folder
            self._auto_save_analysis(state)

            yield "Building results table...", [], state

            for c in chars:
                count = sum(1 for t in tags.values() if t.character_name == c)
                emo = list(set([t.emotion for t in tags.values() if t.character_name == c]))
                table_data.append([c, count, ", ".join(sorted(emo))])

            yield f"Analysis Complete! Saved to {self.analysis_json_path}", table_data, state

        except Exception as e:
            import traceback
            self._log(f"Analysis error: {e}\n{traceback.format_exc()}")
            yield f"Error: {e}", table_data, state

    def load_project(self, project_dir_text, state):
        """Load an existing project from its folder (analysis.json + voices)."""
        if not state:
            state = {}

        project_dir = project_dir_text.strip() if project_dir_text else ""
        if not project_dir or not os.path.isdir(project_dir):
            return f"Error: Folder not found: {project_dir}", "", [], state

        json_path = os.path.join(project_dir, "analysis.json")
        if not os.path.exists(json_path):
            return f"Error: No analysis.json found in {project_dir}", "", [], state

        try:
            self._log(f"Loading project from: {project_dir}")
            self.project_dir = project_dir

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._book_title = data.get("book_title", "Audiobook")
            self._book_author = data.get("book_author", "")
            chars = data.get("chars", [])

            # Fallback: extract from tags
            if not chars:
                tags = data.get("tags", {})
                unique_chars = set()
                for t_data in tags.values():
                    c = t_data.get("char") or t_data.get("character_name")
                    if c:
                        unique_chars.add(c)
                chars = sorted(list(unique_chars))

            self._characters = chars
            raw_tags = data.get("tags", {})
            self._tags = raw_tags

            state["analyzed"] = True
            state["chars"] = chars
            state["tags"] = raw_tags

            # Scan for existing voices
            voices_dir = self.voices_dir
            narrator_wav = os.path.join(voices_dir, "narrator.wav")
            if os.path.exists(narrator_wav):
                self.narrator_wav_path = narrator_wav
                self._log("Found narrator voice in project.")

            for char_name in chars:
                if char_name == "Narrator":
                    continue
                char_wav = os.path.join(voices_dir, f"{char_name.replace(' ', '_')}.wav")
                if os.path.exists(char_wav):
                    self.character_voice_paths[char_name] = char_wav
                    self._log(f"Found voice for {char_name} in project.")

            # Scan for existing segments
            seg_dir = self.segments_dir
            existing_wavs = [f for f in os.listdir(seg_dir) if f.endswith(".wav")] if os.path.exists(seg_dir) else []
            self._log(f"Found {len(existing_wavs)} existing segment WAVs.")

            # Build dataframe
            df_data = []
            for char_name in chars:
                if not char_name:
                    continue
                count = 0
                emotions = set()
                for sid, t_data in raw_tags.items():
                    c_name = t_data.get("char") or t_data.get("character_name")
                    if c_name == char_name:
                        count += 1
                        emo = t_data.get("emotion")
                        if emo:
                            emotions.add(emo)
                df_data.append([str(char_name), int(count), ", ".join(sorted(list(emotions)))])

            if not df_data and chars:
                df_data = [[str(c), 0, ""] for c in chars]

            info = f"Title: {self._book_title}\nAuthor: {self._book_author}\nSegments: {len(existing_wavs)} WAVs\nVoices: {1 if self.narrator_wav_path else 0} narrator + {len(self.character_voice_paths)} chars"

            self._log(f"Project loaded: {len(chars)} chars, {len(existing_wavs)} segments")
            return f"Loaded project: {project_dir}", info, df_data, state

        except Exception as e:
            import traceback
            self._log(f"Load error: {e}\n{traceback.format_exc()}")
            return f"Error: {e}", "", [], state

    # ── Tab 2 Handlers ──────────────────────────────────────────

    def design_narrator(self, desc_text, ref_wav):
        """Design or load the narrator voice. Auto-saves to project/voices/."""
        self.narrator_voice_desc = desc_text

        if not self.project_dir:
            self._ensure_project_dir("")

        out_path = os.path.join(self.voices_dir, "narrator.wav")

        # If user uploaded a WAV, copy it to project
        if ref_wav:
            shutil.copy2(ref_wav, out_path)
            self.narrator_wav_path = out_path
            self.narrator_ref_text = desc_text
            self._log(f"Narrator WAV copied to: {out_path}")
            return f"Narrator voice saved to project.", out_path

        # Otherwise, design it
        engine = self._get_engine()
        try:
            engine.load_model(self._voice_model_variant_design)

            test_text = (
                "Bonjour et bienvenue. Je suis votre narrateur pour ce livre. "
                "Je vais vous guider à travers chaque chapitre avec une voix claire et expressive, "
                "en adaptant le ton selon les scènes et les émotions du récit."
            )

            res_path = engine.design_voice(
                text=test_text,
                instruction=self.narrator_voice_desc,
                language="french",
                output_path=out_path,
            )

            if res_path:
                self.narrator_wav_path = res_path
                self.narrator_ref_text = test_text
                engine.unload_model()
                self._log(f"Narrator voice designed & saved to: {res_path}")
                return f"Voice designed & saved to project.", res_path
            else:
                engine.unload_model()
                return "Voice design failed.", None
        except Exception as e:
            engine.unload_model()
            return f"Error: {e}", None

    def design_all_characters(self, global_desc, state):
        """Design voices for all characters. Auto-saves to project/voices/."""
        if not state.get("analyzed"):
            return "Run Analysis first.", []

        if not self.project_dir:
            self._ensure_project_dir("")

        chars = self._characters
        engine = self._get_engine()
        self._log(f"Designing voices for {len(chars)} characters...")

        try:
            engine.load_model(self._voice_model_variant_design)

            for char_name in chars:
                if char_name == "Narrator":
                    continue

                desc = self.character_voice_descs.get(char_name, global_desc)
                if not desc:
                    continue

                out_path = os.path.join(self.voices_dir, f"{char_name.replace(' ', '_')}.wav")

                if os.path.exists(out_path):
                    self.character_voice_paths[char_name] = out_path
                    self._log(f"Skipping {char_name} (already exists)")
                    continue

                self._log(f"Designing voice for: {char_name}")
                char_test_text = f"Bonjour, je suis {char_name}. Comment allez-vous aujourd'hui? Je suis ravi de faire votre connaissance."

                res_path = engine.design_voice(
                    text=char_test_text,
                    instruction=desc,
                    language="french",
                    output_path=out_path,
                )

                if res_path:
                    self.character_voice_paths[char_name] = res_path
                    self.character_ref_texts[char_name] = char_test_text
                    self._log(f"Voice created for {char_name}")

            engine.unload_model()
            self._log("All voices designed.")

            status = f"Designed voices for {len(self.character_voice_paths)} characters. Saved to: {self.voices_dir}"

            df_data = []
            for char in chars:
                path = self.character_voice_paths.get(char, "Pending")
                df_data.append([char, path if path != "Pending" else self.character_voice_descs.get(char, ""), "Done" if path != "Pending" else "Pending"])

            return status, df_data

        except Exception as e:
            engine.unload_model()
            self._log(f"Batch Design Error: {e}")
            return f"Error: {e}", []

    # ── Tab 3 Handlers ──────────────────────────────────────────

    def _normalize_tags(self, state):
        """Ensure tags are plain dicts, not SpeechTag objects."""
        tags = state.get("tags", {}) if state else {}
        if not tags:
            return {}
        first_val = next(iter(tags.values()), None)
        if first_val is None:
            return {}
        if hasattr(first_val, 'emotion'):
            self._log("Normalizing tags from Objects to Dictionaries...")
            normalized = {}
            for sid, tag in tags.items():
                normalized[sid] = {
                    "speaker": getattr(tag, 'speaker_type', 'narrator'),
                    "char": getattr(tag, 'character_name', None),
                    "emotion": getattr(tag, 'emotion', 'neutral'),
                    "emotion_instruction": getattr(tag, 'emotion_instruction', ''),
                    "text": getattr(tag, 'text', ''),
                }
            return normalized
        return dict(tags)

    def _build_segments_from_tags(self):
        """Build segment list from tags (no EPUB re-parse needed)."""
        segs = []
        for sid in sorted(self._tags.keys()):
            segs.append({"id": sid, "text": self._tags[sid].get("text", "")})
        return segs

    def _generate_loop(self, all_segs, silence_duration, skip_existing=False):
        """Core generation loop. Yields (progress, phase, logs, m4a_file)."""
        engine = self._get_engine()
        total = len(all_segs)
        seg_dir = self.segments_dir
        os.makedirs(seg_dir, exist_ok=True)

        if skip_existing:
            already_done = sum(
                1 for s in all_segs
                if os.path.exists(os.path.join(seg_dir, f"{s['id'] if isinstance(s, dict) else s.id}.wav"))
            )
            self._log(f"Resume: {already_done}/{total} segments already exist.")
            yield 5, f"Resuming: {already_done} already done", self._get_logs(), None
        else:
            already_done = 0

        self._log("Loading Base Model (this takes time)...")
        yield 8, "Loading Base Model...", self._get_logs(), None
        engine.load_model(self._voice_model_variant_base)

        generated_count = already_done
        failed_count = 0

        for i, seg in enumerate(all_segs):
            seg_id = seg.id if hasattr(seg, 'id') else seg.get("id", "")
            out_path = os.path.join(seg_dir, f"{seg_id}.wav")

            # Skip if already generated (resume)
            if skip_existing and os.path.exists(out_path):
                continue

            tag_data = self._tags.get(seg_id, {})
            char_name = tag_data.get("char") or tag_data.get("character_name") or "Narrator"
            emotion = tag_data.get("emotion", "calm")

            if char_name not in self._characters and char_name != "Narrator":
                char_name = "Narrator"

            # Reference audio
            strategy = self.voice_strategy
            if strategy == "single_narrator":
                ref_audio = self.narrator_wav_path
            else:
                ref_audio = self.narrator_wav_path
                if char_name != "Narrator" and char_name in self.character_voice_paths:
                    ref_audio = self.character_voice_paths[char_name]

            if not ref_audio:
                self._log(f"Skipping {seg_id}: No reference audio for {char_name}")
                failed_count += 1
                continue

            # Text
            text = seg.text if hasattr(seg, 'text') else seg.get("text", "")
            if not text.strip():
                text = tag_data.get("text", "")
            if not text.strip():
                continue

            # Emotion instruction
            from audiobook_ai.analysis.character_analyzer import EMOTION_INSTRUCTIONS_FR
            emotion_instr = EMOTION_INSTRUCTIONS_FR.get(emotion, EMOTION_INSTRUCTIONS_FR["calm"])

            # Ref text
            if strategy == "single_narrator" or char_name == "Narrator":
                ref_text = self.narrator_ref_text or text
            else:
                ref_text = self.character_ref_texts.get(char_name, self.narrator_ref_text or text)

            self._log(f"Generating [{char_name}] -> {seg_id} ...")

            try:
                gen_path = engine.generate_voice_clone(
                    text=text,
                    ref_audio_path=ref_audio,
                    ref_text=ref_text,
                    language="french",
                    emotion_instruction=emotion_instr,
                    output_path=out_path,
                )
                if gen_path:
                    generated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                self._log(f"Failed: {seg_id}: {e}")
                failed_count += 1

            if (i + 1) % 5 == 0 or i == total - 1:
                pct = 10 + (generated_count / total * 80)
                yield pct, f"Generated {generated_count}/{total} ({failed_count} failed)", self._get_logs(), None

        engine.unload_model()
        self._log(f"Generation complete: {generated_count} OK, {failed_count} failed")

        # ── Assemble M4A ──
        self._log("Assembling final audiobook...")
        yield 92, "Assembling M4A...", self._get_logs(), None

        wav_files = []
        for seg in all_segs:
            seg_id = seg.id if hasattr(seg, 'id') else seg.get("id", "")
            wav_path = os.path.join(seg_dir, f"{seg_id}.wav")
            if os.path.exists(wav_path):
                wav_files.append(wav_path)

        if not wav_files:
            yield 0, "Error: No WAV files generated.", self._get_logs(), None
            return

        # Build chapter titles from segment prefixes
        chapter_titles = []
        seen_chapters = set()
        for seg in all_segs:
            seg_id = seg.id if hasattr(seg, 'id') else seg.get("id", "")
            ch_prefix = seg_id.split("_")[0]
            if ch_prefix not in seen_chapters:
                seen_chapters.add(ch_prefix)
                try:
                    ch_idx = int(ch_prefix.replace("ch", ""))
                except ValueError:
                    ch_idx = len(chapter_titles)
                if ch_idx < len(self._chapters_list):
                    ch = self._chapters_list[ch_idx]
                    title = ch.get("title", f"Chapter {ch_idx+1}") if isinstance(ch, dict) else getattr(ch, "title", f"Chapter {ch_idx+1}")
                    chapter_titles.append(title)
                else:
                    chapter_titles.append(f"Chapter {ch_idx+1}")

        m4a_path = os.path.join(self.project_dir, f"{self._book_title.replace(' ', '_')}.m4a")

        from audiobook_ai.tts.qwen_engine import TTSEngine
        try:
            result_path = TTSEngine.assemble_wav_files(
                wav_files=wav_files,
                output_path=m4a_path,
                silence_duration=silence_duration,
                sample_rate=24000,
                normalize=True,
                book_title=self._book_title,
                chapter_titles=chapter_titles,
            )
            self._log(f"Audiobook saved: {result_path}")
            yield 100, f"Done! {result_path}", self._get_logs(), result_path
        except Exception as e:
            self._log(f"Assembly error: {e}")
            yield 95, f"Assembly failed: {e}", self._get_logs(), None

    def start_generation(self, preview_mode, silence_duration, state):
        """Generate the audiobook from scratch."""
        if not state or not state.get("analyzed"):
            yield 0, "Error: Run Analysis first in Tab 1.", self._get_logs(), None
            return

        if not self.narrator_wav_path:
            yield 0, "Error: Design narrator voice in Tab 2 first.", self._get_logs(), None
            return

        if not self.project_dir:
            yield 0, "Error: Set a project folder in Tab 1 first.", self._get_logs(), None
            return

        self._log("Starting Production Pipeline...")
        self._tags = self._normalize_tags(state)
        self._characters = state.get("chars", [])

        yield 2, "Preparing segments...", self._get_logs(), None

        try:
            all_segs = self._build_segments_from_tags()

            if not all_segs:
                epub_path = state.get("epub_path")
                if not epub_path:
                    yield 0, "Error: No segments and no EPUB path.", self._get_logs(), None
                    return
                from audiobook_ai.core.epub_parser import EPUBParser
                from audiobook_ai.core.text_segmenter import TextSegmenter
                parser = EPUBParser(epub_path)
                parser_data = parser.parse()
                chapters = parser_data.get("chapters", [])
                self._chapters_list = chapters
                seg = TextSegmenter()
                all_segs = []
                for ch in chapters:
                    text = ch.get("text", "") if isinstance(ch, dict) else getattr(ch, "text", "")
                    title = ch.get("title", "") if isinstance(ch, dict) else getattr(ch, "title", "")
                    idx = ch.get("spine_order", 0) if isinstance(ch, dict) else getattr(ch, "spine_order", 0)
                    if text:
                        all_segs.extend(seg.segment_chapter(text, title, idx))

            if preview_mode:
                first_ch = [s for s in all_segs if (s.id if hasattr(s, 'id') else s.get("id", "")).startswith("ch0")]
                if first_ch:
                    all_segs = first_ch
                    self._log(f"PREVIEW MODE: limited to {len(all_segs)} segments")

            self._log(f"Total segments: {len(all_segs)}")

            for progress, phase, logs, m4a_file in self._generate_loop(all_segs, silence_duration, skip_existing=False):
                yield progress, phase, logs, m4a_file

        except Exception as e:
            self._log(f"Fatal Error: {e}")
            yield 0, f"Error: {e}", self._get_logs(), None

    def resume_generation(self, silence_duration, state):
        """Resume generation - skips existing WAVs in segments/ folder."""
        if not state or not state.get("analyzed"):
            yield 0, "Error: Load the project first (Tab 1).", self._get_logs(), None
            return

        if not self.project_dir:
            yield 0, "Error: No project folder set.", self._get_logs(), None
            return

        self._tags = self._normalize_tags(state)
        self._characters = state.get("chars", [])

        seg_dir = self.segments_dir
        existing_wavs = [f for f in os.listdir(seg_dir) if f.endswith(".wav")] if os.path.exists(seg_dir) else []
        self._log(f"Resume: found {len(existing_wavs)} existing WAVs in {seg_dir}")

        yield 2, f"Found {len(existing_wavs)} existing files. Rebuilding...", self._get_logs(), None

        try:
            all_segs = self._build_segments_from_tags()

            if not all_segs:
                yield 0, "Error: No segment data. Load the analysis JSON first.", self._get_logs(), None
                return

            self._log(f"Total segments: {len(all_segs)}")

            for progress, phase, logs, m4a_file in self._generate_loop(all_segs, silence_duration, skip_existing=True):
                yield progress, phase, logs, m4a_file

        except Exception as e:
            self._log(f"Resume Error: {e}")
            yield 0, f"Error: {e}", self._get_logs(), None

    # ── Launch ──────────────────────────────────────────────────

    def launch(self, port=7860, share=False, server_name="0.0.0.0"):
        if self.app is None:
            self.build()
        self.app.queue()

        # Allow Gradio to serve files from the project directories
        # (voices, segments, m4a output) - otherwise it throws InvalidPathError
        allowed = [DEFAULT_PROJECTS_ROOT]
        if self.project_dir and os.path.isdir(self.project_dir):
            allowed.append(self.project_dir)
            allowed.append(os.path.join(self.project_dir, "voices"))
            allowed.append(os.path.join(self.project_dir, "segments"))

        self.app.launch(
            server_name=server_name,
            server_port=port,
            share=share,
            theme=self.theme,
            css=self.css,
            allowed_paths=allowed,
        )
