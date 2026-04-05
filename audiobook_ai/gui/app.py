"""
AudiobookGUI v2 - Redesigned Gradio interface.
Focus: Single Narrator with emotional range (Voice Acting).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import List, Dict, Any, Generator, Tuple

import gradio as gr

logger = logging.getLogger(__name__)

# Available Base Voices for the "Single Narrator" mode
BASE_VOICES = {
    "narrator_male": "Narrator Male (Warm)",
    "narrator_female": "Narrator Female (Clean)",
    "young_male": "Young Male",
    "young_female": "Young Female",
    "elder_male": "Elder Male",
    "elder_female": "Elder Female",
}

EDGE_VOICES = {
    "fr-FR-DeniseNeural": "Female (French)",
    "fr-FR-HenriNeural": "Male (French)",
    "en-US-GuyNeural": "Male (English)",
}

class AudiobookGUI:
    """Redesigned Gradio interface."""

    def __init__(self, config):
        self.config = config
        self.app = None

        # Project State
        self.project = None
        self.epub_parser = None
        self.analysis_results = {} 
        self.char_list = []
        self.chapters_data = []
        self.segments = []

        # Generation state
        self.gen_running = False
        self.voice_mode = "narrator"

    def build(self):
        theme = gr.themes.Soft(primary_hue="violet", secondary_hue="blue")
        css = ".log-box textarea {font-family:monospace; font-size:12px;}"

        with gr.Blocks(theme=theme, title="AIGUIBook", css=css) as self.app:
            
            # Global State Object
            global_state = gr.State({"status": "init"})

            gr.Markdown("# AIGUIBook v2\n### EPUB to Audiobook with AI Voice Acting")
            
            with gr.Tabs():
                # ==========================
                # TAB 1: ANALYSIS
                # ==========================
                with gr.Tab("1. Analysis"):
                    gr.Markdown("Step 1: Load your book and detect characters/emotions.")
                    
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(label="Upload EPUB", file_types=[".epub"])
                            btn_parse = gr.Button("1. Parse EPUB", variant="primary")
                            book_meta = gr.Textbox(label="Book Metadata", lines=4)
                            
                            btn_analyze = gr.Button("2. Run Character Analysis", variant="primary")
                            status_analysis = gr.Textbox(label="Status", lines=2)
                            json_output = gr.JSON(label="Results")
                            
                            btnSave = gr.Button("3. Save Results to JSON", variant="secondary")
                            save_status = gr.Textbox(label="Save Status")

                        with gr.Column():
                            load_analysis_btn = gr.Button("Load Existing Analysis", variant="secondary")
                            analysis_file_input = gr.File(label="Load JSON", file_types=[".json"])
                            load_status_box = gr.Textbox(label="Load Status")

                # ==========================
                # TAB 2: VOICES
                # ==========================
                with gr.Tab("2. Voices"):
                    gr.Markdown("Step 2: Configure Voice Strategy.")
                    
                    mode_selector = gr.Radio(
                        choices=[
                            ("Single Narrator (Dynamic Emotion)", "narrator"),
                            ("Ensemble Cast (Multi-Voice)", "ensemble")
                        ],
                        value="narrator",
                        label="Strategy"
                    )

                    with gr.Group() as narrator_group:
                        gr.Markdown("### Narrator Mode Settings")
                        base_voice_select = gr.Dropdown(
                            choices=list(BASE_VOICES.values()),
                            value=BASE_VOICES["narrator_male"],
                            label="Base Voice Profile"
                        )
                        ref_audio_input = gr.Audio(
                            label="OR Upload Reference Audio (WAV) for Cloning",
                            type="filepath"
                        )
                        gr.Markdown("*In this mode, a single voice is used for everyone. However, the AI will apply different emotions (Angry, Whisper, etc.) based on the text analysis to simulate acting.*")
                        btn_preview_narrator = gr.Button("Test Narrator Voice", variant="primary")
                        audio_preview_narrator = gr.Audio(label="Preview")

                    with gr.Group(visible=False) as ensemble_group:
                        gr.Markdown("### Character Voice Assignment")
                        char_table = gr.Dataframe(
                            headers=["Character", "Count", "Emotions", "Suggested Voice"],
                            datatype=["str", "number", "str", "str"],
                            interactive=True,
                        )
                        btn_auto_assign = gr.Button("Auto-Assign", variant="primary")
                        btn_save_cast = gr.Button("Save Casting")
                        cast_status = gr.Textbox(label="Status")

                # ==========================
                # TAB 3: PREVIEW
                # ==========================
                with gr.Tab("3. Preview"):
                    gr.Markdown("Step 3: Listen to specific segments.")
                    chapter_select = gr.Dropdown(choices=[], label="Chapter")
                    segment_select = gr.Dropdown(choices=[], label="Segment")
                    btn_load_segment = gr.Button("Load Text")
                    
                    with gr.Row():
                        text_preview = gr.Textbox(label="Text", lines=5)
                        meta_preview = gr.Textbox(label="Info", lines=5)
                        
                    btn_preview_seg = gr.Button("Generate Audio", variant="primary")
                    audio_seg_preview = gr.Audio(label="Result")

                # ==========================
                # TAB 4: GENERATE
                # ==========================
                with gr.Tab("4. Generate"):
                    gr.Markdown("Step 4: Render Full Audiobook.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            btn_start = gr.Button("START GENERATION", variant="primary", size="lg")
                            resume_btn = gr.Button("RESUME", variant="secondary", visible=False)
                            check_preview = gr.Checkbox(label="Preview Only (First 3 Chapters)", value=False)
                            check_val = gr.Checkbox(label="Enable Audio Validation", value=True)
                            
                        with gr.Column(scale=2):
                            progress_bar = gr.Slider(value=0, maximum=100, label="Progress")
                            stage_status = gr.Textbox(label="Current Stage", value="Ready")
                            eta_box = gr.Textbox(label="Est. Time Remaining", value="--:--")
                            log_box = gr.Textbox(label="System Log", lines=10, elem_classes=["log-box"])

                # ==========================
                # TAB 5: SETTINGS
                # ==========================
                with gr.Tab("5. Settings"):
                    gr.Markdown("System Configuration.")
                    with gr.Row():
                        with gr.Column():
                            llm_url = gr.Textbox(label="LM Studio URL", value="http://localhost:1234/v1")
                            btn_test_llm = gr.Button("Test LLM Connection")
                            llm_status = gr.Textbox(label="Status")
                        with gr.Column():
                            tts_model = gr.Textbox(label="TTS Model", value="Qwen/Qwen3-TTS-1.7B-Base")
                            btn_save = gr.Button("Save Settings", variant="primary")
                            save_cfg_status = gr.Textbox(label="Status")

            # ==========================
            # EVENTS
            # ==========================
            
            btn_parse.click(fn=self._parse_epub, inputs=[file_input], outputs=[book_meta, json_output, global_state])
            btn_analyze.click(fn=self._run_analysis, inputs=[file_input, global_state], outputs=[status_analysis, json_output, char_table, global_state])
            
            btnSave.click(fn=self._save_analysis, inputs=[global_state], outputs=[save_status])
            load_analysis_btn.click(fn=self._load_analysis, inputs=[analysis_file_input], outputs=[load_status_box, json_output, global_state])

            mode_selector.change(fn=self._toggle_mode, inputs=[mode_selector], outputs=[narrator_group, ensemble_group])
            
            btn_auto_assign.click(fn=self._auto_assign_cast, inputs=[global_state], outputs=[char_table])
            
            btn_preview_narrator.click(fn=self._preview_narrator, inputs=[base_voice_select], outputs=[audio_preview_narrator])
            
            btn_load_segment.click(fn=self._load_segment_text, inputs=[global_state], outputs=[text_preview, meta_preview])
            
            btn_start.click(fn=self._start_generation, inputs=[global_state, check_preview, check_val], outputs=[progress_bar, stage_status, eta_box, log_box, btn_start, resume_btn])

        return self.app

    def _parse_epub(self, file_path, state:dict) -> Tuple[str, dict, dict]:
        if not file_path: return "No file", {}, state
        try:
            from audiobook_ai.core.epub_parser import EPUBParser
            parser = EPUBParser(file_path)
            res = parser.parse()
            self.epub_parser = parser
            self.chapters_data = res.get('chapters', [])
            
            meta = res['metadata']
            text = "Title: %s\nAuthor: %s\nLang: %s\nChapters: %d" % (
                meta.get('title', '?'), meta.get('author', '?'), meta.get('language', '?'), len(self.chapters_data)
            )
            state['parsed'] = True
            state['chapters'] = self.chapters_data
            return text, {}, state
        except Exception as e:
            return f"Error: {e}", {}, state

    def _run_analysis(self, file_path, state:dict) -> Tuple[str, dict, list, dict]:
        if 'parsed' not in state or not state.get('parsed'):
            if not file_path:
                return "Please parse EPUB first.", {}, [], state
            # Auto parse if not done
            text, json_res, st = self._parse_epub(file_path, state)
            state = st
        
        if not self.chapters_data:
            return "No chapters found.", {}, [], state

        try:
            self._log("Starting analysis...")
            from audiobook_ai.core.text_segmenter import TextSegmenter
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            
            seg = TextSegmenter()
            all_segs = []
            for ch in self.chapters_data:
                s = seg.segment_chapter(ch.text, ch.title, ch.spine_order) if hasattr(ch, 'text') else seg.segment_chapter(ch.get('text',''), ch.get('title',''), ch.get('spine_order',0))
                all_segs.extend(s)
            self.segments = all_segs
            
            analyzer = CharacterAnalyzer(self.config.get_section("analysis"))
            tags, chars = analyzer.analyze_segments(all_segs)
            
            self.analysis_results = tags
            self.char_list = chars
            state['analyzed'] = True
            state['tags'] = tags
            state['chars'] = chars
            state['segments'] = all_segs
            
            out_json = {c: {"segments": 0, "emotions": []} for c in chars}
            for tag in tags.values():
                if tag.character_name in out_json:
                    out_json[tag.character_name]["segments"] += 1
                    if tag.emotion not in out_json[tag.character_name]["emotions"]:
                        out_json[tag.character_name]["emotions"].append(tag.emotion)
            
            table_rows = []
            for c, data in out_json.items():
                table_rows.append([c, data['segments'], ", ".join(data['emotions']), "Narrator Male"])
            
            return "Done! %d chars found." % len(chars), out_json, table_rows, state
        except Exception as e:
            self._log("Error: %s" % e)
            return "Error: %s" % e, {}, [], state

    def _save_analysis(self, state:dict) -> str:
        if not state.get('analyzed'):
            return "No analysis to save."
        try:
            path = os.path.join(tempfile.gettempdir(), "aiguibook_analysis.json")
            serializer = {
                'chars': state.get('chars', []),
                'tags': {sid: t.to_dict() for sid, t in state.get('tags', {}).items()}
            }
            with open(path, 'w') as f:
                json.dump(serializer, f, indent=4)
            return "Saved to: " + path
        except Exception as e:
            return "Error: " + str(e)

    def _load_analysis(self, filepath, state:dict) -> Tuple[str, dict, dict]:
        if not filepath: return "No file.", {}, state
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
            
            chars = content.get('chars', [])
            tags_data = content.get('tags', {})
            
            self.char_list = chars
            state['analyzed'] = True
            state['chars'] = chars
            
            out_json = {c: {"segments": 0, "emotions": []} for c in chars}
            # We can't reconstruct the full tag objects easily here without the class, 
            # but for the UI we just show the structure
            for c in chars:
                out_json[c]["status"] = "Loaded from file"
            
            return "Loaded %d characters." % len(chars), out_json, state
        except Exception as e:
            return "Load Error: " + str(e), {}, state

    def _toggle_mode(self, mode):
        if mode == "narrator": return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    def _auto_assign_cast(self, state:dict) -> Tuple[list, str]:
        if 'chars' not in state or not state['chars']:
            return [], "Run Analysis first."
        
        rows = []
        female_hints = ["mme", "mrs", "miss", "madame", "frau", "she", "her"]
        for c in state['chars']:
            lower = c.lower()
            v = "Narrator Male"
            if any(h in lower for h in female_hints): v = "Narrator Female"
            rows.append([c, 0, "", v])
        return rows, "Assigned %d characters." % len(rows)

    def _preview_narrator(self, voice_name):
        # Use Edge TTS for quick preview
        import asyncio, edge_tts
        try:
            v = "fr-FR-DeniseNeural"
            if "Male" in voice_name: v = "fr-FR-HenriNeural"
            path = os.path.join(tempfile.gettempdir(), "preview.mp3")
            async def gen():
                await edge_tts.Communicate("Bonjour. Ceci est un apercu de la voix.", v).save(path)
            asyncio.run(gen())
            return path
        except Exception as e:
            return None

    def _load_segment_text(self, state:dict) -> Tuple[str, str]:
        if 'segments' not in state or not state['segments']:
            return "No segments loaded.", ""
        
        seg = state['segments'][0] # Get first for demo
        text = seg.text if hasattr(seg, 'text') else seg.get('text', 'No text')
        tag = state.get('tags', {}).get(seg.id if hasattr(seg, 'id') else seg.get('id'))
        info = "ID: %s" % (seg.id if hasattr(seg, 'id') else seg.get('id'))
        if tag:
            info += "\nSpeaker: %s\nEmotion: %s" % (tag.character_name or "Narrator", tag.emotion)
        return text, info

    def _start_generation(self, state:dict, preview:bool, val:bool):
        import threading
        # Placeholder for generation logic
        def run():
            yield 50, "Processing Chapters...", "10:00", "Generating...\n"
        
        # For now, just return success
        return 100, "Complete", "00:00", "Done!", None, None

    def _log(self, msg):
        logger.info(msg)

    def launch(self, port=7860, share=False):
        if self.app is None: self.build()
        self.app.queue()
        self.app.launch(server_name="0.0.0.0", server_port=port, share=share)
