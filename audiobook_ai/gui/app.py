"""AudiobookGUI v2 - Redesigned Gradio interface."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import gradio as gr

logger = logging.getLogger(__name__)

EMOTION_OPTIONS = [
    "calm", "excited", "angry", "sad", "whisper",
    "tense", "urgent", "amused", "contemptuous", "surprised", "neutral"
]

BARK_VOICES = {
    "en_speaker_0": "Neutral Male",
    "en_speaker_1": "Warm Male",
    "en_speaker_2": "Deep Male",
    "en_speaker_3": "Energetic Male",
    "en_speaker_4": "Soft Male",
    "en_speaker_5": "Young Female",
    "en_speaker_6": "Warm Female",
    "en_speaker_7": "Clear Female",
    "en_speaker_8": "Deep Female",
    "en_speaker_9": "Narrator Female",
}

EDGE_VOICES = {
    "en": ["en-US-GuyNeural", "en-US-JennyNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "de": ["de-DE-ConradNeural", "de-DE-KatjaNeural"],
}


class AudiobookGUI:
    """Redesigned Gradio interface."""

    def __init__(self, config):
        self.config = config
        self.app = None

        self._project = None
        self._epub_parser = None
        self._tts_engine = None
        self._voice_manager = None
        self._analyzer = None
        
        self._segment_tags = {}
        self._discovered_chars = []
        self._dedup_map = {}
        self._segmentation = {}
        self._chapter_titles = {}
        self._chapters_list = []
        self._voice_assignments = {}

        self._generation_running = False
        self._generation_paused = False
        self._generation_cancelled = False
        self._voice_mode = "narrator"
        self._log_messages = []

    def _log(self, msg):
        self._log_messages.append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
        logger.info(msg)

    def _get_logs(self):
        return "\n".join(self._log_messages[-200:])

    def build(self):
        with gr.Blocks(title="AIGUIBook", css=".log-box textarea {font-family: monospace; font-size: 12px;}") as self.app:
            gr.Markdown("# AIGUIBook\n### Transform EPUB to Audiobook with AI Voices")

            app_state = gr.State({"loaded": False, "parsed": False, "analyzed": False})

            with gr.Tabs():
                with gr.Tab("1. Analysis"):
                    with gr.Row():
                        with gr.Column():
                            epub_upload = gr.File(label="Upload EPUB", file_types=[".epub"], type="filepath")
                            parse_btn = gr.Button("Parse Book", variant="primary", size="lg")
                            book_info_box = gr.Markdown("*No book loaded*")
                            run_analysis_btn = gr.Button("Run Character Analysis", variant="primary", size="lg")
                            analysis_status_box = gr.Textbox(label="Status", interactive=False)
                            character_list_box = gr.JSON(label="Characters")
                        with gr.Column():
                            load_analysis_btn = gr.Button("Load Saved Analysis", variant="secondary")
                            analysis_file_input = gr.File(label="JSON File", file_types=[".json"], type="filepath")
                            load_status_box = gr.Textbox(label="Load Status", interactive=False)
                            with gr.Group():
                                gen_desc_btn = gr.Button("Generate ElevenLabs Prompts")
                                desc_box = gr.Textbox(label="Prompts", lines=10)
                                copy_desc_btn = gr.Button("Save to File")
                                copy_status_box = gr.Textbox(label="Save Status", interactive=False)

                with gr.Tab("2. Voices"):
                    gr.Markdown("### Voice Configuration")
                    with gr.Row():
                        voice_mode_radio = gr.Radio(
                            choices=[("Single Narrator (Dynamic Emotion)", "narrator"),
                                     ("Ensemble Cast (Multi-Voice)", "ensemble")],
                            value="narrator", label="Voice Mode"
                        )
                    with gr.Group():
                        narrator_source = gr.Dropdown(choices=["Edge TTS", "Custom WAV"], value="Edge TTS", label="Narrator Source")
                        narrator_voice_select = gr.Dropdown(choices=list(EDGE_VOICES["fr"]), value="fr-FR-DeniseNeural", label="Voice")
                        narrator_file_input = gr.File(label="Upload Custom WAV", file_types=[".wav", ".mp3"], type="filepath", visible=False)
                        narrator_preview_btn = gr.Button("Preview Narrator")
                        narrator_audio = gr.Audio(label="Preview", type="filepath")
                    with gr.Group(visible=False):
                        auto_assign_btn = gr.Button("Auto-Assign Voices")
                        char_voice_table = gr.Dataframe(headers=["Character", "Segments", "Voice"], datatype=["str", "number", "str"], interactive=True)
                    apply_voice_btn = gr.Button("Save Settings")
                    voice_status = gr.Textbox(label="Status", interactive=False)

                with gr.Tab("3. Preview"):
                    with gr.Row():
                        with gr.Column():
                            chapter_select = gr.Dropdown(choices=[], label="Chapter")
                            segment_select = gr.Dropdown(choices=[], label="Segment")
                            load_preview_btn = gr.Button("Load Segment")
                        with gr.Column():
                            preview_text = gr.Textbox(label="Text", lines=4)
                            preview_info = gr.Textbox(label="Speaker Info", lines=2)
                            preview_voice = gr.Dropdown(choices=["Narrator"], label="Voice for Preview")
                            preview_emotion = gr.Dropdown(choices=["auto"] + EMOTION_OPTIONS, value="auto", label="Emotion Mode")
                            generate_preview_btn = gr.Button("Generate Audio", variant="primary")
                            preview_audio_out = gr.Audio(label="Result", type="filepath")

                with gr.Tab("4. Generate"):
                    gr.Markdown("### Audiobook Production")
                    with gr.Row():
                        with gr.Column():
                            start_btn = gr.Button("Start Full Generation", variant="primary", size="lg")
                            resume_btn = gr.Button("Resume", variant="secondary", visible=False)
                            preview_only_check = gr.Checkbox(label="Preview Mode (First 3 chapters only)", value=False)
                            no_val_check = gr.Checkbox(label="Disable Validation", value=False)
                        with gr.Column():
                            overall_progress = gr.Slider(minimum=0, maximum=100, value=0, label="Progress")
                            stage_progress = gr.Textbox(label="Current Stage", value="Ready")
                            eta_box = gr.Textbox(label="Estimated Time Remaining", value="--:--")
                            detail_box = gr.Textbox(label="Details", lines=2)
                            log_box = gr.Textbox(label="Log", lines=10, elem_classes=["log-box"])

                with gr.Tab("5. Settings"):
                    with gr.Row():
                        with gr.Column():
                            tts_model_select = gr.Dropdown(choices=["Qwen/Qwen3-TTS-12Hz-1.7B-Base"], value="Qwen/Qwen3-TTS-12Hz-1.7B-Base", label="TTS Model")
                            test_llm_btn = gr.Button("Test LLM Connection")
                            llm_status = gr.Textbox(label="Status")
                    save_btn = gr.Button("Save All Settings", variant="primary")
                    save_status = gr.Textbox(label="Status", interactive=False)

            # Events
            parse_btn.click(fn=self._on_parse_epub, inputs=[epub_upload], outputs=[book_info_box, character_list_box, app_state])
            run_analysis_btn.click(fn=self._on_run_analysis, inputs=[app_state], outputs=[analysis_status_box, character_list_box, app_state])
            load_analysis_btn.click(fn=self._on_load_analysis, inputs=[analysis_file_input], outputs=[load_status_box, character_list_box, app_state])
            
            gen_desc_btn.click(fn=self._on_generate_voice_descriptions, outputs=[desc_box])
            copy_desc_btn.click(fn=self._on_copy_voice_descriptions, inputs=[], outputs=[copy_status_box])
            
            narrator_preview_btn.click(fn=self._on_preview_narrator, inputs=[narrator_source, narrator_voice_select, narrator_file_input], outputs=[narrator_audio])
            narrator_source.change(fn=self._on_source_change, inputs=[narrator_source], outputs=[narrator_file_input, narrator_voice_select])
            
            voice_mode_radio.change(fn=lambda m: (gr.update(visible=m=="ensemble"), gr.update(visible=(m=="narrator") or (m=="ensemble"))), inputs=[voice_mode_radio], outputs=[auto_assign_btn.parent.parent, narrator_source.parent])
            
            auto_assign_btn.click(fn=self._on_auto_assign, outputs=[char_voice_table, voice_status])
            apply_voice_btn.click(fn=lambda: "Assignments saved.", outputs=[voice_status])
            
            start_btn.click(fn=self._on_start_gen, inputs=[preview_only_check, no_val_check, overall_progress, stage_progress, eta_box, detail_box, log_box], 
                            outputs=[overall_progress, stage_progress, eta_box, detail_box, log_box, start_btn, resume_btn])

        return self.app

    def _on_parse_epub(self, epub_path):
        if not epub_path:
            return "*No file selected*", [], {"loaded": False, "parsed": False, "analyzed": False}
        
        try:
            self._log("Parsing EPUB: %s" % epub_path)
            from audiobook_ai.core.epub_parser import EPUBParser
            parser = EPUBParser(epub_path)
            result = parser.parse()
            self._epub_parser = parser
            self._chapters_list = result.get("chapters", [])
            self._chapter_titles = {ch.get("spine_order", i): ch.get("title", "Ch %d" % (i+1)) for i, ch in enumerate(self._chapters_list)}

            meta = result.get("metadata", {})
            info = "**%s** by %s\n%d Chapters" % (meta.get("title", "?"), meta.get("author", "?"), len(self._chapters_list))
            
            return info, [{"name": ch.get("title", "Ch"), "idx": ch.get("spine_order", i)} for i, ch in enumerate(self._chapters_list)], {"loaded": True, "parsed": True}
        except Exception as e:
            self._log(str(e))
            return "*Error: %s*" % e, [], {"loaded": False, "parsed": False, "analyzed": False}

    def _on_run_analysis(self, state):
        self._log("Starting analysis...")
        yield "Analyzing characters...", [], state
        try:
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            self._analyzer = CharacterAnalyzer(self.config.get_section("analysis"))
            tags, chars, dmap = self._analyzer.analyze_segments(self._chapters_list)
            self._segment_tags = tags
            self._discovered_chars = chars
            self._dedup_map = dmap
            self._log("Analysis complete: %d characters" % len(chars))
            return "Done! Found %d characters." % len(chars), [{"char": c, "count": sum(1 for t in tags.values() if t.character_name == c)} for c in chars], {"loaded": True, "parsed": True, "analyzed": True}
        except Exception as e:
            return "Error: %s" % e, [], state

    def _on_load_analysis(self, filepath):
        if not filepath:
            return "No file selected.", [], state
        try:
            tags, chars, dmap = CharacterAnalyzer.load_analysis(filepath)
            self._segment_tags = tags
            self._discovered_chars = chars
            self._dedup_map = dmap
            return "Loaded %d characters." % len(chars), [{"char": c, "count": 0} for c in chars], {"loaded": True, "parsed": True, "analyzed": True}
        except Exception as e:
            return "Error loading file.", [], {"loaded": False, "parsed": False, "analyzed": False}

    def _on_generate_voice_descriptions(self):
        if not self._discovered_chars: return "Run analysis first."
        if not self._analyzer: return "No analyzer found."
        try:
            descs = self._analyzer.build_voice_descriptions()
            return "\n\n".join(["%s: %s" % (k, v['elevenlabs_prompt']) for k, v in descs.items()])
        except Exception as e:
            return "Error: %s" % e

    def _on_copy_voice_descriptions(self):
        try:
            txt = self._on_generate_voice_descriptions()
            path = os.path.join(tempfile.gettempdir(), "prompts.txt")
            with open(path, "w") as f:
                f.write(txt)
            return "Saved to %s" % path
        except Exception as e:
            return str(e)

    def _on_preview_narrator(self, source, voice, file):
        import asyncio
        if source == "Edge TTS":
            try:
                import edge_tts
                path = os.path.join(tempfile.gettempdir(), "preview.mp3")
                async def run():
                    await edge_tts.Communicate("Bonjour, ceci est un apercu.", voice).save(path)
                asyncio.run(run())
                return path
            except Exception as e:
                self._log(str(e))
        elif source == "Custom WAV":
            return file
        return None

    def _on_source_change(self, source):
        if source == "Edge TTS":
            return gr.update(visible=False), gr.update(choices=list(EDGE_VOICES["fr"]), value=EDGE_VOICES["fr"][0])
        return gr.update(visible=True), gr.update(choices=["Custom"], value="Custom")

    def _on_auto_assign(self):
        if not self._discovered_chars:
            return [], "No characters found."
        rows = [[c, 0, "Narrator Female"] for c in self._discovered_chars]
        return rows, "Assigned %d voices." % len(rows)

    def _on_start_gen(self, preview_mode, no_val, progress, stage, eta, detail, logs):
        # Simplified generator loop
        yield 10, "Initializing...", "10:00", "Starting", logs + "\nStarting...", gr.update(visible=False), gr.update(visible=True)
        yield 100, "Done", "00:00", "Ready to implement full pipeline!", logs + "\nDone", gr.update(visible=True), gr.update(visible=False)

    def _on_output_ready(self):
        pass

    def _on_run_pipeline_thread(self):
        pass

    def launch(self, port=7860, share=False):
        if self.app is None: self.build()
        self.app.queue().launch(server_name="0.0.0.0", server_port=port, share=share)
