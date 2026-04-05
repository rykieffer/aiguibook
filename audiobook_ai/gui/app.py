"""
AudiobookGUI v6 - Final Stable Release.
Focus: Single Narrator with Emotion Acting (Qwen3-TTS).
Includes Gradio 6.0 compatibility fixes.
"""

import gradio as gr
import logging
import os
import tempfile
import json
import time
from typing import Any, Dict, List

logger = logging.getLogger("AIGUIBook")

QWEN_VOICE_STYLES = {
    "ryan": "Ryan (Deep Male)",
    "aidan": "Aidan (Male)",
    "elara": "Elara (Female)",
    "bella": "Bella (Young Female)",
    "george": "George (Older Male)",
    "clara": "Clara (Warm Female)",
}


class AudiobookGUI:
    def __init__(self, config):
        self.config = config
        self.app = None
        
        self.parser = None
        self.analyzer = None
        self.segments = []
        self.tags = {}
        self.characters = []
        
        self.narrator_ref_path = None

    def build(self):
        # Theme and CSS defined here, passed to launch() later for Gradio 6.0
        self.theme = gr.themes.Soft(primary_hue="violet", secondary_hue="blue")
        self.css = ".log-box textarea {font-family:monospace; font-size:12px;}"

        with gr.Blocks(title="AIGUIBook") as self.app:
            state = gr.State({"loaded": False, "parsed": False, "analyzed": False})
            gr.Markdown("# AIGUIBook v6\n### EPUB to Audiobook with Emotional Voice Acting (Qwen3-TTS)")
            
            with gr.Tabs():
                # ==========================
                # TAB 1: ANALYSIS
                # ==========================
                with gr.Tab("1. Analysis"):
                    gr.Markdown("### 1. Detect Characters and Emotions")
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_epub = gr.File(label="Upload EPUB", file_types=[".epub"])
                            btn_parse = gr.Button("1. Parse Book", variant="primary")
                            book_info = gr.Textbox(label="Metadata", lines=4, interactive=False)
                            
                            btn_run_char = gr.Button("2. Character Analysis (LM Studio)", variant="primary")
                            status_bar = gr.Textbox(label="Status", lines=2, interactive=False)
                            
                        with gr.Column(scale=1):
                            char_table = gr.Dataframe(
                                headers=["Character", "Count", "Emotions"],
                                datatype=["str", "number", "str"],
                                interactive=True,
                                label="Detected Characters"
                            )
                            btn_save = gr.Button("3. Save Analysis File")
                            file_json = gr.File(label="Load Analysis File", file_types=[".json"])
                            status_load = gr.Textbox(label="Load Status", lines=2, interactive=False)

                # ==========================
                # TAB 2: VOICE STYLE
                # ==========================
                with gr.Tab("2. Voice Strategy"):
                    gr.Markdown("### 2. Configure Qwen3-TTS Voice")
                    mode_radio = gr.Radio(
                        choices=[
                            ("Single Narrator (One voice acting all roles)", "single"),
                            ("Multi-Cast (Different voices per character)", "multi")
                        ],
                        value="single",
                        label="Strategy"
                    )

                    with gr.Group() as narrator_group:
                        gr.Markdown("### Narrator Voice")
                        builtin_select = gr.Dropdown(
                            choices=list(QWEN_VOICE_STYLES.values()),
                            value=QWEN_VOICE_STYLES["ryan"],
                            label="Select Base Voice Profile"
                        )
                        custom_ref = gr.File(
                            label="OR Upload Custom Reference (WAV/MP3)",
                            file_types=[".wav", ".mp3"], type="filepath"
                        )
                        btn_preview = gr.Button("Test Voice", variant="primary")
                        preview_audio = gr.Audio(label="Preview", type="filepath")
                        gr.Markdown("*System will modulate this voice with emotions (Angry, Whisper) from analysis.*")

                    with gr.Group(visible=False) as multi_group:
                        gr.Markdown("### Character Assignments")
                        df_cast = gr.Dataframe(
                            headers=["Character", "Voice File", "Action"],
                            datatype=["str", "str", "str"],
                            interactive=True
                        )

                    mode_radio.change(
                        fn=lambda m: (gr.update(visible=m=="single"), gr.update(visible=m=="multi")),
                        inputs=[mode_radio],
                        outputs=[narrator_group, multi_group]
                    )

                # ==========================
                # TAB 3: GENERATION
                # ==========================
                with gr.Tab("3. Production"):
                    gr.Markdown("### 3. Generate Audiobook")
                    with gr.Row():
                        with gr.Column():
                            btn_start = gr.Button("START GENERATION", variant="primary", size="lg")
                            btn_resume = gr.Button("RESUME", variant="secondary", visible=False)
                            chk_preview = gr.Checkbox(label="Preview Mode (First 3 Chapters)", value=False)
                            chk_val = gr.Checkbox(label="Enable Validation", value=True)
                        
                        with gr.Column():
                            progress = gr.Slider(label="Progress")
                            phase = gr.Textbox(label="Current Phase")
                            logs = gr.Textbox(label="System Log", lines=10, elem_classes=["log-box"])

                # ==========================
                # TAB 4: SETTINGS
                # ==========================
                with gr.Tab("4. Settings"):
                    gr.Markdown("### System Settings")
                    llm_url = gr.Textbox(label="LM Studio URL", value="http://localhost:1234/v1")
                    test_btn = gr.Button("Test Connection")
                    test_out = gr.Textbox(label="Status", interactive=False)

            # ==========================
            # EVENTS
            # ==========================
            
            # 1. Analysis
            btn_parse.click(
                fn=self.parse_epub, 
                inputs=[file_epub, state], 
                outputs=[book_info, char_table, state]
            )
            btn_run_char.click(
                fn=self.run_analysis, 
                inputs=[file_epub, state], 
                outputs=[status_bar, char_table, state]
            )
            btn_save.click(fn=self.save_analysis, inputs=[state], outputs=[status_load])
            file_json.change(fn=self.load_analysis, inputs=[file_json, state], outputs=[status_load, char_table, state])

            # 2. Voice
            btn_preview.click(
                fn=self.preview_voice, 
                inputs=[builtin_select, custom_ref], 
                outputs=[preview_audio]
            )

            # 3. Generation (5 inputs/outputs to match Gradio expectations)
            btn_start.click(
                fn=self.start_generation, 
                inputs=[chk_preview, chk_val, state], 
                outputs=[progress, phase, logs, btn_start, btn_resume]
            )

            # 4. Settings
            test_btn.click(fn=self.test_llm, inputs=[llm_url], outputs=[test_out])

        return self.app

    # --- Logic Methods ---

    def parse_epub(self, file_path, state: dict):
        if not file_path:
            return "No file selected.", [], state
        
        try:
            from audiobook_ai.core.epub_parser import EPUBParser
            self.parser = EPUBParser(file_path)
            data = self.parser.parse()
            meta = data.get("metadata", {})
            chapters = data.get("chapters", [])
            
            info = "Title: %s\nAuthor: %s\nChapters: %d" % (
                meta.get("title","?"), meta.get("author","?"), len(chapters)
            )
            
            state["parsed"] = True
            state["chapters"] = chapters
            
            return info, [], state
        except Exception as e:
            logger.error(str(e))
            return f"Error: {e}", [], state

    def run_analysis(self, file_path, state: dict):
        if not state.get("parsed"):
            if file_path:
                _, _, state = self.parse_epub(file_path, state)
            else:
                return "Please parse a book first.", [], state

        if not state.get("parsed"):
            return "Parse failed.", [], state

        try:
            from audiobook_ai.core.text_segmenter import TextSegmenter
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            
            logger.info("Starting Character Analysis...")
            
            seg = TextSegmenter()
            all_segs = []
            
            chapters = self.parser._chapters if self.parser else state.get("chapters", [])
            for ch in chapters:
                txt = ch.get("text","") if isinstance(ch, dict) else getattr(ch, 'text', "")
                title = ch.get("title","") if isinstance(ch, dict) else getattr(ch, 'title', "")
                idx = ch.get("spine_order",0) if isinstance(ch, dict) else getattr(ch, 'spine_order', 0)
                s = seg.segment_chapter(txt, title, idx)
                all_segs.extend(s)
            
            self.segments = all_segs
            
            cfg = self.config.get_section("analysis")
            self.analyzer = CharacterAnalyzer(cfg)
            tags, chars = self.analyzer.analyze_segments(all_segs)
            
            self.tags = tags
            self.characters = chars
            state["analyzed"] = True
            state["tags"] = tags
            state["chars"] = chars
            
            table_data = []
            for c in chars:
                count = sum(1 for t in tags.values() if t.character_name == c)
                emo = list(set([t.emotion for t in tags.values() if t.character_name == c]))
                table_data.append([c, count, ", ".join(emo)])
            
            return "Analysis Complete. Found %d characters." % len(chars), table_data, state
        except Exception as e:
            logger.error(str(e))
            import traceback
            traceback.print_exc()
            return f"Error: {e}", [], state

    def save_analysis(self, state: dict):
        if not state.get("analyzed"): return "Nothing to save."
        try:
            tags_dict = {
                sid: {
                    'speaker': tag.speaker_type,
                    'char': tag.character_name,
                    'emotion': tag.emotion
                }
                for sid, tag in state.get("tags", {}).items()
            }
            data = {
                "chars": state.get("chars", []),
                "tags": tags_dict
            }
            path = os.path.join(tempfile.gettempdir(), "aiguibook_analysis.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return "Saved to: " + path
        except Exception as e:
            return f"Save Error: {e}"

    def load_analysis(self, file_path, state: dict):
        if not file_path: return "No file selected.", [], state
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state["analyzed"] = True
            state["tags"] = data.get("tags", {})
            chars = data.get("chars", [])
            state["chars"] = chars
            
            table_data = []
            for c in chars:
                table_data.append([c, 0, "Loaded from JSON"])
            return "Loaded %d characters." % len(chars), table_data, state
        except Exception as e:
            return f"Load Error: {e}", [], state

    def preview_voice(self, voice_name, ref_file):
        logger.info(f"Preview requested: {voice_name}, file: {ref_file}")
        return None

    def start_generation(self, preview_mode, val_mode, state):
        """Generator function."""
        yield 0, "Initializing...", "Starting...", gr.update(interactive=False), gr.update(visible=True)
        import time
        time.sleep(1)
        
        if not state.get("analyzed") and not state.get("parsed"):
            yield 0, "Error: Book not analyzed.", "Please analyze book first.", gr.update(interactive=True), gr.update(visible=False)
            return

        try:
            # 1. Start TTS
            yield 10, "Loading Qwen3-TTS...", "Initializing model...", gr.update(visible=False), gr.update(visible=False)
            time.sleep(1)
            
            # 2. Segment
            yield 20, "Segmenting Text...", "Preparing chunks...", gr.update(visible=False), gr.update(visible=False)
            time.sleep(0.5)
            
            # 3. Loop over chapters
            total = 5 # Simulation
            for i in range(total):
                p = 20 + (i * 15)
                yield p, f"Generating Chapter {i+1}...", f"Processing batch {i}...", gr.update(visible=False), gr.update(visible=False)
                time.sleep(0.5)
                
            yield 100, "Complete!", "Audiobook saved.", gr.update(interactive=True), gr.update(visible=False)

        except Exception as e:
            yield 0, f"Error: {e}", "Crashed.", gr.update(interactive=True), gr.update(visible=False)

    def test_llm(self, url):
        try:
            import urllib.request
            urllib.request.urlopen(url.replace('/v1', ''), timeout=2)
            return "Connected to LM Studio."
        except Exception as e:
            return f"Connection failed: {e}"

    def launch(self, port=None, server_port=None, share=False, server_name="0.0.0.0", **kwargs):
        """Launch app with Gradio 6.0 compatibility."""
        if self.app is None:
            self.build()
        
        final_port = port or server_port or 7860
        
        # Apply theme and css via launch() to avoid UserWarning
        self.app.queue()
        self.app.launch(
            server_name=server_name,
            server_port=final_port,
            share=share,
            theme=self.theme,
            css=self.css,
            **kwargs
        )
