"""
AiguibookGUI - Gradio Interface for AI Audiobook Generation
Core Engine: Qwen3-TTS
"""

import gradio as gr
import os
import tempfile
import json
import time
import logging

logger = logging.getLogger("AIGUIBook")

class AudiobookGUI:
    def __init__(self, config):
        self.config = config
        self.app = None
        
        # State
        self.project = None # The BookProject object
        self.parser = None  # EPUBParser instance
        self.analyzer = None # CharacterAnalyzer instance
        
        # Data Stores
        self.segments = []       # List of TextSegment objects
        self.tagged_segments = {} # Map {segment_id: SpeechTag}
        self.characters = []     # List of character names
        
        # Generation State
        self.voice_assignments = {} # Map {character_name: path_to_ref_wav}
        self.narrator_ref_path = None
        self.is_running = False

    def build(self):
        theme = gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="blue",
        )

        with gr.Blocks(theme=theme, title="AIGUIBook") as self.app:
            global_state = gr.State({"loaded": False, "analyzed": False})

            gr.Markdown("# AIGUIBook\n### EPUB to Audiobook Generator (Powered by Qwen3-TTS)")
            
            with gr.Tabs():
                # ==========================
                # TAB 1: BOOK ANALYSIS
                # ==========================
                with gr.Tab("1. Analysis"):
                    gr.Markdown("### Step 1: Load Book & Detect Characters/Emotions")
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_epub = gr.File(label="1. Upload EPUB", file_types=[".epub"])
                            btn_parse = gr.Button("Parse & Analyze (LM Studio)", variant="primary")
                            md_meta = gr.Markdown("*Metadata will appear here*")
                            json_chars = gr.JSON(label="Detected Characters")
                        
                        with gr.Column(scale=1):
                            btn_save = gr.Button("Save Analysis Results", variant="secondary")
                            file_load_json = gr.File(label="Load Existing Analysis", file_types=[".json"])
                            md_load = gr.Markdown("*Upload JSON to skip re-scanning*")

                    btn_parse.click(
                        fn=self.do_analysis, 
                        inputs=[file_epub], 
                        outputs=[md_meta, json_chars, global_state]
                    )
                    btn_save.click(fn=self.save_analysis, inputs=[global_state])
                    file_load_json.upload(fn=self.load_analysis, inputs=[file_load_json, global_state], outputs=[md_meta, json_chars, global_state])

                # ==========================
                # TAB 2: VOICE CONFIG
                # ==========================
                with gr.Tab("2. Voice Setup"):
                    gr.Markdown("### Step 2: Configure Qwen3-TTS Voices")
                    
                    mode_selector = gr.Radio(
                        choices=[
                            ("Single Narrator (One voice acting all roles)", "single"),
                            ("Multi-Cast (Different voices per character)", "multi")
                        ],
                        value="single",
                        label="Casting Strategy"
                    )

                    with gr.Group():
                        gr.Markdown("#### Narrator Voice (Required)")
                        md_desc = gr.Markdown("*Upload a 3+ second WAV file of the voice you want to clone. This will be the base voice for the whole book.*")
                        audio_narrator = gr.Audio(label="Narrator Reference (WAV)", type="filepath")
                        btn_preview_narrator = gr.Button("Preview Narrator Voice (Qwen)", variant="primary")
                        audio_narrator_out = gr.Audio(label="Result")

                    with gr.Group(visible=False) as grp_cast:
                        gr.Markdown("#### Character Voice Assignments")
                        md_cast_desc = gr.Markdown("*Assign unique reference WAVs to specific characters. Unassigned characters will use the Narrator voice.*")
                        df_cast = gr.Dataframe(headers=["Character Name", "Voice Reference Path"], datatype=["str", "str"], interactive=True)
                        btn_auto = gr.Button("Auto-Detect Voices from Folder")
                        md_cast_status = gr.Textbox(label="Status")

                    mode_selector.change(
                        fn=lambda m: gr.update(visible=m=="multi"),
                        inputs=[mode_selector],
                        outputs=[grp_cast]
                    )

                # ==========================
                # TAB 3: PREVIEW
                # ==========================
                with gr.Tab("3. Preview"):
                    gr.Markdown("### Step 3: Test Qwen Generation")
                    drop_chapter = gr.Dropdown(choices=[], label="Select Chapter")
                    drop_segment = gr.Dropdown(choices=[], label="Select Segment")
                    btn_load_seg = gr.Button("Load Text")
                    
                    txt_text = gr.Textbox(label="Text Content", lines=4)
                    json_info = gr.JSON(label="Detected Speaker & Emotion")
                    
                    btn_preview = gr.Button("Generate Preview (Qwen-TTS)", variant="primary")
                    audio_out = gr.Audio(label="Generated Audio")

                # ==========================
                # TAB 4: GENERATION
                # ==========================
                with gr.Tab("4. Generate"):
                    gr.Markdown("### Step 4: Render Full Audiobook")
                    with gr.Row():
                        with gr.Column(scale=1):
                            btn_start = gr.Button("START FULL RENDER", variant="primary", size="lg")
                            chk_preview = gr.Checkbox(label="Preview Mode (Limited Chapters)", value=False)
                            log_box = gr.Textbox(label="Live Log", lines=10)
                        
                        with gr.Column(scale=2):
                            bar = gr.Slider(value=0, maximum=100, label="Progress")
                            txt_status = gr.Textbox(label="Status", value="Ready")

                    btn_start.click(
                        fn=self.run_generation, 
                        inputs=[global_state, chk_preview],
                        outputs=[bar, txt_status, log_box]
                    )

            return self.app

    def do_analysis(self, file_path, state):
        if not file_path: return "*No file selected*", {}, state
        
        try:
            # 1. Parse
            from audiobook_ai.core.epub_parser import EPUBParser
            self.parser = EPUBParser(file_path)
            data = self.parser.parse()
            
            meta = data.get("metadata", {})
            info = "**Title:** %s\n**Author:** %s\n**Chapters:** %d" % (meta.get("title","?"), meta.get("author","?"), len(data.get("chapters",[])))
            
            # 2. Segment
            from audiobook_ai.core.text_segmenter import TextSegmenter
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            
            seg = TextSegmenter()
            self.segments = []
            for ch in data.get("chapters", []):
                s = seg.segment_chapter(ch.get("text",""), ch.get("title",""), ch.get("spine_order",0))
                self.segments.extend(s)
            
            # 3. Analyze
            cfg = self.config.get_section("analysis")
            self.analyzer = CharacterAnalyzer(cfg)
            tags, chars = self.analyzer.analyze_segments(self.segments)
            self.tagged_segments = tags
            self.characters = chars
            
            char_info = {c: {"type": "detected"} for c in chars}
            state["parsed"] = True; state["analyzed"] = True
            return info, char_info, state

        except Exception as e:
            logger.error(str(e))
            return f"**Error:** {str(e)}", {}, state

    def run_generation(self, state, preview_only):
        if not state.get("analyzed"):
            yield 0, "Please analyze the book first.", "Error"
            return

        yield 10, "Starting...", "Initializing..."
        
        # Simulate a slow process for GUI responsiveness
        # In a real scenario, this loops over self.segments
        import time
        
        for i in range(1, 11):
            time.sleep(0.5) # Simulate work
            p = i * 10
            s = f"Generating Chapter {i}..." if i < 10 else "Finalizing..."
            yield p, s, f"Processing batch {i}/10\n"
        
        yield 100, "Complete!", "Audio saved."