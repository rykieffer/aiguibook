"""
AudiobookGUI v7 - Full Production Logic with VoiceDesign & VoiceClone.
Focus: Single source of truth for Qwen3-TTS API.
"""

from __future__ import annotations

import gradio as gr
import logging
import os
import tempfile
import json
import time
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AIGUIBook")

QWEN_VOICE_STYLES = {
    "ryan": "Ryan (Deep Male)",
    "aidan": "Aidan (Male)",
    "elara": "Elara (Female)",
    "bella": "Bella (Young Female)",
    "george": "George (Older Male)",
    "clara": "Clara (Warm Female)",
}

# Default Descriptions
DEFAULT_NARRATOR_DESC = "A warm, deep male voice, French accent, authoritative yet gentle."

class AudiobookGUI:
    def __init__(self, config):
        self.config = config
        self.app = None
        
        # Global State
        self._log_messages = []
        
        # Project Data
        self._epub_parser = None
        self._chapters_list = []
        self._segments = []
        self._tags = {}
        self._characters = []
        self._dedup_map = {}

        # Voice Data
        self._voice_model_variant_design = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        self._voice_model_variant_base = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self._engine = None # TTSEngine instance
        self.narrator_voice_desc = DEFAULT_NARRATOR_DESC
        self.narrator_wav_path = None
        self.narrator_ref_text = ""  # Text spoken in narrator ref audio # Either uploaded or designed
        self.character_voice_paths = {}
        self.character_ref_texts = {}  # {name: text spoken in ref audio} # { "Character Name": "path/to/audio.wav" }
        self.character_voice_descs = {} # { "Character Name": "Voice Description" }

        self._log("AIGUIBook v7 initialized.")
        # Optimization: Enable cuDNN auto-tuner for faster GPU kernels
        try:
            import torch
            torch.backends.cudnn.benchmark = True
        except: pass
        
        # Default theme/css to avoid AttributeError if launch is called before build
        import gradio as gr
        self.theme = gr.themes.Soft()
        self.css = "" 

    def _log(self, msg):
        self._log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    def _get_logs(self):
        return "\n".join(self._log_messages[-100:])

    def _get_engine(self):
        """Lazy load the TTSEngine."""
        if self._engine is None:
            from audiobook_ai.tts.qwen_engine import TTSEngine
            self._engine = TTSEngine()
        return self._engine

    def build(self):
        theme = gr.themes.Soft(primary_hue="violet", secondary_hue="blue")
        css = ".log-box textarea {font-family: monospace; font-size: 12px;}"
        
        # Store them as instance attributes for the launch() method
        self.theme = theme
        self.css = css

        with gr.Blocks(title="AIGUIBook") as self.app:
            gr.Markdown("# AIGUIBook v7\n### EPUB to Audiobook with AI Voice Design (Qwen3-TTS)")
            
            state = gr.State({"loaded": False, "parsed": False, "analyzed": False})
            
            with gr.Tabs():
                # ==========================
                # TAB 1: ANALYSIS
                # ==========================
                with gr.Tab("1. Analysis"):
                    gr.Markdown("### 1. Load EPUB and Detect Characters/Emotions")
                    with gr.Row():
                        with gr.Column():
                            file_epub = gr.File(label="Upload EPUB", file_types=[".epub"])
                            btn_parse = gr.Button("1. Parse Book", variant="primary")
                            book_info = gr.Textbox(label="Metadata", lines=4, interactive=False)
                        
                        with gr.Column():
                            btn_analyze = gr.Button("2. Analyze Characters (LM Studio)", variant="primary")
                            status_bar = gr.Textbox(label="Status", lines=2)
                            char_list_df = gr.Dataframe(label="Detected Characters", headers=["Character", "Count", "Emotions"], interactive=False)
                            
                            btn_save_json = gr.Button("3. Save Analysis JSON")
                            status_save = gr.Textbox(label="Save Status")
                            file_load_json = gr.File(label="Load Existing Analysis", file_types=[".json"])
                            btn_load_json = gr.Button("Load JSON")
                            status_load = gr.Textbox(label="Load Status")

                    file_epub.change(fn=self.parse_epub, inputs=[file_epub, state], outputs=[book_info, char_list_df, state])
                    btn_parse.click(fn=self.parse_epub, inputs=[file_epub, state], outputs=[book_info, char_list_df, state])
                    btn_analyze.click(
                        fn=self.run_analysis, inputs=[file_epub, state],
                        outputs=[status_bar, char_list_df, state]
                    )
                    btn_save_json.click(
                        fn=self.save_analysis_json, inputs=[state], outputs=[status_save]
                    )
                    file_load_json.change(fn=self.load_analysis_json, inputs=[file_load_json, state],
                        outputs=[status_load, char_list_df, state])
                    btn_load_json.click(fn=self.load_analysis_json, inputs=[file_load_json, state],
                        outputs=[status_load, char_list_df, state])

                # ==========================
                # TAB 2: VOICE DESIGN
                # ==========================
                with gr.Tab("2. Voice Design"):
                    gr.Markdown("### 2. Generate Voice References using AI")
                    gr.Markdown("Describe the voice for the Narrator and Characters. The AI will generate unique WAV files.")
                    
                    with gr.Row():
                        # --- NARRATOR ---
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("#### Voice Strategy")
                                voice_strategy_radio = gr.Radio(
                                    choices=[
                                        ("Single Narrator Mode (Voice Acting)", "single_narrator"),
                                        ("Full Ensemble Cast (Multi-Voice)", "full_ensemble")
                                    ],
                                    value="single_narrator",
                                    label="Production Mode"
                                )
                                gr.Markdown("* **Single Narrator**: Uses the narrator voice for ALL characters, but applies the detected emotions (e.g. angry, sad) for acting. Good for traditional audiobooks.*")
                                gr.Markdown("* **Full Ensemble**: Uses the specific voice designed for each character. Good for radio-play styles.*")
                                
                                gr.Markdown("#### Narrator Voice")
                                txt_narrator_desc = gr.Textbox(
                                    label="Voice Description", 
                                    value=DEFAULT_NARRATOR_DESC,
                                    lines=3
                                )
                                file_narrator_ref = gr.File(label="OR Upload Reference (WAV)", file_types=[".wav"], type="filepath")
                                btn_design_narrator = gr.Button("Design / Load Narrator", variant="primary")
                                btn_save_narrator = gr.Button("💾 Save Narrator to Library", variant="secondary")
                                status_narrator = gr.Textbox(label="Narrator Status", interactive=False)
                                audio_narrator_preview = gr.Audio(label="Narrator Preview", interactive=False)

                        # --- CHARACTERS ---
                        with gr.Column(scale=2):
                            with gr.Group():
                                gr.Markdown("#### Character Voices")
                                md_char_info = gr.Markdown("*Run Analysis in Tab 1 first to see characters.*")
                                
                                # Dynamic inputs will be rendered here in the future.
                                # For now, let's use a Dataframe for assignment.
                                df_char_voices = gr.Dataframe(
                                    headers=["Character", "Voice Description", "Action"],
                                    datatype=["str", "str", "str"],
                                    interactive=False, # We won't edit here yet, just show
                                    label="Characters"
                                )
                                
                                txt_char_desc_global = gr.Textbox(
                                    label="Global Character Description",
                                    placeholder="e.g., A young energetic male, French accent",
                                    lines=2
                                )
                                btn_design_all_chars = gr.Button("Design ALL Character Voices", variant="primary")
                                btn_save_chars = gr.Button("💾 Save All Voices to Library", variant="secondary")
                                status_chars = gr.Textbox(label="Characters Status", interactive=False)

                    btn_design_narrator.click(
                        fn=self.design_narrator, 
                        inputs=[txt_narrator_desc, file_narrator_ref], 
                        outputs=[status_narrator, audio_narrator_preview]
                    )

                    btn_design_all_chars.click(
                        fn=self.design_all_characters,
                        inputs=[txt_char_desc_global, state],
                        outputs=[status_chars, df_char_voices]
                    )
                    
                    # When analyzing, update DF
                    self._state_ref = state
                    self._df_chars_ref = df_char_voices
                    self._md_char_info_ref = md_char_info

                    voice_strategy_radio.change(
                        fn=lambda v: setattr(self, 'voice_strategy', v),
                        inputs=[voice_strategy_radio],
                        outputs=[]
                    )
                    self.voice_strategy = "single_narrator"
                # ==========================
                # TAB 3: PRODUCTION
                # ==========================
                with gr.Tab("3. Production"):
                    gr.Markdown("### 3. Generate Audiobook")
                    
                    with gr.Row():
                        with gr.Column():
                            chk_preview = gr.Checkbox(label="Preview Mode (First Chapter)", value=True)
                            silence_slider = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.75, step=0.25,
                                label="Silence Between Segments (seconds)"
                            )
                            output_dir_input = gr.Textbox(
                                label="Output Folder (leave empty for auto, or paste path to resume)",
                                placeholder="/tmp/aiguibook_gen_xxxxx",
                                lines=1
                            )
                            btn_start_prod = gr.Button("START GENERATION", variant="primary", size="lg")
                            btn_resume_prod = gr.Button("RESUME from Folder", variant="secondary", size="lg")
                        
                        with gr.Column():
                            progress = gr.Slider(value=0, label="Progress")
                            phase = gr.Textbox(label="Current Phase", value="Ready")
                            logs = gr.Textbox(label="System Log", lines=10, elem_classes=["log-box"])
                            m4a_out_prod = gr.File(label="Final Audiobook (M4A)", interactive=False)

                    btn_start_prod.click(
                        fn=self.start_generation,
                        inputs=[chk_preview, silence_slider, output_dir_input, state],
                        outputs=[progress, phase, logs, m4a_out_prod]
                    )
                    btn_resume_prod.click(
                        fn=self.resume_generation,
                        inputs=[silence_slider, output_dir_input, state],
                        outputs=[progress, phase, logs, m4a_out_prod]
                    )

                    # Tab 3 Selection: update state
                    # We will assume if they go to tab 3, we need to know the analysis state is there.
                    # State handles this.

            # Helper: When analysis finishes, update the Voice Tab
            # We need to hook this properly.

        return self.app

    # --- Tab 1 Handlers ---
    def parse_epub(self, file_epub, state):
        # Ensure state exists
        if not state: state = {}
        
        if not file_epub:
            return "No file selected.", [], state
        
        self._log(f"Parsing EPUB: {os.path.basename(file_epub)}")
        try:
            from audiobook_ai.core.epub_parser import EPUBParser
            parser = EPUBParser(file_epub)
            data = parser.parse()
            self._epub_parser = parser
            self._chapters_list = data.get("chapters", [])
            meta = data.get("metadata", {})
            
            # UPDATE STATE: Don't create a new one!
            state["parsed"] = True
            state["epub_path"] = file_epub  # Save path for Tab 3
            state["meta"] = meta
            
            # Don't wipe analyzed status if it was already true
            if "analyzed" not in state: state["analyzed"] = False
            
            info = f"Title: {meta.get('title', '?')}\nAuthor: {meta.get('author', '?')}\nChapters: {len(self._chapters_list)}"
            self._log(f"Parsed {info}")
            return info, [], state
        except Exception as e:
            self._log(f"Parse Error: {e}")
            return f"Error: {e}", [], state

    def run_analysis(self, file_epub, state):
        """Run the full character analysis pipeline with live progress."""
        table_data = []
        
        try:
            # Ensure state exists
            if not state:
                state = {"parsed": False, "analyzed": False}
            
            # Parse if needed
            if not state.get("parsed"):
                if file_epub:
                    info, _, state = self.parse_epub(file_epub, state)
                    if not state.get("parsed"):
                        yield "Parse failed.", [], state
                        return
                else:
                    yield "Please upload a book first.", [], state
                    return
            
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
            
            # LIVE PROGRESS LOOP: Safe iteration over analyzer generator
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
            
            yield "Building results table...", [], state
            
            # Build Dataframe data with counts and emotions
            for c in chars:
                count = sum(1 for t in tags.values() if t.character_name == c)
                emo = list(set([t.emotion for t in tags.values() if t.character_name == c]))
                table_data.append([c, count, ", ".join(sorted(emo))])
            
            yield f"Analysis Complete! Found {len(chars)} characters.", table_data, state
            
        except Exception as e:
            import traceback
            self._log(f"Analysis error: {e}\n{traceback.format_exc()}")
            yield f"Error: {e}", table_data, state


    def save_analysis_json(self, state):
        """Save analysis to JSON for future use."""
        if not state.get("analyzed"):
            return "No analysis data to save. Run analysis first."
        try:
            tags_dict = {
                sid: {
                    'speaker': tag.speaker_type,
                    'char': tag.character_name,
                    'emotion': tag.emotion,
                    'text': getattr(tag, 'text', '')
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
            return f"Saved to: {path}"
        except Exception as e:
            return f"Save Error: {e}"

    def load_analysis_json(self, file_load_json, state):
        if not state: state = {}
        try:
            import json
            
            # Handle path input safely
            path = ""
            if isinstance(file_load_json, str):
                path = file_load_json
            elif isinstance(file_load_json, dict):
                path = file_load_json.get("name") or file_load_json.get("path")
            
            if not path:
                self._log("No file path received.")
                return "No file selected", [], state
            
            self._log(f"Loading JSON: {path}")
            
            with open(path, "r") as f:
                data = json.load(f)
            
            # Extract characters
            chars = data.get("chars", [])
            
            # Fallback: extract from tags if chars list is missing
            if not chars:
                tags = data.get("tags", {})
                unique_chars = set()
                for t_data in tags.values():
                    # Check both key names
                    c = t_data.get("char") or t_data.get("character_name")
                    if c:
                        unique_chars.add(c)
                chars = sorted(list(unique_chars))
            
            self._characters = chars
            state["analyzed"] = True
            state["chars"] = chars
            
            # Store tags
            raw_tags = data.get("tags", {})
            self._tags = raw_tags
            state["tags"] = raw_tags
            
            # Build Dataframe Data: [[Name (str), Count (int), Emotions (str)]]
            df_data = []
            for char_name in chars:
                if not char_name: continue
                count = 0
                emotions = set()
                for sid, t_data in raw_tags.items():
                    c_name = t_data.get("char") or t_data.get("character_name")
                    if c_name == char_name:
                        count += 1
                        emo = t_data.get("emotion")
                        if emo: emotions.add(emo)
                
                df_data.append([str(char_name), int(count), ", ".join(sorted(list(emotions)))])
            
            # Ensure we return a valid Dataframe structure even if logic fails
            if not df_data and chars:
                df_data = [[str(c), 0, ""] for c in chars]
            
            self._log(f"Loaded {len(chars)} characters. Dataframe rows: {len(df_data)}.")
            self._log(f"First row preview: {df_data[0] if df_data else 'None'}")
            
            return f"Loaded {len(chars)} characters.", df_data, state

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._log(f"Error loading JSON: {e}\n{tb}")
            return f"Error: {e}", [], state

    def design_narrator(self, desc_text, ref_wav):
        """
        Design the narrator voice.
        If ref_wav is provided, use it. Otherwise use Description.
        """
        self.narrator_voice_desc = desc_text
        
        # If user uploaded a file, use it
        if ref_wav:
            self.narrator_wav_path = ref_wav
            self.narrator_ref_text = desc_text  # Use description as approximate ref_text
            return f"Using uploaded file", ref_wav

        engine = self._get_engine()
        try:
            # Load VoiceDesign Model
            engine.load_model(self._voice_model_variant_design)
            
            out_path = os.path.join(tempfile.gettempdir(), "narrator_voice.wav")
            # Longer, more expressive text (~12 seconds)
            test_text = "Bonjour et bienvenue. Je suis votre narrateur pour ce livre. Je vais vous guider à travers chaque chapitre avec une voix claire et expressive, en adaptant le ton selon les scènes et les émotions du récit."
            
            res_path = engine.design_voice(
                text=test_text,
                instruction=self.narrator_voice_desc,
                language="french",
                output_path=out_path
            )
            
            if res_path:
                self.narrator_wav_path = res_path
                self.narrator_ref_text = test_text  # Save the text we used for the ref audio
                # Unload Design model
                engine.unload_model()
                return f"Voice designed successfully!", res_path
            else:
                engine.unload_model()
                return "Voice design failed.", None
        except Exception as e:
            engine.unload_model()
            return f"Error: {e}", None

    def save_narrator_voice(self):
        """Save narrator voice to permanent library."""
        import shutil
        if not self.narrator_wav_path or not os.path.exists(self.narrator_wav_path):
            return "Error: No narrator voice generated to save."
        
        lib_dir = os.path.join(os.path.expanduser("~"), "audiobooks", "voices", "narrator")
        os.makedirs(lib_dir, exist_ok=True)
        
        # Generate persistent filename
        fname = f"narrator_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        dest_path = os.path.join(lib_dir, fname)
        
        shutil.copy2(self.narrator_wav_path, dest_path)
        self._log(f"Saved narrator voice to {dest_path}")
        return f"Saved to: {dest_path}"

    def save_all_character_voices(self):
        """Save all character voices to permanent library."""
        import shutil
        if not self.character_voice_paths:
            return "Error: No character voices generated to save."
        
        lib_dir = os.path.join(os.path.expanduser("~"), "audiobooks", "voices", "characters")
        os.makedirs(lib_dir, exist_ok=True)
        
        saved_count = 0
        for char_name, wav_path in self.character_voice_paths.items():
            if os.path.exists(wav_path):
                clean_name = char_name.replace(" ", "_").lower()
                fname = f"{clean_name}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
                dest_path = os.path.join(lib_dir, fname)
                shutil.copy2(wav_path, dest_path)
                saved_count += 1
        
        msg = f"Saved {saved_count} voices to: {lib_dir}"
        self._log(msg)
        return msg

    def design_all_characters(self, global_desc, state):
        if not state.get("analyzed"):
            return "Run Analysis first.", []
        
        chars = self._characters
        engine = self._get_engine()
        
        self._log(f"Designing voices for {len(chars)} characters...")
        
        try:
            engine.load_model(self._voice_model_variant_design)
            
            for char_name in chars:
                if char_name == "Narrator": continue
                
                # Use global desc or specific if available
                desc = self.character_voice_descs.get(char_name, global_desc)
                if not desc: continue
                
                self._log(f"Designing voice for: {char_name}")
                out_path = os.path.join(tempfile.gettempdir(), f"char_{char_name.replace(' ', '_')}.wav")
                
                # If already exists, skip
                if os.path.exists(out_path):
                    self.character_voice_paths[char_name] = out_path
                    self._log(f"Skipping {char_name} (already exists)")
                    continue

                char_test_text = f"Bonjour, je suis {char_name}. Comment allez-vous aujourd'hui? Je suis ravi de faire votre connaissance."
                res_path = engine.design_voice(
                    text=char_test_text,
                    instruction=desc,
                    language="french",
                    output_path=out_path
                )
                
                if res_path:
                    self.character_voice_paths[char_name] = res_path
                    self.character_ref_texts[char_name] = char_test_text  # Save ref text
                    self._log(f"Voice created for {char_name}.")
            
            engine.unload_model()
            self._log("All voices designed.")
            
            # Return status and updated Dataframe
            status = f"Designed voices for {len(self.character_voice_paths)} characters."
            
            # Prepare Dataframe
            df_data = []
            for char in chars:
                path = self.character_voice_paths.get(char, self.character_voice_descs.get(char, "Pending"))
                df_data.append([char, path, "Done" if path else "Pending"])
            
            return status, df_data
            
        except Exception as e:
            engine.unload_model()
            self._log(f"Batch Design Error: {e}")
            return f"Error: {e}", []

    # --- Tab 3 Handlers (Production) ---
    def _normalize_tags(self, state):
        """Ensure tags are plain dicts, not SpeechTag objects."""
        tags = state.get("tags", {}) if state else {}
        if not tags:
            return {}
        first_val = next(iter(tags.values()), None)
        if first_val is None:
            return {}
        if hasattr(first_val, 'emotion'):
            # Convert SpeechTag objects to dicts
            self._log("Normalizing tags from Objects to Dictionaries...")
            normalized = {}
            for sid, tag in tags.items():
                normalized[sid] = {
                    "speaker": getattr(tag, 'speaker_type', 'narrator'),
                    "char": getattr(tag, 'character_name', None),
                    "emotion": getattr(tag, 'emotion', 'neutral'),
                    "emotion_instruction": getattr(tag, 'emotion_instruction', ""),
                    "text": getattr(tag, 'text', ""),
                }
            return normalized
        return dict(tags)

    def _build_segments_from_tags(self):
        """Build a simple segment list from tags (no EPUB re-parse needed)."""
        segs = []
        for sid in sorted(self._tags.keys()):
            segs.append({"id": sid, "text": self._tags[sid].get("text", "")})
        return segs

    def _generate_loop(self, all_segs, output_dir, silence_duration, skip_existing=False):
        """Core generation loop. Yields progress updates."""
        engine = self._get_engine()
        
        # Count how many need generation
        total = len(all_segs)
        if skip_existing:
            already_done = sum(
                1 for s in all_segs
                if os.path.exists(os.path.join(output_dir, f"{s['id'] if isinstance(s, dict) else s.id}.wav"))
            )
            self._log(f"Resume: {already_done}/{total} segments already exist, skipping those.")
            yield 5, f"Resuming: {already_done} already done", self._get_logs(), None
        else:
            already_done = 0
        
        # Load Base Model
        self._log("Loading Base Model (this takes time)...")
        yield 8, "Loading Base Model...", self._get_logs(), None
        engine.load_model(self._voice_model_variant_base)
        
        generated_count = already_done
        failed_count = 0

        for i, seg in enumerate(all_segs):
            seg_id = seg.id if hasattr(seg, 'id') else seg.get("id", "")
            out_path = os.path.join(output_dir, f"{seg_id}.wav")
            
            # Skip if already generated (resume support)
            if skip_existing and os.path.exists(out_path):
                continue
            
            # Determine tag data
            tag_data = self._tags.get(seg_id, {})
            char_name = tag_data.get("char") or tag_data.get("character_name") or "Narrator"
            emotion = tag_data.get("emotion", "calm")
            
            if char_name not in self._characters and char_name != "Narrator":
                char_name = "Narrator"
            
            # Determine reference audio
            strategy = getattr(self, 'voice_strategy', 'single_narrator')
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
            
            # Get text
            text = seg.text if hasattr(seg, 'text') else seg.get("text", "")
            if not text.strip():
                # Try to get from tags
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
                    output_path=out_path
                )
                if gen_path:
                    generated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                self._log(f"Failed: {seg_id}: {e}")
                failed_count += 1
            
            # Yield every 5 segments
            if (i + 1) % 5 == 0 or i == total - 1:
                pct = 10 + (generated_count / total * 80)
                yield pct, f"Generated {generated_count}/{total} ({failed_count} failed)", self._get_logs(), None
        
        engine.unload_model()
        self._log(f"Generation complete: {generated_count} OK, {failed_count} failed")
        
        # --- ASSEMBLE M4A ---
        self._log("Assembling final audiobook...")
        yield 92, "Assembling M4A...", self._get_logs(), None
        
        # Collect all WAV files in order
        wav_files = []
        for seg in all_segs:
            seg_id = seg.id if hasattr(seg, 'id') else seg.get("id", "")
            wav_path = os.path.join(output_dir, f"{seg_id}.wav")
            if os.path.exists(wav_path):
                wav_files.append(wav_path)
        
        if not wav_files:
            yield 0, "Error: No WAV files generated.", self._get_logs(), None
            return
        
        # Determine book title
        book_title = "Audiobook"
        if hasattr(self, '_epub_parser') and self._epub_parser:
            try:
                data = self._epub_parser.parse() if not self._chapters_list else {"metadata": {}}
                meta = data.get("metadata", {})
                book_title = meta.get("title", "Audiobook")
            except:
                pass
        
        # Build chapter titles from segment chapter prefixes
        chapter_titles = []
        seen_chapters = set()
        for seg in all_segs:
            seg_id = seg.id if hasattr(seg, 'id') else seg.get("id", "")
            ch_prefix = seg_id.split("_")[0]  # e.g. "ch0"
            if ch_prefix not in seen_chapters:
                seen_chapters.add(ch_prefix)
                ch_idx = int(ch_prefix.replace("ch", ""))
                if ch_idx < len(self._chapters_list):
                    ch = self._chapters_list[ch_idx]
                    title = ch.get("title", f"Chapter {ch_idx+1}") if isinstance(ch, dict) else getattr(ch, "title", f"Chapter {ch_idx+1}")
                    chapter_titles.append(title)
                else:
                    chapter_titles.append(f"Chapter {ch_idx+1}")
        
        m4a_path = os.path.join(output_dir, f"{book_title.replace(' ', '_')}.m4a")
        
        from audiobook_ai.tts.qwen_engine import TTSEngine
        try:
            result_path = TTSEngine.assemble_wav_files(
                wav_files=wav_files,
                output_path=m4a_path,
                silence_duration=silence_duration,
                sample_rate=24000,
                normalize=True,
                book_title=book_title,
                chapter_titles=chapter_titles,
            )
            self._log(f"Audiobook saved: {result_path}")
            yield 100, f"Done! {result_path}", self._get_logs(), result_path
        except Exception as e:
            self._log(f"Assembly error: {e}")
            yield 95, f"Assembly failed: {e}", self._get_logs(), None

    def start_generation(self, preview_mode, silence_duration, output_dir_text, state):
        """Generate the audiobook from scratch."""
        if not state or not state.get("analyzed"):
            yield 0, "Error: Run Analysis first in Tab 1.", self._get_logs(), None
            return
        
        if not self.narrator_wav_path:
            yield 0, "Error: Design narrator voice in Tab 2 first.", self._get_logs(), None
            return
        
        self._log("Starting Production Pipeline...")
        self._tags = self._normalize_tags(state)
        self._characters = state.get("chars", [])
        
        # Determine output directory
        if output_dir_text and os.path.isdir(output_dir_text):
            output_dir = output_dir_text
            self._log(f"Using existing output folder: {output_dir}")
        else:
            output_dir = tempfile.mkdtemp(prefix="aiguibook_gen_")
            self._log(f"Created output folder: {output_dir}")
        
        yield 2, "Preparing segments...", self._get_logs(), None
        
        try:
            # Build segments from tags (text is embedded in JSON)
            all_segs = self._build_segments_from_tags()
            
            if not all_segs:
                # Fallback: re-parse EPUB
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
            
            # Preview mode: limit to first chapter
            if preview_mode:
                first_ch = [s for s in all_segs if (s.id if hasattr(s, 'id') else s.get("id", "")).startswith("ch0")]
                if first_ch:
                    all_segs = first_ch
                    self._log(f"PREVIEW MODE: limited to {len(all_segs)} segments")
            
            self._log(f"Total segments: {len(all_segs)}")
            
            # Run generation loop
            for progress, phase, logs, m4a_file in self._generate_loop(all_segs, output_dir, silence_duration, skip_existing=False):
                yield progress, phase, logs, m4a_file
                
        except Exception as e:
            self._log(f"Fatal Error: {e}")
            yield 0, f"Error: {e}", self._get_logs(), None

    def resume_generation(self, silence_duration, output_dir_text, state):
        """Resume generation from an existing output folder."""
        if not state or not state.get("analyzed"):
            yield 0, "Error: Run Analysis first in Tab 1.", self._get_logs(), None
            return
        
        if not output_dir_text or not os.path.isdir(output_dir_text):
            yield 0, "Error: Paste the output folder path to resume.", self._get_logs(), None
            return
        
        self._tags = self._normalize_tags(state)
        self._characters = state.get("chars", [])
        output_dir = output_dir_text
        
        self._log(f"Resuming generation from: {output_dir}")
        
        # Count existing WAVs
        existing_wavs = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
        self._log(f"Found {len(existing_wavs)} existing WAV files in folder")
        
        yield 2, f"Found {len(existing_wavs)} existing files. Rebuilding segment list...", self._get_logs(), None
        
        try:
            # Rebuild segment list from tags
            all_segs = self._build_segments_from_tags()
            
            if not all_segs:
                yield 0, "Error: No segment data in tags. Load the analysis JSON first.", self._get_logs(), None
                return
            
            self._log(f"Total segments: {len(all_segs)}")
            
            # Run with skip_existing=True
            for progress, phase, logs, m4a_file in self._generate_loop(all_segs, output_dir, silence_duration, skip_existing=True):
                yield progress, phase, logs, m4a_file
                
        except Exception as e:
            self._log(f"Resume Error: {e}")
            yield 0, f"Error: {e}", self._get_logs(), None

    def launch(self, port=7860, share=False, server_name="0.0.0.0"):
        if self.app is None: self.build()
        self.app.queue()
        
        # Move theme and css to launch() to avoid Gradio 6.0 warning
        theme = self.theme
        css = self.css
        
        self.app.launch(
            server_name=server_name,
            server_port=port, 
            share=share,
            theme=theme,
            css=css
        )
