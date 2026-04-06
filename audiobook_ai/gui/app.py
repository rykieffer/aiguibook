"""
AudiobookGUI v6 - Critical Fixes for Gradio Argument Type Mismatches.
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
        self._log_messages = []  # Ensure it's initialized
        self.config = config
        self.app = None
        
        self.parser = None
        self._last_epub_path = None
        self.analyzer = None
        self.segments = []
        self.tags = {}
        self.characters = []
        self.narrator_ref_path = None


    def _log(self, msg):
        """Add a log message to internal list and logger."""
        import time
        import logging
        logger = logging.getLogger("AIGUIBook")
        logger.info(msg)
        
        if not hasattr(self, '_log_messages'):
            self._log_messages = []
        self._log_messages.append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))

    def build(self):
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
                # TAB 2: VOICE STRATEGY
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
            # EVENTS (Inputs and Outputs strictly aligned)
            # ==========================
            
            # 1. Analysis
            # parse_epub outputs: [book_info (text), char_table (list), state (dict)]
            btn_parse.click(
                fn=self.parse_epub, 
                inputs=[file_epub, state], 
                outputs=[book_info, char_table, state]
            )
            
            # run_analysis outputs: [status_bar (text), char_table (list), state (dict)]
            # Note: We do NOT yield progress integers to avoid TypeErrors in Dataframe
            btn_run_char.click(
                fn=self.run_analysis, 
                inputs=[file_epub, state], 
                outputs=[status_bar, char_table, state]
            )
            
            btn_save.click(
                fn=self.save_analysis, 
                inputs=[state], 
                outputs=[status_load] # Just text
            )
            file_json.change(
                fn=self.load_analysis, 
                inputs=[file_json, state], 
                outputs=[status_load, char_table, state]
            )

            # 2. Voice
            btn_preview.click(
                fn=self.preview_voice, 
                inputs=[builtin_select, custom_ref], 
                outputs=[preview_audio]
            )

            # 3. Generation
            btn_start.click(
                fn=self.start_generation, 
                inputs=[chk_preview, chk_val, state], 
                outputs=[progress, phase, logs, btn_start, btn_resume]
            )

            # 4. Settings
            test_btn.click(
                fn=self.test_llm, 
                inputs=[llm_url], 
                outputs=[test_out]
            )

        return self.app

    # --- Logic Methods ---

    def parse_epub(self, file_path, state: dict):
        """Returns: (info_string: str, table_data: list, state: dict)"""
        # Ensure return types are correct to avoid Gradio crashes
        
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
            # Save chapters to state for analysis later
            state["chapters"] = chapters
            
            # Return must match outputs: [Textbox, Dataframe, State]
            return info, [], state
            
        except Exception as e:
            logger.error(str(e))
            return f"**Error:** {str(e)}", [], state

    def run_analysis(self, file_path, state: dict):
        table_data = []
        try:
            if not state.get("parsed"):
                if file_path: info, _, state = self.parse_epub(file_path, state)
                else: yield "Please upload a book first.", [], state; return
            if not state.get("parsed"): yield "Parse failed.", [], state; return

            yield "Segmenting text...", [], state
            from audiobook_ai.core.text_segmenter import TextSegmenter
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            
            seg = TextSegmenter()
            all_segs = []
            chapters = self.parser._chapters if self.parser else state.get("chapters", [])
            for ch in chapters:
                txt = ch.get("text","") if isinstance(ch, dict) else getattr(ch, 'text', "")
                title = ch.get("title","") if isinstance(ch, dict) else getattr(ch, 'title', "")
                idx = ch.get("spine_order",0) if isinstance(ch, dict) else getattr(ch, 'spine_order', 0)
                all_segs.extend(seg.segment_chapter(txt, title, idx))
            
            self.segments = all_segs
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
            self.characters = chars
            self.dedup_map = _dedup_map
            state["analyzed"] = True
            state["tags"] = tags
            state["chars"] = chars
            
            yield "Building results table...", [], state
            for c in chars:
                count = sum(1 for t in tags.values() if t.character_name == c)
                emo = list(set([t.emotion for t in tags.values() if t.character_name == c]))
                table_data.append([c, count, ", ".join(emo)])
            
            yield f"Analysis Complete! Found {len(chars)} characters.", table_data, state
        except Exception as e:
            yield f"Error: {e}", table_data, state
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
            
            # CONVERSION STEP: Convert dict tags to SpeechTag objects
            tags_objects = {}
            from audiobook_ai.analysis.character_analyzer import SpeechTag, EMOTION_INSTRUCTIONS_FR
            
            for sid, t in tags_dict.items():
                speaker = t.get("speaker", "narrator")
                char_name = t.get("char")
                emo = t.get("emotion", "neutral")
                instr = EMOTION_INSTRUCTIONS_FR.get(emo, EMOTION_INSTRUCTIONS_FR["neutral"])
                
                tags_objects[sid] = SpeechTag(
                    segment_id=sid, speaker_type=speaker, 
                    character_name=char_name, emotion=emo, 
                    voice_id="narrator", emotion_instruction=instr
                )

            state["tags"] = tags_objects # CRITICAL: Store objects so generation works
            
            # Recalculate Table Data for UI
            table_data = []
            for c in chars:
                # Count tags matching this character
                c_tags = [t for t in tags_objects.values() if t.character_name == c]
                count = len(c_tags)
                emos = list(set([t.emotion for t in c_tags if t.emotion]))
                table_data.append([c, count, ", ".join(emos)])
            
            return "Loaded %d characters and %d tags." % (len(chars), len(tags_objects)), table_data, state
        except Exception as e:
            return f"Load Error: {e}", [], state

    def preview_voice(self, voice_name, ref_file):
        logger.info(f"Preview requested: {voice_name}, file: {ref_file}")
        return None

    def start_generation(self, preview_mode, val_mode, state):
        """Full audiobook generation pipeline with verbose logging."""
        import time
        import threading

        # Initialize TTS engine from config
        from audiobook_ai.tts.qwen_engine import TTSEngine
        from audiobook_ai.core.epub_parser import EPUBParser
        from audiobook_ai.core.text_segmenter import TextSegmenter
        from audiobook_ai.audio.validation import WhisperValidator
        from audiobook_ai.audio.assembly import AudioAssembly

        self._gen_running = True
        self._gen_cancelled = False
        self._gen_paused = False

        log = []
        def add_log(msg):
            log.append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
            self._log(msg)

        yield 0, "Starting...", "\n".join(log), gr.update(interactive=False), gr.update(visible=True)

        try:
            # Check state
            if not state.get("analyzed") and not state.get("tags"):
                yield 0, "Error: No analysis data. Run analysis first.", "\n".join(log), gr.update(interactive=True), gr.update(visible=False)
                return

            add_log("Generation started")
            yield 5, "Loading analysis data...", "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            # 1. Get tags and dedup map
            tags = state.get("tags", {})
            chars = state.get("chars", [])
            dedup_map = state.get("dedup_map", {})

            # Convert dict tags to SpeechTag objects
            from audiobook_ai.analysis.character_analyzer import SpeechTag
            self.tags = {}
            for sid, t in tags.items():
                # Handle both SpeechTag objects and dict (robustness)
                if hasattr(t, 'emotion'):
                    # It is a SpeechTag object
                    speaker = t.speaker_type
                    char_name = t.character_name
                    emo = t.emotion
                    instr = t.emotion_instruction
                else:
                    # It is a Dict (legacy/JSON fallback)
                    speaker = t.get("speaker", "narrator")
                    char_name = t.get("char")
                    emo = t.get("emotion", "neutral")
                    instr = EMOTION_INSTRUCTIONS_FR.get(emo, EMOTION_INSTRUCTIONS_FR["neutral"])
                # Map emotion instruction
                from audiobook_ai.analysis.character_analyzer import EMOTION_INSTRUCTIONS_FR
                instr = EMOTION_INSTRUCTIONS_FR.get(emo, EMOTION_INSTRUCTIONS_FR["neutral"])
                self.tags[sid] = SpeechTag(
                    segment_id=sid, speaker_type=speaker, character_name=char_name,
                    emotion=emo, voice_id="narrator", emotion_instruction=instr,
                )

            add_log("Loaded %d segment tags, %d characters" % (len(self.tags), len(chars)))
            yield 10, "Analysis loaded", "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            # 2. Load the EPUB and re-segment
            add_log("Loading EPUB for text extraction...")
            yield 15, "Re-loading EPUB...", "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            epub_path = self._last_epub_path if hasattr(self, '_last_epub_path') else None
            if not epub_path:
                add_log("ERROR: No EPUB path available. Please re-parse the book.")
                yield 0, "Error: Re-parse book first.", "\n".join(log), gr.update(interactive=True), gr.update(visible=False)
                return

            parser = EPUBParser(epub_path)
            parser.parse()
            add_log("EPUB parsed: %d chapters" % len(parser._chapters))

            seg = TextSegmenter(max_words=150, min_words=20)
            chapters_with_text = {}  # {chapter_idx: [TextSegment, ...]}
            total_segments = 0

            for ch in parser._chapters:
                idx = getattr(ch, 'spine_order', 0)
                text = getattr(ch, 'text', '')
                title = getattr(ch, 'title', 'Chapter %d' % (idx+1))
                segs = seg.segment_chapter(text, title, idx)
                if segs:
                    chapters_with_text[idx] = segs
                    total_segments += len(segs)

            add_log("Book segmented into %d audio segments" % total_segments)
            yield 20, "Segmented: %d segments" % total_segments, "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            # 3. Initialize Qwen3-TTS engine
            add_log("Initializing Qwen3-TTS engine...")
            add_log("NOTE: If model not cached, it will download (this may take several minutes)")
            yield 25, "Initializing TTS engine...", "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            tts_config = self.config.get_section("tts")
            self.tts_engine = TTSEngine(
                model_path=tts_config.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                device=tts_config.get("device", "cuda"),
                dtype=tts_config.get("dtype", "bfloat16"),
                batch_size=tts_config.get("batch_size", 4),
            )

            def tts_progress(pct, status):
                add_log("  TTS: %s" % status)

            self.tts_engine.initialize()
            add_log("Qwen3-TTS engine initialized successfully")
            yield 30, "TTS Ready", "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            # 4. Setup output directory
            work_dir = os.path.join("/tmp", "aiguibook_gen")
            output_dir = os.path.join(os.path.expanduser("~"), "audiobooks")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Create book project
            from audiobook_ai.core.project import BookProject
            book_title = parser._metadata.get("title", "Untitled")
            self._project = BookProject(
                book_title=book_title, work_dir=work_dir, output_dir=output_dir,
            )
            self._project.create()
            self._project.book_metadata = parser._metadata
            self._project.total_chapters = len(parser._chapters)

            # 5. Generate audio per segment
            add_log("Starting audio generation for %d segments..." % total_segments)
            generated_count = 0
            failed_count = 0
            start_time = time.time()
            seg_audio_paths = {}  # {chapter_idx: [(seg_path, seg_text, seg_tag), ...]}

            # Limit to preview mode if requested
            chapter_indices = sorted(chapters_with_text.keys())
            if preview_mode:
                chapter_indices = chapter_indices[:3]
                total_to_gen = sum(len(chapters_with_text[i]) for i in chapter_indices)
                add_log("PREVIEW MODE: Limiting to first %d chapters (%d segments)" % (len(chapter_indices), total_to_gen))
            else:
                total_to_gen = total_segments

            for ch_idx in chapter_indices:
                segs = chapters_with_text[ch_idx]
                ch_title = getattr(next((c for c in parser._chapters if getattr(c, 'spine_order', 0) == ch_idx), None), 'title', 'Chapter %d' % (ch_idx+1))
                add_log("=== Chapter %d: %s (%d segments) ===" % (ch_idx+1, ch_title, len(segs)))

                ch_audio_dir = os.path.join(self._project.segments_dir, str(ch_idx))
                os.makedirs(ch_audio_dir, exist_ok=True)

                for si, seg_obj in enumerate(segs):
                    if self._gen_cancelled:
                        add_log("Generation cancelled by user")
                        yield generated_count/total_to_gen*100, "Cancelled", "\n".join(log), gr.update(interactive=True), gr.update(visible=False)
                        return

                    while self._gen_paused:
                        time.sleep(0.5)
                        if self._gen_cancelled:
                            return

                    seg_id = seg_obj.id
                    tag = self.tags.get(seg_id)
                    emotion = tag.emotion if tag else "neutral"
                    instr = tag.emotion_instruction if tag else "Parlez d'un ton neutre et naturel"

                    add_log("  [%d/%d] %s (emotion: %s)" % (generated_count+1, total_to_gen, seg_id, emotion))

                    # Output path
                    out_path = os.path.join(ch_audio_dir, "%s.wav" % seg_id)

                    try:
                        audio_path, dur = self.tts_engine.generate(
                            text=seg_obj.text,
                            language="French",
                            emotion_instruction=instr,
                            output_path=out_path,
                            progress_callback=tts_progress,
                        )
                        if audio_path and os.path.exists(audio_path):
                            seg_audio_paths.setdefault(ch_idx, []).append((audio_path, seg_obj.text, tag))
                            generated_count += 1
                            self._project.set_segment_status(seg_id, "generated", {"duration": dur, "path": audio_path})
                            add_log("    -> Generated: %.1fs" % dur)
                        else:
                            add_log("    -> FAILED: no output")
                            failed_count += 1
                    except Exception as e:
                        add_log("    -> ERROR: %s" % e)
                        failed_count += 1

                    # Update progress
                    pct = 30 + (generated_count / total_to_gen * 50)
                    elapsed = time.time() - start_time
                    if generated_count > 0:
                        avg = elapsed / generated_count
                        remaining = avg * (total_to_gen - generated_count)
                        eta = "%02d:%02d remaining" % (remaining/60, remaining%60)
                    else:
                        eta = "calculating..."

                    yield pct, "Generating Ch%d: %d/%d (%s)" % (ch_idx+1, generated_count, total_to_gen, eta), "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            add_log("Generation phase complete. Generated: %d, Failed: %d" % (generated_count, failed_count))

            if generated_count == 0:
                yield 0, "Error: No segments generated. Check TTS setup.", "\n".join(log), gr.update(interactive=True), gr.update(visible=False)
                return

            yield 85, "Generation done, assembling audiobook...", "\n".join(log), gr.update(visible=False), gr.update(visible=False)

            # 6. Assembly
            add_log("Assembling final audiobook...")
            from audiobook_ai.audio.assembly import AudioAssembly
            assembly = AudioAssembly(self._project, self.config)

            # Map chapter indices to titles
            ch_titles = {}
            for ch in parser._chapters:
                idx = getattr(ch, 'spine_order', 0)
                ch_titles[idx] = getattr(ch, 'title', 'Chapter %d' % (idx+1))

            try:
                if preview_mode:
                    # Only assembly selected chapters
                    filtered_chapters_with_text = {i: chapters_with_text[i] for i in chapter_indices}
                    output_path = assembly.assemble_full_m4b(
                        chapter_paths={i: None for i in chapter_indices},  # Will be auto-scanned
                        chapter_titles=ch_titles,
                    )
                else:
                    output_path = assembly.assemble_full_m4b(chapter_titles=ch_titles)

                add_log("FINAL OUTPUT: %s" % output_path)
                yield 100, "COMPLETE! Output: %s" % os.path.basename(output_path), "\n".join(log), gr.update(interactive=True), gr.update(visible=False)

            except Exception as e:
                add_log("ASSEMBLY ERROR: %s" % e)
                import traceback
                add_log(traceback.format_exc())
                yield 90, "Assembly error: %s" % e, "\n".join(log), gr.update(interactive=True), gr.update(visible=False)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            add_log("FATAL ERROR: %s" % e)
            add_log(tb)
            yield 0, "Error: %s" % e, "\n".join(log), gr.update(interactive=True), gr.update(visible=False)

    def test_llm(self, url):
        try:
            import urllib.request
            urllib.request.urlopen(url.replace('/v1', ''), timeout=2)
            return "Connected to LM Studio."
        except Exception as e:
            return f"Connection failed: {e}"

    def launch(self, port=None, server_port=None, share=False, server_name="0.0.0.0", **kwargs):
        if self.app is None:
            self.build()
        
        final_port = port or server_port or 7860
        
        self.app.queue()
        self.app.launch(
            server_name=server_name,
            server_port=final_port,
            share=share,
            theme=self.theme,
            css=self.css,
            **kwargs
        )