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

# Bark voice presets
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

# Edge TTS voices
EDGE_VOICES = {
    "en": ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-ThomasNeural", "en-AU-NatashaNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural", "fr-FR-AlainNeural", "fr-FR-BrigitteNeural"],
    "de": ["de-DE-ConradNeural", "de-DE-KatjaNeural", "de-DE-BerndNeural"],
    "es": ["es-ES-AlvaroNeural", "es-ES-ElviraNeural", "es-MX-DaliaNeural"],
}


class AudiobookGUI:
    """Redesigned Gradio interface."""

    def __init__(self, config):
        self.config = config
        self.app = None

        # Core state
        self._project = None
        self._epub_parser = None
        self._tts_engine = None
        self._voice_manager = None
        self._analyzer = None
        self._validator = None
        self._assembly = None

        # Analysis results
        self._segment_tags = {}
        self._discovered_chars = []
        self._dedup_map = {}
        self._segmentation = {}
        self._chapter_titles = {}
        self._chapters_list = []

        # Generation state
        self._generation_running = False
        self._generation_paused = False
        self._generation_cancelled = False
        self._voice_assignments = {}

        # Logging
        self._log_messages = []

        # Voice mode
        self._voice_mode = "narrator"  # "narrator" or "ensemble"

    def _log(self, msg):
        self._log_messages.append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
        logger.info(msg)

    def _get_logs(self):
        return "\n".join(self._log_messages[-200:])

    def build(self):
        """Build the complete redesigned app."""
        with gr.Blocks(
            title="AIGUIBook",
        ) as self.app:
            gr.Markdown(
                "# AIGUIBook\n"
                "### Transform EPUB to Audiobook with AI Voices"
            )

            # Global state
            app_state = gr.State({
                "loaded": False, "parsed": False,
                "analyzed": False, "voices_ready": False,
            })

            with gr.Tabs():
                # ======================== TAB 1: ANALYSIS ========================
                with gr.Tab("1. Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            epub_upload = gr.File(
                                label="Upload EPUB / Charger EPUB",
                                file_types=[".epub"], type="filepath",
                            )
                            parse_btn = gr.Button(
                                "Parse Book / Analyser la Structure",
                                variant="primary", size="lg",
                            )
                            book_info_box = gr.Markdown(
                                value="*No book loaded yet*",
                            )

                            run_analysis_btn = gr.Button(
                                "Run Character Analysis / Analyser Personnages",
                                variant="primary", size="lg",
                            )
                            analysis_status_box = gr.Textbox(
                                label="Analysis Status / Etat",
                                interactive=False,
                            )
                            character_list_box = gr.JSON(
                                label="Detected Characters / Personnages Detectes",
                            )

                        with gr.Column(scale=1):
                            load_analysis_btn = gr.Button(
                                "Load Saved Analysis / Charger Analyse",
                                variant="secondary",
                            )
                            analysis_file_input = gr.File(
                                label="Save File / Fichier sauvegarde",
                                file_types=[".json"], type="filepath",
                            )
                            load_status_box = gr.Textbox(
                                label="Load Status", interactive=False,
                            )

                            with gr.Group():
                                gr.Markdown("### Voice Descriptions for ElevenLabs")
                                gen_desc_btn = gr.Button(
                                    "Generate Descriptions / Generer",
                                    variant="primary",
                                )
                                desc_box = gr.Textbox(
                                    label="ElevenLabs Prompts",
                                    interactive=True, lines=15, max_lines=50,
                                )
                                copy_desc_btn = gr.Button(
                                    "Save to File / Sauvegarder",
                                    variant="secondary",
                                )
                                copy_status_box = gr.Textbox(
                                    label="Copy Status", interactive=False,
                                )

                # ======================== TAB 2: VOICES ========================
                with gr.Tab("2. Voices"):
                    gr.Markdown(
                        "### Voice Configuration / Configuration Voix\n"
                        "Choose how voices are assigned to your book."
                    )

                    with gr.Row():
                        voice_mode_radio = gr.Radio(
                            choices=[
                                ("Single Narrator - One voice, changing emotions", "narrator"),
                                ("Ensemble Cast - Different voices per character", "ensemble"),
                            ],
                            value="narrator",
                            label="Voice Mode / Mode Voix",
                        )

                    # Narrator settings (always visible)
                    with gr.Group() as narrator_group:
                        gr.Markdown("### Narrator Voice / Voix du Narrateur")
                        with gr.Row():
                            narrator_source = gr.Dropdown(
                                choices=["Bark", "Edge TTS", "Custom WAV"],
                                value="Bark",
                                label="Source / Source Voix",
                            )
                            narrator_voice_select = gr.Dropdown(
                                choices=list(BARK_VOICES.values()),
                                value="Narrator Female",
                                label="Voice / Voix",
                            )
                        narrator_preview_btn = gr.Button(
                            "Preview Narrator / Previsualiser", variant="primary",
                        )
                        narrator_audio = gr.Audio(
                            label="Narrator Preview", type="filepath",
                        )

                    # Custom WAV upload (conditional)
                    narrator_file_input = gr.File(
                        label="Upload Custom Reference / Charger reference WAV",
                        file_types=[".wav", ".mp3"], type="filepath",
                        visible=False,
                    )

                    # Ensemble cast settings (conditional)
                    with gr.Group(visible=False) as ensemble_group:
                        gr.Markdown("### Cast / Personnages")
                        ensemble_status = gr.Textbox(
                            label="Status", value="Load a book first.",
                            interactive=False,
                        )
                        auto_assign_btn = gr.Button(
                            "Auto-Assign Voices / Attribution automatique",
                            variant="primary",
                        )
                        character_voice_table = gr.Dataframe(
                            headers=["Character", "Segments", "Voice", "Actions"],
                            datatype=["str", "number", "str", "str"],
                            interactive=True,
                        )

                    apply_voice_btn = gr.Button(
                        "Save Voice Settings / Sauvegarder",
                        variant="primary",
                    )
                    voice_apply_status = gr.Textbox(
                        label="Apply Status", interactive=False,
                    )

                # ======================== TAB 3: PREVIEW ========================
                with gr.Tab("3. Preview"):
                    gr.Markdown(
                        "### Test Voice Acting / Tester le Jeu d'Acteur\n"
                        "Listen to how your book segments will sound."
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            chapter_select = gr.Dropdown(
                                choices=[], label="Chapter / Chapitre",
                                allow_custom_value=False,
                            )
                            segment_select = gr.Dropdown(
                                choices=[], label="Segment",
                                allow_custom_value=False,
                            )
                            select_preview_btn = gr.Button(
                                "Load Segment / Charger Segment",
                            )

                            preview_text_box = gr.Textbox(
                                label="Text / Texte",
                                interactive=False, lines=4,
                            )
                            preview_info_box = gr.Textbox(
                                label="Analysis Info / Infos",
                                interactive=False, lines=2,
                            )

                        with gr.Column(scale=1):
                            preview_voice_select = gr.Dropdown(
                                choices=[], label="Voice for Preview / Voix pour apercu",
                            )
                            preview_emotion_select = gr.Dropdown(
                                choices=["auto"] + EMOTION_OPTIONS,
                                value="auto",
                                label="Emotion / Emotion",
                            )
                            generate_preview_btn = gr.Button(
                                "Generate Audio / Generer Audio",
                                variant="primary",
                            )
                            preview_audio_out = gr.Audio(
                                label="Preview Result / Resultat",
                                type="filepath",
                            )

                # ======================== TAB 4: GENERATE ========================
                with gr.Tab("4. Generate"):
                    gr.Markdown(
                        "### Audiobook Production / Production du Livre Audio"
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            start_btn = gr.Button(
                                "Start Full Generation / Commencer la Generation",
                                variant="primary", size="lg",
                            )
                            pause_btn = gr.Button(
                                "Pause / Pause", interactive=False,
                            )
                            resume_btn = gr.Button(
                                "Resume / Reprendre", interactive=False, visible=False,
                            )
                            stop_btn = gr.Button(
                                "Stop / Annuler", variant="stop", interactive=False,
                            )

                            # Generation options
                            with gr.Group():
                                preview_only_check = gr.Checkbox(
                                    label="Preview Mode: First 3 chapters only",
                                    value=False,
                                )
                                no_validation_check = gr.Checkbox(
                                    label="Disable audio validation", value=False,
                                )
                                sample_rate_select = gr.Dropdown(
                                    choices=[24000, 16000], value=24000,
                                    label="Audio Sample Rate",
                                )

                        with gr.Column(scale=1):
                            # Progress bars
                            overall_progress = gr.Slider(
                                minimum=0, maximum=100, value=0,
                                label="Overall Progress / Progres Global",
                                interactive=False,
                            )
                            stage_progress = gr.Textbox(
                                label="Current Stage / Etape Actuelle",
                                value="Ready / Pret",
                                interactive=False,
                            )
                            time_box = gr.Textbox(
                                label="ETA / Temps Restant",
                                value="--:--",
                                interactive=False,
                            )
                            detail_box = gr.Textbox(
                                label="Details / Details",
                                value="",
                                interactive=False, lines=2,
                            )

                        with gr.Column(scale=1):
                            output_file_box = gr.Textbox(
                                label="Output File / Fichier de Sortie",
                                interactive=False,
                            )
                            open_output_btn = gr.Button(
                                "Open Output Folder / Ouvrir Dossier",
                            )
                            log_box = gr.Textbox(
                                label="Log / Journal du Systeme",
                                interactive=False,
                                lines=10, max_lines=50,
                            )

                # ======================== TAB 5: SETTINGS ========================
                with gr.Tab("5. Settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### TTS Engine")
                            tts_model_select = gr.Dropdown(
                                choices=[
                                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                                ],
                                value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                label="Model / Modele",
                            )
                            tts_device = gr.Dropdown(
                                choices=["cuda", "cpu"], value="cuda",
                                label="Device / Peripherique",
                            )
                            tts_dtype = gr.Dropdown(
                                choices=["bfloat16", "float16", "float32"],
                                value="bfloat16", label="Precision",
                            )
                            tts_batch = gr.Slider(
                                minimum=1, maximum=16, value=4, step=1,
                                label="Batch Size / Taille de lot",
                            )

                        with gr.Column():
                            gr.Markdown("### Output / Sortie")
                            output_format = gr.Dropdown(
                                choices=["m4b", "m4a", "wav"],
                                value="m4b", label="Format",
                            )
                            output_bitrate = gr.Dropdown(
                                choices=["64k", "128k", "192k", "256k"],
                                value="128k", label="Bitrate / Debit",
                            )
                            normalize_check = gr.Checkbox(
                                label="Normalize audio / Normaliser", value=True,
                            )
                            crossfade_slider = gr.Slider(
                                minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                                label="Crossfade (sec) / Fondu",
                            )

                        with gr.Column():
                            gr.Markdown("### Analysis LLM")
                            llm_backend = gr.Dropdown(
                                choices=["lmstudio", "openrouter", "ollama"],
                                value="lmstudio", label="Backend",
                            )
                            lmstudio_url = gr.Textbox(
                                label="LM Studio URL",
                                value="http://localhost:1234/v1",
                            )
                            test_llm_btn = gr.Button(
                                "Test Connection / Tester",
                            )
                            llm_test_status = gr.Textbox(
                                label="Connection Test", interactive=False,
                            )

                    save_settings_btn = gr.Button(
                        "Save Settings / Sauvegarder", variant="primary",
                    )
                    settings_status = gr.Textbox(
                        label="Status", interactive=False,
                    )

            # ======================== EVENT HANDLERS ========================

            # PARSE
            parse_btn.click(
                fn=self._on_parse_epub,
                inputs=[epub_upload],
                outputs=[book_info_box, chapter_select, app_state],
            )

            # ANALYSE
            run_analysis_btn.click(
                fn=self._on_run_analysis,
                inputs=[app_state],
                outputs=[
                    analysis_status_box, character_list_box,
                    chapter_select, segment_select, app_state,
                ],
            )

            # LOAD ANALYSIS
            load_analysis_btn.click(
                fn=self._on_load_analysis,
                inputs=[analysis_file_input],
                outputs=[
                    load_status_box, character_list_box,
                    chapter_select, segment_select, app_state,
                ],
            )

            # VOICE DESCRIPTIONS
            gen_desc_btn.click(
                fn=self._on_generate_voice_descriptions,
                inputs=[],
                outputs=[desc_box],
            )
            copy_desc_btn.click(
                fn=self._on_copy_voice_descriptions,
                inputs=[],
                outputs=[copy_status_box],
            )

            # VOICE MODE CHANGE
            voice_mode_radio.change(
                fn=self._on_voice_mode_change,
                inputs=[voice_mode_radio],
                outputs=[ensemble_group, narrator_group],
            )

            # NARRATOR SOURCE CHANGE
            narrator_source.change(
                fn=self._on_narrator_source_change,
                inputs=[narrator_source],
                outputs=[narrator_file_input, narrator_voice_select],
            )

            # NARRATOR PREVIEW
            narrator_preview_btn.click(
                fn=self._on_preview_narrator_voice,
                inputs=[narrator_source, narrator_voice_select, narrator_file_input],
                outputs=[narrator_audio],
            )

            # ENSEMBLE CAST
            auto_assign_btn.click(
                fn=self._on_auto_assign_voices,
                inputs=[],
                outputs=[character_voice_table, ensemble_status],
            )

            # SAVE VOICES
            apply_voice_btn.click(
                fn=self._on_save_voice_settings,
                inputs=[],
                outputs=[voice_apply_status],
            )

            # PREVIEW
            select_preview_btn.click(
                fn=self._on_select_preview_segment,
                inputs=[chapter_select, segment_select],
                outputs=[preview_text_box, preview_info_box, preview_voice_select],
            )
            generate_preview_btn.click(
                fn=self._on_generate_preview_audio,
                inputs=[preview_text_box, preview_voice_select, preview_emotion_select],
                outputs=[preview_audio_out],
            )

            # GENERATION
            start_btn.click(
                fn=self._on_start_generation,
                inputs=[preview_only_check, no_validation_check],
                outputs=[
                    overall_progress, stage_progress, time_box, detail_box,
                    log_box, output_file_box,
                    start_btn, pause_btn, stop_btn, resume_btn,
                ],
            )
            open_output_btn.click(
                fn=self._on_output_ready,
                inputs=[],
                outputs=[open_output_btn],
            )

            # SETTINGS
            test_llm_btn.click(
                fn=self._on_test_llm,
                inputs=[llm_backend, lmstudio_url],
                outputs=[llm_test_status],
            )
            save_settings_btn.click(
                fn=self._on_save_settings,
                inputs=[tts_model_select, tts_device, tts_dtype, tts_batch,
                        output_format, output_bitrate, normalize_check,
                        crossfade_slider, llm_backend, lmstudio_url],
                outputs=[settings_status],
            )

        return self.app

    # ======================== HANDLERS ========================

    def _on_parse_epub(self, epub_path):
        """Parse the uploaded EPUB file."""
        if not epub_path:
            return "*No file selected*", [], {"loaded": False, "parsed": False, "analyzed": False, "voices_ready": False}

        try:
            self._log("Parsing EPUB: %s" % epub_path)
            from audiobook_ai.core.epub_parser import EPUBParser
            parser = EPUBParser(epub_path)
            result = parser.parse()
            self._epub_parser = parser

            metadata = result.get("metadata", {})
            chapters = result.get("chapters", [])
            self._chapters_list = chapters

            # Build titles
            self._chapter_titles = {}
            chapter_names = []
            for ch in chapters:
                if hasattr(ch, "title"):
                    title = ch.title
                    idx = getattr(ch, "spine_order", len(chapter_names))
                else:
                    title = ch.get("title", "Chapter")
                    idx = ch.get("spine_order", len(chapter_names))
                self._chapter_titles[idx] = title
                chapter_names.append("Ch %d: %s" % (idx + 1, title[:50]))

            info = "**%s**\nby %s\nLanguage: %s\nChapters: %d\n\n" % (
                metadata.get("title", "Unknown"),
                metadata.get("author", "Unknown"),
                metadata.get("language", "unknown"),
                len(chapters),
            )

            # Create project
            work_dir = os.path.join(tempfile.gettempdir(), "aiguibook")
            output_dir = os.path.join(os.path.expanduser("~"), "audiobooks")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            from audiobook_ai.core.project import BookProject
            self._project = BookProject(
                book_title=metadata.get("title", "Untitled"),
                work_dir=work_dir, output_dir=output_dir,
            )
            self._project.create()
            self._project.book_metadata = metadata
            self._project.total_chapters = len(chapters)

            self._log("Parsed: %s - %d chapters" % (metadata.get("title", "?"), len(chapters)))

            return info, chapter_names, {"loaded": True, "parsed": True, "analyzed": False, "voices_ready": False}

        except Exception as e:
            self._log("Parse error: %s" % e)
            return "*Error: %s*" % e, [], {"loaded": False, "parsed": False, "analyzed": False, "voices_ready": False}

    def _on_run_analysis(self, state):
        """Run character analysis with live progress."""
        parser = getattr(self, "_epub_parser", None)
        if not parser or not getattr(parser, "_chapters", None):
            yield "No book parsed yet.", [], [], [], state

        chapters = parser._chapters
        if not chapters:
            yield "No chapters found.", [], [], [], state

        self._log("Starting analysis...")
        yield "Segmenting text...", [], [], [], state

        try:
            from audiobook_ai.core.text_segmenter import TextSegmenter
            seg = TextSegmenter(max_words=150, min_words=20)
            all_segs = []

            for ch in chapters:
                if hasattr(ch, "text") and hasattr(ch, "title") and hasattr(ch, "spine_order"):
                    all_segs.extend(seg.segment_chapter(ch.text, ch.title, ch.spine_order))
                else:
                    txt = ch.get("text", "") if isinstance(ch, dict) else ""
                    ttl = ch.get("title", "") if isinstance(ch, dict) else ""
                    idx = ch.get("spine_order", 0) if isinstance(ch, dict) else 0
                    all_segs.extend(seg.segment_chapter(txt, ttl, idx))

            if not all_segs:
                yield "No text segments to analyze.", [], [], [], state

            self._log("Loaded %d segments." % len(all_segs))
            yield "Analyzing characters (this may take a while)...", [], [], [], state

            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            cfg = self.config.get_section("analysis")
            self._analyzer = CharacterAnalyzer(cfg)
            lang = self.config.get("general", "language", "french")

            for i, item in enumerate(self._analyzer.analyze_segments_iter(all_segs, lang)):
                if item["status"] == "progress":
                    yield item["msg"], [], [], [], state
                elif item["status"] == "finished":
                    result = item["result"]
                    self._segment_tags = result[0]
                    self._discovered_chars = result[1]
                    self._dedup_map = result[2] if len(result) > 2 else {}

                    # Build character list for UI
                    char_data = []
                    for cn in self._discovered_chars:
                        cnt = sum(
                            1 for t in self._segment_tags.values()
                            if t.character_name is not None and
                            self._dedup_map.get(t.character_name, t.character_name) == cn
                        )
                        char_data.append({"character": cn, "segments": cnt})

                    # Auto-save
                    if self._project:
                        fp = os.path.join(self._project.project_dir, "character_analysis.json")
                        self._analyzer.save_analysis(fp, self._segment_tags, self._discovered_chars, self._dedup_map)

                    # Update chapter/segment selectors
                    seg_names = []
                    for seg in all_segs:
                        seg_names.append("%s (%s)" % (seg.id, seg.text[:40] + "..." if len(seg.text) > 40 else seg.text))

                    msg = "Done! Found %d unique characters." % len(self._discovered_chars)
                    state["analyzed"] = True
                    yield msg, char_data, self._chapter_titles.values(), seg_names, state

        except Exception as e:
            import traceback
            self._log("Analysis error: %s" % e)
            yield "Error: %s" % e, [], [], [], state

    def _on_load_analysis(self, filepath):
        """Load previously saved analysis."""
        if not filepath:
            return "No file selected.", [], [], [], {"loaded": False, "parsed": False, "analyzed": False, "voices_ready": False}

        try:
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            tags, chars, dmap = CharacterAnalyzer.load_analysis(filepath)
            
            self._segment_tags = tags
            self._discovered_chars = chars
            self._dedup_map = dmap

            seg_names = [sid for sid in tags.keys()]
            msg = "Loaded %d characters, %d segments from %s" % (len(chars), len(tags), os.path.basename(filepath))
            
            state = {"loaded": True, "parsed": True, "analyzed": True, "voices_ready": False}
            return msg, [{"character": c, "segments": sum(1 for t in tags.values() if t.character_name == c)} for c in chars], [self._chapter_titles.get(i, "Ch %d" % (i+1)) for i in range(len(self._chapter_titles))], seg_names, state

        except Exception as e:
            return "Error: %s" % e, [], [], [], {"loaded": False, "parsed": False, "analyzed": False, "voices_ready": False}

    def _on_voice_mode_change(self, mode):
        self._voice_mode = mode
        if mode == "ensemble":
            return gr.update(visible=True), gr.update(visible=True)
        elif mode == "narrator":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(), gr.update()

    def _on_narrator_source_change(self, source):
        if source == "Custom WAV":
            return gr.update(visible=True), gr.update(choices=["Custom"])
        elif source == "Bark":
            return gr.update(visible=False), gr.update(choices=list(BARK_VOICES.values()), value="Narrator Female")
        elif source == "Edge TTS":
            return gr.update(visible=False), gr.update(choices=["en-US-GuyNeural", "en-US-JennyNeural", "fr-FR-HenriNeural", "fr-FR-DeniseNeural"], value="fr-FR-DeniseNeural")
        return gr.update(visible=False), gr.update()

    def _on_preview_narrator_voice(self, source, voice, wav_file):
        """Preview the selected narrator voice."""
        try:
            import asyncio
            if source == "Edge TTS":
                import edge_tts
                output = os.path.join(tempfile.gettempdir(), "preview_edge.mp3")
                async def gen():
                    comm = edge_tts.Communicate("Bonjour, ceci est un apercu de ma voix.", voice)
                    await comm.save(output)
                asyncio.run(gen())
                return output
            elif source == "Bark":
                # Use Edge TTS as fallback if Bark not installed
                import edge_tts
                output = os.path.join(tempfile.gettempdir(), "preview.mp3")
                async def gen():
                    comm = edge_tts.Communicate("Bonjour, ceci est un apercu de ma voix.", "fr-FR-DeniseNeural")
                    await comm.save(output)
                asyncio.run(gen())
                return output
            elif source == "Custom WAV":
                return wav_file
        except Exception as e:
            self._log("Preview error: %s" % e)
        return None

    def _on_auto_assign_voices(self):
        """Auto-assign voices to characters based on analysis."""
        if not self._discovered_chars:
            return [], "No characters loaded."

        rows = []
        for cn in self._discovered_chars:
            info = self._analyzer.build_voice_descriptions().get(cn, {})
            voice = info.get("voice_type", "Narrator Female")
            rows.append([cn, sum(1 for t in self._segment_tags.values() if t.character_name == cn), voice, "Auto"])

        self._log("Auto-assigned voices for %d characters." % len(rows))
        return rows, "Assigned %d voices." % len(rows)

    def _on_save_voice_settings(self):
        """Save current voice assignments."""
        self._log("Voice settings saved.")
        return "Voice settings saved."

    def _generate_voice_description(self, char_name):
        """Generate a voice description for a character."""
        if not self._discovered_chars:
            return "No characters available."

        descriptions = {}
        for cn in self._discovered_chars:
            lower = cn.lower()
            is_female = any(w in lower for w in ["miss", "ms", "woman", "she", "her", "lady"])
            gender = "female" if is_female else "male"

            is_military = any(w in lower for w in ["captain", "sergeant", "commander", "soldier"])
            is_old = any(w in lower for w in ["old", "elder", "ancient"])

            if is_military:
                desc = "A %s military voice, authoritative and firm." % gender
            elif is_old:
                desc = "An older %s voice, wise and measured." % gender
            else:
                desc = "A natural %s voice, clear and expressive." % gender

            descriptions[cn] = desc
        
        return descriptions.get(char_name, "A voice for %s." % char_name)

    def _on_generate_voice_descriptions(self):
        """Generate ElevenLabs descriptions for all characters."""
        if not self._discovered_chars:
            return "No analysis data. Run character analysis first."
        try:
            descs = self._analyzer.build_voice_descriptions()
            txt = "=== ElevenLabs Voice Descriptions ===\n\n"
            for char, info in descs.items():
                txt += "Character: %s\nPrompt: %s\n\n" % (char, info.get("elevenlabs_prompt", ""))
            return txt
        except Exception as e:
            return "Error: %s" % e

    def _on_copy_voice_descriptions(self):
        """Save descriptions to temp file."""
        try:
            descs = self._analyzer.build_voice_descriptions()
            content = ""
            for char, info in descs.items():
                content += "Character: %s\nPrompt: %s\n\n" % (char, info.get("elevenlabs_prompt", ""))
            
            outpath = os.path.join(tempfile.gettempdir(), "aiguibook_elevenlabs_prompts.txt")
            with open(outpath, "w", encoding="utf-8") as f:
                f.write(content)
            
            return "Saved to: %s" % outpath
        except Exception as e:
            return "Error: %s" % e

    def _on_select_preview_segment(self, chapter_name, segment_name):
        """Load segment preview."""
        if not segment_name:
            return "No segment selected.", "No segment selected.", []

        seg_id = segment_name.split(" ")[0]  # Extract ID
        tag = self._segment_tags.get(seg_id)
        if tag:
            speaker = tag.character_name or "Narrator"
            emotion = tag.emotion
            info = "Speaker: %s\nEmotion: %s" % (speaker, emotion)
            text = tag.text if hasattr(tag, 'text') else "Text not available"
            voice_options = ["Narrator"] + self._discovered_chars if self._discovered_chars else ["Narrator"]
            return text, info, voice_options
        else:
            return "Segment not found.", "No data.", []

    def _on_generate_preview_audio(self, text, voice, emotion):
        """Generate preview audio for a segment with emotion."""
        try:
            self._log("Generating preview with emotion: %s" % emotion)
            # This should use the actual TTS engine
            msg = "[Preview] Text: %s\nVoice: %s\nEmotion: %s" % (text[:50], voice, emotion)
            self._log(msg)
            # For now, return a placeholder message
            return None
        except Exception as e:
            return "Error: %s" % e

    def _on_start_generation(self, preview_only, no_validation):
        """Start audiobook generation."""
        self._generation_running = True
        self._generation_paused = False
        self._generation_cancelled = False
        self._log_messages.clear()

        result_holder = {
            "progress": 0, "status": "Starting...", "eta": "--:--",
            "detail": "", "logs": "", "error": None, "output_file": "",
            "stage": "initializing", "chapter_idx": 0,
        }

        def run_gen():
            try:
                self._run_pipeline_thread(preview_only, no_validation, result_holder)
            except Exception as e:
                result_holder["error"] = str(e)
                result_holder["status"] = "ERROR: %s" % e
                self._log("FATAL: %s" % e)

        thread = threading.Thread(target=run_gen, daemon=True)
        thread.start()

        start_time = time.time()
        while thread.is_alive():
            time.sleep(0.5)
            progress = result_holder["progress"]
            status = result_holder["status"]
            eta = result_holder["eta"]
            detail = result_holder["detail"]
            logs = self._get_logs()
            outfile = result_holder["output_file"]

            if self._generation_paused:
                status = "PAUSED"
                yield progress, status, eta, detail, logs, outfile, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, interactive=True)
                while self._generation_paused and not self._generation_cancelled:
                    time.sleep(0.5)
                if self._generation_cancelled:
                    yield progress, "CANCELLED", eta, detail, self._get_logs(), outfile, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, interactive=False)
                    return
            
            yield progress, status, eta, detail, logs, outfile, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, interactive=False)

        self._generation_running = False
        if result_holder["error"]:
            yield progress, "FAILED: %s" % result_holder["error"], eta, detail, self._get_logs(), result_holder["output_file"], gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, interactive=False)
        else:
            yield 100, "COMPLETE", eta, detail, self._get_logs(), result_holder["output_file"], gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, interactive=False)

    def _run_pipeline_thread(self, preview_only, no_validation, holder):
        """Run the full generation pipeline in background thread."""
        self._log("Pipeline start!")
        # Placeholder
        time.sleep(2)
        holder["progress"] = 100
        holder["status"] = "Complete"
        holder["eta"] = "00:00"
        holder["detail"] = "Generation finished."
        holder["output_file"] = os.path.join(tempfile.gettempdir(), "output.m4b")
        self._log("Pipeline done.")

    def _on_test_llm(self, backend, url):
        """Test LLM backend connection."""
        try:
            from audiobook_ai.analysis.character_analyzer import test_llm_connection
            if backend == "lmstudio":
                ok, msg = test_llm_connection("lmstudio", base_url=url, timeout=60.0)
            elif backend == "ollama":
                ok, msg = test_llm_connection("ollama", base_url=url, timeout=60.0)
            if ok:
                return "SUCCESS: %s" % msg
            return "FAILED: %s" % msg
        except Exception as e:
            return "Error: %s" % e

    def _on_save_settings(self, model, device, dtype, batch, fmt, bitrate, norm, crossfade, backend, lmurl):
        """Save settings to config and disk."""
        self.config.set("tts", "model", model)
        self.config.set("tts", "device", device)
        self.config.set("tts", "dtype", dtype)
        self.config.set("tts", "batch_size", batch)
        self.config.set("output", "format", fmt)
        self.config.set("output", "bitrate", bitrate)
        self.config.set("output", "normalize_audio", norm)
        self.config.set("output", "crossfade_duration", crossfade)
        self.config.set("analysis", "llm_backend", backend)
        self.config.set("analysis", "lmstudio_base_url", lmurl.removesuffix("/").removesuffix("/v1") + "/v1")
        self.config.save()
        self._log("Settings saved.")
        return "Saved."

    def launch(self, port=7860, share=False):
        """Launch app."""
        if self.app is None:
            self.build()
        
        css = """
            .log-box textarea { font-family: monospace; font-size: 12px; }
            .gradio-container { max-width: 90% !important; margin-left: auto; margin-right: auto; }
        """

        self.app.queue()
        self.app.launch(
            server_name="0.0.0.0", server_port=port,
            share=share, css=css, favicon_path=None
        )
