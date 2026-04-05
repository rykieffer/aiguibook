"""AudiobookGUI - Main Gradio application for AIGUIBook."""

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

EMOTIONS_FR = {
    "calm": "Calm",
    "excited": "Excité",
    "angry": "En colère",
    "sad": "Triste",
    "whisper": "Chuchotement",
    "tense": "Tendu",
    "urgent": "Urgent",
    "amused": "Amusé",
    "contemptuous": "Méprisant",
    "surprised": "Surpris",
    "neutral": "Neutre",
}


class AudiobookGUI:
    """Gradio-based web interface for the audiobook generator."""

    def __init__(self, config):
        self.config = config
        self.app = None

        self._project = None
        self._epub_parser = None
        self._tts_engine = None
        self._voice_manager = None
        self._analyzer = None
        self._validator = None
        self._assembly = None
        self._segment_tags = {}
        self._discovered_chars = []
        self._segmentation = {}
        self._voice_assignments = {}
        self._generation_running = False
        self._generation_paused = False
        self._generation_cancelled = False
        self._chapter_titles = {}
        self._chapters_list = []
        self._log_messages = []
        self._dedup_map = {}

    def _log(self, msg: str):
        """Add a log message."""
        self._log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    def _get_logs(self) -> str:
        return "\n".join(self._log_messages[-200:])

    def build(self) -> gr.Blocks:
        """Build the complete Gradio application."""
        with gr.Blocks(
            title="AIGUIBook - AI Audiobook Generator",
        ) as self.app:
            gr.Markdown(
                "# AIGUIBook - AI Audiobook Generator\n"
                "### Transform your EPUB into an audiobook with AI voices"
            )

            app_state = gr.State({
                "loaded": False, "parsed": False,
                "analyzed": False, "voices_assigned": False,
            })

            with gr.Tabs():
                with gr.Tab("Setup / Configuration"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            epub_file_upload = gr.File(
                                label="EPUB File / Fichier EPUB",
                                file_types=[".epub"], type="filepath",
                            )
                            parse_btn = gr.Button(
                                "Parse EPUB / Analyser EPUB",
                                variant="primary", size="lg",
                            )

                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Book Info / Informations du Livre")
                                book_title_display = gr.Textbox(
                                    label="Title / Titre", interactive=False,
                                )
                                book_author_display = gr.Textbox(
                                    label="Author / Auteur", interactive=False,
                                )
                                book_language_display = gr.Textbox(
                                    label="Language / Langue", interactive=False,
                                )
                                book_chapters_display = gr.Textbox(
                                    label="Chapters / Chapitres", interactive=False,
                                )

                    # Parse result area
                    parse_output = gr.Markdown("")

                    # Voice settings
                    with gr.Row():
                        with gr.Column(scale=1):
                            output_format_select = gr.Dropdown(
                                choices=["m4b", "m4a", "flac"], value="m4b",
                                label="Output Format / Format de Sortie",
                            )
                            language_select = gr.Dropdown(
                                choices=["french", "english", "spanish", "german"],
                                value="french",
                                label="Primary Language / Langue Principale",
                            )
                            tts_model_select = gr.Dropdown(
                                choices=[
                                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                                ],
                                value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                label="TTS Model / Modèle TTS",
                            )
                        with gr.Column(scale=1):
                            narrator_ref_input = gr.File(
                                label="Narrator Reference Audio / Audio de Référence du Narrateur",
                                file_types=[".wav", ".mp3", ".ogg", ".flac"],
                                type="filepath",
                            )
                            setup_default_voices_btn = gr.Button(
                                "Create Default Voices / Créer Voix par Défaut",
                            )
                            voices_status_display = gr.Textbox(
                                label="Voice Status / État des Voix",
                                interactive=False, lines=4,
                            )

                    # Analysis section
                    analyse_btn = gr.Button(
                        "Run Character Analysis / Analyse Personnages",
                        variant="primary",
                    )
                    analysis_status = gr.Textbox(
                        label="Analysis Status / État", interactive=False,
                    )
                    character_list = gr.JSON(
                        label="Characters and Suggestions / Personnages",
                    )

                    with gr.Row():
                        load_analysis_btn = gr.Button(
                            "Load Saved Analysis / Charger Analyse Sauvegardée",
                            variant="secondary",
                        )
                        analysis_file_input = gr.File(
                            label="Analysis JSON File / Fichier JSON d'Analyse",
                            file_types=[".json"], type="filepath",
                        )
                    analysis_load_status = gr.Textbox(
                        label="Load Status / État du Chargement", interactive=False,
                    )

                    # ElevenLabs section
                    with gr.Group():
                        gr.Markdown("### ElevenLabs Voice Descriptions / Descriptions ElevenLabs")
                        generate_voice_desc_btn = gr.Button(
                            "Generate Voice Descriptions / Générer Descriptions",
                            variant="primary",
                        )
                        voice_descriptions_box = gr.Textbox(
                            label="ElevenLabs Prompts (copy to use in ElevenLabs Voice Design)",
                            interactive=True, lines=15, max_lines=50,
                        )
                        copy_voice_desc_btn = gr.Button(
                            "Save to File / Sauvegarder (for easy copy)",
                            variant="secondary",
                        )
                        copy_status = gr.Textbox(
                            label="Save Status", interactive=False,
                        )

                # TAB 2: Voices
                with gr.Tab("Voices / Voix"):
                    gr.Markdown(
                        "### Character Voices / Voix des Personnages\n"
                        "Assign microphone reference audio (3s+) to each character for voice cloning."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("#### Narrator Voice / Voix Narrateur")
                                narrator_voice_select = gr.Dropdown(
                                    choices=["narrator_male", "narrator_female",
                                           "custom", "single_voice"],
                                    value="narrator_male",
                                    label="Narrator Voice / Voix du Narrateur",
                                )
                                narrator_ref_voice_input = gr.File(
                                    label="Custom Narrator Reference / Référence Personnalisée",
                                    file_types=[".wav", ".mp3"], type="filepath",
                                    visible=False,
                                )
                            with gr.Group():
                                gr.Markdown("#### Character Voices / Voix des Personnages")
                                voices_status = gr.Textbox(
                                    label="Status / État",
                                    value="No characters loaded. Run character analysis first.",
                                    interactive=False, lines=2,
                                )
                                character_voices_container = gr.Column()
                                refresh_voices_btn = gr.Button(
                                    "Refresh Character List / Rafraîchir",
                                    variant="secondary",
                                )

                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Voice Design / Création de Voix")
                                voice_design_name = gr.Textbox(
                                    label="Voice Name / Nom de la Voix",
                                    placeholder="e.g., marcus, elise, narrator",
                                )
                                voice_design_desc = gr.Textbox(
                                    label="Voice Description / Description",
                                    placeholder="e.g., Deep male voice, French accent",
                                    lines=3,
                                )
                                voice_design_sample = gr.Textbox(
                                    label="Sample Text / Texte d'Exemple",
                                    value="Bonjour, ceci est un test de ma voix générée.",
                                    lines=2,
                                )
                                create_voice_design_btn = gr.Button(
                                    "Create Voice / Créer Voix", variant="primary",
                                )
                                voice_design_output = gr.Audio(
                                    label="Generated Voice Preview / Aperçu",
                                    type="filepath",
                                )

                            with gr.Group():
                                gr.Markdown("### Available Voices / Voix Disponibles")
                                available_voices_list = gr.Dropdown(
                                    choices=[], label="Select Voice to Preview",
                                    interactive=True,
                                )
                                preview_voice_btn = gr.Button("Preview Voice / Écouter")
                                preview_voice_output = gr.Audio(label="Preview")

                    voice_design_status = gr.Textbox(
                        label="Status / État", interactive=False,
                    )

                # TAB 3: Preview
                with gr.Tab("Preview / Aperçu"):
                    gr.Markdown("### Test TTS Generation / Tester la Synthèse Vocale")
                    with gr.Row():
                        with gr.Column():
                            preview_text = gr.Textbox(
                                label="Text to Speak / Texte à Parler",
                                value="Bonjour, ceci est un test de la synthèse vocale.",
                                lines=4,
                            )
                            preview_voice = gr.Dropdown(
                                choices=["narrator_male", "narrator_female"],
                                value="narrator_male", label="Voice / Voix",
                            )
                            preview_language = gr.Dropdown(
                                choices=["French", "English", "Spanish", "German"],
                                value="French", label="Language / Langue",
                            )
                            preview_emotion = gr.Dropdown(
                                choices=EMOTION_OPTIONS, value="calm",
                                label="Emotion / Émotion",
                            )
                            preview_generate_btn = gr.Button(
                                "Generate Preview / Générer l'Aperçu", variant="primary",
                            )
                        with gr.Column():
                            preview_audio = gr.Audio(
                                label="Generated Audio / Audio Généré",
                                type="filepath",
                            )
                            preview_speed = gr.Slider(
                                minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                                label="Speed / Vitesse",
                            )

                # TAB 4: Generate
                with gr.Tab("Generate / Générer"):
                    gr.Markdown("### Full Audiobook Generation / Génération Complète")
                    with gr.Row():
                        with gr.Column():
                            start_btn = gr.Button(
                                "Start Generation / Commencer",
                                variant="primary", size="lg",
                            )
                            pause_btn = gr.Button("Pause / Pause", interactive=False)
                            resume_btn = gr.Button(
                                "Resume / Reprendre", interactive=False, visible=False,
                            )
                            stop_btn = gr.Button(
                                "Stop / Arrêter", variant="stop", interactive=False,
                            )
                            preview_only_check = gr.Checkbox(
                                label="Preview Only (first 3 chapters)", value=False,
                            )
                            no_validation_check = gr.Checkbox(
                                label="Disable Validation / Désactiver Validation",
                                value=False,
                            )
                        with gr.Column():
                            overall_progress = gr.Slider(
                                minimum=0, maximum=100, value=0,
                                label="Overall Progress / Progrès Global",
                                interactive=False,
                            )
                            current_status = gr.Textbox(
                                label="Current Status / État Actuel",
                                value="Ready / Prêt", interactive=False,
                            )
                            estimated_time = gr.Textbox(
                                label="Estimated Time / Temps Estimé",
                                value="--:--", interactive=False,
                            )
                            progress_text = gr.Textbox(
                                label="Progress Text",
                                value="0/0 segments (0%)",
                                interactive=False, lines=2,
                            )
                        with gr.Column():
                            chapter_progress = gr.JSON(
                                label="Chapter Progress / Progrès par Chapitre",
                                value={},
                            )
                            recent_audio = gr.Audio(
                                label="Recent Segment / Segment Récent",
                                type="filepath",
                            )
                    log_output = gr.Textbox(
                        label="Log / Journal", interactive=False,
                        lines=12, max_lines=50,
                    )

                # TAB 5: Settings
                with gr.Tab("Settings / Paramètres"):
                    gr.Markdown("### Application Settings / Paramètres")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### TTS Settings")
                            tts_batch_size = gr.Slider(
                                minimum=1, maximum=16, value=4, step=1,
                                label="Batch Size",
                            )
                            tts_dtype_select = gr.Dropdown(
                                choices=["bfloat16", "float16", "float32"],
                                value="bfloat16", label="Precision / Précision",
                            )
                            tts_device_select = gr.Dropdown(
                                choices=["cuda", "cpu"], value="cuda",
                                label="Device / Périphérique",
                            )
                        with gr.Column():
                            gr.Markdown("#### Validation")
                            validation_enabled = gr.Checkbox(
                                label="Enable Validation", value=True,
                            )
                            max_wer = gr.Slider(
                                minimum=5, maximum=50, value=15, step=1,
                                label="Max WER %",
                            )
                            max_retries = gr.Slider(
                                minimum=0, maximum=5, value=2, step=1,
                                label="Max Retries / Tentatives Max",
                            )
                        with gr.Column():
                            gr.Markdown("#### Output")
                            output_bitrate = gr.Dropdown(
                                choices=["64k", "128k", "192k", "256k", "320k"],
                                value="128k", label="Bitrate",
                            )
                            output_crossfade = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                                label="Crossfade (s) / Fondu Enchaîné (s)",
                            )
                            normalize_audio = gr.Checkbox(
                                label="Normalize Audio / Normaliser", value=True,
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### LLM Settings / Paramètres LLM")
                            llm_backend = gr.Dropdown(
                                choices=["lmstudio", "openrouter", "ollama"],
                                value="lmstudio", label="LLM Backend / Moteur LLM",
                            )
                            with gr.Row(equal_height=True):
                                lmstudio_url_field = gr.Textbox(
                                    label="LM Studio URL",
                                    value="http://localhost:1234/v1",
                                    placeholder="http://localhost:1234/v1", scale=2,
                                )
                                refresh_models_btn = gr.Button(
                                    "Refresh Models / Rafraîchir",
                                    variant="secondary", size="sm", scale=1,
                                )
                            llm_model_dropdown = gr.Dropdown(
                                choices=[], value=None,
                                label="Model (auto-detected) / Modèle (auto-detecté)",
                                info="Click Refresh to detect models",
                                allow_custom_value=True,
                            )
                            with gr.Row(equal_height=True):
                                test_llm_btn = gr.Button(
                                    "Test Connection / Tester",
                                    variant="primary", size="sm", scale=1,
                                )
                                llm_test_status = gr.Textbox(
                                    label="Connection Status",
                                    value="", interactive=False,
                                    lines=2, scale=4,
                                )

                            with gr.Accordion("OpenRouter API (optional)", open=False):
                                openrouter_key = gr.Textbox(
                                    label="OpenRouter API Key",
                                    type="password", value="",
                                )
                                openrouter_model = gr.Textbox(
                                    label="OpenRouter Model",
                                    value="openai/gpt-4o-mini",
                                )
                            with gr.Accordion("Ollama (optional)", open=False):
                                ollama_base = gr.Textbox(
                                    label="Ollama Base URL",
                                    value="http://localhost:11434",
                                )
                                ollama_model = gr.Textbox(
                                    label="Ollama Model", value="qwen3:32b",
                                )

                    save_config_btn = gr.Button(
                        "Save Configuration / Sauvegarder", variant="primary",
                    )
                    load_config_btn = gr.Button("Load Config / Charger")
                    config_status = gr.Textbox(label="Status", interactive=False)

            # ================== EVENT HANDLERS ==================

            parse_btn.click(
                fn=self._on_parse_epub,
                inputs=[epub_file_upload, language_select, tts_model_select,
                        narrator_ref_input],
                outputs=[
                    book_title_display, book_author_display,
                    book_language_display, book_chapters_display,
                    parse_output, app_state,
                ],
            )

            analyse_btn.click(
                fn=self._on_run_analysis,
                inputs=[app_state],
                outputs=[analysis_status, character_list, app_state],
            )

            load_analysis_btn.click(
                fn=self._on_load_analysis,
                inputs=[analysis_file_input],
                outputs=[analysis_load_status, character_list, app_state],
            )

            generate_voice_desc_btn.click(
                fn=self._on_generate_voice_descriptions,
                inputs=[],
                outputs=[voice_descriptions_box],
            )

            copy_voice_desc_btn.click(
                fn=self._on_copy_voice_descriptions,
                inputs=[],
                outputs=[copy_status],
            )

            setup_default_voices_btn.click(
                fn=self._on_setup_default_voices,
                inputs=[],
                outputs=[voices_status_display],
            )

            refresh_voices_btn.click(
                fn=self._refresh_character_voices,
                inputs=[narrator_voice_select],
                outputs=[character_voices_container, voices_status],
            )

            create_voice_design_btn.click(
                fn=self._on_create_voice_design,
                inputs=[voice_design_name, voice_design_desc, voice_design_sample],
                outputs=[voice_design_output, voice_design_status, available_voices_list],
            )

            preview_voice_btn.click(
                fn=self._on_preview_voice,
                inputs=[available_voices_list, voice_design_sample],
                outputs=[preview_voice_output],
            )

            narrator_voice_select.change(
                fn=self._on_narrator_voice_change,
                inputs=[narrator_voice_select],
                outputs=[narrator_ref_voice_input],
            )

            preview_generate_btn.click(
                fn=self._on_generate_preview,
                inputs=[preview_text, preview_voice, preview_language,
                        preview_emotion, preview_speed],
                outputs=[preview_audio],
            )

            start_btn.click(
                fn=self._on_start_generation,
                inputs=[preview_only_check, no_validation_check, narrator_voice_select],
                outputs=[
                    overall_progress, current_status, estimated_time,
                    progress_text, chapter_progress, recent_audio,
                    log_output, start_btn, pause_btn, stop_btn, resume_btn,
                ],
            )

            save_config_btn.click(
                fn=self._on_save_settings,
                inputs=[tts_batch_size, tts_dtype_select, tts_device_select,
                        validation_enabled, max_wer, max_retries,
                        output_bitrate, output_crossfade, normalize_audio,
                        llm_backend, lmstudio_url_field, llm_model_dropdown,
                        openrouter_key, openrouter_model,
                        ollama_model, ollama_base],
                outputs=[config_status],
            )

            load_config_btn.click(
                fn=self._on_load_settings,
                inputs=[],
                outputs=[tts_batch_size, tts_dtype_select, tts_device_select,
                         validation_enabled, max_wer, max_retries,
                         output_bitrate, output_crossfade, normalize_audio,
                         llm_backend, openrouter_key, openrouter_model,
                         ollama_model, ollama_base, config_status,
                         llm_model_dropdown],
            )

            refresh_models_btn.click(
                fn=self._on_refresh_models,
                inputs=[llm_backend, lmstudio_url_field, openrouter_key, ollama_base],
                outputs=[llm_model_dropdown, llm_test_status],
            )

            test_llm_btn.click(
                fn=self._on_test_llm_connection,
                inputs=[llm_backend, llm_model_dropdown, lmstudio_url_field,
                        openrouter_key, openrouter_model, ollama_model, ollama_base],
                outputs=[llm_test_status],
            )

        return self.app

    def _on_parse_epub(self, epub_path, language, tts_model, narrator_ref):
        """Handle EPUB upload and parsing."""
        if not epub_path:
            return "", "", "", "", "Error: No EPUB file selected", {}

        try:
            self._log("Parsing EPUB: " + str(epub_path))
            self.config.set("general", "language", language)
            self.config.set("tts", "model", tts_model)
            if narrator_ref:
                self.config.set("voices", "narrator_ref", narrator_ref)

            from audiobook_ai.core.epub_parser import EPUBParser
            parser = EPUBParser(epub_path)
            result = parser.parse()
            self._epub_parser = parser

            metadata = result["metadata"]
            chapters_raw = result.get("chapters", [])

            # Build chapter list and titles
            self._chapters_list = []
            self._chapter_titles = {}
            for i, ch in enumerate(chapters_raw):
                self._chapters_list.append(ch)
                if hasattr(ch, "spine_order"):
                    idx = ch.spine_order
                    title = getattr(ch, "title", "Chapter %d" % (idx + 1))
                else:
                    idx = ch.get("spine_order", i)
                    title = ch.get("title", "Chapter %d" % (i + 1))
                self._chapter_titles[idx] = title

            title = metadata.get("title", "Unknown")
            author = metadata.get("author", "Unknown")
            lang = metadata.get("language", "unknown")
            num_chapters = len(chapters_raw)

            self._log("Parsed: %s by %s (%d chapters)" % (title, author, num_chapters))

            work_dir = os.path.join(tempfile.gettempdir(), "aiguibook")
            output_dir = os.path.join(os.path.expanduser("~"), "audiobooks")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            from audiobook_ai.core.project import BookProject
            self._project = BookProject(
                book_title=title, work_dir=work_dir, output_dir=output_dir,
            )
            self._project.create()
            self._project.book_metadata = metadata
            self._project.total_chapters = num_chapters

            msg = "Successfully parsed:\nTitle: %s\nAuthor: %s\nLanguage: %s\nChapters: %d" % (
                title, author, lang, num_chapters,
            )
            return (
                title, author, lang,
                "%d chapters" % num_chapters,
                msg,
                {"loaded": True, "parsed": True, "analyzed": False},
            )

        except Exception as e:
            self._log("EPUB parse error: %s" % e)
            return "", "", "", "", "Error parsing EPUB: %s" % e, {
                "loaded": False, "parsed": False, "analyzed": False,
            }

    def _on_run_analysis(self, state):
        """Generator-based analysis to provide live progress updates."""
        self._ensure_voices_initialized()
        parser = getattr(self, "_epub_parser", None)
        if not parser or not getattr(parser, "_chapters", None):
            yield "No book loaded. Parse an EPUB first.", [], state
            return
        chapters = parser._chapters
        if not chapters:
            yield "No chapters found.", [], state
            return

        self._log("Starting character and emotion analysis...")
        yield "Initializing...", [], state

        try:
            from audiobook_ai.core.text_segmenter import TextSegmenter
            seg = TextSegmenter(max_words=150, min_words=20)
            all_segs = []

            self._log("Segmenting text...")
            yield "Segmenting book into chunks...", [], state

            for ch in chapters:
                if hasattr(ch, "text") and hasattr(ch, "title") and hasattr(ch, "spine_order"):
                    all_segs.extend(seg.segment_chapter(ch.text, ch.title, ch.spine_order))
                else:
                    txt = ch.get("text", "") if isinstance(ch, dict) else str(ch)
                    ttl = ch.get("title", "") if isinstance(ch, dict) else "Unknown"
                    idx = ch.get("spine_order", 0) if isinstance(ch, dict) else 0
                    all_segs.extend(seg.segment_chapter(txt, ttl, idx))

            if not all_segs:
                yield "No text segments to analyze.", [], state
                return

            total_segs = len(all_segs)
            self._log("Total %d chunks. Initializing LLM..." % total_segs)
            yield "Starting analysis of %d chunks..." % total_segs, [], state

            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            cfg = self.config.get_section("analysis")
            self._analyzer = CharacterAnalyzer(cfg)
            lang = self.config.get("general", "language", "french")

            for item in self._analyzer.analyze_segments_iter(all_segs, language=lang):
                status = item.get("status", "")
                msg = item.get("msg", "")

                if status in ("init", "analyzing", "progress",
                              "batch_start", "batch_done"):
                    yield msg, [], state
                elif status == "finished":
                    result = item["result"]
                    self._segment_tags = result[0]
                    self._discovered_chars = result[1]
                    self._dedup_map = result[2] if len(result) > 2 else {}
                    state["analyzed"] = True

                    chars = []
                    tags = list(self._segment_tags.values())
                    for cn in sorted(self._discovered_chars):
                        emotion_list = list(set(
                            t.emotion for t in tags
                            if t.character_name is not None and
                            self._dedup_map.get(t.character_name, t.character_name) == cn
                        ))
                        seg_count = sum(
                            1 for t in tags
                            if t.character_name is not None and
                            self._dedup_map.get(t.character_name, t.character_name) == cn
                        )
                        chars.append({
                            "character": cn,
                            "suggested_voice": "narrator_male",
                            "segments": seg_count,
                            "emotions": emotion_list,
                        })

                    # Auto-save analysis
                    if self._project:
                        analysis_path = os.path.join(
                            self._project.project_dir, "character_analysis.json",
                        )
                        self._analyzer.save_analysis(
                            analysis_path,
                            self._segment_tags,
                            self._discovered_chars,
                            self._dedup_map,
                        )

                    self._log("Analysis complete. Found %d unique characters." % len(chars))
                    final_msg = "Done! %d characters, %d segments. Saved." % (
                        len(chars), total_segs,
                    )
                    yield final_msg, chars, state

        except Exception as e:
            import traceback
            self._log("Analysis error: %s\n%s" % (e, traceback.format_exc()))
            yield "Error: %s" % e, [], state

    def _on_load_analysis(self, analysis_file_path):
        """Load a previously saved character analysis from a JSON file."""
        if not analysis_file_path:
            yield "No file selected.", [], {"loaded": False}
            return

        try:
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            segment_tags, char_list, dedup_map = CharacterAnalyzer.load_analysis(
                analysis_file_path,
            )
            self._segment_tags = segment_tags
            self._discovered_chars = char_list
            self._dedup_map = dedup_map

            chars = []
            tags = list(self._segment_tags.values())
            for cn in sorted(self._discovered_chars):
                emotion_list = list(set(
                    t.emotion for t in tags
                    if t.character_name is not None and
                    self._dedup_map.get(t.character_name, t.character_name) == cn
                ))
                seg_count = sum(
                    1 for t in tags
                    if t.character_name is not None and
                    self._dedup_map.get(t.character_name, t.character_name) == cn
                )
                chars.append({
                    "character": cn,
                    "suggested_voice": "narrator_male",
                    "segments": seg_count,
                    "emotions": emotion_list,
                })

            self._log("Loaded analysis: %d characters, %d segments" % (
                len(char_list), len(segment_tags),
            ))
            state = {"loaded": True, "parsed": True, "analyzed": True}
            yield "Loaded %d characters" % len(char_list), chars, state
        except Exception as e:
            self._log("Error loading analysis: %s" % e)
            yield "Error: %s" % e, [], {"loaded": False}

    def _refresh_character_voices(self, narrator_voice):
        """Rebuild the character voices UI from discovered characters."""
        if not self._discovered_chars:
            return gr.update(), "No characters discovered yet. Run analysis first."

        char_data = []
        for cn in self._discovered_chars:
            seg_count = len(self._analyzer.get_character_segments(cn))
            emotions = list(set(
                t.emotion for t in self._segment_tags.values()
                if t.character_name == cn
            ))
            char_data.append({
                "character": cn, "segments": seg_count,
                "emotions": list(emotions),
                "voice_assigned": "narrator_male",
                "has_reference": "No",
            })

        status = "Found %d characters:\n" % len(self._discovered_chars)
        for c in char_data:
            status += "  - %s: %d segments, emotions=%s\n" % (
                c["character"], c["segments"], c["emotions"],
            )
        return gr.update(value=char_data), status

    def _on_setup_default_voices(self):
        """Create default voice profiles."""
        try:
            from audiobook_ai.tts.voice_manager import VoiceManager
            voices_dir = os.path.join(
                self._project.project_dir if self._project else tempfile.gettempdir(),
                "voices",
            )
            os.makedirs(voices_dir, exist_ok=True)
            self._voice_manager = VoiceManager(voices_dir)
            created = self._voice_manager.create_default_voices()
            voices_info = self._voice_manager.list_voices()
            return "Created %d default voices:\n%s" % (
                len(created),
                "\n".join("  - %s" % n for n in sorted(voices_info.keys())),
            )
        except Exception as e:
            self._log("Error creating default voices: %s" % e)
            return "Error: %s" % e

    def _ensure_voices_initialized(self):
        """Ensure voice manager is initialized and return voice list."""
        if self._voice_manager is None:
            voices_dir = os.path.join(
                self._project.project_dir if self._project else tempfile.gettempdir(),
                "voices",
            )
            os.makedirs(voices_dir, exist_ok=True)
            from audiobook_ai.tts.voice_manager import VoiceManager
            self._voice_manager = VoiceManager(voices_dir)
        return self._voice_manager.list_voices()

    def _on_create_voice_design(self, name, description, sample_text):
        """Create voice using VoiceDesign model."""
        if not name.strip():
            return None, "Error: Voice name required", []
        if not description.strip():
            return None, "Error: Voice description required", []

        try:
            self._log("Creating voice: %s - %s" % (name, description))
            voices = self._ensure_voices_initialized()
            if self._tts_engine is None or not self._tts_engine._initialized:
                self._log("Initializing TTS engine for VoiceDesign...")
                self._initialize_tts()

            voice_path = self._voice_manager.create_voice_with_design(
                name=name.strip(), description=description,
                example_text=sample_text or "This is a test of the generated voice.",
                tts_model=self._tts_engine,
            )
            voices = self._voice_manager.list_voices()
            return voice_path, "Voice created: %s" % name, list(voices.keys())
        except Exception as e:
            self._log("Voice design error: %s" % e)
            return None, "Error: %s" % e, (
                self._voice_manager.list_voices() if self._voice_manager else []
            )

    def _on_preview_voice(self, voice_name, sample_text):
        """Preview a voice with sample text."""
        if not voice_name:
            return None
        try:
            voices = self._voice_manager.list_voices()
            if voice_name in voices:
                ref_path = voices[voice_name].get("ref_audio", "")
                if ref_path and os.path.exists(ref_path):
                    return ref_path
        except Exception:
            pass
        return None

    def _on_narrator_voice_change(self, narrator_voice):
        """Show/hide custom narrator ref input based on selection."""
        return gr.update(visible=(narrator_voice == "custom"))

    def _on_generate_voice_descriptions(self):
        """Generate and return ElevenLabs voice descriptions for characters."""
        if not self._analyzer or not self._discovered_chars:
            return "No character data available. Run character analysis first."
        try:
            descriptions = self._analyzer.build_voice_descriptions()
            output = "=== ElevenLabs Voice Descriptions ===\n\n"
            for char, info in descriptions.items():
                output += "Character: %s\n" % char
                output += "ElevenLabs Prompt: %s\n" % info["elevenlabs_prompt"]
                output += "Speaking Segments: %s\n" % info["segment_count"]
                output += "\n" + "-" * 40 + "\n\n"
            return output
        except Exception as e:
            return "Error generating descriptions: %s" % e

    def _on_copy_voice_descriptions(self):
        """Save ElevenLabs descriptions to file for easy copying."""
        try:
            desc = self._analyzer.build_voice_descriptions()
            text_content = ""
            for char, info in desc.items():
                text_content += "Character: %s\nPrompt: %s\n\n" % (
                    char, info["elevenlabs_prompt"],
                )

            filepath = os.path.join(
                tempfile.gettempdir(), "elevenlabs_voice_prompts.txt",
            )
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text_content)
            return ("Saved to %s\n\nOpen this file, copy the prompts, "
                    "and paste into ElevenLabs Voice Design." % filepath)
        except Exception as e:
            return "Error: %s" % e

    def _on_generate_edge_tts_sample(self, voice_dropdown, sample_text, language):
        """Generate a voice sample using Edge TTS (free, no API key needed)."""
        import asyncio
        import edge_tts

        try:
            if not sample_text or not sample_text.strip():
                return None

            voice_id = voice_dropdown.split(" (")[0]
            output_mp3 = os.path.join(
                tempfile.gettempdir(),
                "edge_tts_%d.mp3" % int(time.time()),
            )

            async def _gen():
                comm = edge_tts.Communicate(sample_text, voice_id)
                await comm.save(output_mp3)

            asyncio.run(_gen())

            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(output_mp3)
            wav_path = output_mp3.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav")
            if os.path.exists(output_mp3):
                os.remove(output_mp3)

            self._log("Edge TTS sample generated: %s" % wav_path)
            return wav_path
        except Exception as e:
            self._log("Edge TTS error: %s" % e)
            raise

    def _on_generate_preview(self, text, voice, language, emotion, speed):
        """Generate a TTS preview."""
        if not text or not text.strip():
            return None

        try:
            self._log("Preview: voice=%s, lang=%s, emotion=%s" % (voice, language, emotion))
            if self._tts_engine is None or not self._tts_engine._initialized:
                self._initialize_tts()

            from audiobook_ai.analysis.character_analyzer import (
                EMOTION_INSTRUCTIONS_FR,
                EMOTION_INSTRUCTIONS_EN,
            )
            if language.lower() in ("french", "fr"):
                emotion_instr = EMOTION_INSTRUCTIONS_FR.get(
                    emotion, EMOTION_INSTRUCTIONS_FR["calm"],
                )
            else:
                emotion_instr = EMOTION_INSTRUCTIONS_EN.get(
                    emotion, EMOTION_INSTRUCTIONS_EN["calm"],
                )

            ref_audio, ref_text = "", ""
            if self._voice_manager:
                ref_audio, ref_text = self._voice_manager.get_voice(voice)

            output_path = os.path.join(
                tempfile.gettempdir(),
                "aiguibook_preview_%d.wav" % int(time.time()),
            )
            path, dur = self._tts_engine.generate(
                text=text, language=language,
                ref_audio=ref_audio or None, ref_text=ref_text or None,
                emotion_instruction=emotion_instr, output_path=output_path,
            )
            self._log("Preview generated: %.1fs, %s" % (dur, path))
            return path
        except Exception as e:
            self._log("Preview error: %s" % e)
            return None

    def _initialize_tts(self):
        """Initialize TTS engine."""
        if self._tts_engine is not None and self._tts_engine._initialized:
            return
        try:
            self._log("Initializing TTS engine...")
            from audiobook_ai.tts.qwen_engine import TTSEngine
            self._tts_engine = TTSEngine(
                model_path=self.config.get("tts", "model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                device=self.config.get("tts", "device", "cuda"),
                dtype=self.config.get("tts", "dtype", "bfloat16"),
                batch_size=self.config.get("tts", "batch_size", 4),
            )
            self._tts_engine.initialize()
            self._log("TTS engine initialized successfully")
        except Exception as e:
            self._log("TTS init error: %s" % e)
            raise

    def _on_start_generation(self, preview_only, no_validation, narrator_voice):
        """Start audiobook generation pipeline."""
        self._generation_running = True
        self._generation_paused = False
        self._generation_cancelled = False
        self._log_messages.clear()

        result_holder = {
            "progress": 0, "status": "Starting...",
            "est_time": "--:--", "progress_text": "0/0",
            "chapters": {}, "recent_audio": None, "logs": "", "error": None,
        }

        def run_pipeline():
            try:
                self._run_pipeline_thread(
                    preview_only=preview_only,
                    no_validation=no_validation,
                    narrator_voice=narrator_voice,
                    result_holder=result_holder,
                )
            except Exception as e:
                result_holder["error"] = str(e)
                result_holder["status"] = "ERROR: %s" % e
                self._log("FATAL ERROR: %s" % e)

        gen_thread = threading.Thread(target=run_pipeline, daemon=True)
        gen_thread.start()

        start_time = time.time()
        while gen_thread.is_alive():
            time.sleep(0.5)
            elapsed = time.time() - start_time
            prog = result_holder.get("progress", 0)
            status = result_holder.get("status", "Running...")
            est = result_holder.get("est_time", "--:--")
            txt = result_holder.get("progress_text", "0/0")
            chaps = result_holder.get("chapters", {})
            recent = result_holder.get("recent_audio", None)
            logs = self._get_logs()

            if self._generation_paused:
                status = "PAUSED"
                yield (
                    prog, status, est, txt, chaps, recent, logs,
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(visible=True, interactive=True),
                )
                while self._generation_paused and not self._generation_cancelled:
                    time.sleep(0.5)
                if self._generation_cancelled:
                    yield (
                        prog, "CANCELLED", "--:--", txt, chaps, recent,
                        self._get_logs(), gr.update(interactive=True),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                    )
                    return

            elapsed_mins = int(elapsed / 60)
            elapsed_secs = int(elapsed % 60)
            est_time_display = est if not est.startswith("--") else (
                "Elapsed: %02d:%02d" % (elapsed_mins, elapsed_secs)
            )
            yield (
                prog, status, est_time_display, txt, chaps, recent, logs,
                gr.update(interactive=False), gr.update(interactive=True),
                gr.update(interactive=True), gr.update(visible=False),
            )

        self._generation_running = False
        if result_holder.get("error"):
            yield (
                result_holder.get("progress", 0),
                "FAILED: %s" % result_holder["error"],
                "--:--", result_holder.get("progress_text", "0/0"),
                result_holder.get("chapters", {}),
                result_holder.get("recent_audio", None),
                self._get_logs(),
                gr.update(interactive=True), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(visible=False),
            )
        else:
            yield (
                100, "COMPLETE / TERMINÉ !", "--:--",
                result_holder.get("progress_text", "Done!"),
                result_holder.get("chapters", {}),
                result_holder.get("recent_audio", None),
                self._get_logs(),
                gr.update(interactive=True), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(visible=False),
            )

    def _run_pipeline_thread(self, preview_only, no_validation, narrator_voice, result_holder):
        """Run the full audiobook pipeline in a thread. Placeholder."""
        self._log("Pipeline started")
        result_holder["progress"] = 100
        result_holder["status"] = "Complete"

    def _on_save_settings(self, batch_size, dtype, device,
                          val_enabled, max_wer, max_retries,
                          bitrate, crossfade, normalize,
                          llm_back, lmstudio_url_val, lmstudio_model_val,
                          api_key, llm_model,
                          ollama_model_val, ollama_url_val):
        """Save all settings."""
        try:
            self.config.set("tts", "batch_size", int(batch_size))
            self.config.set("tts", "dtype", dtype)
            self.config.set("tts", "device", device)
            self.config.set("validation", "enabled", val_enabled)
            self.config.set("validation", "max_wer", int(max_wer))
            self.config.set("validation", "max_retries", int(max_retries))
            self.config.set("output", "bitrate", str(bitrate))
            self.config.set("output", "crossfade_duration", float(crossfade))
            self.config.set("output", "normalize_audio", normalize)
            self.config.set("analysis", "llm_backend", llm_back)
            self.config.set("analysis", "lmstudio_base_url", lmstudio_url_val)
            self.config.set("analysis", "lmstudio_model", lmstudio_model_val or "")
            if api_key:
                self.config.set("analysis", "openrouter_api_key", api_key)
            self.config.set("analysis", "openrouter_model", llm_model)
            self.config.set("analysis", "ollama_model", ollama_model_val)
            self.config.set("analysis", "ollama_base_url", ollama_url_val)
            self.config.save()
            return "Configuration saved / Configuration sauvegardee"
        except Exception as e:
            return "Error saving: %s" % e

    def _on_load_settings(self):
        """Load current settings into the UI."""
        try:
            batch = self.config.get("tts", "batch_size", 4)
            dtype = self.config.get("tts", "dtype", "bfloat16")
            device = self.config.get("tts", "device", "cuda")
            val_enabled = self.config.get("validation", "enabled", True)
            max_wer = self.config.get("validation", "max_wer", 15)
            max_retries_val = self.config.get("validation", "max_retries", 2)
            bitrate = self.config.get("output", "bitrate", "128k")
            crossfade = self.config.get("output", "crossfade_duration", 0.5)
            normalize = self.config.get("output", "normalize_audio", True)
            llm_backend_val = self.config.get("analysis", "llm_backend", "lmstudio")
            lmstudio_url_v = self.config.get("analysis", "lmstudio_base_url", "http://localhost:1234/v1")
            lmstudio_model_v = self.config.get("analysis", "lmstudio_model", "")
            api_key_val = self.config.get("analysis", "openrouter_api_key", "")
            llm_model_val = self.config.get("analysis", "openrouter_model", "openai/gpt-4o-mini")
            ollama_model_val = self.config.get("analysis", "ollama_model", "qwen3:32b")
            ollama_url_val = self.config.get("analysis", "ollama_base_url", "http://localhost:11434")

            model_choices, model_value = [], None
            if llm_backend_val == "lmstudio":
                from audiobook_ai.analysis.character_analyzer import get_llm_models_from_backend
                ok, models, _ = get_llm_models_from_backend(
                    "lmstudio", base_url=lmstudio_url_v,
                )
                if ok and models:
                    model_choices, model_value = models, lmstudio_model_v or models[0]
                    self.config.set("analysis", "lmstudio_model", model_value)

            return (
                batch, dtype, device, val_enabled, max_wer, max_retries_val,
                bitrate, crossfade, normalize, llm_backend_val,
                api_key_val, llm_model_val,
                ollama_model_val, ollama_url_val,
                "Configuration loaded. Models auto-detected.",
                gr.update(choices=model_choices, value=model_value),
            )
        except Exception as e:
            return (
                4, "bfloat16", "cuda", True, 15, 2,
                "128k", 0.5, True, "lmstudio",
                "", "openai/gpt-4o-mini",
                "qwen3:32b", "http://localhost:11434",
                "Error: %s" % e,
                gr.update(choices=[], value=None),
            )

    def _on_refresh_models(self, backend, lmstudio_url, openrouter_api_key, ollama_base):
        """Hit the LLM backend to discover available models."""
        from audiobook_ai.analysis.character_analyzer import get_llm_models_from_backend

        if backend == "lmstudio":
            ok, models, err = get_llm_models_from_backend(
                "lmstudio", base_url=lmstudio_url or "http://localhost:1234/v1",
            )
            if ok and models:
                return (
                    gr.update(choices=models, value=models[0]),
                    "Found %d: %s" % (len(models), ", ".join(models)),
                )
            return gr.update(choices=[], value=None), "Failed: %s" % err

        elif backend == "ollama":
            ok, models, err = get_llm_models_from_backend(
                "ollama", base_url=ollama_base or "http://localhost:11434",
            )
            if ok and models:
                return (
                    gr.update(choices=models, value=models[0]),
                    "Found %d: %s" % (len(models), ", ".join(models)),
                )
            return gr.update(choices=[], value=None), "Failed: %s" % err

        elif backend == "openrouter":
            key = openrouter_api_key or self.config.get("analysis", "openrouter_api_key", "")
            if not key:
                return gr.update(choices=[], value=None), "No OpenRouter API key set"
            ok, models, err = get_llm_models_from_backend("openrouter", api_key=key)
            if ok and models:
                top = models[:30]
                return (
                    gr.update(choices=top, value=top[0] if top else None),
                    "Found %d models (showing top %d)" % (len(models), len(top)),
                )
            return gr.update(choices=[], value=None), "Failed: %s" % err

        return gr.update(choices=[], value=None), "Unknown backend: %s" % backend

    def _on_test_llm_connection(self, backend, selected_model, lmstudio_url,
                                openrouter_api_key, openrouter_model,
                                ollama_model_val, ollama_base):
        """Test LLM connection by sending a simple chat request."""
        from audiobook_ai.analysis.character_analyzer import (
            test_llm_connection, get_llm_models_from_backend,
        )

        if backend == "lmstudio":
            model = selected_model or self.config.get("analysis", "lmstudio_model", "")
            if not model and lmstudio_url:
                ok, models, _ = get_llm_models_from_backend(
                    "lmstudio", base_url=lmstudio_url,
                )
                if ok and models:
                    model = models[0]
            if not model:
                return "No model selected. Click Refresh Models first."
            base = lmstudio_url or "http://localhost:1234/v1"
            if not base.rstrip("/").endswith("/v1"):
                base = base.rstrip("/") + "/v1"
            ok, msg = test_llm_connection(
                "lmstudio", base_url=base, model=model, timeout=60.0,
            )
            if ok:
                self.config.set("analysis", "lmstudio_model", model)
                return "SUCCESS (LM Studio): %s" % msg
            return "FAILED (LM Studio): %s" % msg

        elif backend == "ollama":
            model = selected_model or ollama_model_val or "qwen3:32b"
            base = ollama_base or "http://localhost:11434"
            ok, msg = test_llm_connection(
                "ollama", base_url=base, model=model, timeout=60.0,
            )
            if ok:
                self.config.set("analysis", "ollama_model", model)
                return "SUCCESS (Ollama): %s" % msg
            return "FAILED (Ollama): %s" % msg

        elif backend == "openrouter":
            model = selected_model or openrouter_model or "openai/gpt-4o-mini"
            key = openrouter_api_key or self.config.get("analysis", "openrouter_api_key", "")
            if not key:
                return "No OpenRouter API key."
            ok, msg = test_llm_connection(
                "openrouter", model=model, api_key=key, timeout=60.0,
            )
            if ok:
                self.config.set("analysis", "openrouter_model", model)
                return "SUCCESS (OpenRouter): %s" % msg
            return "FAILED (OpenRouter): %s" % msg

        return "Unknown backend: %s" % backend

    def launch(self, port=7860, share=False, server_name="0.0.0.0", **kwargs):
        """Launch the Gradio application."""
        if self.app is None:
            self.build()
        css_text = (
            ".log-box textarea {font-family: monospace !important; font-size: 12px !important;}\n"
            ".progress-text {font-size: 18px !important; font-weight: bold !important;}\n"
            ".status-badge {border-radius: 8px !important;}\n"
        )
        from gradio.themes import Soft, GoogleFont
        theme = Soft(
            primary_hue="violet",
            secondary_hue="blue",
            font=GoogleFont("Inter"),
        )
        self.app.queue()
        self.app.launch(
            theme=theme, css=css_text,
            server_name=server_name, server_port=port, share=share,
        )
