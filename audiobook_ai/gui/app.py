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
        """
        Args:
            config: AudiobookConfig instance
        """
        self.config = config
        self.app = None

        # State shared across tabs
        self._project = None
        self._tts_engine = None
        self._voice_manager = None
        self._analyzer = None
        self._validator = None
        self._assembly = None
        self._segment_tags = {}  # {segment_id: SpeechTag}
        self._discovered_chars = []
        self._segmentation = {}  # {chapter_idx: [TextSegment]}
        self._voice_assignments = {}  # {character_name: voice_id}
        self._generation_running = False
        self._generation_paused = False
        self._generation_cancelled = False
        self._chapter_titles = {}
        self._chapters_list = []
        self._log_messages = []

    def _log(self, msg: str):
        """Add a log message."""
        self._log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    def _get_logs(self) -> str:
        return "\n".join(self._log_messages[-200:])

    def build(self) -> gr.Blocks:
        """Build the complete Gradio application."""
        with gr.Blocks(
            title="AIGUIBook - AI Audiobook Generator / Générateur de Livres Audio IA",
            analytics_enabled=False,
        ) as self.app:

            gr.Markdown(
                "# AIGUIBook / Générateur de Livres Audio IA\n"
                "### Transform your EPUB into an audiobook with AI voices\n"
                "### Transformez votre EPUB en livre audio avec des voix IA"
            )

            # Global state
            app_state = gr.State({
                "loaded": False,
                "parsed": False,
                "analyzed": False,
                "voices_assigned": False,
            })

            with gr.Tabs():
                # ==================== TAB 1: Setup ====================
                with gr.Tab("Setup / Configuration"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            epub_file_upload = gr.File(
                                label="EPUB File / Fichier EPUB",
                                file_types=[".epub"],
                                type="filepath",
                            )
                            parse_btn = gr.Button(
                                "Parse EPUB / Analyser EPUB",
                                variant="primary",
                                size="lg",
                            )

                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Book Info / Informations du Livre")
                                book_title_display = gr.Textbox(
                                    label="Title / Titre",
                                    interactive=False,
                                )
                                book_author_display = gr.Textbox(
                                    label="Author / Auteur",
                                    interactive=False,
                                )
                                book_language_display = gr.Textbox(
                                    label="Language / Langue",
                                    interactive=False,
                                )
                                book_chapters_display = gr.Textbox(
                                    label="Chapters / Chapitres",
                                    interactive=False,
                                )

                    with gr.Row():
                        with gr.Column():
                            output_format_select = gr.Dropdown(
                                choices=["m4b", "m4a", "flac"],
                                value="m4b",
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
                                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                                ],
                                value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                label="TTS Model / Modèle TTS",
                            )

                        with gr.Column():
                            narrator_ref_input = gr.File(
                                label="Narrator Reference Audio / Audio de Référence du Narrateur\n(.wav, .mp3, min 3s)",
                                file_types=[".wav", ".mp3", ".ogg", ".flac"],
                                type="filepath",
                            )
                            setup_default_voices_btn = gr.Button(
                                "Create Default Voices / Créer Voix par Défaut",
                            )
                            voices_status_display = gr.Textbox(
                                label="Voice Status / État des Voix",
                                interactive=False,
                                lines=4,
                            )

                    parse_output = gr.Markdown("")

                    analyse_btn = gr.Button("Run Character Analysis / Analyse Personnages", variant="primary")
                    analysis_status = gr.Textbox(label="Analysis Status / État", interactive=False)
                    character_list = gr.JSON(label="Characters and Suggestions / Personnages")

                    analyse_btn.click(
                        fn=self._on_run_analysis,
                        inputs=[app_state],
                        outputs=[analysis_status, character_list, app_state],
                    )
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

                    setup_default_voices_btn.click(
                        fn=self._on_setup_default_voices,
                        inputs=[],
                        outputs=[voices_status_display],
                    )

                # ==================== TAB 2: Voices ====================
                with gr.Tab("Voices / Voix"):
                    gr.Markdown(
                        "### Character Voices / Voix des Personnages\n"
                        "Assign voices to each detected character and the narrator."
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            characters_list = gr.JSON(
                                label="Detected Characters / Personnages Détectés",
                                value=[],
                                visible=False,
                            )

                            with gr.Group():
                                gr.Markdown("#### Narrator Voice")
                                narrator_voice_select = gr.Dropdown(
                                    choices=["narrator_male", "narrator_female", "custom", "single_voice"],
                                    value="narrator_male",
                                    label="Narrator Voice / Voix du Narrateur",
                                )
                                narrator_ref_voice_input = gr.File(
                                    label="Custom Narrator Reference / Référence Personnalisée",
                                    file_types=[".wav", ".mp3"],
                                    type="filepath",
                                    visible=False,
                                )

                            with gr.Group():
                                gr.Markdown("#### Character Voice Assignments")
                                voice_assignments_row = gr.Column(visible=True)

                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Voice Design / Création de Voix")
                                voice_design_name = gr.Textbox(
                                    label="Voice Name / Nom de la Voix",
                                    placeholder="e.g., marcus, elise, narrator",
                                )
                                voice_design_desc = gr.Textbox(
                                    label="Voice Description / Description",
                                    placeholder="e.g., Deep male voice, French accent, warm and authoritative",
                                    lines=3,
                                )
                                voice_design_sample = gr.Textbox(
                                    label="Sample Text / Texte d'Exemple",
                                    placeholder="Text to test this voice...",
                                    lines=3,
                                )
                                create_voice_design_btn = gr.Button(
                                    "Create Voice / Créer Voix",
                                    variant="primary",
                                )
                                voice_design_output = gr.Audio(
                                    label="Generated Voice Preview / Aperçu de la Voix",
                                    type="filepath",
                                )

                            with gr.Group():
                                gr.Markdown("### Available Voices / Voix Disponibles")
                                available_voices_list = gr.Dropdown(
                                    choices=[],
                                    label="Select Voice to Preview",
                                    interactive=True,
                                )
                                preview_voice_btn = gr.Button("Preview Voice / Écouter")
                                preview_voice_output = gr.Audio(label="Preview")

                    voice_design_status = gr.Textbox(
                        label="Status / État", interactive=False,
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

                # ==================== TAB 3: Preview ====================
                with gr.Tab("Preview / Aperçu"):
                    gr.Markdown(
                        "### Test TTS Generation / Tester la Synthèse Vocale"
                    )

                    with gr.Row():
                        with gr.Column():
                            preview_text = gr.Textbox(
                                label="Text to Speak / Texte à Parler",
                                placeholder="Entrez le texte que vous souhaitez tester...",
                                value="Bonjour, ceci est un test de la synthèse vocale. Comment trouvez-vous cette voix ?",
                                lines=4,
                            )
                            preview_voice = gr.Dropdown(
                                choices=["narrator_male", "narrator_female"],
                                value="narrator_male",
                                label="Voice / Voix",
                            )
                            preview_language = gr.Dropdown(
                                choices=["French", "English", "Spanish", "German"],
                                value="French",
                                label="Language / Langue",
                            )
                            preview_emotion = gr.Dropdown(
                                choices=EMOTION_OPTIONS,
                                value="calm",
                                label="Emotion / Émotion",
                            )
                            preview_generate_btn = gr.Button(
                                "Generate Preview / Générer l'Aperçu",
                                variant="primary",
                            )

                        with gr.Column():
                            preview_audio = gr.Audio(
                                label="Generated Audio / Audio Généré",
                                type="filepath",
                            )
                            preview_speed = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Speed / Vitesse",
                            )

                    preview_generate_btn.click(
                        fn=self._on_generate_preview,
                        inputs=[preview_text, preview_voice, preview_language,
                                preview_emotion, preview_speed],
                        outputs=[preview_audio],
                    )

                # ==================== TAB 4: Generate ====================
                with gr.Tab("Generate / Générer"):
                    gr.Markdown(
                        "### Full Audiobook Generation / Génération Complète du Livre Audio"
                    )

                    with gr.Row():
                        with gr.Column():
                            start_btn = gr.Button(
                                "Start Generation / Commencer",
                                variant="primary",
                                size="lg",
                            )
                            pause_btn = gr.Button("Pause / Pause", interactive=False)
                            resume_btn = gr.Button("Resume / Reprendre", interactive=False, visible=False)
                            stop_btn = gr.Button("Stop / Arrêter", variant="stop", interactive=False)

                            preview_only_check = gr.Checkbox(
                                label="Preview Only (first 3 chapters) / Aperçu Seulement (3 premiers chapitres)",
                                value=False,
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
                                value="Ready / Prêt",
                                interactive=False,
                            )
                            estimated_time = gr.Textbox(
                                label="Estimated Time / Temps Estimé",
                                value="--:--",
                                interactive=False,
                            )
                            progress_text = gr.Textbox(
                                label="Progress Text / Texte de Progrès",
                                value="0/0 segments (0%)",
                                interactive=False,
                                lines=2,
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
                        label="Log / Journal",
                        interactive=False,
                        lines=12,
                        max_lines=50,
                        elem_classes=["log-box"],
                    )

                    # Button event handlers for Start/Pause/Resume/Stop
                    start_btn.click(
                        fn=self._on_start_generation,
                        inputs=[preview_only_check, no_validation_check,
                                narrator_voice_select],
                        outputs=[
                            overall_progress, current_status, estimated_time,
                            progress_text, chapter_progress, recent_audio,
                            log_output, start_btn, pause_btn, stop_btn,
                            resume_btn,
                        ],
                    )

                # ==================== TAB 5: Settings ====================
                with gr.Tab("Settings / Paramètres"):
                    gr.Markdown("### Application Settings / Paramètres de l'Application")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### TTS Settings / Paramètres TTS")
                            tts_batch_size = gr.Slider(
                                minimum=1, maximum=16, value=4, step=1,
                                label="Batch Size / Taille de Lot",
                            )
                            tts_dtype_select = gr.Dropdown(
                                choices=["bfloat16", "float16", "float32"],
                                value="bfloat16",
                                label="Precision / Précision",
                            )
                            tts_device_select = gr.Dropdown(
                                choices=["cuda", "cpu"],
                                value="cuda",
                                label="Device / Périphérique",
                            )

                        with gr.Column():
                            gr.Markdown("#### Validation / Validation")
                            validation_enabled = gr.Checkbox(
                                label="Enable Validation / Activer Validation",
                                value=True,
                            )
                            max_wer = gr.Slider(
                                minimum=5, maximum=50, value=15, step=1,
                                label="Max WER % / WER Max %",
                            )
                            max_retries = gr.Slider(
                                minimum=0, maximum=5, value=2, step=1,
                                label="Max Retries / Tentatives Max",
                            )

                        with gr.Column():
                            gr.Markdown("#### Output / Sortie")
                            output_bitrate = gr.Dropdown(
                                choices=["64k", "128k", "192k", "256k", "320k"],
                                value="128k",
                                label="Bitrate",
                            )
                            output_crossfade = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                                label="Crossfade (s) / Fondu Enchaîné (s)",
                            )
                            normalize_audio = gr.Checkbox(
                                label="Normalize Audio / Normaliser l'Audio",
                                value=True,
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### LLM Settings / Paramètres LLM")
                            llm_backend = gr.Dropdown(
                                choices=["lmstudio", "openrouter", "ollama"],
                                value="lmstudio",
                                label="LLM Backend / Moteur LLM",
                            )
                            with gr.Row():
                                lmstudio_url_field = gr.Textbox(
                                    label="LM Studio URL",
                                    value="http://localhost:1234",
                                    placeholder="http://localhost:1234",
                                )
                                lmstudio_model_field = gr.Textbox(
                                    label="LM Studio Model",
                                    value="gemma-4-26b-a4b",
                                    placeholder="gemma-4-26b-a4b",
                                )

                            with gr.Accordion("OpenRouter API (optional)", open=False):
                                openrouter_key = gr.Textbox(
                                    label="OpenRouter API Key",
                                    type="password",
                                    value="",
                                )
                                openrouter_model = gr.Textbox(
                                    label="OpenRouter Model / Modèle",
                                    value="qwen/qwen3.6-plus:free",
                                )

                            with gr.Accordion("Ollama (optional)", open=False):
                                ollama_model = gr.Textbox(
                                    label="Ollama Model / Modèle",
                                    value="qwen3:32b",
                                )
                                ollama_url_field = gr.Textbox(
                                    label="Ollama Base URL",
                                    value="http://localhost:11434",
                                )

                    save_config_btn = gr.Button(
                        "Save Configuration / Sauvegarder", variant="primary"
                    )
                    load_config_btn = gr.Button("Load Config / Charger")
                    reset_config_btn = gr.Button("Reset Config / Réinitialiser")
                    config_status = gr.Textbox(label="Status / État", interactive=False)

                    save_config_btn.click(
                        fn=self._on_save_settings,
                        inputs=[tts_batch_size, tts_dtype_select, tts_device_select,
                                validation_enabled, max_wer, max_retries,
                                output_bitrate, output_crossfade, normalize_audio,
                                llm_backend, lmstudio_url_field, lmstudio_model_field,
                                openrouter_key, openrouter_model,
                                ollama_model, ollama_url_field],
                        outputs=[config_status],
                    )

                    load_config_btn.click(
                        fn=self._on_load_settings,
                        inputs=[],
                        outputs=[tts_batch_size, tts_dtype_select, tts_device_select,
                                 validation_enabled, max_wer, max_retries,
                                 output_bitrate, output_crossfade, normalize_audio,
                                 llm_backend, openrouter_key, openrouter_model,
                                 ollama_model, ollama_url_field, config_status],
                    )

            return self.app

    def _on_parse_epub(
        self, epub_path, language, tts_model, narrator_ref,
    ):
        """Handle EPUB upload and parsing."""
        if not epub_path:
            return "", "", "", "", "Error: No EPUB file selected", {}, ""

        try:
            self._log(f"Parsing EPUB: {epub_path}")

            # Update config
            self.config.set("general", "language", language)
            self.config.set("tts", "model", tts_model)

            if narrator_ref:
                self.config.set("voices", "narrator_ref", narrator_ref)

            # Parse EPUB
            from audiobook_ai.core.epub_parser import EPUBParser
            parser = EPUBParser(epub_path)
            result = parser.parse()

            metadata = result["metadata"]
            chapters = result.get("chapters", [])
            self._chapters_list = chapters

            # Store parser reference
            self._epub_parser = parser

            # Store chapter titles
            self._chapter_titles = {}
            for ch in chapters:
                self._chapter_titles[ch.spine_order] = ch.title

            title = metadata.get("title", "Unknown")
            author = metadata.get("author", "Unknown")
            lang = metadata.get("language", "unknown")
            num_chapters = len(chapters)

            self._log(f"Parsed: {title} by {author} ({num_chapters} chapters)")

            # Create project
            work_dir = os.path.join(tempfile.gettempdir(), "aiguibook")
            output_dir = os.path.join(os.path.expanduser("~"), "audiobooks")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            from audiobook_ai.core.project import BookProject
            self._project = BookProject(
                book_title=title,
                work_dir=work_dir,
                output_dir=output_dir,
            )
            self._project.create()
            self._project.book_metadata = metadata
            self._project.total_chapters = num_chapters

            # Load saved state if any
            saved = self._project.load_state()
            if saved:
                self._log(f"Loaded saved state: {len(saved.get('segment_status_map', {}))} segments")

            state_data = {
                "loaded": True,
                "parsed": True,
                "analyzed": False,
                "voices_assigned": False,
            }

            return (
                title, author, lang,
                f"{num_chapters} chapters / {num_chapters} chapitres",
                f"Successfully parsed / Analysé avec succès:\n"
                f"Title: {title}\nAuthor: {author}\n"
                f"Language: {lang}\nChapters: {num_chapters}",
                state_data,
            )

        except Exception as e:
            logger.error(f"EPUB parse error: {e}", exc_info=True)
            self._log(f"ERROR: {e}")
            return (
                "", "", "", "",
                f"Error parsing EPUB: {e}",
                {"loaded": False, "parsed": False, "analyzed": False, "voices_assigned": False},
            )

    def _on_run_analysis(self, state):
        """Run character/emotion analysis."""
        self._ensure_voices_initialized()
        if not self._epub_parser or not getattr(self._epub_parser, '_chapters', None):
            return "No book loaded.", [], state
        chapters = self._epub_parser._chapters
        if not chapters:
            return "No chapters found.", [], state
        self._log("Analyzing characters and emotions...")
        try:
            from audiobook_ai.core.text_segmenter import TextSegmenter
            seg = TextSegmenter(max_words=150, min_words=20)
            all_segs = []
            for ch in chapters:
                all_segs.extend(seg.segment_chapter(ch.text, ch.title, ch.spine_order))
            if not all_segs:
                return "No text segments.", [], state
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            cfg = self.config.get_section("analysis")
            self._analyzer = CharacterAnalyzer(cfg)
            lang = self.config.get("general", "language", "french")
            self._segment_tags, self._discovered_chars = self._analyzer.analyze_segments(all_segs, language=lang)
            state["analyzed"] = True
            chars = []
            tags = list(self._segment_tags.values())
            for cn in self._discovered_chars:
                s = self._voice_manager.suggest_voice_for_character(cn, tags)
                chars.append({"name": cn, "suggested_voice": s["suggested_voice"], "confidence": s["confidence"], "description": s["description"]})
            self._log(f"Found {len(chars)} characters.")
            return f"Analyzed {len(all_segs)} segments. Found {len(chars)} characters.", chars, state
        except Exception as e:
            self._log(f"Error: {e}")
            return f"Failed: {e}", [], state

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
            status = f"Created {len(created)} default voices:\n" + "\n".join(
                f"  - {name}" for name in sorted(voices_info.keys())
            )
            return status
        except Exception as e:
            self._log(f"Error creating default voices: {e}")
            return f"Error: {e}"

    def _ensure_voices_initialized(self) -> Dict[str, dict]:
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
            return None, "Error: Voice name is required", self._voice_manager.list_voices() if self._voice_manager else {}

        if not description.strip():
            return None, "Error: Voice description is required", self._voice_manager.list_voices() if self._voice_manager else {}

        try:
            self._log(f"Creating voice: {name} - {description}")

            # Ensure voice manager exists
            voices = self._ensure_voices_initialized()

            # Ensure TTS engine exists for voice generation
            if self._tts_engine is None or not self._tts_engine._initialized:
                self._log("Initializing TTS engine for VoiceDesign...")
                self._initialize_tts()

            voice_path = self._voice_manager.create_voice_with_design(
                name=name.strip(),
                description=description,
                example_text=sample_text or "This is a test of the generated voice. Comment trouvez-vous cette voix ?",
                tts_model=self._tts_engine,
            )

            voices = self._voice_manager.list_voices()
            return voice_path, f"Voice created: {name}", list(voices.keys())

        except Exception as e:
            self._log(f"Voice design error: {e}")
            return None, f"Error: {e}", self._voice_manager.list_voices() if self._voice_manager else []

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
            return None
        except Exception:
            return None

    def _on_narrator_voice_change(self, narrator_voice):
        """Show/hide custom narrator ref input based on selection."""
        return gr.update(visible=(narrator_voice == "custom"))

    def _on_generate_preview(self, text, voice, language, emotion, speed):
        """Generate a TTS preview."""
        if not text.strip():
            return None

        try:
            self._log(f"Preview: voice={voice}, lang={language}, emotion={emotion}")

            # Initialize TTS if needed
            if self._tts_engine is None or not self._tts_engine._initialized:
                self._initialize_tts()

            from audiobook_ai.analysis.character_analyzer import EMOTION_INSTRUCTIONS_FR, EMOTION_INSTRUCTIONS_EN
            if language.lower() in ("french", "fr"):
                emotion_instr = EMOTION_INSTRUCTIONS_FR.get(emotion, EMOTION_INSTRUCTIONS_FR["calm"])
            else:
                emotion_instr = EMOTION_INSTRUCTIONS_EN.get(emotion, EMOTION_INSTRUCTIONS_EN["calm"])

            # Get voice reference
            ref_audio, ref_text = "", ""
            if self._voice_manager:
                ref_audio, ref_text = self._voice_manager.get_voice(voice)

            output_path = os.path.join(tempfile.gettempdir(), f"aiguibook_preview_{int(time.time())}.wav")

            path, dur = self._tts_engine.generate(
                text=text,
                language=language,
                ref_audio=ref_audio or None,
                ref_text=ref_text or None,
                emotion_instruction=emotion_instr,
                output_path=output_path,
            )
            self._log(f"Preview generated: {dur:.1f}s, {path}")
            return path
        except Exception as e:
            self._log(f"Preview error: {e}")
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
            self._log(f"TTS init error: {e}")
            raise

    def _on_start_generation(
        self, preview_only, no_validation, narrator_voice,
    ):
        """Start audiobook generation pipeline."""
        self._generation_running = True
        self._generation_paused = False
        self._generation_cancelled = False
        self._log_messages.clear()

        # Use gr.Progress for updates
        progress = gr.Progress()

        def update_progress(frac, msg):
            """Update the Gradio progress."""
            # This will be called from the thread

        # Run in a thread to not freeze the UI
        result_holder = {
            "progress": 0,
            "status": "Starting...",
            "est_time": "--:--",
            "progress_text": "0/0",
            "chapters": {},
            "recent_audio": None,
            "logs": "",
            "error": None,
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
                logger.error(f"Pipeline error: {e}", exc_info=True)
                result_holder["error"] = str(e)
                result_holder["status"] = f"ERROR: {e}"
                self._log(f"FATAL ERROR: {e}")

        # Start generation thread
        gen_thread = threading.Thread(target=run_pipeline, daemon=True)
        gen_thread.start()

        # Wait a moment and collect initial state
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

            # Check for pause/cancel
            if self._generation_paused:
                status = "PAUSED / EN PAUSE"
                yield (
                    prog, status, est, txt, chaps, recent, logs,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(visible=True, interactive=True),
                )
                while self._generation_paused and not self._generation_cancelled:
                    time.sleep(0.5)
                    if self._generation_cancelled:
                        break
                if self._generation_cancelled:
                    yield (
                        prog, "CANCELLED / ANNULÉ", "--:--", txt,
                        chaps, recent, self._get_logs(),
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                    )
                    return

            elapsed_mins = int(elapsed / 60)
            elapsed_secs = int(elapsed % 60)
            if not est.startswith("--"):
                est_time_display = est
            else:
                est_time_display = f"Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}"

            yield (
                prog, status, est_time_display, txt, chaps, recent, logs,
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(visible=False),
            )

        # Thread finished
        self._generation_running = False
        if result_holder.get("error"):
            yield (
                result_holder.get("progress", 0),
                f"FAILED: {result_holder['error']}",
                "--:--",
                result_holder.get("progress_text", "0/0"),
                result_holder.get("chapters", {}),
                result_holder.get("recent_audio", None),
                self._get_logs(),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=False),
            )
        else:
            yield (
                100,
                "COMPLETE / TERMINÉ !",
                "--:--",
                result_holder.get("progress_text", "Done!"),
                result_holder.get("chapters", {}),
                result_holder.get("recent_audio", None),
                self._get_logs(),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=False),
            )

    def _run_pipeline_thread(self, preview_only, no_validation, narrator_voice, result_holder):
        """Run the full audiobook pipeline in a thread."""
        try:
            # Check prerequisites
            if not self._project:
                result_holder["status"] = "Error: No project loaded. Parse an EPUB first."
                return

            # Initialize TTS
            self._log("Initializing TTS engine...")
            result_holder["status"] = "Loading TTS model / Chargement du modèle..."
            self._initialize_tts()

            # Initialize voice manager
            self._ensure_voices_initialized()

            # Initialize validator
            if not no_validation:
                from audiobook_ai.audio.validation import WhisperValidator
                self._validator = WhisperValidator(
                    device=self.config.get("tts", "device", "cuda"),
                )
                self._log("Whisper validator initialized")

            # Segment chapters
            self._log("Segmenting chapters...")
            result_holder["status"] = "Segmenting text / Segmentation..."

            # We need the original Chapter objects, not the dict versions
            from audiobook_ai.core.text_segmenter import TextSegmenter

            # Re-parse chapters into proper objects
            segments_by_chapter = {}
            total_segments = 0

            # Get chapters from parser if available, otherwise create from project
            if hasattr(self, '_epub_parser') and self._epub_parser:
                chapters = self._epub_parser.chapters
            else:
                chapters = []

            segmenter = TextSegmenter(max_words=150, min_words=20)

            for chapter in chapters:
                segs = segmenter.segment_chapter(
                    chapter.text,
                    chapter.title,
                    chapter.spine_order,
                )
                if segs:
                    segments_by_chapter[chapter.spine_order] = segs
                    chapter.segments = segs
                    # Register with project
                    self._project.set_chapter_segments(
                        chapter.spine_order,
                        [s.id for s in segs],
                    )
                    total_segments += len(segs)

            self._segmentation = segments_by_chapter
            self._log(f"Total segments to generate: {total_segments}")

            # Limit to preview if requested
            if preview_only:
                max_chapter = sorted(self._segmentation.keys())[:3]
                limited = {k: v for k, v in self._segmentation.items() if k in max_chapter}
                self._segmentation = limited
                total_segments = sum(len(v) for v in self._segmentation.values())
                self._log(f"Preview mode: limited to {total_segments} segments")

            # Analyze characters
            self._log("Analyzing characters and emotions...")
            result_holder["status"] = "Analyzing characters / Analyse des personnages..."

            all_segs = []
            for chapter_idx, segs in sorted(self._segmentation.items()):
                all_segs.extend(segs)

            if all_segs:
                from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
                analysis_config = self.config.get_section("analysis")
                self._analyzer = CharacterAnalyzer(analysis_config)
                language = self.config.get("general", "language", "french")
                self._segment_tags, self._discovered_chars = self._analyzer.analyze_segments(
                    all_segs, language=language,
                )
                self._log(f"Analysis complete: {len(self._discovered_chars)} characters found")

            # Set up voice assignments
            self._setup_voice_assignments(narrator_voice)

            # Generate audio per segment
            result_holder["status"] = "Generating audio / Génération audio..."
            generated_count = 0
            start_time = time.time()
            chapter_progress = {}

            for chapter_idx in sorted(self._segmentation.keys()):
                segs = self._segmentation[chapter_idx]
                ch_done = 0
                ch_total = len(segs)
                chapter_progress[f"Ch. {chapter_idx}"] = f"0/{ch_total}"
                result_holder["chapters"] = dict(chapter_progress)

                for seg in segs:
                    if self._generation_cancelled:
                        self._log("Generation cancelled by user")
                        return

                    while self._generation_paused:
                        time.sleep(1)
                        if self._generation_cancelled:
                            return

                    if self._segment_tags.get(seg.id) is None:
                        self._segment_tags[seg.id] = None

                    self._project.set_segment_status(seg.id, "generating")

                    # Get voice info
                    voice_id = "narrator"
                    emotion_instr = "Parlez d'un ton calme et naturel"

                    tag = self._segment_tags.get(seg.id)
                    if tag:
                        voice_id = tag.voice_id
                        emotion_instr = tag.emotion_instruction
                    else:
                        voice_id = "narrator"

                    # Get reference audio
                    ref_audio, ref_text = "", ""
                    if voice_id and self._voice_manager:
                        ref_audio, ref_text = self._voice_manager.get_voice(voice_id)
                    if not ref_audio:
                        # Fallback to narrator ref
                        narrator_ref = self.config.get("voices", "narrator_ref", "")
                        if narrator_ref and os.path.exists(narrator_ref):
                            ref_audio = narrator_ref

                    # Get output path
                    output_path = self._project.get_segment_audio_path(
                        chapter_idx, seg.id, voice_id,
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    language_str = "French"
                    if self.config.get("general", "language", "french") != "french":
                        lang_map = {
                            "english": "English",
                            "french": "French",
                            "spanish": "Spanish",
                            "german": "German",
                        }
                        language_str = lang_map.get(
                            self.config.get("general", "language", "french"),
                            "French",
                        )

                    try:
                        # Generate
                        audio_path, duration = self._tts_engine.generate(
                            text=seg.text,
                            language=language_str,
                            ref_audio=ref_audio or None,
                            ref_text=ref_text or None,
                            emotion_instruction=emotion_instr,
                            output_path=output_path,
                        )

                        self._project.set_segment_status(
                            seg.id, "generated",
                            metadata={"duration": duration, "audio_path": audio_path},
                        )

                        # Validate
                        if not no_validation and self._validator:
                            self._project.set_segment_status(seg.id, "validating")
                            result = self._validator.validate(
                                audio_path, seg.text,
                                language=self.config.get("general", "language", "french"),
                                max_wer=self.config.get("validation", "max_wer", 15),
                            )
                            if result.passed:
                                self._project.set_segment_status(seg.id, "validated")
                            else:
                                self._project.set_segment_status(seg.id, "failed")
                                self._log(
                                    f"Validation failed for {seg.id}: WER={result.wer:.1f}%"
                                )

                        generated_count += 1
                        ch_done += 1
                        chapter_progress[f"Ch. {chapter_idx}"] = f"{ch_done}/{ch_total}"
                        result_holder["chapters"] = dict(chapter_progress)

                        result_holder["recent_audio"] = audio_path

                    except Exception as e:
                        self._log(f"Error generating segment {seg.id}: {e}")
                        self._project.set_segment_status(seg.id, "error")

                    # Update progress
                    done, total, pct = self._project.get_progress()
                    result_holder["progress"] = pct
                    result_holder["progress_text"] = f"{done}/{total} segments ({pct}%)"

                    elapsed = time.time() - start_time
                    if done > 0:
                        est_total = elapsed / (done / total) if total > 0 else elapsed
                        remaining = est_total - elapsed
                        mins = int(remaining / 60)
                        secs = int(remaining % 60)
                        result_holder["est_time"] = f"~{mins:02d}:{secs:02d} remaining"

                    result_holder["status"] = f"Generating {seg.id} ({done}/{total})"

            # Assemble final audiobook
            self._log("Assembling final audiobook...")
            result_holder["status"] = "Assembling / Assemblage..."

            try:
                from audiobook_ai.audio.assembly import AudioAssembly
                self._assembly = AudioAssembly(self._project, self.config)

                # Assemble
                output_path = self._assembly.assemble_full_m4b(
                    chapter_titles=self._chapter_titles,
                )
                self._log(f"Final audiobook created: {output_path}")
                result_holder["status"] = f"Complete! Output: {output_path}"
                result_holder["progress"] = 100

            except Exception as e:
                self._log(f"Assembly error: {e}")
                result_holder["status"] = f"Assembly error: {e}"
                result_holder["error"] = str(e)

        except Exception as e:
            self._log(f"Pipeline fatal error: {e}")
            result_holder["error"] = str(e)
            result_holder["status"] = f"FATAL: {e}"

    def _setup_voice_assignments(self, narrator_voice: str):
        """Set up default voice assignments based on analysis."""
        self._voice_assignments = {}

        # Narrator assignment
        if narrator_voice and narrator_voice != "single_voice":
            if narrator_voice.startswith("narrator_"):
                self._voice_assignments["narrator"] = narrator_voice
            elif narrator_voice == "custom":
                narrator_ref = self.config.get("voices", "narrator_ref", "")
                if narrator_ref:
                    name = "narrator_custom"
                    self._voice_assignments["narrator"] = name
                    try:
                        self._voice_manager.register_speaker(
                            name, narrator_ref, ref_text=""
                        )
                    except Exception as e:
                        self._log(f"Warning: Could not register narrator voice: {e}")
        else:
            # Single voice mode
            self._voice_assignments["narrator"] = "narrator"
            self._voice_assignments["single_voice"] = "narrator"

        # Character assignments default to narrator
        for char_name in self._discovered_chars:
            voice_name = char_name.lower().replace(" ", "_")
            self._voice_assignments[char_name] = voice_name

            # Check if we have a voice for this character
            if self._voice_manager:
                existing = self._voice_manager.get_voice(voice_name)
                if not existing[0]:
                    # No specific voice, will use narrator fallback
                    self._voice_assignments[char_name] = "narrator"

        self._log(f"Voice assignments: {len(self._voice_assignments)} mappings")

    # ==================== Settings Tab Handlers ====================

    def _on_save_settings(
        self, batch_size, dtype, device,
        val_enabled, max_wer, max_retries,
        bitrate, crossfade, normalize,
        llm_back,
        lmstudio_url_val, lmstudio_model_val,
        api_key, llm_model,
        ollama_model_val, ollama_url_val,
    ):
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
            self.config.set("analysis", "lmstudio_model", lmstudio_model_val)
            if api_key:
                self.config.set("analysis", "openrouter_api_key", api_key)
            self.config.set("analysis", "openrouter_model", llm_model)
            self.config.set("analysis", "ollama_model", ollama_model_val)
            self.config.set("analysis", "ollama_base_url", ollama_url_val)

            self.config.save()
            return "Configuration saved / Configuration sauvegardée"
        except Exception as e:
            return f"Error saving: {e}"

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
            lmstudio_url_v = self.config.get("analysis", "lmstudio_base_url", "http://localhost:1234")
            lmstudio_model_v = self.config.get("analysis", "lmstudio_model", "gemma-4-26b-a4b")
            api_key_val = self.config.get("analysis", "openrouter_api_key", "")
            llm_model_val = self.config.get("analysis", "openrouter_model", "qwen/qwen3.6-plus:free")
            ollama_model_val = self.config.get("analysis", "ollama_model", "qwen3:32b")
            ollama_url_val = self.config.get("analysis", "ollama_base_url", "http://localhost:11434")

            return (
                batch, dtype, device,
                val_enabled, max_wer, max_retries_val,
                bitrate, crossfade, normalize,
                llm_backend_val,
                lmstudio_url_v, lmstudio_model_v,
                api_key_val, llm_model_val,
                ollama_model_val, ollama_url_val,
                "Configuration loaded / Configuration chargée",
            )
        except Exception as e:
            return (4, "bfloat16", "cuda", True, 15, 2,
                    "128k", 0.5, True, "lmstudio",
                    "http://localhost:1234", "gemma-4-26b-a4b",
                    "", "qwen/qwen3.6-plus:free",
                    "qwen3:32b", "http://localhost:11434",
                    f"Error: {e}")

    def launch(self, port=7860, share=False, server_name="0.0.0.0", **kwargs):
        """Launch the Gradio application.

        Args:
            port: Port to serve on
            share: Create a public shareable link
            server_name: Host to bind to
        """
        if self.app is None:
            self.build()

        # Store theme/css for Gradio 6.0 launch() method
        css_text = """
            .log-box textarea {font-family: monospace !important; font-size: 12px !important;}
            .progress-text {font-size: 18px !important; font-weight: bold !important;}
            .status-badge {border-radius: 8px !important;}
        """
        
        from gradio.themes import Soft, GoogleFont
        theme = Soft(
            primary_hue="violet",
            secondary_hue="blue",
            font=GoogleFont("Inter"),
        )

        self.app.queue()
        self.app.launch(
            theme=theme,
            css=css_text,
            server_name=server_name,
            server_port=port,
            share=share,
        )
