"""CLI entry point for headless AIGUIBook usage."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import click

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
logger = logging.getLogger("aiguibook.cli")


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def main(ctx, verbose, config):
    """AIGUIBook CLI - AI-powered audiobook generator from EPUB

    CLI for generating multi-voice audiobooks with AI character analysis,
    emotion detection, and Whisper validation.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)

    from audiobook_ai.core.config import AudiobookConfig
    cfg = AudiobookConfig(config_path=config)
    cfg.load()
    ctx.obj["config"] = cfg


@main.command()
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True),
              help="Input EPUB file")
@click.option("--output", "-o", "output_dir", default=None,
              help="Output directory (default: ~/audiobooks)")
@click.option("--config", "-c", "config_path", default=None, type=click.Path(exists=True),
              help="Config file")
@click.option("--voices", "-v", "voices_file", default=None, type=click.Path(exists=True),
              help="JSON file with voice assignments")
@click.option("--no-validation", is_flag=True, help="Skip Whisper validation")
@click.option("--preview-only", is_flag=True, help="Only process first 3 chapters")
@click.option("--language", "-l", default="french", type=click.Choice(["french", "english"]),
              help="Primary language")
@click.option("--narrator-ref", default=None, type=click.Path(exists=True),
              help="Narrator reference audio")
@click.pass_context
def generate(ctx, input_file, output_dir, config_path, voices_file,
             no_validation, preview_only, language, narrator_ref):
    """Generate audiobook from EPUB (full pipeline).

    Pipeline: EPUB -> Parse -> Segment -> Analyze -> Assign Voices
              -> Generate Audio -> Validate -> Assemble -> M4B
    """
    console.print(Panel(
        "[bold cyan]AIGUIBook - Full Audiobook Generation[/bold cyan]\n"
        f"Input: {input_file}",
        title="AIGUIBook",
        border_style="cyan",
    ))

    config = ctx.obj["config"] if not config_path else None
    if config_path:
        from audiobook_ai.core.config import AudiobookConfig
        config = AudiobookConfig(config_path=config_path)
        config.load()
    config.set("general", "language", language)
    if narrator_ref:
        config.set("voices", "narrator_ref", narrator_ref)

    # Step 1: Parse EPUB
    console.print("\n[bold yellow]Step 1/7: Parsing EPUB...[/bold yellow]")
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), TimeElapsedColumn()) as progress:
        task = progress.add_task("Loading and parsing EPUB...", total=None)

        from audiobook_ai.core.epub_parser import EPUBParser
        parser = EPUBParser(input_file)
        result = parser.parse()

        chapters = parser.chapters
        metadata = parser.metadata

        progress.update(task, description=f"Parsed {len(chapters)} chapters")

    title = metadata.get("title", "Unknown")
    author = metadata.get("author", "Unknown")
    console.print(f"  [green]Title:[/green] {title}")
    console.print(f"  [green]Author:[/green] {author}")
    console.print(f"  [green]Language:[/green] {metadata.get('language', 'unknown')}")
    console.print(f"  [green]Chapters:[/green] {len(chapters)}")

    # Setup project
    work_dir = os.path.join(os.path.expanduser("~"), ".aiguibook", "work")
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "audiobooks")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    from audiobook_ai.core.project import BookProject
    project = BookProject(book_title=title, work_dir=work_dir, output_dir=output_dir)
    project.create()
    project.book_metadata = metadata
    project.total_chapters = len(chapters)

    # Step 2: Segment
    console.print("\n[bold yellow]Step 2/7: Segmenting text...[/bold yellow]")
    from audiobook_ai.core.text_segmenter import TextSegmenter

    segmenter = TextSegmenter(max_words=150, min_words=20)
    segments_by_chapter = segmenter.segment_full_book(chapters)

    total_segments = sum(len(segs) for segs in segments_by_chapter.values())
    console.print(f"  Total segments: [bold]{total_segments}[/bold]")

    # Preview only mode
    if preview_only:
        limited_chapters = sorted(segments_by_chapter.keys())[:3]
        segments_by_chapter = {k: v for k, v in segments_by_chapter.items() if k in limited_chapters}
        total_segments = sum(len(v) for v in segments_by_chapter.values())
        console.print(f"  [yellow]Preview mode: limited to {total_segments} segments (first 3 chapters)[/yellow]")

    for ch_idx in sorted(segments_by_chapter.keys()):
        segs = segments_by_chapter[ch_idx]
        chapter = next((c for c in chapters if c.spine_order == ch_idx), None)
        ch_title = chapter.title if chapter else f"Chapter {ch_idx+1}"
        project.set_chapter_segments(ch_idx, [s.id for s in segs])

    # Step 3: Analyze characters
    console.print("\n[bold yellow]Step 3/7: Analyzing characters and emotions...[/bold yellow]")
    all_segs = []
    for ch_idx in sorted(segments_by_chapter.keys()):
        all_segs.extend(segments_by_chapter[ch_idx])

    analysis_config = config.get_section("analysis")
    segment_tags = {}
    discovered_chars = []

    if all_segs:
        try:
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            analyzer = CharacterAnalyzer(analysis_config)
            segment_tags, discovered_chars = analyzer.analyze_segments(
                all_segs, language=language,
            )
            console.print(f"  Characters found: {len(discovered_chars)}")
            for char_name in discovered_chars:
                console.print(f"    - [cyan]{char_name}[/cyan]: {analyzer.get_character_segments(char_name)}")
        except Exception as e:
            console.print(f"  [red]Character analysis failed: {e}. Using narrator for all segments.[/red]")

    # Step 4: Load voices
    console.print("\n[bold yellow]Step 4/7: Setting up voices...[/bold yellow]")
    voices_dir = os.path.join(project.project_dir, "voices")
    os.makedirs(voices_dir, exist_ok=True)
    from audiobook_ai.tts.voice_manager import VoiceManager
    voice_manager = VoiceManager(voices_dir)
    created = voice_manager.create_default_voices()
    console.print(f"  Default voices available: {', '.join(created)}")

    # Load voice assignments from file
    voice_assignments = {}
    if voices_file:
        try:
            with open(voices_file, 'r') as f:
                voice_assignments = json.load(f)
            console.print(f"  Loaded {len(voice_assignments)} voice assignments from file")
        except Exception as e:
            console.print(f"  [red]Could not load voice assignments: {e}[/red]")

    for char_name, vid in voice_assignments.items():
        ref_audio, ref_text = voice_manager.get_voice(vid)
        if ref_audio and os.path.exists(ref_audio):
            console.print(f"  [green]{char_name}[/green] -> {vid} ({ref_audio})")
        else:
            console.print(f"  [yellow]{char_name}[/yellow] -> {vid} (no reference, will use TTS)")

    # Step 5: Initialize TTS
    console.print("\n[bold yellow]Step 5/7: Initializing TTS model...[/bold yellow]")
    from audiobook_ai.tts.qwen_engine import TTSEngine
    tts_config = config.get_section("tts")
    tts = TTSEngine(
        model_path=tts_config.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        device=tts_config.get("device", "cuda"),
        dtype=tts_config.get("dtype", "bfloat16"),
        batch_size=tts_config.get("batch_size", 4),
    )
    tts.initialize()
    console.print("  [green]TTS model loaded successfully[/green]")

    # Step 6: Generate audio
    console.print("\n[bold yellow]Step 6/7: Generating audio...[/bold yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task_total = progress.add_task("Overall progress", total=total_segments)
        generated = 0
        failed = 0

        from audiobook_ai.analysis.character_analyzer import EMOTION_INSTRUCTIONS_FR, EMOTION_INSTRUCTIONS_EN

        lang_code = language.lower()
        emotion_dict = EMOTION_INSTRUCTIONS_FR if lang_code == "french" else EMOTION_INSTRUCTIONS_EN
        language_str = "French" if lang_code == "french" else "English"

        for ch_idx in sorted(segments_by_chapter.keys()):
            segs = segments_by_chapter[ch_idx]
            chapter = next((c for c in chapters if c.spine_order == ch_idx), None)
            ch_title = chapter.title if chapter else f"Chapter {ch_idx+1}"

            ch_task = progress.add_task(f"[cyan]{ch_title}[/cyan]", total=len(segs))

            for seg in segs:
                tag = segment_tags.get(seg.id)
                voice_id = "narrator"
                emotion_instr = emotion_dict.get("calm", "Parlez d'un ton calme")

                if tag:
                    voice_id = tag.voice_id
                    emotion_instr = tag.emotion_instruction

                ref_audio, ref_text = "", ""
                if voice_id:
                    if voice_id in voice_assignments:
                        ref_audio, ref_text = voice_manager.get_voice(voice_assignments[voice_id])
                    else:
                        ref_audio, ref_text = voice_manager.get_voice(voice_id)

                narrator_ref_cfg = config.get("voices", "narrator_ref", "")
                if not ref_audio and narrator_ref_cfg and os.path.exists(narrator_ref_cfg):
                    ref_audio = narrator_ref_cfg

                output_path = project.get_segment_audio_path(ch_idx, seg.id, voice_id)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    task_desc = f"[bold green]{seg.id}[/bold green] (voice: {voice_id})"
                    progress.update(task_total, description=task_desc)

                    audio_path, duration = tts.generate(
                        text=seg.text,
                        language=language_str,
                        ref_audio=ref_audio if ref_audio else None,
                        ref_text=ref_text if ref_text else None,
                        emotion_instruction=emotion_instr,
                        output_path=output_path,
                    )

                    project.set_segment_status(
                        seg.id, "generated",
                        metadata={"duration": duration, "audio_path": audio_path},
                    )
                    generated += 1

                    # Validation
                    if not no_validation:
                        try:
                            from audiobook_ai.audio.validation import WhisperValidator
                            validator = WhisperValidator(device=config.get("tts", "device", "cuda"))
                            val_result = validator.validate(
                                audio_path, seg.text,
                                language=language,
                                max_wer=config.get("validation", "max_wer", 15),
                            )
                            if val_result.passed:
                                project.set_segment_status(seg.id, "validated")
                            else:
                                project.set_segment_status(seg.id, "failed")
                                console.print(f"\n  [red]Validation failed: {seg.id} (WER={val_result.wer:.1f}%)[/red]")
                        except Exception as ve:
                            console.print(f"  [yellow]Validation skipped: {ve}[/yellow]")

                except Exception as e:
                    console.print(f"\n  [red]ERROR generating {seg.id}: {e}[/red]")
                    project.set_segment_status(seg.id, "error")
                    failed += 1

                progress.update(task_total, advance=1)
                progress.update(ch_task, advance=1)

        console.print(f"\n  Generated: [green]{generated}[/green], Failed: [red]{failed}[/red]")

    # Step 7: Assemble
    console.print("\n[bold yellow]Step 7/7: Assembling audiobook...[/bold yellow]")
    chapter_titles = {}
    for ch in chapters:
        chapter_titles[ch.spine_order] = ch.title

    from audiobook_ai.audio.assembly import AudioAssembly
    assembly = AudioAssembly(project, config)
    output_path = assembly.assemble_full_m4b(chapter_titles=chapter_titles)

    console.print(f"\n  [bold green]Audiobook created: {output_path}[/bold green]")

    # Summary
    done, total, pct = project.get_progress()
    console.print(Panel(
        f"[bold]Title:[/bold] {title}\n"
        f"[bold]Author:[/bold] {author}\n"
        f"[bold]Output:[/bold] {output_path}\n"
        f"[bold]Segments:[/bold] {done}/{total} ({pct}%)\n"
        f"[bold]Failed:[/bold] {failed}",
        title="Generation Complete",
        border_style="green",
    ))


@main.command()
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True),
              help="Input EPUB file")
@click.pass_context
def parse(ctx, input_file):
    """Parse an EPUB file and display metadata/chapters."""
    console.print(Panel(
        f"[bold cyan]Parsing EPUB: {input_file}[/bold cyan]",
        title="AIGUIBook - EPUB Parser",
        border_style="cyan",
    ))

    from audiobook_ai.core.epub_parser import EPUBParser

    parser = EPUBParser(input_file)
    result = parser.parse()

    metadata = result["metadata"]
    chapters = result.get("chapters", [])
    toc = result.get("toc", [])

    # Display metadata
    table = Table(title="Book Metadata")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for key, val in metadata.items():
        table.add_row(key, str(val))
    console.print(table)

    # Display chapters
    console.print(f"\n[bold]Chapters ({len(chapters)}):[/bold]")
    ch_table = Table()
    ch_table.add_column("#", style="dim")
    ch_table.add_column("Title", style="bold")
    ch_table.add_column("Words", style="yellow")
    for ch in chapters:
        word_count = len(ch.get("text", "").split())
        ch_table.add_row(
            str(ch.get("spine_order", "?")),
            ch.get("title", "Untitled"),
            str(word_count),
        )
    console.print(ch_table)

    # Display TOC
    if toc:
        console.print(f"\n[bold]Table of Contents:[/bold]")
        toc_table = Table()
        toc_table.add_column("Title", style="cyan")
        toc_table.add_column("Href", style="dim")
        toc_table.add_column("Children", style="yellow")
        for entry in toc:
            toc_table.add_row(
                entry.get("title", ""),
                entry.get("href", ""),
                str(len(entry.get("children", []))),
            )
        console.print(toc_table)

    # Summary
    full_text = parser.get_full_text()
    total_words = len(full_text.split())
    console.print(f"\n[bold green]Total chapters: {len(chapters)}, Word count: {total_words}[/bold green]")


@main.command()
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True),
              help="Input EPUB file")
@click.option("--language", "-l", default="french", type=click.Choice(["french", "english"]))
@click.option("--output", "-o", "output_file", default=None,
              help="Save analysis results as JSON")
@click.pass_context
def analyze(ctx, input_file, language, output_file):
    """Run character/emotion analysis on an EPUB file."""
    console.print(Panel(
        f"[bold cyan]Character analysis: {input_file}[/bold cyan]",
        title="AIGUIBook - Character Analyzer",
        border_style="cyan",
    ))

    # Parse EPUB
    from audiobook_ai.core.epub_parser import EPUBParser
    parser = EPUBParser(input_file)
    result = parser.parse()
    chapters = parser.chapters

    # Segment
    from audiobook_ai.core.text_segmenter import TextSegmenter
    segmenter = TextSegmenter(max_words=150, min_words=20)
    segments_by_chapter = segmenter.segment_full_book(chapters)

    all_segs = []
    for ch_idx in sorted(segments_by_chapter.keys()):
        all_segs.extend(segments_by_chapter[ch_idx])

    if not all_segs:
        console.print("[yellow]No segments found[/yellow]")
        return

    console.print(f"Total segments to analyze: [bold]{len(all_segs)}[/bold]")

    # Analyze
    config = ctx.obj["config"]
    analysis_config = config.get_section("analysis")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Running character analysis...", total=None)

        try:
            from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer
            analyzer = CharacterAnalyzer(analysis_config)
            segment_tags, discovered_chars = analyzer.analyze_segments(
                all_segs, language=language,
            )
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            raise SystemExit(1)

    # Display results
    console.print(f"\n[bold]Discovered Characters ({len(discovered_chars)}):[/bold]")
    for char_name in discovered_chars:
        segs = analyzer.get_character_segments(char_name)
        console.print(f"  [cyan]{char_name}[/cyan]: {len(segs)} segments")

    # Display segment tags summary
    emotion_counts = {}
    narrator_count = 0
    dialogue_count = 0
    for tag in segment_tags.values():
        if tag.speaker_type == "narrator":
            narrator_count += 1
        else:
            dialogue_count += 1
        emotion = tag.emotion
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
        emotion_counts[emotion] += 1

    summary_table = Table(title="Analysis Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total segments", str(len(segment_tags)))
    summary_table.add_row("Narrator segments", str(narrator_count))
    summary_table.add_row("Dialogue segments", str(dialogue_count))
    summary_table.add_row("Characters", str(len(discovered_chars)))
    for emotion, count in sorted(emotion_counts.items()):
        summary_table.add_row(f"  - {emotion}", str(count))
    console.print(summary_table)

    # Save to file
    if output_file:
        result_data = {
            "characters": discovered_chars,
            "segments": {sid: tag.to_dict() for sid, tag in segment_tags.items()},
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Analysis saved to: {output_file}[/green]")


@main.command()
@click.option("--download-models", is_flag=True, help="Download required models")
@click.pass_context
def setup(ctx, download_models):
    """First-time setup: download models and test GPU."""
    console.print(Panel(
        "[bold cyan]AIGUIBook Setup[/bold cyan]\n"
        "Configuring AIGUIBook for first use",
        border_style="cyan",
    ))

    # Test GPU
    console.print("\n[bold]System Check:[/bold]")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        console.print(f"  PyTorch: [green]{torch.__version__}[/green]")
        console.print(f"  CUDA available: [green]{cuda_available}[/green]")
        if cuda_available:
            console.print(f"  GPU: [green]{torch.cuda.get_device_name(0)}[/green]")
            console.print(f"  GPU memory: [green]{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB[/green]")
        else:
            console.print("  [yellow]No GPU available. TTS will be very slow on CPU.[/yellow]")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")
        return

    # Test ffmpeg
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            console.print(f"  FFmpeg: [green]installed[/green]")
        else:
            console.print("  [red]FFmpeg not working properly[/red]")
    except FileNotFoundError:
        console.print("  [red]FFmpeg not found. Install with: apt install ffmpeg[/red]")

    # Download models
    if download_models:
        console.print("\n[bold]Downloading TTS model...[/bold]")
        console.print("  This may take several minutes depending on your internet connection.")
        console.print("  Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base")

        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                local_dir=os.path.expanduser("~/.cache/huggingface/hub/Qwen3-TTS-12Hz-1.7B-Base"),
            )
            console.print(f"  [green]Model downloaded to: {model_path}[/green]")
        except Exception as e:
            console.print(f"  [red]Model download failed: {e}[/red]")
            console.print("  You can also run the model directly in Gradio mode - it will download on first use.")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("  Run 'aiguibook-gui' to start the GUI or 'python main.py'")
    console.print("  Run 'aiguibook generate --input your_book.epub' to generate from CLI")


@main.command("voices")
@click.option("--list", "list_voices", is_flag=True, help="List available voices")
@click.option("--create", type=str, nargs=2, metavar="NAME REF_PATH", help="Create voice from reference")
@click.option("--create-design", type=str, nargs=3, metavar="NAME DESCRIPTION SAMPLE", help="Design voice")
@click.option("--delete", type=str, help="Delete a voice profile")
@click.pass_context
def voices_cmd(ctx, list_voices, create, create_design, delete):
    """List, create, and manage voice profiles."""
    config = ctx.obj["config"]
    voices_dir = os.path.join(os.path.expanduser("~"), ".aiguibook", "voices")
    os.makedirs(voices_dir, exist_ok=True)

    from audiobook_ai.tts.voice_manager import VoiceManager
    vm = VoiceManager(voices_dir)

    if list_voices:
        voices = vm.list_voices()
        if not voices:
            console.print("[yellow]No voice profiles found. Run 'voices --create' to add one.[/yellow]")
            return

        table = Table(title="Available Voices")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Ref Audio", style="green")
        for name, info in voices.items():
            table.add_row(
                name,
                info.get("type", "unknown"),
                "Yes" if info.get("exists") else "No",
            )
        console.print(table)

    if create:
        name, ref_path = create
        try:
            if not os.path.exists(ref_path):
                console.print(f"[red]Reference not found: {ref_path}[/red]")
                return
            vm.register_speaker(name, ref_path)
            console.print(f"[green]Voice '{name}' created from {ref_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    if create_design:
        name, description, sample = create_design
        try:
            # Need TTS model for voice design
            console.print("[yellow]Loading TTS model for VoiceDesign...[/yellow]")
            from audiobook_ai.tts.qwen_engine import TTSEngine
            tts_config = config.get_section("tts")
            tts = TTSEngine(
                model_path=tts_config.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                device=tts_config.get("device", "cuda"),
                dtype=tts_config.get("dtype", "bfloat16"),
            )
            tts.initialize()

            path = vm.create_voice_with_design(name, description, sample, tts_model=tts)
            console.print(f"[green]Voice designed: {name} -> {path}[/green]")
        except Exception as e:
            console.print(f"[red]Voice design failed: {e}[/red]")

    if delete:
        if vm.delete_voice(delete):
            console.print(f"[green]Voice '{delete}' deleted[/green]")
        else:
            console.print(f"[yellow]Voice '{delete}' not found[/yellow]")

    if not (list_voices or create or create_design or delete):
        console.print("Use --list, --create, --create-design, or --delete")


if __name__ == "__main__":
    main()
