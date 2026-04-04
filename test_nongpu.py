#!/usr/bin/env python3
"""Test non-GPU components of AIGUIBook."""
import sys, os, json, logging, tempfile, shutil, subprocess

sys.path.insert(0, "/home/hermes/audiobook-ai")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

PASS = FAIL = SKIP = 0

def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS: {name}")

def fail(name, msg=""):
    global FAIL
    FAIL += 1
    print(f"  FAIL: {name} -- {msg}")

def skip_r(name, reason):
    global SKIP
    SKIP += 1
    print(f"  SKIP: {name} -- {reason}")

print("=" * 60)
print("AIGUIBook Non-GPU Component Tests")
print("=" * 60)

# --- 1. Imports ---
print("\n[1] Module imports")
for mod_name, import_str in [
    ("epub_parser", "from audiobook_ai.core.epub_parser import EPUBParser"),
    ("text_segmenter", "from audiobook_ai.core.text_segmenter import TextSegmenter"),
    ("project", "from audiobook_ai.core.project import BookProject"),
    ("config", "from audiobook_ai.core.config import AudiobookConfig"),
    ("character_analyzer", "from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer, SpeechTag"),
    ("voice_manager", "from audiobook_ai.tts.voice_manager import VoiceManager"),
    ("assembly", "from audiobook_ai.audio.assembly import AudioAssembly"),
    ("validation", "from audiobook_ai.audio.validation import WhisperValidator"),
]:
    try:
        exec(import_str)
        ok(f"{mod_name} import")
    except Exception as e:
        fail(f"{mod_name} import", str(e))

# --- 2. Config system ---
print("\n[2] Config system")
try:
    from audiobook_ai.core.config import AudiobookConfig
    cfg = AudiobookConfig()
    ok("default config created")
    ok("llm_backend=lstudio", ("lmstudio" == cfg.get("analysis", "llm_backend")))
    ok("lmstudio url", cfg.get("analysis", "lmstudio_base_url") == "http://localhost:1234")
    ok("language=french", cfg.get("general", "language") == "french")
    ok("tts model", "Qwen3-TTS" in cfg.get("tts", "model", ""))
    ok("format=m4b", cfg.get("output", "format") == "m4b")
    ok("validation enabled", cfg.get("validation", "enabled") is True)

    tmpf = os.path.join(tempfile.gettempdir(), "test_cfg.yaml")
    cfg.save(tmpf)
    cfg2 = AudiobookConfig()
    cfg2.load(tmpf)
    ok("config roundtrip", cfg2.get("analysis", "llm_backend") == "lmstudio")
    os.unlink(tmpf)
except Exception as e:
    fail("config system", f"{type(e).__name__}: {e}")

# --- 3. Text segmenter ---
print("\n[3] Text segmenter")
try:
    from audiobook_ai.core.text_segmenter import TextSegmenter
    seg = TextSegmenter(max_words=150, min_words=20)

    # French text with guillemets
    french_text = (
        "C'etait une belle matinee d'ete. Le soleil brillait sur la ville. "
        "Antoine sortit de chez lui avec un large sourire. "
        "Il avait attendu ce moment pendant des annees. "
        "La porte se referma derriere lui dans un claquement sec. "
        "\u00ab Je n'ai jamais eu si peur de ma vie \u00bb, murmura-t-il. "
        "Le vent soufflait fort sur les toits. "
        "Marie le regarda avec surprise. "
        "\u00ab Es-tu sur de vouloir faire ca ? \u00bb demanda-t-elle doucement."
    )
    segments = seg.segment_chapter(french_text, "Chapitre 1", 0)
    ok("segments created", ) if len(segments) > 0 else fail("segments created", f"got {len(segments)}")

    too_long = [s for s in segments if s.word_count > 150]
    ok("no segment > max_words") if len(too_long) == 0 else fail("no segment > max_words", f"{len(too_long)} too long")

    print(f"    Generated {len(segments)} segments:")
    for s in segments[:3]:
        print(f"      [{s.id}] ({s.word_count}w): {s.text[:70]}")

    # Empty text
    empty_segs = seg.segment_chapter("", "", 99)
    ok("empty text returns empty") if len(empty_segs) == 0 else fail("empty text returns empty", f"got {len(empty_segs)}")

except Exception as e:
    import traceback
    fail("text segmenter", f"{type(e).__name__}: {e}")

# --- 4. BookProject ---
print("\n[4] BookProject")
try:
    from audiobook_ai.core.project import BookProject

    work_dir = os.path.join(tempfile.gettempdir(), "aigui_test_proj")
    output_dir = os.path.join(tempfile.gettempdir(), "aigui_test_out")
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    proj = BookProject("Test Book", work_dir, output_dir)
    proj.create()
    ok("project created") if os.path.exists(work_dir) else fail("project created")

    proj.set_segment_status("ch0_s001", "generated")
    status = proj.get_segment_status("ch0_s001")
    ok("segment status") if status == "generated" else fail("segment status", f"got {status}")

    total = 10
    proj.set_chapter_segments(0, [f"ch0_s{str(i).zfill(3)}" for i in range(total)])
    for i in range(total):
        proj.set_segment_status(f"ch0_s{str(i).zfill(3)}", "generated")

    done, t, pct = proj.get_progress()
    ok("progress tracking") if (t == 10 and done == 10) else fail("progress tracking", f"{done}/{t}")

    proj.save_state({"test_key": "test_value"})
    loaded = proj.load_state()
    ok("state roundtrip") if loaded.get("test_key") == "test_value" else fail("state roundtrip")

    generated = proj.get_generated_segments()
    ok("generated list") if len(generated) == total else fail("generated list", f"got {len(generated)}")

    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
except Exception as e:
    import traceback
    fail("BookProject", f"{type(e).__name__}: {e}")

# --- 5. VoiceManager ---
print("\n[5] VoiceManager")
try:
    from audiobook_ai.tts.voice_manager import VoiceManager
    import numpy as np, soundfile as sf

    voices_dir = os.path.join(tempfile.gettempdir(), "aigui_test_voices")
    shutil.rmtree(voices_dir, ignore_errors=True)
    os.makedirs(voices_dir, exist_ok=True)

    vm = VoiceManager(voices_dir)
    ok("VoiceManager created")

    voice_list = vm.list_voices()
    ok("default voices >= 7") if len(voice_list) >= 7 else fail("default voices >= 7", f"got {len(voice_list)}")
    print(f"    Voices: {list(voice_list.keys())}")

    # Register a fake voice
    fake_wav = os.path.join(voices_dir, "test_voice.wav")
    audio = np.random.randn(24000 * 3).astype(np.float32) * 0.01
    sf.write(fake_wav, audio, 24000)
    vm.register_speaker("test_char", fake_wav, "Test reference transcript.")
    voices_after = vm.list_voices()
    ok("custom voice registered") if "test_char" in voices_after else fail("custom voice registered")

    ref_path, ref_text = vm.get_voice("test_char")
    ok("get_voice path") if os.path.exists(ref_path) else fail("get_voice path")
    ok("get_voice text") if ref_text == "Test reference transcript." else fail("get_voice text", f"got {ref_text}")

    shutil.rmtree(voices_dir, ignore_errors=True)
except Exception as e:
    import traceback
    fail("VoiceManager", f"{type(e).__name__}: {e}")

# --- 6. CharacterAnalyzer (without GPU, just test init with lmstudio) ---
print("\n[6] CharacterAnalyzer")
try:
    from audiobook_ai.analysis.character_analyzer import CharacterAnalyzer

    cfg = AudiobookConfig()
    analysis_cfg = cfg.get_section("analysis")

    # Test with lmstudio backend (LMStudio not running, should still parse config)
    ok("lmstudio backend configured") if analysis_cfg.get("llm_backend") == "lmstudio" else fail("lmstudio backend", f"got {analysis_cfg.get('llm_backend')}")

    # Test SpeechTag dataclass
    from audiobook_ai.analysis.character_analyzer import SpeechTag
    tag = SpeechTag(
        segment_id="ch0_s001",
        speaker_type="dialogue",
        character_name="Antoine",
        emotion="fearful",
        voice_id="antoine",
        emotion_instruction="Parlez avec peur et tremblement",
    )
    d = tag.to_dict()
    ok("SpeechTag serialization") if d["character_name"] == "Antoine" else fail("SpeechTag serialization")
    ok("SpeechTag has emotion_instr in to_dict") if "emotion_instruction" in d else fail("emotion_instruction in to_dict")

except Exception as e:
    fail("CharacterAnalyzer prep", f"{type(e).__name__}: {e}")

# --- 7. FFmpeg check ---
print("\n[7] FFmpeg availability")
try:
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
    ok("FFmpeg found") if r.returncode == 0 else fail("FFmpeg found")
    if r.returncode == 0:
        print(f"    {r.stdout.splitlines()[0]}")
except FileNotFoundError:
    fail("FFmpeg found", "not in PATH")

# --- 8. AudioAssembly (non-GPU parts) ---
print("\n[8] AudioAssembly (structure check)")
try:
    from audiobook_ai.audio.assembly import AudioAssembly
    ok("AudioAssembly import")

    # Check methods exist
    methods = ["assemble_chapter", "assemble_full_m4b", "normalize_audio", "concatenate_audio", "add_chapter_metadata"]
    for m in methods:
        ok(f"method: {m}") if hasattr(AudioAssembly, m) else fail(f"missing method: {m}")

except Exception as e:
    fail("AudioAssembly", f"{type(e).__name__}: {e}")

# --- 9. WhisperValidator (structure check) ---
print("\n[9] WhisperValidator (structure check)")
try:
    from audiobook_ai.audio.validation import WhisperValidator
    ok("WhisperValidator import")
    methods = ["validate", "get_validation_summary"]
    for m in methods:
        ok(f"method: {m}") if hasattr(WhisperValidator, m) else fail(f"missing method: {m}")
except Exception as e:
    fail("WhisperValidator", f"{type(e).__name__}: {e}")

# --- 10. Gradio app (structure check) ---
print("\n[10] Gradio app (structure check)")
try:
    from audiobook_ai.gui.app import AudiobookGUI
    ok("AudiobookGUI import")
    methods = ["build", "launch", "_on_parse_epub", "_run_pipeline_thread",
               "_on_create_voice_design", "_on_preview_voice",
               "_on_generate_preview", "_on_save_settings"]
    for m in methods:
        ok(f"method: {m}") if hasattr(AudiobookGUI, m) else fail(f"missing method: {m}")
except Exception as e:
    fail("AudiobookGUI", f"{type(e).__name__}: {e}")


# --- SUMMARY ---
print("\n" + "=" * 60)
total = PASS + FAIL + SKIP
print(f"TOTAL: {total} tests | {PASS} passed | {FAIL} failed | {SKIP} skipped")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
