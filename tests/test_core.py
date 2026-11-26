"""Unit tests for Stable Audio Lab core modules."""

import unittest

from core.compat import get_device_info, setup_environment
from core.engine import (
    _INFERENCE_LOCK,
    _MODEL_CACHE,
    GenerationConfig,
    GenerationResult,
    apply_post_processing,
    delete_output_file,
    generate_audio,
    get_loaded_model_name,
    list_recent_outputs,
    parse_cfg_sweep,
    read_wav_metadata,
    slugify,
    unload_model,
    validate_cfg_sweep,
)
from core.exceptions import (
    GenerationCancelledError,
    GenerationError,
    ModelNotFoundError,
    StableAudioError,
)
from core.registry import MODELS, ModelSpec, get_model_spec
from core.storage import CacheStatus, check_model_cache, is_model_cached


class TestSlugify(unittest.TestCase):
    """Test slugify function for prompt sanitization."""

    def test_basic_slug(self):
        self.assertEqual(slugify("Upbeat funky bassline"), "upbeat_funky_bassline")

    def test_special_characters(self):
        self.assertEqual(
            slugify("80s Japanese City-Pop! (w/ bright slap bass)"),
            "80s_japanese_city_pop_w_bright",
        )

    def test_empty_and_whitespace(self):
        self.assertEqual(slugify(""), "audio")
        self.assertEqual(slugify("    "), "audio")
        self.assertEqual(slugify("!!!???"), "audio")

    def test_trailing_punctuation_stripped(self):
        slug = slugify("Fast aggressive heavy metal...")
        self.assertFalse(slug.endswith("_"))
        self.assertFalse(slug.startswith("_"))


class TestRegistry(unittest.TestCase):
    """Test model registry and specifications."""

    def test_models_exist(self):
        self.assertIn("small-music", MODELS)
        self.assertIn("small-sfx", MODELS)
        self.assertIn("medium", MODELS)

    def test_get_model_spec(self):
        spec = get_model_spec("small-music")
        self.assertIsInstance(spec, ModelSpec)
        self.assertEqual(spec.name, "small-music")
        self.assertGreater(spec.max_duration, 0)
        self.assertGreater(spec.default_duration, 0)
        self.assertTrue(len(spec.examples) > 0)

    def test_get_model_spec_invalid(self):
        with self.assertRaises(ModelNotFoundError):
            get_model_spec("nonexistent-model")


class TestCompat(unittest.TestCase):
    """Test platform and hardware compatibility helpers."""

    def test_setup_environment_idempotent(self):
        setup_environment()
        setup_environment()

    def test_device_info(self):
        info = get_device_info()
        self.assertIsInstance(info, str)
        self.assertTrue(len(info) > 0)
        self.assertIn("Hardware", info)


class TestStorage(unittest.TestCase):
    """Test model cache status checking."""

    def test_check_model_cache_structure(self):
        st = check_model_cache("small-music")
        self.assertIsInstance(st, CacheStatus)
        self.assertIsInstance(st.downloaded, bool)
        self.assertIsInstance(st.size_gb, float)
        self.assertIsInstance(st.status_text, str)

        d = st.to_dict()
        self.assertIn("downloaded", d)
        self.assertIn("size_gb", d)
        self.assertIn("status_text", d)

    def test_is_model_cached(self):
        self.assertIsInstance(is_model_cached("small-music"), bool)


class TestGenerationConfig(unittest.TestCase):
    """Test GenerationConfig dataclass and validation."""

    def test_valid_config(self):
        cfg = GenerationConfig(prompt="funky bass", duration=15.0, steps=8, cfg_scale=1.0)
        cfg.validate()
        self.assertEqual(cfg.prompt, "funky bass")

    def test_empty_prompt_raises(self):
        with self.assertRaises(GenerationError):
            GenerationConfig(prompt="").validate()
        with self.assertRaises(GenerationError):
            GenerationConfig(prompt="   ").validate()

    def test_negative_duration_raises(self):
        with self.assertRaises(GenerationError):
            GenerationConfig(prompt="rock", duration=-5.0).validate()
        with self.assertRaises(GenerationError):
            GenerationConfig(prompt="rock", duration=0.0).validate()

    def test_invalid_steps_raises(self):
        with self.assertRaises(GenerationError):
            GenerationConfig(prompt="rock", steps=0).validate()
        with self.assertRaises(GenerationError):
            GenerationConfig(prompt="rock", steps=-1).validate()

    def test_invalid_cfg_raises(self):
        with self.assertRaises(GenerationError):
            GenerationConfig(prompt="rock", cfg_scale=-0.5).validate()

    def test_invalid_model_raises(self):
        with self.assertRaises(ModelNotFoundError):
            GenerationConfig(model_name="nonexistent-model", prompt="rock").validate()

    def test_chunked_decode_default_true(self):
        cfg = GenerationConfig(prompt="test")
        self.assertTrue(cfg.chunked_decode)
        cfg_off = GenerationConfig(prompt="test", chunked_decode=False)
        self.assertFalse(cfg_off.chunked_decode)

    def test_seed_resolution(self):
        cfg = GenerationConfig(prompt="rock", seed=-1)
        resolved = cfg.resolve_seed()
        self.assertGreaterEqual(resolved, 0)

        cfg_fixed = GenerationConfig(prompt="rock", seed=42)
        self.assertEqual(cfg_fixed.resolve_seed(), 42)


class TestGenerationResult(unittest.TestCase):
    """Test GenerationResult namedtuple and tuple unpacking."""

    def test_result_unpacking(self):
        res = GenerationResult(
            output_path="outputs/test.wav",
            status_message="Generated 15s in 2s",
            duration=15.0,
            elapsed=2.0,
            speed=4.0,
            seed=123,
        )
        self.assertEqual(res.output_path, "outputs/test.wav")
        self.assertEqual(res.duration, 15.0)

        # Verify backwards-compatible tuple unpacking
        path, status = res
        self.assertEqual(path, "outputs/test.wav")
        self.assertEqual(status, "Generated 15s in 2s")


class TestParseCfgSweep(unittest.TestCase):
    """Test parse_cfg_sweep parsing and validation."""

    def test_default_sweep(self):
        self.assertEqual(parse_cfg_sweep(""), [1.0, 1.5, 2.0, 3.0])
        self.assertEqual(parse_cfg_sweep("   "), [1.0, 1.5, 2.0, 3.0])

    def test_custom_sweep(self):
        self.assertEqual(parse_cfg_sweep("1.0, 2.5, 4.0, 6.0"), [1.0, 2.5, 4.0, 6.0])

    def test_max_five_values(self):
        self.assertEqual(parse_cfg_sweep("1, 2, 3, 4, 5, 6"), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_invalid_tokens_ignored(self):
        self.assertEqual(parse_cfg_sweep("1.0, abc, 2.5, -1, 3.0"), [1.0, 2.5, 3.0])


class TestValidateCfgSweep(unittest.TestCase):
    """Test validate_cfg_sweep for strict user input validation."""

    def test_valid_sweeps(self):
        valid, vals, msg = validate_cfg_sweep("1.0, 2.0")
        self.assertTrue(valid)
        self.assertEqual(vals, [1.0, 2.0])

        valid, vals, msg = validate_cfg_sweep("1.0, 1.5, 2.0, 3.0, 4.5")
        self.assertTrue(valid)
        self.assertEqual(len(vals), 5)
        self.assertIn("Valid", msg)

    def test_empty_or_whitespace(self):
        valid, vals, msg = validate_cfg_sweep("")
        self.assertFalse(valid)
        self.assertEqual(vals, [])
        self.assertIn("Please enter 2 to 5 numbers", msg)

        valid, vals, msg = validate_cfg_sweep("   ,   ")
        self.assertFalse(valid)

    def test_too_few_values(self):
        valid, vals, msg = validate_cfg_sweep("2.5")
        self.assertFalse(valid)
        self.assertEqual(vals, [2.5])
        self.assertIn("Too few values", msg)

    def test_too_many_values(self):
        valid, vals, msg = validate_cfg_sweep("1.0, 2.0, 3.0, 4.0, 5.0, 6.0")
        self.assertFalse(valid)
        self.assertEqual(len(vals), 6)
        self.assertIn("Too many values", msg)

    def test_out_of_bounds_values(self):
        valid, vals, msg = validate_cfg_sweep("0.5, 2.0")
        self.assertFalse(valid)
        self.assertIn("out of range", msg)

        valid, vals, msg = validate_cfg_sweep("1.0, 16.0")
        self.assertFalse(valid)
        self.assertIn("out of range", msg)

    def test_non_numeric_tokens(self):
        valid, vals, msg = validate_cfg_sweep("1.0, abc, 2.0")
        self.assertFalse(valid)
        self.assertIn("Invalid numeric value", msg)
        self.assertIn("'abc'", msg)

    def test_trailing_leading_commas_and_spaces(self):
        valid, vals, msg = validate_cfg_sweep(" , 1.0, 2.5 , 3.0 , ")
        self.assertTrue(valid)
        self.assertEqual(vals, [1.0, 2.5, 3.0])


class TestMetadataAndConfig(unittest.TestCase):
    """Test metadata config flag and RIFF INFO audio metadata roundtrip."""

    def test_embed_metadata_default_true(self):
        cfg = GenerationConfig(prompt="test")
        self.assertTrue(cfg.embed_metadata)

    def test_embed_metadata_can_be_disabled(self):
        cfg = GenerationConfig(prompt="test", embed_metadata=False)
        self.assertFalse(cfg.embed_metadata)

    def test_metadata_roundtrip(self):
        import json
        import tempfile
        import numpy as np
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            meta = {
                "model": "small-music",
                "prompt": "synthwave beat",
                "negative_prompt": "",
                "duration": 5.0,
                "steps": 8,
                "cfg_scale": 1.5,
                "seed": 123456,
            }
            with sf.SoundFile(tmp_path, mode="w", samplerate=44100, channels=2, subtype="PCM_16") as f:
                f.comment = json.dumps(meta)
                f.title = "synthwave beat"
                f.artist = "Stable Audio Lab"
                f.write(np.zeros((4410, 2), dtype=np.float32))

            with sf.SoundFile(tmp_path) as f:
                self.assertEqual(f.title, "synthwave beat")
                self.assertEqual(f.artist, "Stable Audio Lab")
                loaded = json.loads(f.comment)
                self.assertEqual(loaded["seed"], 123456)
                self.assertEqual(loaded["cfg_scale"], 1.5)
                self.assertEqual(loaded["model"], "small-music")
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestExceptions(unittest.TestCase):
    """Test domain exception hierarchy."""

    def test_hierarchy(self):
        self.assertTrue(issubclass(ModelNotFoundError, StableAudioError))
        self.assertTrue(issubclass(GenerationError, StableAudioError))
        self.assertTrue(issubclass(GenerationCancelledError, StableAudioError))
        self.assertTrue(issubclass(ModelNotFoundError, ValueError))
        self.assertTrue(issubclass(GenerationError, ValueError))


class TestGenerationLifecycle(unittest.TestCase):
    """Test output clearing on valid generation start and retention on error."""

    def test_invalid_sweep_blocks_without_yielding_clear(self):
        import gradio as gr
        from app import generate

        gen = generate(
            model_name="small-sfx",
            prompt="wind sounds",
            negative_prompt="",
            duration=5.0,
            steps=8,
            cfg=1.0,
            cfg_sweep_text="1.0",
            is_sweep=True,
            embed_metadata=False,
            seed=42,
        )
        with self.assertRaises(gr.Error) as ctx:
            next(gen)
        self.assertIn("Invalid CFG sweep input", str(ctx.exception))

    def test_empty_prompt_blocks_immediately(self):
        import gradio as gr
        from app import generate

        gen = generate(
            model_name="small-sfx",
            prompt="   ",
            negative_prompt="",
            duration=5.0,
            steps=8,
            cfg=1.0,
            cfg_sweep_text="1.0, 2.0",
            is_sweep=True,
            embed_metadata=False,
            seed=42,
        )
        with self.assertRaises(gr.Error) as ctx:
            next(gen)
        self.assertIn("Prompt cannot be empty", str(ctx.exception))

    def test_valid_start_clears_outputs_immediately(self):
        from app import generate

        gen = generate(
            model_name="small-sfx",
            prompt="ocean waves",
            negative_prompt="",
            duration=5.0,
            steps=8,
            cfg=1.0,
            cfg_sweep_text="1.0, 2.0, 3.0",
            is_sweep=True,
            embed_metadata=False,
            seed=42,
        )
        first_step = next(gen)
        out_single, s1, s2, s3, s4, s5, _last_seed, status = first_step
        self.assertIsNone(out_single)
        self.assertIsNone(s1["value"])
        self.assertIsNone(s2["value"])
        self.assertIsNone(s3["value"])
        self.assertIsNone(s4["value"])
        self.assertIsNone(s5["value"])
        self.assertIn("Initializing small-sfx", status)


class TestRobustnessAndCancellation(unittest.TestCase):
    """Test model unloading, cancellation, heartbeat, thread-safe progress, and OOM recovery."""

    def tearDown(self):
        _MODEL_CACHE.clear()

    def test_unload_model_clears_cache(self):
        self.assertIsNone(get_loaded_model_name())
        _MODEL_CACHE["dummy"] = object()
        self.assertEqual(get_loaded_model_name(), "dummy")
        evicted = unload_model()
        self.assertEqual(evicted, "dummy")
        self.assertEqual(len(_MODEL_CACHE), 0)
        self.assertIsNone(get_loaded_model_name())
        self.assertIsNone(unload_model())

    def test_unload_model_raises_when_inference_locked(self):
        _INFERENCE_LOCK.acquire()
        try:
            with self.assertRaises(GenerationError):
                unload_model()
        finally:
            _INFERENCE_LOCK.release()

    def test_read_wav_metadata_and_list_recent_outputs(self):
        import json
        import os
        import tempfile
        import time
        import numpy as np
        import soundfile as sf

        self.assertIsNone(read_wav_metadata("nonexistent_file_12345.wav"))
        self.assertEqual(list_recent_outputs("nonexistent_dir_12345"), [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav1 = os.path.join(tmp_dir, "older_no_meta.wav")
            wav2 = os.path.join(tmp_dir, "newer_with_meta.wav")

            with sf.SoundFile(wav1, mode="w", samplerate=44100, channels=2, subtype="PCM_16") as f:
                f.write(np.zeros((4410, 2), dtype=np.float32))

            time.sleep(0.05)

            meta = {
                "model": "medium",
                "prompt": "cinematic strings",
                "negative_prompt": "distortion",
                "duration": 30.0,
                "steps": 8,
                "cfg_scale": 2.0,
                "seed": 999,
            }
            with sf.SoundFile(wav2, mode="w", samplerate=44100, channels=2, subtype="PCM_16") as f:
                f.comment = json.dumps(meta)
                f.write(np.zeros((4410, 2), dtype=np.float32))

            self.assertIsNone(read_wav_metadata(wav1))
            parsed = read_wav_metadata(wav2)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["model"], "medium")
            self.assertEqual(parsed["seed"], 999)

            records = list_recent_outputs(tmp_dir, limit=10)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["filename"], "newer_with_meta.wav")
            self.assertTrue(records[0]["has_metadata"])
            self.assertEqual(records[0]["prompt"], "cinematic strings")
            self.assertEqual(records[1]["filename"], "older_no_meta.wav")
            self.assertFalse(records[1]["has_metadata"])

    def test_generate_audio_cancelled_releases_lock(self):
        import threading

        cancel_event = threading.Event()
        cancel_event.set()
        with self.assertRaises(GenerationCancelledError):
            generate_audio(
                model_name="small-sfx",
                prompt="ocean waves",
                duration=5.0,
                steps=4,
                cancel_event=cancel_event,
            )
        # Lock must remain unheld after cancellation
        acquired = _INFERENCE_LOCK.acquire(blocking=False)
        self.assertTrue(acquired)
        _INFERENCE_LOCK.release()

    def test_app_generate_progress_proxy_and_heartbeat(self):
        import time
        from unittest.mock import patch
        from app import generate

        progress_calls = []

        def fake_progress(value, desc=None, unit=None):
            progress_calls.append((value, desc, unit))

        def fake_generate_audio(config, progress=None, status_callback=None, cancel_event=None):
            if status_callback:
                status_callback("[load] [1/4] Building architecture...")
            if progress:
                progress(None, desc="Building...")
            # Stay silent longer than the 2s heartbeat threshold
            time.sleep(2.2)
            if progress:
                progress((1, 2), desc="Sampling", unit="steps")
                progress((2, 2), desc="Sampling", unit="steps")
            return GenerationResult(
                output_path="outputs/fake.wav",
                status_message="Generated 5.0s audio in 2.20s (4.0 steps/s) | Model: small-sfx | Seed: 42 | Saved: fake.wav",
                duration=5.0,
                elapsed=2.2,
                speed=4.0,
                seed=42,
            )

        with patch("app.generate_audio", side_effect=fake_generate_audio):
            outputs = list(
                generate(
                    model_name="small-sfx",
                    prompt="ocean waves",
                    negative_prompt="",
                    duration=5.0,
                    steps=2,
                    cfg=1.0,
                    cfg_sweep_text="1.0, 2.0",
                    is_sweep=False,
                    embed_metadata=False,
                    seed=42,
                    progress=fake_progress,
                )
            )

        # Verify progress events were executed on main thread with expected payloads
        self.assertIn((None, "Building...", None), progress_calls)
        self.assertIn(((1, 2), "Sampling", "steps"), progress_calls)
        self.assertIn(((2, 2), "Sampling", "steps"), progress_calls)

        # Verify at least one heartbeat update appended live elapsed seconds
        statuses = [item[-1] for item in outputs]
        self.assertTrue(any("Building architecture... (" in s and "s)" in s for s in statuses))
        self.assertEqual(outputs[-1][0], "outputs/fake.wav")

    def test_app_generate_sets_cancel_event_on_close(self):
        import time
        from unittest.mock import patch
        from app import generate

        captured = {}

        def blocking_generate_audio(config, progress=None, status_callback=None, cancel_event=None):
            captured["event"] = cancel_event
            if status_callback:
                status_callback("[load] [1/4] Working...")
            while not cancel_event.is_set():
                time.sleep(0.05)
            raise GenerationCancelledError("Generation cancelled.")

        with patch("app.generate_audio", side_effect=blocking_generate_audio):
            gen = generate(
                model_name="small-sfx",
                prompt="ocean waves",
                negative_prompt="",
                duration=5.0,
                steps=4,
                cfg=1.0,
                cfg_sweep_text="1.0, 2.0",
                is_sweep=False,
                embed_metadata=False,
                seed=42,
            )
            next(gen)  # Initial clear (before worker starts)
            next(gen)  # Worker started and entered try/finally loop
            gen.close()
            time.sleep(0.1)

        self.assertIn("event", captured)
        self.assertTrue(captured["event"].is_set())

    def test_app_generate_oom_and_memory_error_recovery(self):
        from unittest.mock import patch
        import gradio as gr
        import torch
        from app import generate

        for exc_cls in (torch.cuda.OutOfMemoryError, MemoryError):
            _MODEL_CACHE["dummy"] = object()

            def failing_generate_audio(config, progress=None, status_callback=None, cancel_event=None):
                raise exc_cls("out of memory")

            with patch("app.generate_audio", side_effect=failing_generate_audio):
                gen = generate(
                    model_name="small-sfx",
                    prompt="ocean waves",
                    negative_prompt="",
                    duration=5.0,
                    steps=4,
                    cfg=1.0,
                    cfg_sweep_text="1.0, 2.0",
                    is_sweep=False,
                    embed_metadata=False,
                    seed=42,
                )
                with self.assertRaises(gr.Error):
                    list(gen)
            self.assertEqual(len(_MODEL_CACHE), 0)

    def test_generation_config_audio_and_postproc_validation(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            GenerationConfig(prompt="lo-fi remix", init_audio_path=tmp_path, init_noise_level=0.6).validate()
            GenerationConfig(prompt="fill gap", inpaint_audio_path=tmp_path, inpaint_start=1.0, inpaint_end=4.0).validate()

            with self.assertRaises(GenerationError):
                GenerationConfig(
                    prompt="both",
                    init_audio_path=tmp_path,
                    inpaint_audio_path=tmp_path,
                    inpaint_start=1.0,
                    inpaint_end=2.0,
                ).validate()

            with self.assertRaises(GenerationError):
                GenerationConfig(prompt="remix", init_audio_path=tmp_path, init_noise_level=0.0).validate()
            with self.assertRaises(GenerationError):
                GenerationConfig(prompt="remix", init_audio_path=tmp_path, init_noise_level=1.5).validate()

            with self.assertRaises(GenerationError):
                GenerationConfig(prompt="inpaint", inpaint_audio_path=tmp_path, inpaint_start=5.0, inpaint_end=3.0).validate()
            with self.assertRaises(GenerationError):
                GenerationConfig(prompt="inpaint", duration=4.0, inpaint_audio_path=tmp_path, inpaint_start=4.0, inpaint_end=6.0).validate()

            with self.assertRaises(GenerationError):
                GenerationConfig(prompt="remix", init_audio_path="does_not_exist_12345.wav").validate()

            with self.assertRaises(GenerationError):
                GenerationConfig(prompt="fade", fade_out_seconds=-1.0).validate()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_apply_post_processing_normalize_and_fade(self):
        import numpy as np

        sr = 1000
        audio = np.full((2000, 2), 0.25, dtype=np.float32)
        processed = apply_post_processing(audio, sample_rate=sr, normalize=True, fade_out_seconds=1.0)
        expected_peak = 10.0 ** (-1.0 / 20.0)
        self.assertAlmostEqual(float(np.max(np.abs(processed))), expected_peak, places=4)
        self.assertAlmostEqual(float(np.abs(processed[-1, 0])), 0.0, places=5)

        silent = np.zeros((1000, 2), dtype=np.float32)
        silent_out = apply_post_processing(silent, sample_rate=sr, normalize=True, fade_out_seconds=0.5)
        self.assertTrue(np.all(np.isfinite(silent_out)))
        self.assertEqual(float(np.max(np.abs(silent_out))), 0.0)

    def test_delete_output_file_safety(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "sample.wav")
            txt_path = os.path.join(tmp_dir, "notes.txt")
            with open(wav_path, "wb") as f:
                f.write(b"RIFF")
            with open(txt_path, "wb") as f:
                f.write(b"hi")

            self.assertTrue(delete_output_file(wav_path, output_dir=tmp_dir))
            self.assertFalse(os.path.exists(wav_path))
            self.assertFalse(delete_output_file(wav_path, output_dir=tmp_dir))

            with self.assertRaises(GenerationError):
                delete_output_file(txt_path, output_dir=tmp_dir)
            with self.assertRaises(GenerationError):
                delete_output_file(os.path.join(tmp_dir, "..", "outside.wav"), output_dir=tmp_dir)

    def test_app_seed_batch_and_audio_mode_validation(self):
        from unittest.mock import patch
        import gradio as gr
        from app import generate

        with self.assertRaises(gr.Error):
            next(
                generate(
                    model_name="small-sfx",
                    prompt="ocean waves",
                    negative_prompt="",
                    duration=5.0,
                    steps=4,
                    cfg=1.0,
                    cfg_sweep_text="1.0, 2.0",
                    is_sweep="Single",
                    embed_metadata=True,
                    seed=42,
                    audio_mode="Audio-to-Audio (Remix)",
                    source_audio=None,
                )
            )

        seen_seeds = []
        progress_calls = []

        def fake_progress(value, desc=None, unit=None):
            progress_calls.append((value, desc, unit))

        def fake_gen(config, progress=None, status_callback=None, cancel_event=None):
            seen_seeds.append(config.seed)
            if progress:
                progress((1, 4), desc="Sampling", unit="steps")
            return GenerationResult(
                output_path=f"outputs/s_{config.seed}.wav",
                status_message="ok",
                duration=config.duration,
                elapsed=0.1,
                speed=10.0,
                seed=config.seed,
            )

        with patch("app.generate_audio", side_effect=fake_gen):
            outputs = list(
                generate(
                    model_name="small-sfx",
                    prompt="ocean waves",
                    negative_prompt="",
                    duration=5.0,
                    steps=4,
                    cfg=1.5,
                    cfg_sweep_text="1.0, 2.0",
                    is_sweep="Seed Batch",
                    embed_metadata=True,
                    seed=100,
                    seed_batch_count=3,
                    progress=fake_progress,
                )
            )

        self.assertEqual(seen_seeds, [100, 101, 102])
        last = outputs[-1]
        self.assertEqual(last[1]["value"], "outputs/s_100.wav")
        self.assertEqual(last[2]["value"], "outputs/s_101.wav")
        self.assertEqual(last[3]["value"], "outputs/s_102.wav")
        self.assertEqual(last[6], 100)
        self.assertIn("Completed Seed Batch: 3 variations", last[-1])
        self.assertTrue(any("[1/3 · Seed 100] Sampling" == desc for _, desc, _ in progress_calls))
        self.assertTrue(any("[3/3 · Seed 102] Sampling" == desc for _, desc, _ in progress_calls))

    def test_accordion_label_helpers(self):
        from app import (
            AUDIO_ACCORDION_LABEL,
            POSTPROC_ACCORDION_LABEL,
            audio_accordion_label,
            postproc_accordion_label,
        )

        self.assertEqual(audio_accordion_label("Off"), AUDIO_ACCORDION_LABEL)
        self.assertIn("Remix active", audio_accordion_label("Audio-to-Audio (Remix)"))
        self.assertIn("Inpainting active", audio_accordion_label("Inpainting"))
        self.assertEqual(postproc_accordion_label(True, False, 0.0), POSTPROC_ACCORDION_LABEL)
        summary = postproc_accordion_label(False, True, 1.5)
        self.assertIn("Normalize", summary)
        self.assertIn("Fade 1.5s", summary)
        self.assertIn("Metadata off", summary)

    def test_app_per_slot_progress_targets(self):
        import inspect
        import gradio as gr
        from app import create_app

        demo = create_app()
        progress_deps = [
            d for d in demo.fns.values()
            if getattr(d, "show_progress", None) == "full" and getattr(d, "show_progress_on", None)
        ]
        # 3 model tabs * (1 single audio slot + 5 sweep audio slots) = 18 individual Audio targets
        self.assertEqual(len(progress_deps), 18)
        for dep in progress_deps:
            targets = dep.show_progress_on
            self.assertEqual(len(targets), 1)
            self.assertIsInstance(targets[0], gr.Audio)
            # Direct return functions (non-generators) prevent intermediate yields from resetting gr.Progress
            self.assertFalse(inspect.isgeneratorfunction(dep.fn))

    def test_run_job_blocking_routes_progress_and_heartbeat(self):
        import threading
        import time
        from unittest.mock import patch
        from app import _LIVE_LOGS, _format_live_logs, _run_job_blocking

        progress_calls = []

        def fake_progress(val, desc=None, unit=None):
            progress_calls.append((val, desc, unit))

        def fake_generate_audio(config, progress=None, status_callback=None, cancel_event=None):
            if status_callback:
                status_callback("[load] [1/4] Building...")
            if progress:
                progress((1, 4), desc="Building...", unit="stage")
            time.sleep(2.2)
            if progress:
                progress((1, 2), desc="Sampling · 2.0 st/s (ETA: 1s)", unit="steps")
                progress((2, 2), desc="Sampling · 2.0 st/s (ETA: 0s)", unit="steps")
            return GenerationResult(
                output_path="outputs/ok.wav",
                status_message="done",
                duration=5.0,
                elapsed=2.2,
                speed=4.0,
                seed=42,
            )

        logs = []
        _LIVE_LOGS.pop("test-tab", None)
        with patch("app.generate_audio", side_effect=fake_generate_audio):
            kind, payload, _ = _run_job_blocking(
                {"model_name": "small-sfx", "prompt": "waves", "duration": 5.0, "steps": 2},
                fake_progress,
                logs,
                threading.Event(),
                live_log_key="test-tab",
            )

        self.assertEqual(kind, "done")
        self.assertEqual(payload.output_path, "outputs/ok.wav")
        self.assertTrue(any(val == (0, None) and "Building..." in (desc or "") and "s)" in (desc or "") for val, desc, _ in progress_calls))
        self.assertIn(((1, 2), "Sampling · 2.0 st/s (ETA: 1s)", "steps"), progress_calls)
        self.assertIn(((2, 2), "Sampling · 2.0 st/s (ETA: 0s)", "steps"), progress_calls)
        self.assertIn("[load] [1/4] Building...", _format_live_logs("test-tab"))
        _LIVE_LOGS.pop("test-tab", None)

    def test_read_playhead_js_helper(self):
        from app import _read_playhead_js

        js_code = _read_playhead_js("source-audio-small-music")
        self.assertIn("source-audio-small-music", js_code)
        self.assertIn("currentTime", js_code)

    def test_mcp_and_download_models_helpers(self):
        import download_models
        import mcp_server

        args = download_models.parse_args(["--models", "small-sfx", "--force"])
        self.assertEqual(args.models, ["small-sfx"])
        self.assertTrue(args.force)

        models_info = mcp_server.list_models_tool()
        self.assertIn("loaded_model", models_info)
        self.assertEqual(len(models_info["models"]), 3)

        unload_res = mcp_server.unload_model_tool()
        self.assertIn("status", unload_res)

        recent = mcp_server.list_recent_outputs_tool(limit=5)
        self.assertIsInstance(recent, list)


if __name__ == "__main__":
    unittest.main()
