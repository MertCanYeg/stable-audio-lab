"""Unit tests for Stable Audio Lab core modules."""

import unittest

from core.compat import get_device_info, setup_environment
from core.engine import (
    GenerationConfig,
    GenerationResult,
    generate_audio,
    parse_cfg_sweep,
    slugify,
    validate_cfg_sweep,
)
from core.exceptions import (
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
        self.assertTrue(issubclass(ModelNotFoundError, ValueError))
        self.assertTrue(issubclass(GenerationError, ValueError))


if __name__ == "__main__":
    unittest.main()
