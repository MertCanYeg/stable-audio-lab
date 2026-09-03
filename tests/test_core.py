"""Unit tests for Stable Audio Lab core modules."""

import unittest

from core.compat import get_device_info, setup_environment
from core.engine import GenerationConfig, GenerationResult, generate_audio, slugify
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


class TestExceptions(unittest.TestCase):
    """Test domain exception hierarchy."""

    def test_hierarchy(self):
        self.assertTrue(issubclass(ModelNotFoundError, StableAudioError))
        self.assertTrue(issubclass(GenerationError, StableAudioError))
        self.assertTrue(issubclass(ModelNotFoundError, ValueError))
        self.assertTrue(issubclass(GenerationError, ValueError))


if __name__ == "__main__":
    unittest.main()
