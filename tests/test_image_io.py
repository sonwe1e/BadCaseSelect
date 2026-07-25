from __future__ import annotations

import numpy as np

from vfi_hard_miner.image_io import read_rgb01, write_image_atomic


def test_rgb_round_trip(tmp_path):
    source = np.zeros((5, 7, 3), dtype=np.float32)
    source[2, 3] = (1.0, 0.5, 0.0)
    path = tmp_path / "image.png"
    write_image_atomic(path, source)
    restored = read_rgb01(path)
    assert restored.shape == source.shape
    assert restored.dtype == np.float32
    assert np.max(np.abs(restored - source)) <= 1.0 / 255.0


def test_uint8_cache_form_round_trips_bit_identical_to_read_rgb01(tmp_path):
    from PIL import Image

    from vfi_hard_miner.image_io import read_rgb_uint8, rgb_uint8_to_float32

    rng = np.random.default_rng(3)
    array = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(array).save(path)

    cached = read_rgb_uint8(path)
    assert cached.dtype == np.uint8
    assert cached.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(cached, array)
    np.testing.assert_array_equal(rgb_uint8_to_float32(cached), read_rgb01(path))


def test_jpeg_writer_honors_high_quality_444_output(tmp_path):
    from PIL import Image, JpegImagePlugin

    array = np.zeros((24, 32, 3), dtype=np.uint8)
    array[:, :16] = (255, 64, 16)
    path = tmp_path / "diagnostic.jpg"

    write_image_atomic(path, array, quality=92, subsampling=0)

    with Image.open(path) as restored:
        assert restored.format == "JPEG"
        assert restored.size == (32, 24)
        assert JpegImagePlugin.get_sampling(restored) == 0
