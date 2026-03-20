"""Tests for download_hdris.py helper functions."""

import os
import sys
import struct
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from download_hdris import (
    hdri_url,
    hdri_filename,
    validate_exr_magic,
    _HDRIS,
    _BASE_URL,
    _EXR_MAGIC,
)


class TestHdriUrl:
    def test_url_pattern(self):
        url = hdri_url("studio_small_09")
        assert url == f"{_BASE_URL}/studio_small_09_2k.exr"

    def test_each_hdri_has_valid_url(self):
        for name, _ in _HDRIS:
            url = hdri_url(name)
            assert url.startswith("https://")
            assert url.endswith(".exr")
            assert name in url


class TestHdriFilename:
    def test_filename_pattern(self):
        assert hdri_filename("studio_small_09") == "studio_small_09_2k.exr"

    def test_all_filenames_unique(self):
        filenames = [hdri_filename(name) for name, _ in _HDRIS]
        assert len(filenames) == len(set(filenames))


class TestValidateExrMagic:
    def test_valid_exr(self, tmp_path):
        exr_file = tmp_path / "test.exr"
        exr_file.write_bytes(_EXR_MAGIC + b"\x00" * 100)
        assert validate_exr_magic(str(exr_file)) is True

    def test_invalid_magic(self, tmp_path):
        bad_file = tmp_path / "bad.exr"
        bad_file.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 100)
        assert validate_exr_magic(str(bad_file)) is False

    def test_nonexistent_file(self, tmp_path):
        assert validate_exr_magic(str(tmp_path / "nope.exr")) is False

    def test_too_short_file(self, tmp_path):
        short_file = tmp_path / "short.exr"
        short_file.write_bytes(b"\x76\x2f")
        assert validate_exr_magic(str(short_file)) is False

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.exr"
        empty.write_bytes(b"")
        assert validate_exr_magic(str(empty)) is False


class TestHdriList:
    def test_five_hdris(self):
        assert len(_HDRIS) == 5

    def test_unique_names(self):
        names = [name for name, _ in _HDRIS]
        assert len(names) == len(set(names))

    def test_all_have_descriptions(self):
        for name, desc in _HDRIS:
            assert len(name) > 0
            assert len(desc) > 0
