"""Tests for per-scene stratified validation splitting."""

import pytest

from deni_train.data.splits import (
    scene_name_from_file,
    scene_name_from_pair,
    stratified_split,
    stratified_split_files,
)


class TestSceneNameFromPair:
    def test_flat_naming(self):
        pair = ("data/ABeautifulGame_2f90895c_input.exr",
                "data/ABeautifulGame_2f90895c_target.exr")
        assert scene_name_from_pair(pair) == "ABeautifulGame"

    def test_flat_naming_underscore_scene(self):
        pair = ("data/cornell_box_bbd5b2ff_input.exr",
                "data/cornell_box_bbd5b2ff_target.exr")
        assert scene_name_from_pair(pair) == "cornell_box"

    def test_directory_naming(self):
        pair = ("data/my_scene/input.exr", "data/my_scene/target.exr")
        assert scene_name_from_pair(pair) == "my_scene"

    def test_unknown_fallback(self):
        pair = ("data/weirdfile.exr", "data/weirdfile2.exr")
        assert scene_name_from_pair(pair) == "unknown"

    def test_windows_paths(self):
        pair = ("C:\\data\\Sponza_abcd1234_input.exr",
                "C:\\data\\Sponza_abcd1234_target.exr")
        assert scene_name_from_pair(pair) == "Sponza"

    def test_multi_underscore_scene(self):
        pair = ("data/a_beautiful_game_12345678_input.exr",
                "data/a_beautiful_game_12345678_target.exr")
        assert scene_name_from_pair(pair) == "a_beautiful_game"

    def test_exposure_wedge_naming(self):
        pair = ("data/Sponza_abcd1234_ev+0_input.exr",
                "data/Sponza_abcd1234_ev+0_target.exr")
        assert scene_name_from_pair(pair) == "Sponza"

    def test_exposure_wedge_negative_offset(self):
        pair = ("data/cornell_box_bbd5b2ff_ev-2_input.exr",
                "data/cornell_box_bbd5b2ff_ev-2_target.exr")
        assert scene_name_from_pair(pair) == "cornell_box"


class TestStratifiedSplit:
    def _make_pairs(self, scene_counts: dict[str, int]) -> list[tuple[str, str]]:
        """Create fake pairs for testing, sorted by filename (as ExrDataset does)."""
        pairs = []
        for scene, count in scene_counts.items():
            for i in range(count):
                hex_id = f"{i:08x}"
                pairs.append((
                    f"data/{scene}_{hex_id}_input.exr",
                    f"data/{scene}_{hex_id}_target.exr",
                ))
        return sorted(pairs)

    def test_all_scenes_represented_in_val(self):
        pairs = self._make_pairs({"SceneA": 20, "SceneB": 15, "SceneC": 10})
        train_idx, val_idx = stratified_split(pairs)
        # Every scene should have at least one val sample
        val_scenes = {scene_name_from_pair(pairs[i]) for i in val_idx}
        assert val_scenes == {"SceneA", "SceneB", "SceneC"}

    def test_no_overlap(self):
        pairs = self._make_pairs({"SceneA": 20, "SceneB": 10})
        train_idx, val_idx = stratified_split(pairs)
        assert set(train_idx).isdisjoint(set(val_idx))

    def test_covers_all_indices(self):
        pairs = self._make_pairs({"SceneA": 20, "SceneB": 10})
        train_idx, val_idx = stratified_split(pairs)
        assert sorted(train_idx + val_idx) == list(range(len(pairs)))

    def test_val_is_roughly_10_percent(self):
        pairs = self._make_pairs({"SceneA": 100, "SceneB": 50})
        train_idx, val_idx = stratified_split(pairs)
        # Each scene gives ~10% to val
        assert 10 <= len(val_idx) <= 20

    def test_minimum_one_val_per_scene(self):
        pairs = self._make_pairs({"Tiny": 3})
        train_idx, val_idx = stratified_split(pairs)
        assert len(val_idx) >= 1

    def test_single_pair_scene(self):
        pairs = self._make_pairs({"OnlyOne": 1})
        train_idx, val_idx = stratified_split(pairs)
        # With 1 pair, 10% rounds to 0, but min is 1 → all in val
        assert len(val_idx) == 1
        assert len(train_idx) == 0

    def test_two_scene_proportions(self):
        pairs = self._make_pairs({"Big": 40, "Small": 10})
        train_idx, val_idx = stratified_split(pairs)
        val_scenes = {}
        for i in val_idx:
            scene = scene_name_from_pair(pairs[i])
            val_scenes[scene] = val_scenes.get(scene, 0) + 1
        # Big: 40 // 10 = 4 val, Small: 10 // 10 = 1 val
        assert val_scenes["Big"] == 4
        assert val_scenes["Small"] == 1


class TestSceneNameFromFile:
    def test_flat_naming(self):
        assert scene_name_from_file("data/ABeautifulGame_2f90895c.safetensors") == "ABeautifulGame"

    def test_flat_naming_underscore_scene(self):
        assert scene_name_from_file("data/cornell_box_bbd5b2ff.safetensors") == "cornell_box"

    def test_directory_naming(self):
        assert scene_name_from_file("data/my_scene/variation.safetensors") == "my_scene"

    def test_windows_paths(self):
        assert scene_name_from_file("C:\\data\\Sponza_abcd1234.safetensors") == "Sponza"

    def test_multi_underscore_scene(self):
        assert scene_name_from_file("data/a_beautiful_game_12345678.safetensors") == "a_beautiful_game"

    def test_no_hex_id_falls_back_to_parent(self):
        assert scene_name_from_file("data/my_scene/foo.safetensors") == "my_scene"

    def test_root_no_parent_returns_unknown(self):
        assert scene_name_from_file("standalone.safetensors") == "unknown"


class TestStratifiedSplitFiles:
    def _make_files(self, scene_counts: dict[str, int]) -> list[str]:
        """Create fake safetensors file paths, sorted."""
        files = []
        for scene, count in scene_counts.items():
            for i in range(count):
                hex_id = f"{i:08x}"
                files.append(f"data/{scene}_{hex_id}.safetensors")
        return sorted(files)

    def test_all_scenes_in_val(self):
        files = self._make_files({"SceneA": 20, "SceneB": 15})
        train_idx, val_idx = stratified_split_files(files)
        val_scenes = {scene_name_from_file(files[i]) for i in val_idx}
        assert val_scenes == {"SceneA", "SceneB"}

    def test_no_overlap(self):
        files = self._make_files({"SceneA": 20, "SceneB": 10})
        train_idx, val_idx = stratified_split_files(files)
        assert set(train_idx).isdisjoint(set(val_idx))

    def test_covers_all_indices(self):
        files = self._make_files({"SceneA": 20, "SceneB": 10})
        train_idx, val_idx = stratified_split_files(files)
        assert sorted(train_idx + val_idx) == list(range(len(files)))

    def test_matches_pair_split_proportions(self):
        """Safetensors split should produce same index counts as EXR pair split."""
        scenes = {"SceneA": 20, "SceneB": 10, "SceneC": 5}
        files = self._make_files(scenes)
        pairs = []
        for scene, count in scenes.items():
            for i in range(count):
                hex_id = f"{i:08x}"
                pairs.append((
                    f"data/{scene}_{hex_id}_input.exr",
                    f"data/{scene}_{hex_id}_target.exr",
                ))
        pairs = sorted(pairs)

        _, val_pairs = stratified_split(pairs)
        _, val_files = stratified_split_files(files)
        assert len(val_pairs) == len(val_files)
