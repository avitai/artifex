"""The minimal ShapeNet fallback writes its placeholder where the loader reads it."""

from __future__ import annotations

from pathlib import Path

import pytest

from artifex.benchmarks.datasets import geometric
from artifex.generative_models.core.configuration import DataConfig


def test_placeholder_model_is_written_inside_the_dataset_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the cube export fails, the placeholder lands in data_path, not a redirected copy."""

    def fail_export(**kwargs: object) -> None:
        raise ValueError("mesh export unavailable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(geometric.trimesh.creation, "box", fail_export)
    dataset = object.__new__(geometric.ShapeNetDataset)
    dataset.config = DataConfig(
        name="minimal_shapenet",
        dataset_name="shapenet",
        data_dir=Path("shapenet"),
        split="train",
        num_workers=2,
        metadata={"synsets": ["02691156"]},
    )
    dataset.data_path = Path("shapenet")

    dataset._create_minimal_dataset()

    assert (tmp_path / "shapenet/02691156/minimal_001/model.obj").is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shapenet"]
