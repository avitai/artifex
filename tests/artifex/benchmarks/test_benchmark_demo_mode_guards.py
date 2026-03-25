from pathlib import Path

import flax.nnx as nnx
import pytest

from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset
from artifex.benchmarks.datasets.geometric import ShapeNetDataset
from artifex.benchmarks.datasets.protein_dataset import SyntheticProteinDataset
from artifex.benchmarks.suites.protein_benchmarks import ProteinBenchmarkSuite
from artifex.generative_models.core.configuration import DataConfig


def test_crossdocked_requires_explicit_demo_mode() -> None:
    config = DataConfig(name="crossdocked", dataset_name="crossdocked")

    with pytest.raises(RuntimeError, match="demo"):
        CrossDockedDataset(data_path="./tmp/crossdocked", config=config, rngs=nnx.Rngs(42))


def test_synthetic_protein_dataset_requires_explicit_demo_mode() -> None:
    config = DataConfig(name="protein", dataset_name="synthetic_protein")

    with pytest.raises(RuntimeError, match="demo"):
        SyntheticProteinDataset(data_path="./tmp/protein", config=config, rngs=nnx.Rngs(42))


def test_shapenet_supported_mode_fails_fast_without_assets(tmp_path: Path) -> None:
    config = DataConfig(
        name="shapenet",
        dataset_name="shapenet",
        metadata={"num_points": 64},
    )

    with pytest.raises(RuntimeError, match="benchmark-grade ShapeNet assets"):
        ShapeNetDataset(
            data_path=str(tmp_path / "missing_shapenet"), config=config, rngs=nnx.Rngs(42)
        )


def test_protein_benchmark_suite_requires_explicit_demo_mode() -> None:
    with pytest.raises(RuntimeError, match="demo"):
        ProteinBenchmarkSuite(num_samples=4, random_seed=42)
