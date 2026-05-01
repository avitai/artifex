"""Tests for protein dataset backed by datarax DataSourceModule."""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.data_source import DataSourceModule

from artifex.data.protein import (
    AA_TYPES,
    create_synthetic_protein_dataset,
    protein_collate_fn,
    ProteinDataset,
    ProteinDatasetConfig,
    ProteinStructure,
)


def test_protein_structure_init():
    """Test ProteinStructure initialization."""
    atom_positions = np.random.normal(size=(10, 4, 3))
    atom_mask = np.ones((10, 4))
    aatype = np.random.randint(0, 20, size=10)

    structure = ProteinStructure.from_numpy(
        {
            "atom_positions": atom_positions,
            "atom_mask": atom_mask,
            "aatype": aatype,
            "residue_index": np.arange(10),
        }
    )

    assert isinstance(structure.atom_positions, jax.Array)
    assert isinstance(structure.atom_mask, jax.Array)
    assert isinstance(structure.aatype, jax.Array)
    assert isinstance(structure.residue_index, jax.Array)

    assert structure.atom_positions.shape == (10, 4, 3)
    assert structure.atom_mask.shape == (10, 4)
    assert structure.aatype.shape == (10,)
    assert structure.residue_index.shape == (10,)


def test_protein_dataset_is_datasource_module():
    """Test that ProteinDataset is a proper DataSourceModule subclass."""
    config = ProteinDatasetConfig()
    dataset = ProteinDataset(config)

    assert isinstance(dataset, DataSourceModule)


def test_protein_dataset_init():
    """Test ProteinDataset initialization with config."""
    config = ProteinDatasetConfig(
        max_seq_length=128,
        backbone_atom_indices=(0, 1, 2, 3),
    )
    dataset = ProteinDataset(config, data_dir="dummy_dir")

    assert dataset.config.max_seq_length == 128
    assert dataset.config.backbone_atom_indices == (0, 1, 2, 3)
    assert dataset.config.center_positions is True
    assert dataset.config.normalize_positions is True

    assert len(AA_TYPES) == 20


def test_process_structure():
    """Test protein structure processing."""
    config = ProteinDatasetConfig(
        max_seq_length=8,
        backbone_atom_indices=(0, 1, 2, 3),
    )
    dataset = ProteinDataset(config)

    atom_positions = np.random.normal(size=(10, 4, 3))
    atom_mask = np.ones((10, 4))
    aatype = np.random.randint(0, 20, size=10)

    structure = ProteinStructure(
        atom_positions=jnp.array(atom_positions),
        atom_mask=jnp.array(atom_mask),
        aatype=jnp.array(aatype),
        residue_index=None,
        chain_index=None,
        b_factors=None,
        resolution=None,
        seq_length=None,
    )

    processed = dataset._process_structure(structure)

    # Check truncation
    assert processed.atom_positions.shape[0] <= config.max_seq_length
    assert processed.atom_mask.shape[0] <= config.max_seq_length
    assert processed.aatype.shape[0] <= config.max_seq_length

    # Check residue indices were created
    assert processed.residue_index is not None
    assert processed.residue_index.shape == (config.max_seq_length,)


def test_collate_batch():
    """Test batch collation via standalone protein_collate_fn."""
    max_seq_length = 16

    # Create examples with different sequence lengths
    examples = []
    for seq_len in [10, 12, 8]:
        atom_positions = np.random.normal(size=(seq_len, 4, 3))
        atom_mask = np.ones((seq_len, 4))
        aatype = np.random.randint(0, 20, size=seq_len)

        example = {
            "atom_positions": atom_positions,
            "atom_mask": atom_mask,
            "aatype": aatype,
            "residue_index": np.arange(seq_len),
            "seq_length": seq_len,
        }
        examples.append(example)

    batch = protein_collate_fn(examples, max_seq_length=max_seq_length)

    assert "atom_positions" in batch
    assert "atom_mask" in batch
    assert "aatype" in batch
    assert "residue_index" in batch

    max_len = max(ex["seq_length"] for ex in examples)
    batch_size = len(examples)
    assert batch["atom_positions"].shape == (batch_size, max_len, 4, 3)
    assert batch["atom_mask"].shape == (batch_size, max_len, 4)
    assert batch["aatype"].shape == (batch_size, max_len)
    assert batch["residue_index"].shape == (batch_size, max_len)

    assert isinstance(batch["atom_positions"], jax.Array)
    assert isinstance(batch["atom_mask"], jax.Array)


def test_synthetic_dataset_creation():
    """Test creation of synthetic protein dataset."""
    num_proteins = 5
    min_seq_length = 10
    max_seq_length = 15
    num_atom_types = 4

    dataset = create_synthetic_protein_dataset(
        num_proteins=num_proteins,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_atom_types=num_atom_types,
        random_seed=42,
    )

    assert isinstance(dataset, ProteinDataset)
    assert isinstance(dataset, DataSourceModule)
    assert dataset.config.max_seq_length == max_seq_length
    assert len(dataset.structures) == num_proteins

    for structure in dataset.structures:
        assert isinstance(structure, ProteinStructure)
        assert structure.atom_positions.shape[0] >= min_seq_length
        assert structure.atom_positions.shape[0] <= max_seq_length
        assert structure.atom_positions.shape[1] == num_atom_types
        assert structure.atom_positions.shape[2] == 3
        assert structure.atom_mask.shape == structure.atom_positions.shape[:-1]
        assert structure.aatype.shape == (structure.atom_positions.shape[0],)

    # Test dataset indexing
    example = dataset[0]
    assert "atom_positions" in example
    assert "atom_mask" in example
    assert "aatype" in example
    assert "residue_index" in example
    assert "seq_length" in example

    # Test out of bounds indexing
    with pytest.raises(IndexError):
        dataset[num_proteins]


def test_dataset_iteration():
    """Test that ProteinDataset supports iteration (DataSourceModule contract)."""
    dataset = create_synthetic_protein_dataset(
        num_proteins=3,
        min_seq_length=5,
        max_seq_length=10,
        random_seed=42,
    )

    elements = list(dataset)
    assert len(elements) == 3

    for element in elements:
        assert "atom_positions" in element
        assert "atom_mask" in element
        assert "seq_length" in element


def test_protein_dataset_batched_collation():
    """Test batched collation over a protein dataset.

    The datarax 0.1.3 ``Pipeline`` contract requires ``get_batch_at(start,
    size, key)`` for JIT-compiled batch fetches. Proteins use
    variable-length collation via ``protein_collate_fn``, so batched access
    is exercised through the dataset's own ``get_batch`` method rather than
    through ``Pipeline.step``.
    """
    dataset = create_synthetic_protein_dataset(
        num_proteins=6,
        min_seq_length=5,
        max_seq_length=10,
        random_seed=42,
    )

    batch_size = 3
    batches = [
        dataset.get_batch(list(range(i, i + batch_size)))
        for i in range(0, len(dataset), batch_size)
    ]

    # 6 proteins / batch_size 3 = 2 batches
    assert len(batches) == 2

    for batch in batches:
        assert "atom_positions" in batch
        assert "atom_mask" in batch
        assert "aatype" in batch


@pytest.fixture
def simple_protein_data():
    """Create a simple protein dataset for testing."""
    num_proteins = 5
    max_sequence_length = 10
    num_atom_types = 4

    proteins = []
    for i in range(num_proteins):
        seq_length = np.random.randint(5, max_sequence_length + 1)
        aatype = np.random.randint(0, 20, size=seq_length)
        atom_positions = np.random.normal(0, 1, size=(seq_length, num_atom_types, 3))
        atom_mask = np.ones((seq_length, num_atom_types))
        residue_index = np.arange(seq_length)

        protein = {
            "aatype": aatype,
            "atom_positions": atom_positions,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
            "chain_index": np.zeros(seq_length),
            "sequence": "A" * seq_length,
            "resolution": 2.0,
            "pdb_id": f"TEST{i + 1}",
        }
        proteins.append(protein)

    return proteins


def test_protein_dataset_creation(simple_protein_data):
    """Test creating a protein dataset from in-memory data."""
    config = ProteinDatasetConfig(
        backbone_atom_indices=(0, 1, 2, 3),
        center_positions=False,
        normalize_positions=False,
    )
    dataset = ProteinDataset(config, simple_protein_data)

    assert len(dataset) == len(simple_protein_data)

    item = dataset[0]
    assert "aatype" in item
    assert "atom_positions" in item
    assert "atom_mask" in item


def test_protein_dataset_path_loading_uses_config_contract(simple_protein_data, tmp_path):
    """ProteinDataset should load path-backed data only through the config-based constructor."""
    data_path = tmp_path / "synthetic_proteins.pkl"
    with data_path.open("wb") as handle:
        pickle.dump(simple_protein_data, handle)

    config = ProteinDatasetConfig(
        max_seq_length=12,
        backbone_atom_indices=(0, 1, 2, 3),
        center_positions=False,
        normalize_positions=False,
    )
    dataset = ProteinDataset(config, data_path)

    assert len(dataset) == len(simple_protein_data)
    assert dataset[0]["pdb_id"] == simple_protein_data[0]["pdb_id"]

    batch = dataset.get_batch([0, 1], max_length=12)
    assert batch["atom_positions"].shape == (2, 12, 4, 3)


def test_protein_dataset_batch(simple_protein_data):
    """Test batching protein data."""
    config = ProteinDatasetConfig(
        backbone_atom_indices=(0, 1, 2, 3),
        center_positions=False,
        normalize_positions=False,
    )
    dataset = ProteinDataset(config, simple_protein_data)

    batch_indices = [0, 1]
    batch_data = dataset.get_batch(batch_indices)

    assert "aatype" in batch_data
    assert "atom_positions" in batch_data
    assert "atom_mask" in batch_data

    assert batch_data["aatype"].shape[0] == len(batch_indices)
    assert batch_data["atom_positions"].shape[0] == len(batch_indices)


def test_protein_dataset_padding(simple_protein_data):
    """Test padding protein data to a fixed size."""
    config = ProteinDatasetConfig(
        backbone_atom_indices=(0, 1, 2, 3),
        center_positions=False,
        normalize_positions=False,
    )
    dataset = ProteinDataset(config, simple_protein_data)

    max_length = 15
    batch_indices = [0, 1, 2]
    batch_data = dataset.get_batch(batch_indices, max_length=max_length)

    assert batch_data["aatype"].shape[1] == max_length
    assert batch_data["atom_positions"].shape[1] == max_length
    assert batch_data["atom_mask"].shape[1] == max_length

    for i, idx in enumerate(batch_indices):
        mask_sum = batch_data["atom_mask"][i].sum()
        expected_sum = simple_protein_data[idx]["atom_mask"].sum()
        assert mask_sum == expected_sum, "Padding changed the mask sum"


def test_protein_dataset_stats(simple_protein_data):
    """Test computing dataset statistics."""
    config = ProteinDatasetConfig(
        backbone_atom_indices=(0, 1, 2, 3),
        center_positions=False,
        normalize_positions=False,
    )
    dataset = ProteinDataset(config, simple_protein_data)

    stats = dataset.get_statistics()

    assert "num_examples" in stats
    assert "seq_length_mean" in stats
    assert "seq_length_std" in stats
    assert "seq_length_min" in stats
    assert "seq_length_max" in stats

    seq_lengths = [len(p["aatype"]) for p in simple_protein_data]
    assert stats["num_examples"] == len(simple_protein_data)
    assert np.isclose(stats["seq_length_mean"], np.mean(seq_lengths))
    assert np.isclose(stats["seq_length_std"], np.std(seq_lengths))
    assert stats["seq_length_min"] == min(seq_lengths)
    assert stats["seq_length_max"] == max(seq_lengths)
