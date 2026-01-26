"""Tests for protein dataset."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from artifex.data.protein_dataset import (
    create_synthetic_protein_dataset,
    ProteinDataset,
    ProteinStructure,
)
from artifex.utils.file_utils import get_valid_output_dir


# Create output directory in test_results
OUTPUT_DIR = get_valid_output_dir("protein", "test_results")


def test_protein_structure_init():
    """Test ProteinStructure initialization."""
    # Create sample data
    atom_positions = np.random.normal(size=(10, 4, 3))
    atom_mask = np.ones((10, 4))
    aatype = np.random.randint(0, 20, size=10)

    # Create protein structure from numpy arrays
    structure = ProteinStructure.from_numpy(
        {
            "atom_positions": atom_positions,
            "atom_mask": atom_mask,
            "aatype": aatype,
            "residue_index": np.arange(10),
        }
    )

    # Check that arrays were converted to jax arrays
    assert isinstance(structure.atom_positions, jax.Array)
    assert isinstance(structure.atom_mask, jax.Array)
    assert isinstance(structure.aatype, jax.Array)
    assert isinstance(structure.residue_index, jax.Array)

    # Check shapes
    assert structure.atom_positions.shape == (10, 4, 3)
    assert structure.atom_mask.shape == (10, 4)
    assert structure.aatype.shape == (10,)
    assert structure.residue_index.shape == (10,)


def test_protein_dataset_init():
    """Test ProteinDataset initialization."""
    # Create dataset with default parameters
    dataset = ProteinDataset(
        data_dir="dummy_dir",
        max_seq_length=128,
        backbone_atom_indices=[0, 1, 2, 3],
    )

    # Check attributes
    assert dataset.data_dir == "dummy_dir"
    assert dataset.max_seq_length == 128
    assert dataset.backbone_atom_indices == [0, 1, 2, 3]
    assert dataset.center_positions is True
    assert dataset.normalize_positions is True

    # Check that standard atom types and aa types are defined
    assert len(dataset.atom_types) > 0
    assert len(dataset.aa_types) == 20
    assert len(dataset.aa_type_to_idx) == 20


def test_process_structure():
    """Test protein structure processing."""
    # Create dataset
    dataset = ProteinDataset(
        data_dir="dummy_dir",
        max_seq_length=8,
        backbone_atom_indices=[0, 1, 2, 3],
    )

    # Create a structure to process
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

    # Process the structure
    processed = dataset._process_structure(structure)

    # Check truncation
    assert processed.atom_positions.shape[0] <= dataset.max_seq_length
    assert processed.atom_mask.shape[0] <= dataset.max_seq_length
    assert processed.aatype.shape[0] <= dataset.max_seq_length

    # Check residue indices were created
    assert processed.residue_index is not None
    assert processed.residue_index.shape == (dataset.max_seq_length,)


def test_collate_batch():
    """Test batch collation."""
    # Create dataset
    dataset = ProteinDataset(
        data_dir="dummy_dir",
        max_seq_length=16,
        backbone_atom_indices=[0, 1, 2, 3],
    )

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

    # Collate batch
    batch = dataset.collate_batch(examples)

    # Check that batch has the expected structure
    assert "atom_positions" in batch
    assert "atom_mask" in batch
    assert "aatype" in batch
    assert "residue_index" in batch

    # Check shapes
    max_len = max(ex["seq_length"] for ex in examples)
    batch_size = len(examples)
    assert batch["atom_positions"].shape == (batch_size, max_len, 4, 3)
    assert batch["atom_mask"].shape == (batch_size, max_len, 4)
    assert batch["aatype"].shape == (batch_size, max_len)
    assert batch["residue_index"].shape == (batch_size, max_len)

    # Check types
    assert isinstance(batch["atom_positions"], jax.Array)
    assert isinstance(batch["atom_mask"], jax.Array)
    assert isinstance(batch["aatype"], jax.Array)
    assert isinstance(batch["residue_index"], jax.Array)


def test_synthetic_dataset_creation():
    """Test creation of synthetic protein dataset."""
    # Create synthetic dataset
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

    # Check that dataset has the expected attributes
    assert isinstance(dataset, ProteinDataset)
    assert dataset.max_seq_length == max_seq_length
    assert len(dataset.structures) == num_proteins

    # Check structures
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


@pytest.fixture
def simple_protein_data():
    """Create a simple protein dataset for testing."""
    # Generate random protein data
    num_proteins = 5
    max_sequence_length = 10
    num_atom_types = 4

    proteins = []
    for i in range(num_proteins):
        # Random sequence length between 5 and max_sequence_length
        seq_length = np.random.randint(5, max_sequence_length + 1)

        # Generate random amino acid sequence (integers representing amino acid types)
        aatype = np.random.randint(0, 20, size=seq_length)

        # Generate random atom positions
        atom_positions = np.random.normal(0, 1, size=(seq_length, num_atom_types, 3))

        # All atoms are valid (mask of 1s)
        atom_mask = np.ones((seq_length, num_atom_types))

        # Residue index is just the position in the sequence
        residue_index = np.arange(seq_length)

        # Create a protein example
        protein = {
            "aatype": aatype,
            "atom_positions": atom_positions,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
            "chain_index": np.zeros(seq_length),
            "sequence": "".join(["A" for _ in range(seq_length)]),  # Dummy sequence of alanines
            "resolution": 2.0,  # Dummy resolution value
            "pdb_id": f"TEST{i + 1}",  # Dummy PDB ID
        }

        proteins.append(protein)

    # Log the created data
    with open(os.path.join(OUTPUT_DIR, "test_protein_dataset_data.txt"), "w") as f:
        f.write(f"Created {num_proteins} test proteins\n")
        for i, protein in enumerate(proteins):
            f.write(f"Protein {i + 1}:\n")
            f.write(f"  Sequence length: {len(protein['aatype'])}\n")
            f.write(f"  Shape of atom positions: {protein['atom_positions'].shape}\n")
            f.write(f"  PDB ID: {protein['pdb_id']}\n\n")

    return proteins


def test_protein_dataset_creation(simple_protein_data):
    """Test creating a protein dataset."""
    # Create the dataset
    dataset = ProteinDataset(simple_protein_data)

    # Verify dataset length
    assert len(dataset) == len(simple_protein_data)

    # Test getting an item
    item = dataset[0]
    assert "aatype" in item
    assert "atom_positions" in item
    assert "atom_mask" in item

    # Check shapes match
    reference = simple_protein_data[0]
    assert item["aatype"].shape == reference["aatype"].shape
    assert item["atom_positions"].shape == reference["atom_positions"].shape
    assert item["atom_mask"].shape == reference["atom_mask"].shape

    # Log test results
    with open(os.path.join(OUTPUT_DIR, "test_protein_dataset_creation.txt"), "w") as f:
        f.write("Test protein dataset creation passed\n")
        f.write(f"Dataset length: {len(dataset)}\n")
        f.write(f"Item keys: {list(item.keys())}\n")
        f.write(f"aatype shape: {item['aatype'].shape}\n")
        f.write(f"atom_positions shape: {item['atom_positions'].shape}\n")
        f.write(f"atom_mask shape: {item['atom_mask'].shape}\n")


def test_protein_dataset_batch(simple_protein_data):
    """Test batching protein data."""
    # Create the dataset
    dataset = ProteinDataset(simple_protein_data)

    # Get batch of indices
    batch_indices = [0, 1]
    batch_data = dataset.get_batch(batch_indices)

    # Check that batch has the right keys and shapes
    assert "aatype" in batch_data
    assert "atom_positions" in batch_data
    assert "atom_mask" in batch_data

    # Check batch has correct batch size
    assert batch_data["aatype"].shape[0] == len(batch_indices)
    assert batch_data["atom_positions"].shape[0] == len(batch_indices)

    # Log test results
    with open(os.path.join(OUTPUT_DIR, "test_protein_dataset_batch.txt"), "w") as f:
        f.write("Test protein dataset batching passed\n")
        f.write(f"Batch indices: {batch_indices}\n")
        f.write(f"Batch keys: {list(batch_data.keys())}\n")
        f.write(f"aatype shape: {batch_data['aatype'].shape}\n")
        f.write(f"atom_positions shape: {batch_data['atom_positions'].shape}\n")
        f.write(f"atom_mask shape: {batch_data['atom_mask'].shape}\n")


def test_protein_dataset_padding(simple_protein_data):
    """Test padding protein data to a fixed size."""
    # Create the dataset
    dataset = ProteinDataset(simple_protein_data)

    # Get a batch with padding
    max_length = 15  # Larger than any of our test sequences
    batch_indices = [0, 1, 2]
    batch_data = dataset.get_batch(batch_indices, max_length=max_length)

    # Check all sequences are padded to max_length
    assert batch_data["aatype"].shape[1] == max_length
    assert batch_data["atom_positions"].shape[1] == max_length
    assert batch_data["atom_mask"].shape[1] == max_length

    # Log test results
    with open(os.path.join(OUTPUT_DIR, "test_protein_dataset_padding.txt"), "w") as f:
        f.write("Test protein dataset padding passed\n")
        f.write(f"Batch indices: {batch_indices}\n")
        f.write(f"Max padding length: {max_length}\n")
        f.write(f"Padded aatype shape: {batch_data['aatype'].shape}\n")
        f.write(f"Padded atom_positions shape: {batch_data['atom_positions'].shape}\n")
        f.write(f"Padded atom_mask shape: {batch_data['atom_mask'].shape}\n")

        # Check the actual data to verify padding
        for i, idx in enumerate(batch_indices):
            original_length = simple_protein_data[idx]["aatype"].shape[0]
            f.write(f"\nProtein {idx}:\n")
            f.write(f"  Original length: {original_length}\n")
            f.write(f"  Padded to: {max_length}\n")

            # Verify that the mask is correct (1s for real data, 0s for padding)
            mask_sum = batch_data["atom_mask"][i].sum()
            expected_sum = simple_protein_data[idx]["atom_mask"].sum()
            f.write(f"  Original mask sum: {expected_sum}\n")
            f.write(f"  Padded mask sum: {mask_sum}\n")
            assert mask_sum == expected_sum, "Padding changed the mask sum"


def test_protein_dataset_stats(simple_protein_data):
    """Test computing dataset statistics."""
    # Create the dataset
    dataset = ProteinDataset(simple_protein_data)

    # Compute statistics
    stats = dataset.get_statistics()

    # Check statistics have expected keys
    assert "num_examples" in stats
    assert "seq_length_mean" in stats
    assert "seq_length_std" in stats
    assert "seq_length_min" in stats
    assert "seq_length_max" in stats

    # Log test results
    with open(os.path.join(OUTPUT_DIR, "test_protein_dataset_stats.txt"), "w") as f:
        f.write("Test protein dataset statistics passed\n")
        f.write("Dataset statistics:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")

        # Verify the statistics
        seq_lengths = [len(p["aatype"]) for p in simple_protein_data]
        f.write("\nVerification:\n")
        f.write(f"  num_examples: {len(simple_protein_data)}\n")
        f.write(f"  seq_length_mean: {np.mean(seq_lengths)}\n")
        f.write(f"  seq_length_std: {np.std(seq_lengths)}\n")
        f.write(f"  seq_length_min: {min(seq_lengths)}\n")
        f.write(f"  seq_length_max: {max(seq_lengths)}\n")

        # Assert that the computed statistics match manual calculation
        assert stats["num_examples"] == len(simple_protein_data)
        assert np.isclose(stats["seq_length_mean"], np.mean(seq_lengths))
        assert np.isclose(stats["seq_length_std"], np.std(seq_lengths))
        assert stats["seq_length_min"] == min(seq_lengths)
        assert stats["seq_length_max"] == max(seq_lengths)


if __name__ == "__main__":
    # If running as standalone script, run all tests
    simple_protein_data = simple_protein_data()
    test_protein_dataset_creation(simple_protein_data)
    test_protein_dataset_batch(simple_protein_data)
    test_protein_dataset_padding(simple_protein_data)
    test_protein_dataset_stats(simple_protein_data)

    print(f"All protein dataset tests completed. Results written to {OUTPUT_DIR}")
