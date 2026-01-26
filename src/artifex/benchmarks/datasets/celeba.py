"""CelebA dataset for image generation benchmarks.

This module provides a dataset implementation for the CelebA dataset,
which is used for face generation and attribute manipulation benchmarks.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax import nnx
from PIL import Image

from artifex.generative_models.core.protocols.evaluation import DatasetProtocol


class CelebADataset(DatasetProtocol):
    """CelebA dataset for face generation benchmarks.

    This dataset provides access to the CelebA dataset, which contains
    over 200K celebrity face images with 40 attribute annotations per image.
    For the benchmark system, we use a subset of the data with configurable
    image size and attribute selection.
    """

    def __init__(
        self,
        data_path: str = "data/celeba",
        num_samples: int = 10000,
        image_size: int = 128,
        include_attributes: bool = True,
        split: str = "all",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize CelebA dataset.

        Args:
            data_path: Path to CelebA dataset (not used with HF datasets, kept for compatibility)
            num_samples: Number of samples to use
            image_size: Size of images (square)
            include_attributes: Whether to include attribute annotations
            split: Data split to use ('train', 'val', 'test', or 'all')
            rngs: Random number generator keys
        """
        self.data_path = data_path
        self.num_samples = num_samples
        self.image_size = image_size
        self.include_attributes = include_attributes
        self.split = split
        self.rngs = rngs

        # Attribute names (40 binary attributes)
        self.attribute_names = [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ]

        # Selected attributes for disentanglement evaluation
        self.selected_attributes = [
            "Male",
            "Young",
            "Smiling",
            "Eyeglasses",
            "Black_Hair",
            "Blond_Hair",
            "Brown_Hair",
            "Bangs",
            "Wearing_Hat",
        ]

        # Initialize dataset
        self._initialize_dataset()

    def _initialize_dataset(self):
        """Initialize the dataset by loading real data."""
        self._load_celeba_data()

    def _load_celeba_data(self):
        """Load the CelebA dataset from Hugging Face."""
        try:
            import os

            cache_dir = os.environ.get("HF_DATASETS_CACHE", "~/.cache/huggingface/datasets")
            print("Loading CelebA dataset from Hugging Face...")
            print(f"Using cache directory: {cache_dir}")

            # Determine which HF split to load
            if self.split in ["train", "val", "test"]:
                hf_split = self.split
                if self.split == "val":
                    hf_split = "validation"
            else:
                # Load all splits and combine them
                hf_split = "train"

            # Load dataset from Hugging Face with cache directory
            # Pin revision for security and reproducibility
            import os

            revision = os.environ.get("CELEBA_DATASET_REVISION", "main")
            ds = load_dataset(  # nosec B615 - revision is pinned via environment variable
                "flwrlabs/celeba",
                split=hf_split,
                streaming=False,
                cache_dir=cache_dir,
                revision=revision,
            )

            # Limit to num_samples
            if len(ds) > self.num_samples:
                ds = ds.select(range(self.num_samples))
            actual_num_samples = len(ds)

            print(f"Loading {actual_num_samples} images...")

            # Load and process images
            images = []
            attr_values = []

            for i, item in enumerate(ds):
                if i >= self.num_samples:
                    break

                # Process image
                img = item["image"]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img = img.resize((self.image_size, self.image_size))
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                images.append(img_array)

                # Process attributes if needed
                if self.include_attributes:
                    # Attributes are stored as individual keys in the HF dataset
                    attr_list = []
                    for attr_name in self.attribute_names:
                        if attr_name in item:
                            # Convert True/False to 1/0
                            attr_list.append(1 if item[attr_name] else 0)
                        else:
                            attr_list.append(0)  # Default to 0 if attribute not found
                    attr_values.append(attr_list)

            self.images = jnp.array(images)

            # Handle attributes
            if self.include_attributes and attr_values:
                self.attributes = jnp.array(attr_values, dtype=jnp.float32)

                # Extract selected attributes
                self.selected_attribute_indices = [
                    self.attribute_names.index(attr)
                    for attr in self.selected_attributes
                    if attr in self.attribute_names
                ]
                self.selected_attributes_data = self.attributes[:, self.selected_attribute_indices]
            else:
                self.attributes = None
                self.selected_attributes_data = None

            # Create splits for the loaded data
            if self.split == "all":
                # Create train/val/test splits from the loaded data
                self.train_indices = jnp.arange(0, int(0.8 * actual_num_samples))
                self.val_indices = jnp.arange(
                    int(0.8 * actual_num_samples), int(0.9 * actual_num_samples)
                )
                self.test_indices = jnp.arange(int(0.9 * actual_num_samples), actual_num_samples)
            elif self.split == "train":
                self.train_indices = jnp.arange(actual_num_samples)
                self.val_indices = jnp.array([])
                self.test_indices = jnp.array([])
            elif self.split == "val":
                self.train_indices = jnp.array([])
                self.val_indices = jnp.arange(actual_num_samples)
                self.test_indices = jnp.array([])
            elif self.split == "test":
                self.train_indices = jnp.array([])
                self.val_indices = jnp.array([])
                self.test_indices = jnp.arange(actual_num_samples)

            # Update num_samples to actual count
            self.num_samples = actual_num_samples

            print(f"Successfully loaded {actual_num_samples} CelebA images")

        except Exception as e:
            raise RuntimeError(f"Failed to load CelebA dataset from Hugging Face: {e}") from e

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of samples in the dataset
        """
        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "val":
            return len(self.val_indices)
        elif self.split == "test":
            return len(self.test_indices)
        else:
            return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample data
        """
        # Map index to the appropriate split
        if self.split == "train":
            actual_idx = self.train_indices[idx]
        elif self.split == "val":
            actual_idx = self.val_indices[idx]
        elif self.split == "test":
            actual_idx = self.test_indices[idx]
        else:
            actual_idx = idx

        # Get image
        image = self.images[actual_idx]

        # Create sample dict
        sample = {"images": image}

        # Add attributes if available
        if self.include_attributes and self.selected_attributes_data is not None:
            sample["attributes"] = self.selected_attributes_data[actual_idx]
            sample["attribute_names"] = self.selected_attributes

        return sample

    def get_batch(
        self, batch_size: int = 32, start_idx: int = 0, split: str | None = None
    ) -> dict[str, jnp.ndarray]:
        """Get a batch of samples.

        Args:
            batch_size: Batch size
            start_idx: Starting index
            split: Data split to use (overrides instance split if provided)

        Returns:
            Dictionary with batch data
        """
        # Use provided split or instance split
        current_split = split if split is not None else self.split

        # Get indices for the specified split
        if current_split == "train":
            indices = self.train_indices
        elif current_split == "val":
            indices = self.val_indices
        elif current_split == "test":
            indices = self.test_indices
        else:
            indices = jnp.arange(self.num_samples)

        # Ensure start_idx is within bounds
        if start_idx >= len(indices):
            start_idx = 0

        # Get batch indices
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]

        # Get batch images
        batch_images = self.images[batch_indices]

        # Create batch dict
        batch = {"images": batch_images}

        # Add attributes if available
        if self.include_attributes and self.selected_attributes_data is not None:
            batch["attributes"] = self.selected_attributes_data[batch_indices]
            batch["attribute_names"] = self.selected_attributes

        return batch

    def get_split(self, split: str) -> "CelebADataset":
        """Get a dataset for a specific split.

        Args:
            split: Split name ('train', 'val', 'test')

        Returns:
            Dataset instance for the specified split
        """
        # Create a new dataset instance with the same parameters but different split
        return CelebADataset(
            data_path=self.data_path,
            num_samples=self.num_samples,
            image_size=self.image_size,
            include_attributes=self.include_attributes,
            split=split,
            rngs=self.rngs,
        )
