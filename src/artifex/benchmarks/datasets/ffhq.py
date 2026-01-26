"""FFHQ Dataset Implementation for StyleGAN3 Benchmarks.

This module provides FFHQ (Flickr-Faces-HQ) dataset integration for high-resolution
face generation benchmarks with StyleGAN3.

Key Features:
- High-resolution face image dataset (256x256, 1024x1024)
- Mock data generation for development and testing
- Efficient batching and preprocessing
- Few-shot adaptation capabilities

Note: Dataset classes don't inherit from nnx.Module because they're data
containers, not neural network modules.
"""

from typing import Iterator

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class FFHQDataset:
    """FFHQ dataset for high-resolution face generation."""

    def __init__(
        self,
        data_path: str = "data/ffhq",
        split: str = "train",
        image_size: int = 256,
        channels: int = 3,
        num_samples: int = 70000,
        few_shot_samples: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize FFHQ dataset.

        Args:
            data_path: Path to FFHQ dataset directory
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (256, 512, 1024)
            channels: Number of image channels (3 for RGB)
            num_samples: Total number of samples in dataset
            few_shot_samples: Number of samples for few-shot learning
            rngs: Random number generators
        """
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.channels = channels
        self.num_samples = few_shot_samples or num_samples
        self.rngs = rngs

        # Dataset statistics for normalization
        self.mean = jnp.array([0.485, 0.456, 0.406])
        self.std = jnp.array([0.229, 0.224, 0.225])

        # Initialize dataset
        self._prepare_dataset()

    def _prepare_dataset(self):
        """Prepare dataset indices and metadata."""
        if self.split == "train":
            self.indices = list(range(0, int(self.num_samples * 0.8)))
        elif self.split == "val":
            self.indices = list(range(int(self.num_samples * 0.8), int(self.num_samples * 0.9)))
        else:  # test
            self.indices = list(range(int(self.num_samples * 0.9), self.num_samples))

    def _generate_mock_face_image(self, index: int) -> jnp.ndarray:
        """Generate mock face image with realistic features.

        Args:
            index: Image index for reproducible generation

        Returns:
            Mock face image of shape (height, width, channels)
        """
        # Use index for reproducible generation
        key = jax.random.fold_in(self.rngs.sample(), index)

        # Generate base face structure
        image = jax.random.uniform(
            key, (self.image_size, self.image_size, self.channels), minval=0.0, maxval=1.0
        )

        # Add face-like features
        center_x, center_y = self.image_size // 2, self.image_size // 2

        # Create coordinate grids
        y, x = jnp.meshgrid(jnp.arange(self.image_size), jnp.arange(self.image_size), indexing="ij")

        # Face oval
        face_radius = self.image_size * 0.35
        face_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < face_radius**2

        # Skin tone variation
        skin_base = 0.6 + 0.3 * jax.random.uniform(key, ())
        skin_color = jnp.array([skin_base, skin_base * 0.9, skin_base * 0.8])

        # Apply face base color
        image = jnp.where(face_mask[:, :, None], skin_color[None, None, :] + 0.1 * image, image)

        # Add eyes
        eye_y = center_y - self.image_size * 0.1
        eye_radius = self.image_size * 0.03

        # Left eye
        left_eye_x = center_x - self.image_size * 0.08
        left_eye_mask = ((x - left_eye_x) ** 2 + (y - eye_y) ** 2) < eye_radius**2

        # Right eye
        right_eye_x = center_x + self.image_size * 0.08
        right_eye_mask = ((x - right_eye_x) ** 2 + (y - eye_y) ** 2) < eye_radius**2

        eye_color = jnp.array([0.1, 0.1, 0.1])  # Dark eyes
        image = jnp.where(
            (left_eye_mask | right_eye_mask)[:, :, None], eye_color[None, None, :], image
        )

        # Add mouth
        mouth_y = center_y + self.image_size * 0.15
        mouth_width = self.image_size * 0.06
        mouth_height = self.image_size * 0.02

        mouth_mask = (jnp.abs(x - center_x) < mouth_width) & (jnp.abs(y - mouth_y) < mouth_height)

        mouth_color = jnp.array([0.3, 0.1, 0.1])  # Dark mouth
        image = jnp.where(mouth_mask[:, :, None], mouth_color[None, None, :], image)

        # Add some noise for texture
        noise_key = jax.random.fold_in(key, 1)
        noise = 0.05 * jax.random.normal(
            noise_key, (self.image_size, self.image_size, self.channels)
        )
        image = jnp.clip(image + noise, 0.0, 1.0)

        return image

    def _preprocess_image(self, image: jnp.ndarray) -> jnp.ndarray:
        """Preprocess image for training.

        Args:
            image: Raw image of shape (height, width, channels)

        Returns:
            Preprocessed image
        """
        # Normalize to [-1, 1] range for StyleGAN3
        image = image * 2.0 - 1.0

        return image

    def __call__(self, batch_size: int) -> Iterator[dict[str, jnp.ndarray]]:
        """Generate batches of face images.

        Args:
            batch_size: Number of images per batch

        Yields:
            Batches containing:
            - images: Face images of shape (batch_size, height, width, channels)
            - indices: Image indices for tracking
        """
        num_batches = len(self.indices) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_indices = self.indices[start_idx:end_idx]

            # Generate batch of images
            images = []
            for idx in batch_indices:
                image = self._generate_mock_face_image(idx)
                image = self._preprocess_image(image)
                images.append(image)

            images = jnp.stack(images, axis=0)

            yield {
                "images": images,
                "indices": jnp.array(batch_indices),
            }

    def get_few_shot_batch(self, num_samples: int = 100) -> dict[str, jnp.ndarray]:
        """Get a few-shot batch for domain adaptation.

        Args:
            num_samples: Number of samples for few-shot learning

        Returns:
            Few-shot batch with images and metadata
        """
        # Select random indices for few-shot learning
        key = self.rngs.sample()
        selected_indices = jax.random.choice(
            key, jnp.array(self.indices), shape=(num_samples,), replace=False
        )

        # Generate few-shot images
        images = []
        for idx in selected_indices:
            image = self._generate_mock_face_image(int(idx))
            image = self._preprocess_image(image)
            images.append(image)

        images = jnp.stack(images, axis=0)

        return {
            "images": images,
            "indices": selected_indices,
            "num_samples": num_samples,
        }


class CelebADataset:
    """CelebA dataset for few-shot adaptation experiments."""

    def __init__(
        self,
        data_path: str = "data/celeba",
        split: str = "train",
        image_size: int = 256,
        channels: int = 3,
        num_samples: int = 202599,
        target_attributes: list | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize CelebA dataset.

        Args:
            data_path: Path to CelebA dataset directory
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size
            channels: Number of image channels
            num_samples: Total number of samples
            target_attributes: Specific attributes to filter on
            rngs: Random number generators
        """
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.channels = channels
        self.num_samples = num_samples
        self.target_attributes = target_attributes or []
        self.rngs = rngs

        # CelebA attribute list (40 attributes)
        self.attributes = [
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

        self._prepare_dataset()

    def _prepare_dataset(self):
        """Prepare dataset indices and attribute mappings."""
        if self.split == "train":
            self.indices = list(range(0, int(self.num_samples * 0.8)))
        elif self.split == "val":
            self.indices = list(range(int(self.num_samples * 0.8), int(self.num_samples * 0.9)))
        else:  # test
            self.indices = list(range(int(self.num_samples * 0.9), self.num_samples))

    def _generate_mock_celeba_image(
        self, index: int, attributes: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate mock CelebA image with attributes.

        Args:
            index: Image index
            attributes: Optional attribute vector

        Returns:
            Tuple of (image, attribute_vector)
        """
        # Generate attributes if not provided
        if attributes is None:
            key = jax.random.fold_in(self.rngs.sample(), index)
            attributes = jax.random.bernoulli(key, p=0.3, shape=(len(self.attributes),)).astype(
                jnp.float32
            )

        # Generate base image (similar to FFHQ but with attribute variations)
        key = jax.random.fold_in(self.rngs.sample(), index + 1000)
        image = self._generate_attributed_face(key, attributes)

        return image, attributes

    def _generate_attributed_face(self, key: jnp.ndarray, attributes: jnp.ndarray) -> jnp.ndarray:
        """Generate face image based on attributes.

        Args:
            key: Random key
            attributes: Attribute vector

        Returns:
            Generated face image
        """
        # Start with base face
        image = jax.random.uniform(
            key, (self.image_size, self.image_size, self.channels), minval=0.3, maxval=0.7
        )

        # Modify based on attributes
        center_x, center_y = self.image_size // 2, self.image_size // 2

        # Male/Female variations
        if attributes[20]:  # Male
            # Broader jaw, different hair
            image = image * 0.9  # Slightly darker

        # Smiling
        if attributes[31]:  # Smiling
            # Add smile features
            mouth_y = center_y + self.image_size * 0.1
            for dy in range(-5, 6):
                for dx in range(-15, 16):
                    y, x = mouth_y + dy, center_x + dx
                    if 0 <= y < self.image_size and 0 <= x < self.image_size:
                        # Upward curve for smile
                        curve_factor = 1 + 0.1 * (1 - abs(dx) / 15) * (dy + 5) / 10
                        image = image.at[int(y), int(x)].multiply(curve_factor)

        # Add noise and normalize
        noise_key = jax.random.fold_in(key, 1)
        noise = 0.1 * jax.random.normal(
            noise_key, (self.image_size, self.image_size, self.channels)
        )
        image = jnp.clip(image + noise, 0.0, 1.0)

        # Normalize to [-1, 1] for StyleGAN3
        image = image * 2.0 - 1.0

        return image

    def __call__(self, batch_size: int) -> Iterator[dict[str, jnp.ndarray]]:
        """Generate batches of CelebA images with attributes.

        Args:
            batch_size: Number of images per batch

        Yields:
            Batches containing images and attributes
        """
        num_batches = len(self.indices) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_indices = self.indices[start_idx:end_idx]

            images = []
            attributes_batch = []

            for idx in batch_indices:
                image, attrs = self._generate_mock_celeba_image(idx)
                images.append(image)
                attributes_batch.append(attrs)

            images = jnp.stack(images, axis=0)
            attributes_batch = jnp.stack(attributes_batch, axis=0)

            yield {
                "images": images,
                "attributes": attributes_batch,
                "indices": jnp.array(batch_indices),
            }
