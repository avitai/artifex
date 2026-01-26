"""Data generation utilities for test fixtures.

This module implements the three-tier data generation strategy:
1. Fast random data for smoke tests
2. Synthetic realistic data for integration tests
3. Cached complex data for performance tests
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


class FastDataGenerator:
    """Fast random data for smoke tests.

    Generates simple random data with minimal computational overhead.
    Used for basic functionality testing where data realism is not critical.
    """

    @staticmethod
    def random_image_batch(
        shape: tuple[int, ...], batch_size: int = 8, seed: int = 42
    ) -> jax.Array:
        """Generate random image batch for basic testing.

        Args:
            shape: Image shape (height, width, channels)
            batch_size: Number of images in batch
            seed: Random seed for reproducibility

        Returns:
            Random image batch with values in [0, 1]
        """
        key = jax.random.PRNGKey(seed)
        # Generate values in [0, 1] range
        images = jax.random.uniform(key, (batch_size, *shape), minval=0.0, maxval=1.0)
        return images

    @staticmethod
    def random_point_cloud(
        num_points: int = 1024, dims: int = 3, batch_size: int = 8, seed: int = 42
    ) -> jax.Array:
        """Generate random point cloud for basic geometric tests.

        Args:
            num_points: Number of points in cloud
            dims: Dimensionality of points (typically 3)
            batch_size: Number of point clouds in batch
            seed: Random seed for reproducibility

        Returns:
            Random point cloud batch with values in [-1, 1]
        """
        key = jax.random.PRNGKey(seed)
        points = jax.random.uniform(key, (batch_size, num_points, dims), minval=-1.0, maxval=1.0)
        return points

    @staticmethod
    def random_timesteps(
        max_timesteps: int = 1000, batch_size: int = 8, seed: int = 42
    ) -> jax.Array:
        """Generate random timesteps for diffusion tests.

        Args:
            max_timesteps: Maximum timestep value
            batch_size: Number of timesteps to generate
            seed: Random seed for reproducibility

        Returns:
            Random timesteps as integers in [0, max_timesteps)
        """
        key = jax.random.PRNGKey(seed)
        timesteps = jax.random.randint(key, (batch_size,), 0, max_timesteps)
        return timesteps

    @staticmethod
    def random_noise(shape: tuple[int, ...], batch_size: int = 8, seed: int = 42) -> jax.Array:
        """Generate random Gaussian noise.

        Args:
            shape: Shape of noise arrays
            batch_size: Number of noise arrays
            seed: Random seed for reproducibility

        Returns:
            Gaussian noise with mean 0, std 1
        """
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, (batch_size, *shape))
        return noise

    @staticmethod
    def random_labels(num_classes: int = 10, batch_size: int = 8, seed: int = 42) -> jax.Array:
        """Generate random class labels.

        Args:
            num_classes: Number of possible classes
            batch_size: Number of labels to generate
            seed: Random seed for reproducibility

        Returns:
            Random integer labels in [0, num_classes)
        """
        key = jax.random.PRNGKey(seed)
        labels = jax.random.randint(key, (batch_size,), 0, num_classes)
        return labels


class SyntheticDataGenerator:
    """Generate realistic synthetic data for comprehensive testing.

    Creates data that mimics real-world patterns and structures while being
    deterministic and computationally efficient.
    """

    @staticmethod
    def synthetic_images(
        pattern_type: str = "mnist_like",
        shape: tuple[int, ...] = (28, 28, 1),
        batch_size: int = 8,
        seed: int = 42,
    ) -> jax.Array:
        """Generate synthetic images with realistic patterns.

        Args:
            pattern_type: Type of patterns to generate ('mnist_like', 'natural_like', 'structured')
            shape: Image shape (height, width, channels)
            batch_size: Number of images in batch
            seed: Random seed for reproducibility

        Returns:
            Synthetic image batch with realistic patterns
        """
        key = jax.random.PRNGKey(seed)

        if pattern_type == "mnist_like":
            return SyntheticDataGenerator._generate_digit_like_images(key, shape, batch_size)
        elif pattern_type == "natural_like":
            return SyntheticDataGenerator._generate_natural_like_images(key, shape, batch_size)
        elif pattern_type == "structured":
            return SyntheticDataGenerator._generate_structured_patterns(key, shape, batch_size)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    @staticmethod
    def _generate_digit_like_images(key, shape, batch_size):
        """Generate MNIST-like digit patterns."""
        height, width = shape[:2]
        channels = shape[2] if len(shape) > 2 else 1
        images = jnp.zeros((batch_size, height, width, channels))

        for i in range(batch_size):
            key, subkey = jax.random.split(key)
            # Generate geometric shapes (circles, lines, curves)
            digit_type = i % 10  # Cycle through 10 different patterns
            image = SyntheticDataGenerator._create_geometric_pattern(
                subkey, height, width, digit_type
            )
            if channels == 1:
                images = images.at[i].set(image)
            else:
                # Replicate across channels for RGB
                images = images.at[i].set(jnp.repeat(image, channels, axis=-1))

        # Add realistic noise
        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, images.shape) * 0.05
        return jnp.clip(images + noise, 0.0, 1.0)

    @staticmethod
    def _create_geometric_pattern(key, height, width, pattern_id):
        """Create geometric patterns resembling digits."""
        x = jnp.linspace(0, 1, width)
        y = jnp.linspace(0, 1, height)
        X, Y = jnp.meshgrid(x, y)

        if pattern_id == 0:  # Circle (like '0')
            center_x, center_y = 0.5, 0.5
            radius = 0.3
            circle = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius**2
            ring = circle & (((X - center_x) ** 2 + (Y - center_y) ** 2) > (radius * 0.6) ** 2)
            return ring.astype(jnp.float32)[..., None]
        elif pattern_id == 1:  # Vertical line (like '1')
            line = (jnp.abs(X - 0.5) < 0.08).astype(jnp.float32)
            return line[..., None]
        elif pattern_id == 2:  # Horizontal lines (like '2')
            lines = ((jnp.abs(Y - 0.3) < 0.05) | (jnp.abs(Y - 0.7) < 0.05)).astype(jnp.float32)
            return lines[..., None]
        elif pattern_id == 3:  # Cross (like '3' or '+')
            cross = ((jnp.abs(X - 0.5) < 0.05) | (jnp.abs(Y - 0.5) < 0.05)).astype(jnp.float32)
            return cross[..., None]
        elif pattern_id == 4:  # Square (like '4')
            square = (
                (X > 0.2)
                & (X < 0.8)
                & (Y > 0.2)
                & (Y < 0.8)
                & ((X < 0.3) | (X > 0.7) | (Y < 0.3) | (Y > 0.7))
            ).astype(jnp.float32)
            return square[..., None]
        else:  # Random blob for other patterns
            key, subkey = jax.random.split(key)
            blob_center_x = jax.random.uniform(subkey, (), minval=0.2, maxval=0.8)
            key, subkey = jax.random.split(key)
            blob_center_y = jax.random.uniform(subkey, (), minval=0.2, maxval=0.8)
            blob = jnp.exp(-((X - blob_center_x) ** 2 + (Y - blob_center_y) ** 2) / 0.05)
            return blob[..., None]

    @staticmethod
    def _generate_natural_like_images(key, shape, batch_size):
        """Generate natural image-like patterns."""
        height, width = shape[:2]
        channels = shape[2] if len(shape) > 2 else 1

        images = []
        for i in range(batch_size):
            key, subkey = jax.random.split(key)

            # Create multi-scale noise (like natural textures)
            image = jnp.zeros((height, width))
            for scale in [1, 2, 4, 8]:
                key, noise_key = jax.random.split(key)
                noise_h, noise_w = height // scale, width // scale
                noise = jax.random.normal(noise_key, (noise_h, noise_w))
                # Upsample noise
                upsampled = jnp.repeat(jnp.repeat(noise, scale, axis=0), scale, axis=1)
                upsampled = upsampled[:height, :width]  # Crop to exact size
                image += upsampled / scale

            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

            if channels == 1:
                images.append(image[..., None])
            else:
                # Create correlated color channels
                key, color_key = jax.random.split(key)
                color_variations = jax.random.normal(color_key, (channels,)) * 0.1
                multi_channel = jnp.stack([image + var for var in color_variations], axis=-1)
                multi_channel = jnp.clip(multi_channel, 0, 1)
                images.append(multi_channel)

        return jnp.stack(images)

    @staticmethod
    def _generate_structured_patterns(key, shape, batch_size):
        """Generate structured geometric patterns."""
        height, width = shape[:2]
        channels = shape[2] if len(shape) > 2 else 1

        images = []
        for i in range(batch_size):
            key, subkey = jax.random.split(key)

            # Create checkerboard-like patterns
            x = jnp.arange(width)
            y = jnp.arange(height)
            X, Y = jnp.meshgrid(x, y)

            # Random checkerboard size
            key, size_key = jax.random.split(key)
            check_size = jax.random.randint(size_key, (), 4, 16)

            pattern = ((X // check_size) + (Y // check_size)) % 2

            if channels == 1:
                images.append(pattern.astype(jnp.float32)[..., None])
            else:
                multi_channel = jnp.repeat(pattern[..., None], channels, axis=-1)
                images.append(multi_channel.astype(jnp.float32))

        return jnp.stack(images)

    @staticmethod
    def synthetic_point_clouds(
        cloud_type: str = "sphere", num_points: int = 1024, batch_size: int = 8, seed: int = 42
    ) -> jax.Array:
        """Generate synthetic point clouds with geometric structure.

        Args:
            cloud_type: Type of point cloud ('sphere', 'cube', 'torus', 'bunny')
            num_points: Number of points in each cloud
            batch_size: Number of point clouds in batch
            seed: Random seed for reproducibility

        Returns:
            Synthetic point cloud batch with realistic structure
        """
        key = jax.random.PRNGKey(seed)

        if cloud_type == "sphere":
            return SyntheticDataGenerator._generate_sphere_points(key, num_points, batch_size)
        elif cloud_type == "cube":
            return SyntheticDataGenerator._generate_cube_points(key, num_points, batch_size)
        elif cloud_type == "torus":
            return SyntheticDataGenerator._generate_torus_points(key, num_points, batch_size)
        elif cloud_type == "bunny":
            return SyntheticDataGenerator._generate_bunny_like_points(key, num_points, batch_size)
        else:
            raise ValueError(f"Unknown cloud type: {cloud_type}")

    @staticmethod
    def _generate_sphere_points(key, num_points, batch_size):
        """Generate points on unit sphere surface."""
        points = []
        for _ in range(batch_size):
            key, subkey = jax.random.split(key)
            # Sample from normal distribution and normalize to sphere
            raw_points = jax.random.normal(subkey, (num_points, 3))
            norms = jnp.linalg.norm(raw_points, axis=1, keepdims=True)
            sphere_points = raw_points / (norms + 1e-8)
            points.append(sphere_points)

        return jnp.stack(points)

    @staticmethod
    def _generate_cube_points(key, num_points, batch_size):
        """Generate points on cube surface."""
        points = []
        for _ in range(batch_size):
            key, subkey = jax.random.split(key)

            # Sample points on 6 faces of unit cube
            face_points = num_points // 6
            cube_points = []

            for face in range(6):
                key, face_key = jax.random.split(key)

                if face == 0:  # +X face
                    face_pts = jnp.array(
                        [
                            jnp.ones(face_points),
                            jax.random.uniform(face_key, (face_points,), -1, 1),
                            jax.random.uniform(face_key, (face_points,), -1, 1),
                        ]
                    ).T
                elif face == 1:  # -X face
                    face_pts = jnp.array(
                        [
                            -jnp.ones(face_points),
                            jax.random.uniform(face_key, (face_points,), -1, 1),
                            jax.random.uniform(face_key, (face_points,), -1, 1),
                        ]
                    ).T
                # ... similar for other faces
                else:  # Simplified: random points for remaining faces
                    face_pts = jax.random.uniform(face_key, (face_points, 3), -1, 1)

                cube_points.append(face_pts)

            # Handle remaining points
            remaining = num_points - 6 * face_points
            if remaining > 0:
                key, remain_key = jax.random.split(key)
                extra_pts = jax.random.uniform(remain_key, (remaining, 3), -1, 1)
                cube_points.append(extra_pts)

            points.append(jnp.concatenate(cube_points, axis=0))

        return jnp.stack(points)

    @staticmethod
    def _generate_torus_points(key, num_points, batch_size):
        """Generate points on torus surface."""
        points = []
        R = 1.0  # Major radius
        r = 0.3  # Minor radius

        for _ in range(batch_size):
            key, subkey = jax.random.split(key)

            # Sample angles uniformly
            key, theta_key = jax.random.split(key)
            key, phi_key = jax.random.split(key)

            theta = jax.random.uniform(theta_key, (num_points,), 0, 2 * jnp.pi)
            phi = jax.random.uniform(phi_key, (num_points,), 0, 2 * jnp.pi)

            # Convert to Cartesian coordinates
            x = (R + r * jnp.cos(phi)) * jnp.cos(theta)
            y = (R + r * jnp.cos(phi)) * jnp.sin(theta)
            z = r * jnp.sin(phi)

            torus_points = jnp.stack([x, y, z], axis=1)
            points.append(torus_points)

        return jnp.stack(points)

    @staticmethod
    def _generate_bunny_like_points(key, num_points, batch_size):
        """Generate bunny-like shape points (simplified)."""
        points = []

        for _ in range(batch_size):
            key, subkey = jax.random.split(key)

            # Create a simple bunny-like shape using multiple spheres
            body_points = num_points // 3
            head_points = num_points // 3
            ear_points = num_points - body_points - head_points

            # Body (larger sphere)
            key, body_key = jax.random.split(key)
            body = jax.random.normal(body_key, (body_points, 3))
            body_norms = jnp.linalg.norm(body, axis=1, keepdims=True)
            body = body / (body_norms + 1e-8) * 0.8  # Scale down
            body = body + jnp.array([0, 0, -0.3])  # Offset down

            # Head (smaller sphere)
            key, head_key = jax.random.split(key)
            head = jax.random.normal(head_key, (head_points, 3))
            head_norms = jnp.linalg.norm(head, axis=1, keepdims=True)
            head = head / (head_norms + 1e-8) * 0.5  # Smaller
            head = head + jnp.array([0, 0, 0.4])  # Offset up

            # Ears (elongated)
            key, ear_key = jax.random.split(key)
            ears = jax.random.normal(ear_key, (ear_points, 3))
            ears = ears * jnp.array([0.1, 0.1, 0.8])  # Elongate in Z
            ears = ears + jnp.array([0, 0, 0.8])  # Offset up more

            bunny_points = jnp.concatenate([body, head, ears], axis=0)
            points.append(bunny_points)

        return jnp.stack(points)


class CachedDataManager:
    """Manage cached test data for complex scenarios.

    Provides session and disk caching for expensive data generation operations.
    Uses specification-based caching to ensure deterministic results.
    """

    _session_cache: dict[str, Any] = {}
    _disk_cache_dir = Path("test_artifacts/data_cache")

    @classmethod
    def get_cached_data(cls, data_spec: dict[str, Any], use_disk_cache: bool = True):
        """Get cached data based on specification.

        Args:
            data_spec: Dictionary specifying the data to generate
            use_disk_cache: Whether to use disk caching

        Returns:
            Generated or cached data matching the specification
        """
        cache_key = cls._create_cache_key(data_spec)

        # Check session cache first
        if cache_key in cls._session_cache:
            return cls._session_cache[cache_key]

        # Check disk cache
        if use_disk_cache:
            cls._disk_cache_dir.mkdir(parents=True, exist_ok=True)
            disk_path = cls._disk_cache_dir / f"{cache_key}.npz"
            if disk_path.exists():
                data = cls._load_from_disk(disk_path)
                cls._session_cache[cache_key] = data
                return data

        # Generate new data
        data = cls._generate_data(data_spec)
        cls._session_cache[cache_key] = data

        if use_disk_cache:
            cls._save_to_disk(data, disk_path)

        return data

    @classmethod
    def _create_cache_key(cls, data_spec: dict[str, Any]) -> str:
        """Create a unique cache key from data specification."""
        spec_str = json.dumps(data_spec, sort_keys=True)
        return hashlib.md5(spec_str.encode()).hexdigest()

    @classmethod
    def _load_from_disk(cls, disk_path: Path):
        """Load data from disk cache."""
        import numpy as np

        loaded = np.load(disk_path, allow_pickle=True)

        # Convert back to JAX arrays and reconstruct structure
        if "data_type" in loaded and loaded["data_type"].item() == "dict":
            data = {}
            for key in loaded["keys"]:
                data[key] = jnp.array(loaded[f"data_{key}"])
            return data
        else:
            return jnp.array(loaded["data"])

    @classmethod
    def _save_to_disk(cls, data, disk_path: Path):
        """Save data to disk cache."""
        import numpy as np

        save_dict = {}
        if isinstance(data, dict):
            save_dict["data_type"] = "dict"
            save_dict["keys"] = list(data.keys())
            for key, value in data.items():
                save_dict[f"data_{key}"] = np.array(value)
        else:
            save_dict["data"] = np.array(data)

        np.savez_compressed(disk_path, **save_dict)  # type: ignore

    @classmethod
    def _generate_data(cls, data_spec: dict[str, Any]):
        """Generate data based on specification."""
        data_type = data_spec["type"]

        if data_type == "diffusion_sequence":
            return cls._generate_diffusion_sequence(data_spec)
        elif data_type == "large_point_cloud":
            return cls._generate_large_point_cloud(data_spec)
        elif data_type == "protein_structure":
            return cls._generate_protein_structure(data_spec)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    @classmethod
    def _generate_diffusion_sequence(cls, spec):
        """Generate a sequence of diffusion states."""
        shape = spec["shape"]
        num_timesteps = spec["num_timesteps"]
        batch_size = spec["batch_size"]
        seed = spec.get("seed", 42)

        key = jax.random.PRNGKey(seed)

        # Generate clean data
        x0 = SyntheticDataGenerator.synthetic_images("mnist_like", shape, batch_size, seed)

        # Generate noise schedule
        betas = jnp.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = jnp.cumprod(alphas)

        # Generate noisy versions at different timesteps
        sequence = []
        sample_timesteps = jnp.arange(0, num_timesteps, max(1, num_timesteps // 10))

        for t in sample_timesteps:
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, x0.shape)
            alpha_bar_t = alpha_bars[t]
            xt = jnp.sqrt(alpha_bar_t) * x0 + jnp.sqrt(1 - alpha_bar_t) * noise
            sequence.append(xt)

        return {
            "x0": x0,
            "sequence": jnp.stack(sequence),
            "timesteps": sample_timesteps,
            "betas": betas,
            "alpha_bars": alpha_bars,
        }

    @classmethod
    def _generate_large_point_cloud(cls, spec):
        """Generate large point cloud for performance testing."""
        cloud_type = spec.get("cloud_type", "sphere")
        num_points = spec["num_points"]
        batch_size = spec["batch_size"]
        seed = spec.get("seed", 42)

        return SyntheticDataGenerator.synthetic_point_clouds(
            cloud_type, num_points, batch_size, seed
        )

    @classmethod
    def _generate_protein_structure(cls, spec):
        """Generate protein-like structure data."""
        num_residues = spec.get("num_residues", 256)
        batch_size = spec["batch_size"]
        seed = spec.get("seed", 42)

        key = jax.random.PRNGKey(seed)

        structures = []
        for _ in range(batch_size):
            # Generate backbone atoms (simplified)
            key, subkey = jax.random.split(key)

            # Create a chain-like structure
            t = jnp.linspace(0, 10, num_residues)
            x = jnp.sin(t) + 0.1 * jax.random.normal(subkey, (num_residues,))
            key, subkey = jax.random.split(key)
            y = jnp.cos(t) + 0.1 * jax.random.normal(subkey, (num_residues,))
            key, subkey = jax.random.split(key)
            z = 0.1 * t + 0.1 * jax.random.normal(subkey, (num_residues,))

            protein_coords = jnp.stack([x, y, z], axis=1)
            structures.append(protein_coords)

        return jnp.stack(structures)

    @classmethod
    def clear_cache(cls, disk_cache_only: bool = False):
        """Clear cached data.

        Args:
            disk_cache_only: If True, only clear disk cache, keep session cache
        """
        if not disk_cache_only:
            cls._session_cache.clear()

        if cls._disk_cache_dir.exists():
            import shutil

            shutil.rmtree(cls._disk_cache_dir)
            cls._disk_cache_dir.mkdir(parents=True, exist_ok=True)
