"""Advanced image augmentation for generative models.

This module provides JAX-compatible image augmentation techniques
for training robust generative models.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ImageAugmentationConfig
from artifex.generative_models.extensions.base import AugmentationExtension


class AdvancedImageAugmentation(AugmentationExtension):
    """Advanced image augmentation for generative models.

    This extension only accepts the canonical typed image augmentation config.
    """

    def __init__(
        self,
        config: ImageAugmentationConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize image augmentation module.

        Args:
            config: Typed image augmentation configuration
            rngs: Random number generator keys

        Raises:
            TypeError: If config is not a supported type
        """
        if not isinstance(config, ImageAugmentationConfig):
            raise TypeError(f"config must be ImageAugmentationConfig, got {type(config).__name__}")
        super().__init__(config, rngs=rngs)

        self.rngs = rngs

    def augment(
        self,
        images: jax.Array,
        *,
        key: jax.Array | None = None,
        deterministic: bool = False,
    ) -> jax.Array:
        """Apply the typed image augmentation pipeline."""
        if deterministic or self.config.deterministic or self.config.probability == 0.0:
            return images

        augmented = images
        if self.config.random_flip_horizontal:
            augmented = self.apply_horizontal_flip(augmented)
        if self.config.random_flip_vertical:
            augmented = self.apply_vertical_flip(augmented)
        if self.config.random_rotation > 0:
            augmented = self._apply_random_rotations(
                augmented,
                max_rotation=self.config.random_rotation,
            )
        if self.config.color_jitter:
            augmented = self._apply_color_jitter(augmented)
        return jnp.clip(augmented, 0.0, 1.0)

    def _apply_random_rotations(self, images: jax.Array, *, max_rotation: float) -> jax.Array:
        """Apply random in-plane rotations."""
        batch_size, height, width, _ = images.shape

        angles = jax.random.uniform(
            self.rngs.augment(),
            (batch_size,),
            minval=-max_rotation,
            maxval=max_rotation,
        )
        zeros = jnp.zeros((batch_size,))
        ones = jnp.ones((batch_size,))
        return jax.vmap(self._transform_single_image)(images, angles, zeros, zeros, ones)

    def _transform_single_image(
        self, image: jax.Array, angle: jax.Array, tx: jax.Array, ty: jax.Array, scale: jax.Array
    ) -> jax.Array:
        """Apply geometric transformation to a single image."""
        height, width = image.shape[:2]

        # Create transformation matrix
        # Center coordinates
        cx, cy = width / 2, height / 2

        # Rotation matrix
        cos_a, sin_a = jnp.cos(jnp.radians(angle)), jnp.sin(jnp.radians(angle))

        # Combined transformation matrix (scale, rotate, translate)
        transform_matrix = jnp.array(
            [
                [scale * cos_a, -scale * sin_a, tx + cx - scale * cos_a * cx + scale * sin_a * cy],
                [scale * sin_a, scale * cos_a, ty + cy - scale * sin_a * cx - scale * cos_a * cy],
                [0, 0, 1],
            ]
        )

        # Apply transformation using bilinear interpolation
        transformed = self._apply_affine_transform(image, transform_matrix)

        return transformed

    def _apply_affine_transform(self, image: jax.Array, transform_matrix: jax.Array) -> jax.Array:
        """Apply affine transformation using bilinear interpolation."""
        height, width = image.shape[:2]

        # Create coordinate grid
        y_coords, x_coords = jnp.mgrid[0:height, 0:width]
        coords = jnp.stack([x_coords.flatten(), y_coords.flatten(), jnp.ones(height * width)])

        # Apply inverse transformation
        inv_transform = jnp.linalg.inv(transform_matrix)
        transformed_coords = jnp.dot(inv_transform, coords)

        # Extract x, y coordinates
        x_new = transformed_coords[0].reshape(height, width)
        y_new = transformed_coords[1].reshape(height, width)

        # Bilinear interpolation
        transformed = self._bilinear_interpolate(image, x_new, y_new)

        return transformed

    def _bilinear_interpolate(
        self, image: jax.Array, x_coords: jax.Array, y_coords: jax.Array
    ) -> jax.Array:
        """Perform bilinear interpolation."""
        height, width = image.shape[:2]

        # Clamp coordinates to image bounds
        x_coords = jnp.clip(x_coords, 0, width - 1)
        y_coords = jnp.clip(y_coords, 0, height - 1)

        # Get integer coordinates
        x0 = jnp.floor(x_coords).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, width - 1)
        y0 = jnp.floor(y_coords).astype(jnp.int32)
        y1 = jnp.minimum(y0 + 1, height - 1)

        # Get fractional parts
        fx = x_coords - x0
        fy = y_coords - y0

        # Bilinear interpolation
        if len(image.shape) == 3:  # Color image
            i00 = image[y0, x0]
            i01 = image[y1, x0]
            i10 = image[y0, x1]
            i11 = image[y1, x1]
        else:  # Grayscale
            i00 = image[y0, x0, None]
            i01 = image[y1, x0, None]
            i10 = image[y0, x1, None]
            i11 = image[y1, x1, None]

        # Interpolate
        fx = fx[..., None]
        fy = fy[..., None]

        interpolated = (
            i00 * (1 - fx) * (1 - fy) + i10 * fx * (1 - fy) + i01 * (1 - fx) * fy + i11 * fx * fy
        )

        if len(image.shape) == 2:
            interpolated = interpolated.squeeze(-1)

        return interpolated

    def _apply_color_jitter(self, images: jax.Array) -> jax.Array:
        """Apply typed brightness and contrast jitter."""
        batch_size = images.shape[0]

        brightness_factors = jax.random.uniform(
            self.rngs.augment(),
            (batch_size, 1, 1, 1),
            minval=self.config.brightness_range[0],
            maxval=self.config.brightness_range[1],
        )

        contrast_factors = jax.random.uniform(
            self.rngs.augment(),
            (batch_size, 1, 1, 1),
            minval=self.config.contrast_range[0],
            maxval=self.config.contrast_range[1],
        )

        # Apply brightness
        adjusted = images * brightness_factors

        # Apply contrast (around mean)
        mean_values = jnp.mean(adjusted, axis=(1, 2, 3), keepdims=True)
        adjusted = mean_values + contrast_factors * (adjusted - mean_values)

        # Clamp to valid range
        adjusted = jnp.clip(adjusted, 0.0, 1.0)

        return adjusted

    def _apply_saturation(self, images: jax.Array) -> jax.Array:
        """Apply saturation adjustment to color images."""
        batch_size = images.shape[0]

        # Generate random saturation factors
        saturation_factors = jax.random.uniform(
            self.rngs.augment(),
            (batch_size, 1, 1, 1),
            minval=1 - self.augmentation_config["saturation_range"],
            maxval=1 + self.augmentation_config["saturation_range"],
        )

        # Convert to grayscale (luminance)
        # Use standard RGB to grayscale conversion weights
        gray = 0.299 * images[..., 0:1] + 0.587 * images[..., 1:2] + 0.114 * images[..., 2:3]
        gray = jnp.repeat(gray, 3, axis=-1)

        # Interpolate between grayscale and original
        saturated = gray + saturation_factors * (images - gray)

        return saturated

    def _apply_noise_injection(self, images: jax.Array) -> jax.Array:
        """Apply Gaussian noise injection."""
        noise = (
            jax.random.normal(self.rngs.augment(), images.shape)
            * self.augmentation_config["noise_level"]
        )

        noisy_images = images + noise
        noisy_images = jnp.clip(noisy_images, 0.0, 1.0)

        return noisy_images

    def _apply_blur(self, images: jax.Array) -> jax.Array:
        """Apply random blur to images."""
        batch_size = images.shape[0]

        # Random decision for each image whether to apply blur
        apply_blur = (
            jax.random.uniform(self.rngs.augment(), (batch_size,))
            < self.augmentation_config["blur_probability"]
        )

        # Apply blur to all images, then select based on mask (JIT-compatible)
        blurred_all = jax.vmap(self._gaussian_blur)(images)

        # Broadcast apply_blur to image dimensions
        mask = apply_blur.reshape(batch_size, *((1,) * (images.ndim - 1)))
        return jnp.where(mask, blurred_all, images)

    def _gaussian_blur(self, image: jax.Array, sigma: float = 1.0) -> jax.Array:
        """Apply Gaussian blur to a single image."""
        kernel_size = 3
        kernel = jnp.ones((kernel_size, kernel_size)) / (kernel_size**2)
        pad_h, pad_w = kernel_size // 2, kernel_size // 2

        if len(image.shape) == 3:  # Color image (H, W, C)
            num_channels = image.shape[-1]
            # Use depthwise convolution: (1, C, H, W) with (C, 1, kH, kW) kernel
            image_4d = jnp.transpose(image, (2, 0, 1))[None]  # (1, C, H, W)
            kernel_4d = jnp.tile(kernel[None, None], (num_channels, 1, 1, 1))  # (C, 1, kH, kW)
            result = jax.lax.conv_general_dilated(
                image_4d,
                kernel_4d,
                window_strides=(1, 1),
                padding=((pad_h, pad_h), (pad_w, pad_w)),
                feature_group_count=num_channels,
            )
            return jnp.transpose(result[0], (1, 2, 0))  # Back to (H, W, C)
        # Grayscale
        return self._apply_conv2d(image, kernel)

    def _apply_conv2d(self, image: jax.Array, kernel: jax.Array) -> jax.Array:
        """Apply 2D convolution using JAX's built-in conv operation."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Reshape for lax.conv_general_dilated: (batch=1, channels=1, H, W)
        image_4d = image[None, None, :, :]
        kernel_4d = kernel[None, None, :, :]

        result = jax.lax.conv_general_dilated(
            image_4d,
            kernel_4d,
            window_strides=(1, 1),
            padding=((pad_h, pad_h), (pad_w, pad_w)),
        )
        return result[0, 0]

    def apply_horizontal_flip(self, images: jax.Array) -> jax.Array:
        """Apply horizontal flip with given probability."""
        batch_size = images.shape[0]

        # Random decision for each image
        flip_mask = jax.random.uniform(self.rngs.augment(), (batch_size,)) < self.config.probability

        # Apply flip to all, then select with mask (JIT-compatible)
        flipped_all = jnp.flip(images, axis=2)  # Flip along width axis (NHWC)
        mask = flip_mask.reshape(batch_size, *((1,) * (images.ndim - 1)))
        return jnp.where(mask, flipped_all, images)

    def apply_vertical_flip(self, images: jax.Array) -> jax.Array:
        """Apply vertical flip with the configured probability."""
        batch_size = images.shape[0]
        flip_mask = jax.random.uniform(self.rngs.augment(), (batch_size,)) < self.config.probability
        flipped_all = jnp.flip(images, axis=1)
        mask = flip_mask.reshape(batch_size, *((1,) * (images.ndim - 1)))
        return jnp.where(mask, flipped_all, images)

    def apply_cutout(
        self, images: jax.Array, cutout_size: int = 16, num_cutouts: int = 1
    ) -> jax.Array:
        """Apply cutout augmentation (random rectangular masks)."""
        batch_size, height, width, _channels = images.shape

        result = images
        for _ in range(num_cutouts):
            # Generate random cutout positions for all images in batch
            key = self.rngs.augment()
            key_y, key_x = jax.random.split(key)
            ys = jax.random.randint(key_y, (batch_size,), 0, height - cutout_size + 1)
            xs = jax.random.randint(key_x, (batch_size,), 0, width - cutout_size + 1)

            # Create cutout mask for each image using vmap
            def make_cutout_mask(y: jax.Array, x: jax.Array) -> jax.Array:
                """Create a binary mask with a rectangular cutout."""
                row_mask = (jnp.arange(height) >= y) & (jnp.arange(height) < y + cutout_size)
                col_mask = (jnp.arange(width) >= x) & (jnp.arange(width) < x + cutout_size)
                return ~(row_mask[:, None] & col_mask[None, :])  # (H, W), True = keep

            masks = jax.vmap(make_cutout_mask)(ys, xs)  # (batch, H, W)
            masks = masks[..., None]  # (batch, H, W, 1) for broadcasting over channels
            result = jnp.where(masks, result, 0.0)

        return result

    def create_augmentation_sequence(self, augmentation_types: list[str]) -> list[str]:
        """Create a sequence of augmentations to apply."""
        available_augmentations = ["geometric", "color", "noise", "blur", "flip", "cutout"]

        # Filter valid augmentation types
        valid_augmentations = [aug for aug in augmentation_types if aug in available_augmentations]

        return valid_augmentations
