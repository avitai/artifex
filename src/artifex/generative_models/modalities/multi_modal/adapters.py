"""Multi-modal adapters for different model types.

This module provides adapters that enable existing models to work with
multiple modalities.
"""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.modalities.multi_modal.representations import (
    ModalityFusionProcessor,
    MultiModalProcessor,
)


# Valid model types for multi-modal adapters
VALID_MODEL_TYPES = ("vae", "diffusion", "gan")

# Valid fusion strategies
VALID_FUSION_STRATEGIES = ("concatenate", "attention", "product")


@dataclasses.dataclass(frozen=True)
class MultiModalAdapterConfig:
    """Configuration for multi-modal adapters.

    Attributes:
        name: Name of the adapter configuration
        modalities: List of modality names (e.g., ["image", "text", "audio"])
        model_type: Type of model ("vae", "diffusion", "gan")
        shared_latent_dim: Dimension of shared latent space
        fusion_strategy: How to fuse modalities ("concatenate", "attention", "product")
        latent_dim: Dimension of the latent space (for generators)
        beta: Beta parameter for VAE (KL divergence weight)
    """

    name: str
    modalities: tuple[str, ...]
    model_type: str
    shared_latent_dim: int = 128
    fusion_strategy: str = "concatenate"
    latent_dim: int = 128
    beta: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.modalities:
            raise ValueError("modalities cannot be empty")
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {VALID_MODEL_TYPES}, got '{self.model_type}'"
            )
        if self.fusion_strategy not in VALID_FUSION_STRATEGIES:
            raise ValueError(
                f"fusion_strategy must be one of {VALID_FUSION_STRATEGIES}, "
                f"got '{self.fusion_strategy}'"
            )
        if self.shared_latent_dim <= 0:
            raise ValueError(f"shared_latent_dim must be positive, got {self.shared_latent_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")


class MultiModalAdapter(nnx.Module):
    """Adapter for multi-modal models."""

    def __init__(
        self,
        config: MultiModalAdapterConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-modal adapter.

        Args:
            config: Adapter configuration (MultiModalAdapterConfig)
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.modalities = list(config.modalities)
        self.model_type = config.model_type
        self.shared_latent_dim = config.shared_latent_dim
        self.fusion_strategy = config.fusion_strategy
        self.latent_dim = config.latent_dim

        # Initialize components based on model type
        self._init_model_components(rngs)

    def _init_model_components(self, rngs: nnx.Rngs):
        """Initialize model-specific components.

        Args:
            rngs: Random number generators
        """
        if self.model_type == "vae":
            self._init_vae_components(rngs)
        elif self.model_type == "diffusion":
            self._init_diffusion_components(rngs)
        elif self.model_type == "gan":
            self._init_gan_components(rngs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _init_vae_components(self, rngs: nnx.Rngs):
        """Initialize VAE-specific components."""
        # Modality-specific encoders using nnx.Dict for proper parameter tracking
        encoders: dict[str, nnx.Module] = {}
        for modality in self.modalities:
            encoders[modality] = self._create_encoder(modality, rngs)
        self.encoders = nnx.Dict(encoders)

        # Shared fusion processor
        self.fusion_processor = ModalityFusionProcessor(
            modalities=self.modalities,
            fusion_method=self.fusion_strategy,
            output_dim=self.shared_latent_dim * 2,  # Mean and log_var
            hidden_dim=self.shared_latent_dim,  # Match encoder output
            rngs=rngs,
        )

        # Modality-specific decoders using nnx.Dict
        decoders: dict[str, nnx.Module] = {}
        for modality in self.modalities:
            decoders[modality] = self._create_decoder(modality, rngs)
        self.decoders = nnx.Dict(decoders)

    def _init_diffusion_components(self, rngs: nnx.Rngs):
        """Initialize diffusion-specific components."""
        # Multi-modal processor for conditioning
        self.multi_modal_processor = MultiModalProcessor(
            modalities=self.modalities,
            output_dim=self.shared_latent_dim,
            rngs=rngs,
        )

        # Time embedding
        self.time_embedding = nnx.Sequential(
            nnx.Linear(1, self.shared_latent_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(self.shared_latent_dim, self.shared_latent_dim, rngs=rngs),
        )

        # Denoising network for each modality using nnx.Dict
        denoisers: dict[str, nnx.Module] = {}
        for modality in self.modalities:
            denoisers[modality] = self._create_denoiser(modality, rngs)
        self.denoisers = nnx.Dict(denoisers)

    def _init_gan_components(self, rngs: nnx.Rngs):
        """Initialize GAN-specific components."""
        # Generator components
        self.noise_processor = nnx.Linear(
            self.latent_dim,
            self.shared_latent_dim,
            rngs=rngs,
        )

        # Modality-specific generators using nnx.Dict
        generators: dict[str, nnx.Module] = {}
        for modality in self.modalities:
            generators[modality] = self._create_generator(modality, rngs)
        self.generators = nnx.Dict(generators)

        # Multi-modal discriminator
        self.discriminator_processor = MultiModalProcessor(
            modalities=self.modalities,
            output_dim=1,  # Real/fake score
            rngs=rngs,
        )

    def _create_encoder(self, modality: str, rngs: nnx.Rngs) -> nnx.Module:
        """Create modality-specific encoder.

        Args:
            modality: Modality name
            rngs: Random number generators

        Returns:
            Encoder module
        """
        # Simplified encoder - in practice would be modality-specific
        if modality == "image":
            input_dim = 32 * 32 * 3
        elif modality == "text":
            input_dim = 100  # Embedding dim
        elif modality == "audio":
            input_dim = 16000  # 1 second at 16kHz
        else:
            input_dim = 256

        return nnx.Sequential(
            nnx.Linear(input_dim, 512, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, self.shared_latent_dim, rngs=rngs),
        )

    def _create_decoder(self, modality: str, rngs: nnx.Rngs) -> nnx.Module:
        """Create modality-specific decoder.

        Args:
            modality: Modality name
            rngs: Random number generators

        Returns:
            Decoder module
        """
        # Simplified decoder - dimensions should match expected modality dims
        if modality == "image":
            output_dim = 32 * 32 * 3  # 3072
        elif modality == "text":
            output_dim = 50  # Match expected text embedding dimension
        elif modality == "audio":
            output_dim = 256  # Match expected audio feature dimension
        else:
            output_dim = 256

        return nnx.Sequential(
            nnx.Linear(self.shared_latent_dim, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 512, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, output_dim, rngs=rngs),
        )

    def _create_denoiser(self, modality: str, rngs: nnx.Rngs) -> nnx.Module:
        """Create modality-specific denoiser for diffusion.

        Args:
            modality: Modality name
            rngs: Random number generators

        Returns:
            Denoiser module
        """
        # Simplified denoiser
        return nnx.Sequential(
            nnx.Linear(256 + self.shared_latent_dim, 512, rngs=rngs),  # Input + conditioning
            nnx.relu,
            nnx.Linear(512, 512, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, 256, rngs=rngs),  # Output same as input
        )

    def _create_generator(self, modality: str, rngs: nnx.Rngs) -> nnx.Module:
        """Create modality-specific generator for GAN.

        Args:
            modality: Modality name
            rngs: Random number generators

        Returns:
            Generator module
        """
        # Simplified generator
        if modality == "image":
            output_dim = 32 * 32 * 3
        elif modality == "text":
            output_dim = 100
        elif modality == "audio":
            output_dim = 16000
        else:
            output_dim = 256

        return nnx.Sequential(
            nnx.Linear(self.shared_latent_dim, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 512, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, output_dim, rngs=rngs),
            nnx.tanh,  # Output in [-1, 1]
        )

    def encode(
        self,
        inputs: dict[str, jax.Array],
        *,
        deterministic: bool = False,
    ) -> dict[str, jax.Array]:
        """Encode multi-modal inputs.

        Args:
            inputs: Dictionary of modality inputs
            deterministic: Whether to apply dropout

        Returns:
            Encoded representation
        """
        if self.model_type == "vae":
            # Encode each modality - iterate over nnx.Dict items
            # (avoids membership check issues with nnx.Dict.__contains__)
            encoded = {}
            for modality, encoder in self.encoders.items():
                # Check if input provided (plain Python dict check is JIT-safe)
                if modality in inputs:
                    # Flatten input if needed
                    x = inputs[modality]
                    if x.ndim > 1:
                        x = x.reshape(-1)

                    # For text, pad to expected size
                    if modality == "text" and x.shape[0] < 100:
                        # Pad with zeros to reach size 100
                        padding = jnp.zeros(100 - x.shape[0])
                        x = jnp.concatenate([x, padding])

                    encoded[modality] = encoder(x)

            # Fuse encoded modalities
            fused = self.fusion_processor(encoded, deterministic=deterministic)

            # Split into mean and log_var
            latent_dim = self.shared_latent_dim
            mean = fused[:latent_dim]
            log_var = fused[latent_dim:]

            return {
                "latent": mean,  # For simplicity, just return mean
                "mean": mean,
                "log_var": log_var,
            }
        else:
            # For other model types, just process
            processed = self.multi_modal_processor(inputs, deterministic=deterministic)
            return {"latent": processed}

    def decode(
        self,
        latent: jax.Array,
        target_modalities: list[str] | None = None,
        *,
        deterministic: bool = False,
    ) -> dict[str, jax.Array]:
        """Decode latent to multiple modalities.

        Args:
            latent: Latent representation
            target_modalities: Specific modalities to decode
            deterministic: Whether to apply dropout

        Returns:
            Decoded outputs for each modality
        """
        if target_modalities is None:
            target_modalities = self.modalities

        outputs = {}

        if self.model_type == "vae":
            # Iterate over decoders - only decode modalities we have decoders for
            for modality, decoder in self.decoders.items():
                # Only decode if this modality is in target list
                # (plain Python list check is JIT-safe)
                if modality in target_modalities:
                    decoded = decoder(latent)
                    # Reshape based on modality - ensure batch dimension
                    if modality == "image":
                        # Ensure batch dimension exists
                        if decoded.ndim == 1:
                            # Single sample: reshape to (1, 32, 32, 3)
                            decoded = decoded.reshape(1, 32, 32, 3)
                        else:
                            # Multiple samples: reshape last 3 dims to image shape
                            batch_size = decoded.shape[0]
                            decoded = decoded.reshape(batch_size, 32, 32, 3)
                    elif modality == "text":
                        # Ensure batch dimension for text
                        if decoded.ndim == 1:
                            decoded = decoded[jnp.newaxis, :]
                    elif modality == "audio":
                        # Ensure batch dimension for audio
                        if decoded.ndim == 1:
                            decoded = decoded[jnp.newaxis, :]
                    outputs[modality] = decoded
        elif self.model_type == "gan":
            # Process noise through shared layer
            processed = self.noise_processor(latent)

            # Generate each modality - iterate over nnx.Dict
            for modality, generator in self.generators.items():
                # Only generate if in target list (plain Python list check is JIT-safe)
                if modality in target_modalities:
                    generated = generator(processed)
                    # Reshape based on modality
                    if modality == "image":
                        generated = generated.reshape(32, 32, 3)
                    outputs[modality] = generated

        return outputs

    def forward(
        self,
        inputs: dict[str, jax.Array] | jax.Array,
        *,
        deterministic: bool = False,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """Forward pass through the adapter.

        Args:
            inputs: Multi-modal inputs or latent
            deterministic: Whether to apply dropout
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        if isinstance(inputs, dict):
            # Encode-decode pipeline
            encoded = self.encode(inputs, deterministic=deterministic)
            outputs = self.decode(encoded["latent"], deterministic=deterministic)
            outputs["latent"] = encoded["latent"]
            return outputs
        else:
            # Just decode from latent
            return self.decode(inputs, deterministic=deterministic)


def create_multi_modal_adapter(
    config: MultiModalAdapterConfig,
    *,
    rngs: nnx.Rngs,
) -> MultiModalAdapter:
    """Create a multi-modal adapter from configuration.

    Args:
        config: Adapter configuration (MultiModalAdapterConfig)
        rngs: Random number generators

    Returns:
        Multi-modal adapter instance
    """
    return MultiModalAdapter(config=config, rngs=rngs)


@dataclasses.dataclass(frozen=True)
class MultiModalVAEAdapterConfig(MultiModalAdapterConfig):
    """Configuration for multi-modal VAE adapters.

    Inherits from MultiModalAdapterConfig with model_type fixed to "vae".
    """

    model_type: str = "vae"

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Override to force model_type to "vae"
        if self.model_type != "vae":
            object.__setattr__(self, "model_type", "vae")
        super().__post_init__()


class MultiModalVAEAdapter(MultiModalAdapter):
    """Specialized adapter for multi-modal VAEs."""

    def __init__(
        self,
        config: MultiModalVAEAdapterConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-modal VAE adapter.

        Args:
            config: VAE adapter configuration (MultiModalVAEAdapterConfig)
            rngs: Random number generators
        """
        super().__init__(config=config, rngs=rngs)
        self.beta = config.beta

    def compute_loss(
        self,
        inputs: dict[str, jax.Array],
        outputs: dict[str, jax.Array],
        encoded: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute VAE loss for multi-modal data.

        Args:
            inputs: Original inputs
            outputs: Reconstructed outputs
            encoded: Encoded representation with mean and log_var

        Returns:
            Dictionary of loss values
        """
        losses: dict[str, jax.Array] = {}

        # Reconstruction loss for each modality
        total_recon_loss = 0.0
        for modality in self.modalities:
            if modality in inputs and modality in outputs:
                # MSE loss
                recon_loss = jnp.mean((inputs[modality] - outputs[modality]) ** 2)
                losses[f"{modality}_recon_loss"] = recon_loss
                total_recon_loss += recon_loss

        losses["reconstruction_loss"] = total_recon_loss

        # KL divergence loss
        mean = encoded["mean"]
        log_var = encoded["log_var"]
        kl_loss = -0.5 * jnp.sum(1 + log_var - mean**2 - jnp.exp(log_var), axis=-1).mean()
        losses["kl_loss"] = kl_loss

        # Total loss
        losses["total_loss"] = total_recon_loss + self.beta * kl_loss

        return losses
