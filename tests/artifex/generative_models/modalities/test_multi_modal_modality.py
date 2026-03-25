"""Test multi-modal modality implementation.

Following TDD principles - write tests first, then implement.
"""

import dataclasses

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.modalities import get_modality, list_modalities
from artifex.generative_models.modalities.multi_modal.adapters import (
    create_multi_modal_adapter,
    MultiModalAdapter,
)
from artifex.generative_models.modalities.multi_modal.base import (
    MultiModalModality,
    MultiModalModalityConfig,
    MultiModalRepresentation,
)
from artifex.generative_models.modalities.multi_modal.datasets import (
    create_synthetic_multi_modal_dataset,
)
from artifex.generative_models.modalities.multi_modal.evaluation import (
    compute_multi_modal_metrics,
    MultiModalEvaluationSuite,
)
from artifex.generative_models.modalities.multi_modal.representations import (
    CrossModalProcessor,
    ModalityFusionProcessor,
    MultiModalProcessor,
)


class TestMultiModalModality:
    """Test multi-modal modality functionality."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Create test multi-modal configuration."""
        return MultiModalModalityConfig(
            modalities=["image", "text"],
            fusion_strategy="concatenate",
            alignment_method="attention",
            shared_embedding_dim=256,
        )

    def test_multi_modal_config_validation(self):
        """Test multi-modal configuration validation."""
        # Valid config
        config = MultiModalModalityConfig(
            modalities=["image", "text", "audio"],
            fusion_strategy="attention",
            alignment_method="contrastive",
            shared_embedding_dim=512,
        )
        assert config.modalities == ("image", "text", "audio")
        assert config.fusion_strategy == "attention"

        # Test invalid modality
        with pytest.raises(ValueError, match="Unsupported multi-modal helper modality"):
            MultiModalModalityConfig(
                modalities=["image", "unknown_modality"],
                fusion_strategy="concatenate",
            )

        # Test invalid fusion strategy
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            MultiModalModalityConfig(
                modalities=["image", "text"],
                fusion_strategy="invalid_fusion",
            )

    @pytest.mark.parametrize(
        "unsupported_modality",
        ["protein", "molecular", "tabular", "timeseries"],
    )
    def test_multi_modal_config_rejects_non_helper_modalities(self, unsupported_modality):
        """The helper package should only accept its retained image/text/audio set."""
        with pytest.raises(ValueError, match="Unsupported multi-modal helper modality"):
            MultiModalModalityConfig(
                modalities=["image", unsupported_modality],
                fusion_strategy="concatenate",
            )

    def test_multi_modal_is_not_registry_backed(self, rngs):
        """The multi-modal helper surface should stay outside the shared registry."""
        assert "multi_modal" not in list_modalities()

        with pytest.raises(ValueError, match="Unknown modality 'multi_modal'"):
            get_modality("multi_modal", rngs=rngs)

    def test_multi_modal_config_is_frozen_and_supports_from_dict(self):
        """Multi-modal runtime config should be a frozen typed config document."""
        config = MultiModalModalityConfig.from_dict(
            {
                "name": "multi_modal_runtime",
                "modalities": ["image", "text"],
                "fusion_strategy": "concatenate",
            }
        )

        assert config.modalities == ("image", "text")

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.modalities = ("audio", "text")

    def test_multi_modal_modality_initialization(self, config, rngs):
        """Test multi-modal modality initialization."""
        modality = MultiModalModality(config=config, rngs=rngs)

        assert modality.modalities == ("image", "text")
        assert modality.fusion_strategy == "concatenate"
        assert modality.shared_embedding_dim == 256
        assert hasattr(modality, "modality_instances")
        assert len(modality.modality_instances) == 2

    def test_multi_modal_representation(self):
        """Test multi-modal representation enum."""
        assert MultiModalRepresentation.CONCATENATED.value == "concatenated"
        assert MultiModalRepresentation.ALIGNED.value == "aligned"
        assert MultiModalRepresentation.FUSED.value == "fused"
        assert MultiModalRepresentation.HIERARCHICAL.value == "hierarchical"

    def test_multi_modal_processor(self, rngs):
        """Test basic multi-modal processor."""
        processor = MultiModalProcessor(
            modalities=["image", "text"],
            output_dim=256,
            rngs=rngs,
        )

        # Test with dummy inputs
        image_input = jnp.ones((32, 32, 3))
        text_input = jnp.ones((100,))  # Token embeddings

        inputs = {"image": image_input, "text": text_input}
        output = processor(inputs)

        assert output.shape == (256,)  # Shared embedding dimension

    def test_cross_modal_processor(self, rngs):
        """Test cross-modal alignment processor."""
        processor = CrossModalProcessor(
            source_modality="image",
            target_modality="text",
            alignment_dim=256,
            use_attention=True,
            rngs=rngs,
        )

        # Test alignment
        image_features = jnp.ones((512,))
        text_features = jnp.ones((512,))

        aligned_image, aligned_text = processor(image_features, text_features)

        assert aligned_image.shape == (256,)
        assert aligned_text.shape == (256,)

    def test_modality_fusion_processor(self, rngs):
        """Test modality fusion processor."""
        processor = ModalityFusionProcessor(
            modalities=["image", "text", "audio"],
            fusion_method="attention",
            output_dim=512,
            rngs=rngs,
        )

        # Test fusion
        inputs = {
            "image": jnp.ones((256,)),
            "text": jnp.ones((256,)),
            "audio": jnp.ones((256,)),
        }

        fused = processor(inputs)
        assert fused.shape == (512,)

    def test_multi_modal_dataset(self, rngs):
        """Test multi-modal dataset functionality."""
        dataset = create_synthetic_multi_modal_dataset(
            modalities=("image", "text"),
            num_samples=100,
            alignment_strength=0.8,
            rngs=rngs,
            image_shape=(32, 32, 3),
            text_vocab_size=1000,
            text_sequence_length=50,
        )

        # Test dataset properties
        assert len(dataset) == 100

        # Test sample generation
        sample = dataset[0]
        assert "image" in sample
        assert "text" in sample
        assert "alignment_score" in sample

        assert sample["image"].shape == (32, 32, 3)
        assert sample["text"].shape == (50,)

    def test_synthetic_multi_modal_dataset_factory(self, rngs):
        """Test synthetic multi-modal dataset factory."""
        dataset = create_synthetic_multi_modal_dataset(
            modalities=["image", "text", "audio"],
            num_samples=50,
            rngs=rngs,
        )

        assert len(dataset) == 50

        sample = dataset[0]
        assert all(mod in sample for mod in ["image", "text", "audio"])

    def test_multi_modal_evaluation_suite(self, rngs):
        """Test multi-modal evaluation suite."""
        suite = MultiModalEvaluationSuite(
            modalities=["image", "text"],
            metrics=["alignment", "consistency", "quality"],
            rngs=rngs,
        )

        # Test metric computation
        generated = {
            "image": jnp.ones((10, 32, 32, 3)),
            "text": jnp.ones((10, 50)),
        }
        reference = {
            "image": jnp.ones((10, 32, 32, 3)) * 0.9,
            "text": jnp.ones((10, 50)) * 0.9,
        }

        metrics = suite.evaluate(generated, reference)

        assert "alignment_score" in metrics
        assert "consistency_score" in metrics
        assert "quality_scores" in metrics
        assert "overall_score" in metrics

    def test_multi_modal_text_evaluator_uses_full_text_token_defaults(self, rngs):
        """Text evaluator config should reuse the standard special-token defaults."""
        suite = MultiModalEvaluationSuite(modalities=["text"], rngs=rngs)

        text_params = suite.modality_evaluators["text"].text_params
        assert text_params["vocab_size"] == 10000
        assert text_params["max_length"] == 128
        assert text_params["pad_token_id"] == 0
        assert text_params["unk_token_id"] == 1
        assert text_params["bos_token_id"] == 2
        assert text_params["eos_token_id"] == 3
        assert text_params["handle_oov"] == "unk"

    def test_multi_modal_audio_path_is_supported_end_to_end(self, rngs):
        """The retained helper surface should support image/audio jointly."""
        modality = MultiModalModality(
            config=MultiModalModalityConfig(modalities=("image", "audio")),
            rngs=rngs,
        )

        sample = {
            "image": jnp.ones((32, 32, 3), dtype=jnp.float32),
            "audio": jnp.ones((16000,), dtype=jnp.float32),
        }
        processed = modality.process(sample)

        assert processed["fused_representation"].ndim == 1
        assert set(processed["individual_representations"]) == {"image", "audio"}

        suite = MultiModalEvaluationSuite(modalities=["image", "audio"], rngs=rngs)
        metrics = suite.evaluate(
            {"image": sample["image"][None, ...], "audio": sample["audio"][None, ...]},
            {
                "image": sample["image"][None, ...] * 0.9,
                "audio": sample["audio"][None, ...] * 0.9,
            },
        )

        assert metrics["overall_score"] >= 0.0

    def test_multi_modal_evaluation_rejects_unsupported_modalities(self, rngs):
        """Evaluation should fail explicitly on unsupported helper modalities."""
        with pytest.raises(ValueError, match="Unsupported multi-modal helper modality"):
            MultiModalEvaluationSuite(modalities=["image", "protein"], rngs=rngs)

    def test_compute_multi_modal_metrics(self):
        """Test multi-modal metrics computation function."""
        generated = {
            "image": jnp.ones((5, 28, 28, 1)),
            "text": jnp.ones((5, 20)),
        }
        reference = {
            "image": jnp.ones((5, 28, 28, 1)) * 0.8,
            "text": jnp.ones((5, 20)) * 0.8,
        }

        metrics = compute_multi_modal_metrics(
            generated,
            reference,
            modalities=["image", "text"],
            metric_names=["mse", "alignment"],
        )

        assert "image_mse" in metrics
        assert "text_mse" in metrics
        assert "cross_modal_alignment" in metrics

    def test_multi_modal_adapter(self, rngs):
        """Test multi-modal model adapter."""
        from artifex.generative_models.modalities.multi_modal.adapters import (
            MultiModalAdapterConfig,
        )

        adapter_config = MultiModalAdapterConfig(
            name="test_adapter",
            modalities=("image", "text"),
            model_type="vae",
            shared_latent_dim=128,
        )
        adapter = MultiModalAdapter(config=adapter_config, rngs=rngs)

        # Test adaptation
        inputs = {
            "image": jnp.ones((32, 32, 3)),
            "text": jnp.ones((50,)),
        }

        outputs = adapter.encode(inputs)
        assert "latent" in outputs
        assert outputs["latent"].shape[-1] == 128

        # Test decode
        decoded = adapter.decode(outputs["latent"])
        assert "image" in decoded
        assert "text" in decoded

    def test_create_multi_modal_adapter_factory(self, rngs):
        """Test multi-modal adapter factory function."""
        from artifex.generative_models.modalities.multi_modal.adapters import (
            MultiModalAdapterConfig,
        )

        adapter_config = MultiModalAdapterConfig(
            name="multi_modal_adapter",
            modalities=("image", "audio"),
            model_type="diffusion",
            shared_latent_dim=128,
        )

        adapter = create_multi_modal_adapter(
            config=adapter_config,
            rngs=rngs,
        )

        assert adapter.model_type == "diffusion"
        assert set(adapter.modalities) == {"image", "audio"}

    def test_multi_modal_end_to_end_workflow(self, config, rngs):
        """Test complete multi-modal workflow."""
        from artifex.generative_models.modalities.multi_modal.adapters import (
            MultiModalAdapterConfig,
        )

        # Create modality
        modality = MultiModalModality(config=config, rngs=rngs)

        # Create dataset
        dataset = create_synthetic_multi_modal_dataset(
            modalities=config.modalities,
            num_samples=10,
            rngs=rngs,
        )

        # Process samples
        sample = dataset[0]
        processed = modality.process(sample)

        assert processed is not None
        assert "fused_representation" in processed

        # Create adapter using config dataclass
        adapter_config = MultiModalAdapterConfig(
            name="end_to_end_adapter",
            modalities=tuple(config.modalities),
            model_type="vae",
            shared_latent_dim=100,
        )
        adapter = create_multi_modal_adapter(
            config=adapter_config,
            rngs=rngs,
        )

        # Encode and decode
        encoded = adapter.encode(processed)
        decoded = adapter.decode(encoded["latent"])

        # Evaluate
        suite = MultiModalEvaluationSuite(
            modalities=config.modalities,
            rngs=rngs,
        )

        metrics = suite.evaluate(decoded, sample)
        assert metrics["overall_score"] >= 0.0

    def test_cross_modal_attention(self, rngs):
        """Test cross-modal attention mechanism."""
        from artifex.generative_models.modalities.multi_modal.representations import (
            CrossModalAttention,
        )

        attention = CrossModalAttention(
            query_dim=256,
            key_dim=256,
            value_dim=256,
            num_heads=8,
            rngs=rngs,
        )

        # Test attention between image and text
        image_features = jnp.ones((10, 256))  # 10 patches
        text_features = jnp.ones((20, 256))  # 20 tokens

        attended = attention(
            query=image_features,
            key=text_features,
            value=text_features,
        )

        assert attended.shape == (10, 256)

    def test_multi_modal_consistency_loss(self):
        """Test multi-modal consistency loss computation."""
        from artifex.generative_models.modalities.multi_modal.evaluation import (
            multi_modal_consistency_loss,
        )

        representations = {
            "image": jnp.ones((32, 256)),
            "text": jnp.ones((32, 256)) * 0.9,
            "audio": jnp.ones((32, 256)) * 0.8,
        }

        loss = multi_modal_consistency_loss(representations)
        assert loss > 0.0

    def test_modality_dropout(self, rngs):
        """Test modality dropout for robustness."""
        from artifex.generative_models.modalities.multi_modal.representations import (
            ModalityDropout,
        )

        dropout = ModalityDropout(
            modalities=["image", "text", "audio"],
            dropout_rate=0.5,
            rngs=rngs,
        )

        inputs = {
            "image": jnp.ones((256,)),
            "text": jnp.ones((256,)),
            "audio": jnp.ones((256,)),
        }

        # Test dropout
        outputs_train = dropout(inputs, deterministic=False)
        assert len(outputs_train) <= len(inputs)  # Some modalities dropped

        # Test deterministic mode
        outputs_eval = dropout(inputs, deterministic=True)
        assert len(outputs_eval) == len(inputs)  # All modalities preserved

    def test_hierarchical_fusion(self, rngs):
        """Test hierarchical modality fusion."""
        from artifex.generative_models.modalities.multi_modal.representations import (
            HierarchicalFusion,
        )

        # NNX requires eager initialization - provide modality dimensions upfront
        modality_dims = {
            "image": 256,
            "text": 256,
            "audio": 128,
        }

        fusion = HierarchicalFusion(
            modality_groups=[["image", "text"], ["audio"]],
            group_fusion_dims=[512, 256],
            final_fusion_dim=768,
            modality_dims=modality_dims,
            rngs=rngs,
        )

        inputs = {
            "image": jnp.ones((256,)),
            "text": jnp.ones((256,)),
            "audio": jnp.ones((128,)),
        }

        fused = fusion(inputs)
        assert fused.shape == (768,)
