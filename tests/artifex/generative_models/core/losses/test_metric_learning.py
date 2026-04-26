"""Tests for metric learning loss re-exports from calibrax."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.losses.metric_learning import (
    ArcFaceLoss,
    ContrastiveLoss,
    CosFaceLoss,
    MetricLearningLoss,
    NTXentLoss,
    ProxyAnchorLoss,
    ProxyNCALoss,
    TripletMarginLoss,
)


@pytest.fixture()
def embeddings_and_labels():
    """Create test embeddings and labels with 4 classes, 2 samples each."""
    embeddings = jax.random.normal(jax.random.key(0), (8, 16))
    labels = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
    return embeddings, labels


class TestContrastiveLoss:
    """Tests for ContrastiveLoss re-export."""

    def test_returns_scalar(self, embeddings_and_labels):
        """ContrastiveLoss returns a non-negative scalar."""
        embeddings, labels = embeddings_and_labels
        loss_fn = ContrastiveLoss(margin=1.0)
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()
        assert float(loss) >= 0.0

    def test_inherits_base(self):
        """ContrastiveLoss is a MetricLearningLoss."""
        assert issubclass(ContrastiveLoss, MetricLearningLoss)


class TestTripletMarginLoss:
    """Tests for TripletMarginLoss re-export."""

    def test_returns_scalar(self, embeddings_and_labels):
        """TripletMarginLoss returns a non-negative scalar."""
        embeddings, labels = embeddings_and_labels
        loss_fn = TripletMarginLoss(margin=0.2)
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()
        assert float(loss) >= 0.0


class TestNTXentLoss:
    """Tests for NTXentLoss re-export."""

    def test_returns_scalar(self, embeddings_and_labels):
        """NTXentLoss returns a scalar."""
        embeddings, labels = embeddings_and_labels
        loss_fn = NTXentLoss(temperature=0.5)
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()


class TestProxyLosses:
    """Tests for proxy-based metric learning losses."""

    def test_proxy_nca(self, embeddings_and_labels):
        """ProxyNCALoss returns a scalar and has trainable proxies."""
        embeddings, labels = embeddings_and_labels
        loss_fn = ProxyNCALoss(num_classes=4, embedding_dim=16, rngs=nnx.Rngs(0))
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()
        assert isinstance(loss_fn, nnx.Module)

    def test_proxy_anchor(self, embeddings_and_labels):
        """ProxyAnchorLoss returns a scalar and has trainable proxies."""
        embeddings, labels = embeddings_and_labels
        loss_fn = ProxyAnchorLoss(num_classes=4, embedding_dim=16, rngs=nnx.Rngs(0))
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()


class TestAngularMarginLosses:
    """Tests for angular margin losses."""

    def test_arcface(self, embeddings_and_labels):
        """ArcFaceLoss returns a scalar and has trainable weights."""
        embeddings, labels = embeddings_and_labels
        loss_fn = ArcFaceLoss(num_classes=4, embedding_dim=16, rngs=nnx.Rngs(0))
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()
        assert isinstance(loss_fn, nnx.Module)

    def test_cosface(self, embeddings_and_labels):
        """CosFaceLoss returns a scalar and has trainable weights."""
        embeddings, labels = embeddings_and_labels
        loss_fn = CosFaceLoss(num_classes=4, embedding_dim=16, rngs=nnx.Rngs(0))
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()


class TestImportFromTopLevel:
    """Test that metric learning losses can be imported from top-level losses module."""

    def test_import_from_losses(self):
        """Metric learning classes are importable from the losses package."""
        from artifex.generative_models.core.losses import (
            ArcFaceLoss,
            ContrastiveLoss,
            CosFaceLoss,
            MetricLearningLoss,
            NTXentLoss,
            ProxyAnchorLoss,
            ProxyNCALoss,
            TripletMarginLoss,
        )

        assert ContrastiveLoss is not None
        assert TripletMarginLoss is not None
        assert NTXentLoss is not None
        assert ArcFaceLoss is not None
        assert CosFaceLoss is not None
        assert ProxyNCALoss is not None
        assert ProxyAnchorLoss is not None
        assert MetricLearningLoss is not None


class TestMetricLearningTransformCompatibility:
    """Metric learning re-exports should preserve CalibraX transform behavior."""

    @pytest.mark.parametrize(
        ("loss_cls", "kwargs"),
        [
            (ContrastiveLoss, {"margin": 1.0}),
            (TripletMarginLoss, {"margin": 0.2}),
            (NTXentLoss, {"temperature": 0.5}),
        ],
    )
    def test_stateless_losses_are_jittable_and_differentiable(
        self, loss_cls, kwargs, embeddings_and_labels
    ):
        embeddings, labels = embeddings_and_labels
        loss_fn = loss_cls(**kwargs)

        def scalar_loss(values):
            return loss_fn(values, labels)

        compiled_value = jax.jit(scalar_loss)(embeddings)
        gradients = jax.grad(scalar_loss)(embeddings)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert gradients.shape == embeddings.shape
        assert jnp.all(jnp.isfinite(gradients))

    @pytest.mark.parametrize(
        "loss_cls",
        [ArcFaceLoss, CosFaceLoss, ProxyNCALoss, ProxyAnchorLoss],
    )
    def test_trainable_losses_are_nnx_jittable_and_differentiable(
        self, loss_cls, embeddings_and_labels
    ):
        embeddings, labels = embeddings_and_labels
        loss_module = loss_cls(num_classes=4, embedding_dim=16, rngs=nnx.Rngs(0))

        def scalar_loss(module, values):
            return module(values, labels)

        compiled_value = nnx.jit(scalar_loss)(loss_module, embeddings)
        module_grad = nnx.grad(scalar_loss, argnums=0)(loss_module, embeddings)
        embedding_grad = nnx.grad(scalar_loss, argnums=1)(loss_module, embeddings)

        module_grad_leaves = jax.tree_util.tree_leaves(module_grad)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert module_grad_leaves
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in module_grad_leaves)
        assert embedding_grad.shape == embeddings.shape
        assert jnp.all(jnp.isfinite(embedding_grad))
