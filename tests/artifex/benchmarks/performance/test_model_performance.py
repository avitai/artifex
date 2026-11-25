"""Performance tests for model inference and training.

Every timed call is a compiled program, and every timing waits for the call's whole result. JAX
dispatches asynchronously, so a clock stopped when the call returns measures only the dispatch.
"""

import ast
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeGuard

import jax
import jax.numpy as jnp
import optax
import pytest
from calibrax.profiling import CompilationProfiler, time_calls
from flax import nnx
from substrax.testing.nnx_conformance import Case, check_rows_independent


@nnx.jit
def _encode(model: nnx.Module, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The VAE encoder as one compiled program: the latent mean and log-variance of ``x``."""
    return model.encode(x)


def _blocking(function: Callable[..., object], *args: object) -> Callable[[], object]:
    """A zero-argument call to ``function`` that returns once its whole result is ready."""
    return lambda: jax.block_until_ready(function(*args))


def _is_name_call(node: ast.AST, name: str) -> TypeGuard[ast.Call]:
    """Whether ``node`` calls the plain name ``name``."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def test_every_benchmark_waits_for_its_result() -> None:
    """Every benchmarked call goes through :func:`_blocking`, so no round stops at dispatch."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    benchmarks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and any(a.arg == "benchmark" for a in node.args.args)
    ]
    calls = [node for node in ast.walk(tree) if _is_name_call(node, "benchmark")]

    assert benchmarks, "found no benchmark; the check is reading the wrong file"
    assert len(calls) == len(benchmarks)
    assert [
        call.lineno
        for call in calls
        if not (call.args and _is_name_call(call.args[0], "_blocking"))
    ] == []


def _create_vae_config(
    name: str,
    input_shape: tuple,
    latent_dim: int,
    hidden_dims: tuple,
    use_batch_norm: bool = True,
):
    """Create a VAEConfig with proper nested encoder/decoder configs.

    Args:
        name: Name of the config
        input_shape: Shape of input data
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        use_batch_norm: Whether to use batch normalization (default True).
            Set to False when comparing batch vs individual processing
            since BatchNorm statistics differ by batch size.
    """
    from artifex.generative_models.core.configuration import (
        DecoderConfig,
        EncoderConfig,
        VAEConfig,
    )

    encoder = EncoderConfig(
        name=f"{name}_encoder",
        input_shape=input_shape,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        use_batch_norm=use_batch_norm,
    )

    decoder = DecoderConfig(
        name=f"{name}_decoder",
        latent_dim=latent_dim,
        output_shape=input_shape,
        hidden_dims=tuple(reversed(hidden_dims)),
        activation="relu",
    )

    return VAEConfig(
        name=name,
        encoder=encoder,
        decoder=decoder,
        encoder_type="dense",
        kl_weight=1.0,
    )


@pytest.mark.performance
@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for model operations."""

    def test_vae_inference_performance(self, benchmark):
        """Benchmark VAE inference latency."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Setup
        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))
        batch_size = 8
        test_batch = jnp.ones((batch_size, 32, 32, 3))

        # Compile at the timed shape before the first round.
        jax.block_until_ready(_encode(model, test_batch))

        mu, logvar = benchmark(_blocking(_encode, model, test_batch))
        assert mu.shape == (batch_size, 16)
        assert logvar.shape == (batch_size, 16)

    def test_vae_sampling_performance(self, benchmark):
        """Benchmark VAE sampling performance."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Setup
        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))
        n_samples = 8
        # n_samples sets the output shape, so it is a Python constant of the compiled program.
        sample = nnx.jit(lambda model: model.sample(n_samples=n_samples))
        jax.block_until_ready(sample(model))

        result = benchmark(_blocking(sample, model))

        # Verify result shape
        assert result.shape == (n_samples, 32, 32, 3)

    def test_diffusion_inference_performance(self, benchmark):
        """Benchmark diffusion model inference performance."""
        try:
            from artifex.generative_models.core.configuration import (
                DDPMConfig,
                NoiseScheduleConfig,
                UNetBackboneConfig,
            )
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("Diffusion model not available")

        # Setup
        rng = jax.random.key(42)

        # Create proper nested configs for DDPM
        backbone = UNetBackboneConfig(
            name="test_backbone",
            hidden_dims=(16, 32),
            activation="gelu",
            in_channels=3,
            out_channels=3,
            time_embedding_dim=16,
            num_res_blocks=1,
            attention_resolutions=(),  # No attention for speed
            channel_mult=(1, 2),
        )

        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            schedule_type="linear",
            num_timesteps=10,  # Fewer steps for testing
            beta_start=1e-4,
            beta_end=0.02,
        )

        config = DDPMConfig(
            name="test_ddpm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(16, 16, 3),  # Smaller for performance testing
            loss_type="mse",
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))
        batch_size = 4
        test_batch = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([2, 4, 6, 8])

        predict_noise = nnx.jit(lambda model, x, t: model(x, t))
        jax.block_until_ready(predict_noise(model, test_batch, t))

        result = benchmark(_blocking(predict_noise, model, test_batch, t))

        # Verify result shape (model returns dict with predicted_noise)
        assert result["predicted_noise"].shape == test_batch.shape

    def test_gan_generator_performance(self, benchmark):
        """Benchmark GAN generator performance."""
        try:
            from artifex.generative_models.core.configuration import (
                ConvDiscriminatorConfig,
                ConvGeneratorConfig,
                DCGANConfig,
            )
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("GAN model not available")

        # Setup
        rng = jax.random.key(42)

        # Create proper nested configs for DCGAN
        generator = ConvGeneratorConfig(
            name="test_generator",
            latent_dim=16,
            output_shape=(3, 32, 32),  # channels first for GAN
            hidden_dims=(64, 32),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="test_discriminator",
            input_shape=(3, 32, 32),  # channels first for GAN
            hidden_dims=(32, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = DCGANConfig(
            name="test_dcgan",
            generator=generator,
            discriminator=discriminator,
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, dropout=rng, sample=rng))
        batch_size = 8
        z = jax.random.normal(rng, (batch_size, 16))

        generate = nnx.jit(lambda generator, z: generator(z))
        jax.block_until_ready(generate(model.generator, z))

        result = benchmark(_blocking(generate, model.generator, z))

        # Verify result shape - GAN models typically return (batch, channels, height, width)
        assert result.shape == (batch_size, 3, 32, 32)

    def test_flow_forward_performance(self, benchmark):
        """Benchmark normalizing flow forward pass performance."""
        try:
            from artifex.generative_models.core.configuration import (
                CouplingNetworkConfig,
                RealNVPConfig,
            )
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("Flow model not available")

        # Setup
        rng = jax.random.key(42)
        input_dim = 32 * 32 * 3

        # Create proper nested config for RealNVP
        coupling_network = CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 128),
            activation="relu",
            network_type="mlp",
            scale_activation="tanh",
        )

        config = RealNVPConfig(
            name="test_realnvp",
            coupling_network=coupling_network,
            input_dim=input_dim,
            num_coupling_layers=4,
            mask_type="checkerboard",
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))
        batch_size = 8
        test_batch = jax.random.normal(rng, (batch_size, input_dim))

        forward = nnx.jit(lambda model, x: model.forward(x))
        jax.block_until_ready(forward(model, test_batch))

        z, log_det = benchmark(_blocking(forward, model, test_batch))

        # Verify result shapes
        assert z.shape == test_batch.shape
        assert log_det.shape == (batch_size,)


def _batch_vae(rngs: nnx.Rngs | None = None) -> nnx.Module:
    """The dense VAE the batch tests encode with, in eval mode and without batch norm.

    BatchNorm statistics differ by batch size, so it is off: batched and per-item encodings of
    the same rows must agree.
    """
    from artifex.generative_models.factory import create_model

    if rngs is None:
        rng = jax.random.key(42)
        rngs = nnx.Rngs(params=rng, sample=rng)
    config = _create_vae_config(
        name="test_vae",
        input_shape=(32, 32, 3),
        latent_dim=16,
        hidden_dims=(32, 64),
        use_batch_norm=False,
    )
    model = create_model(config, rngs=rngs)
    model.eval()
    return model


def _compiled_kernels(function: object, *arguments: object) -> int:
    """The number of HLO kernels XLA compiles ``function`` to for these arguments.

    calibrax's ``CompilationProfiler`` reports 0 when compilation fails rather than raising, so a
    count of 0 is refused here instead of being compared.
    """
    kernels = CompilationProfiler().estimate_xla_optimization(function, *arguments).total_kernels
    assert kernels > 0, "calibrax could not compile the function"
    return kernels


@pytest.mark.performance
class TestMemoryUsage:
    """Tests for memory usage and efficiency."""

    def test_vae_memory_footprint(self):
        """Test VAE model memory usage doesn't exceed thresholds."""
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        # Setup
        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(64, 64, 3),  # Larger model for memory testing
            latent_dim=32,
            hidden_dims=(64, 128, 256),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))

        # Test with different batch sizes
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            test_batch = jnp.ones((batch_size, 64, 64, 3))

            # Should not raise memory errors
            mu, logvar = model.encode(test_batch)
            z = model.reparameterize(mu, logvar)
            reconstruction = model.decode(z)

            # Verify shapes
            assert mu.shape == (batch_size, 32)
            assert reconstruction.shape == test_batch.shape

    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32])
    def test_batched_encoding_is_one_vectorized_computation(self, batch_size: int) -> None:
        """A batch is encoded as one computation whose size does not grow with the batch.

        This is the property batch speedup rests on, checked without a clock: XLA compiles the
        encoder to as many kernels for ``2 * batch_size`` rows as for ``batch_size``, so no step
        runs once per row (measured 101 kernels at every batch from 2 to 64, against 121 to 2,107
        for an encoder looping over rows; a single row compiles to a different program, so the
        comparison starts at 2); and each row's encoding depends on that row alone (substrax's
        ``check_rows_independent``, which compares the batched call with rows mapped one at a
        time within its documented ``DEPTH * eps * scale`` bound). Batched and single-row
        matmuls round differently in float32, by up to 17.5 eps here, so no exact equality is
        asserted.
        """
        model = _batch_vae()

        assert _compiled_kernels(model.encode, jnp.ones((batch_size, 32, 32, 3))) == (
            _compiled_kernels(model.encode, jnp.ones((2 * batch_size, 32, 32, 3)))
        )

        check_rows_independent(
            Case(
                build=_batch_vae,
                inputs=lambda key: (jax.random.normal(key, (batch_size, 32, 32, 3)),),
                call=lambda model, x: model.encode(x),
            )
        )

    def test_the_vectorization_check_catches_a_per_row_loop(self) -> None:
        """Positive control: an encoder looping over rows grows with the batch, so it fails."""
        graphdef, state = nnx.split(_batch_vae())

        def per_row(state: nnx.State, x: jax.Array) -> jax.Array:
            module = nnx.merge(graphdef, state)
            return jnp.concatenate([module.encode(x[i : i + 1])[0] for i in range(x.shape[0])])

        assert _compiled_kernels(per_row, state, jnp.ones((16, 32, 32, 3))) > (
            _compiled_kernels(per_row, state, jnp.ones((8, 32, 32, 3)))
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "batch_size,min_speedup",
        [
            (4, 2.0),  # Small batch: expect at least 2x speedup
            (8, 4.0),  # Medium batch: expect at least 4x speedup
            (16, 8.0),  # Larger batch: expect at least 8x speedup
            (32, 16.0),  # Large batch: expect at least 16x speedup
        ],
    )
    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="macOS CPU timing variance makes the batch-speedup ratio non-deterministic",
    )
    def test_batch_processing_efficiency(self, batch_size: int, min_speedup: float):
        """Test that batch processing is more efficient than individual items.

        Batch processing should provide significant speedup over processing
        items individually. The speedup scales roughly linearly with batch size
        as the overhead of kernel launches is amortized.

        The ratio is wall-clock time, so it measures the machine's load as well as the model.
        This test's compiled encoder, median of 15 waited calls, alone on one 24-thread CPU host
        (jax 0.11.1, three runs each) measured 3.90-4.05x at a batch of 4, 7.73-8.12x at 8,
        15.73-21.40x at 16 and 22.26-32.91x at 32. The batched call took about 0.7 ms at every
        size, so at these shapes the ratio is mostly per-call overhead the batch amortizes. It is
        ``slow``, so it runs in the performance leg, one test at a time without coverage. The unit
        leg holds the property it rests on with
        ``test_batched_encoding_is_one_vectorized_computation``.
        """
        # Time at float32 matmul precision, the precision this ratio was measured at.
        with jax.default_matmul_precision("float32"):
            self._run_batch_efficiency_test(batch_size, min_speedup)

    def _run_batch_efficiency_test(self, batch_size: int, min_speedup: float) -> None:
        """Compare the compiled encoder on the batch with the same rows encoded one at a time.

        ``time_calls`` warms each form up at its own shape, so compilation is outside both
        timings, waits for each call's whole result, and reports the median.
        """
        model = _batch_vae()
        test_batch = jnp.ones((batch_size, 32, 32, 3))
        rows = [test_batch[i : i + 1] for i in range(batch_size)]

        batched = time_calls(lambda: _encode(model, test_batch), warmup=3, iterations=15)
        per_row = time_calls(lambda: [_encode(model, row) for row in rows], warmup=3, iterations=15)

        efficiency_ratio = per_row.median_sec / batched.median_sec
        assert efficiency_ratio > min_speedup, (
            f"Batch processing not efficient enough for batch_size={batch_size}: "
            f"{efficiency_ratio:.2f}x (expected >{min_speedup}x)"
        )


@pytest.mark.performance
@pytest.mark.slow
class TestRegressionBenchmarks:
    """Regression tests to ensure performance doesn't degrade."""

    def test_vae_training_step_performance(self):
        """One compiled VAE training step, gradient and optimizer update included, within 1 s.

        The step is ``VAETrainer.train_step`` under ``nnx.jit``, the form its docstring gives.
        ``time_calls`` warms it up, so compilation is outside the timing, and waits for each
        step's whole result; the parameter update belongs to the same compiled program, so it is
        finished when the loss is.
        """
        try:
            from artifex.generative_models.factory import create_model
            from artifex.generative_models.training.trainers import (
                VAETrainer,
                VAETrainingConfig,
            )
        except ImportError:
            pytest.skip("VAE model not available")

        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        train_step = nnx.jit(VAETrainer(VAETrainingConfig()).train_step)
        batch = {"image": jnp.ones((8, 32, 32, 3))}

        timing = time_calls(lambda: train_step(model, optimizer, batch), warmup=3, iterations=10)
        loss, metrics = train_step(model, optimizer, batch)

        max_step_time = 1.0  # 1 second max for small model
        assert timing.median_sec < max_step_time, (
            f"Training step too slow: {timing.median_sec:.3f}s"
        )
        assert jnp.isfinite(loss)
        assert "reconstruction_loss" in metrics
        assert "kl_loss" in metrics

    def test_model_compilation_time(self):
        """Compiling the VAE encoder takes under 10 s.

        The clock covers tracing, lowering and XLA compilation only: JAX's ahead-of-time
        ``lower(...).compile()`` compiles without running the program, and the model is built
        before the clock starts.
        """
        try:
            from artifex.generative_models.factory import create_model
        except ImportError:
            pytest.skip("VAE model not available")

        rng = jax.random.key(42)
        config = _create_vae_config(
            name="test_vae",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(32, 64),
        )

        model = create_model(config, rngs=nnx.Rngs(params=rng, sample=rng))
        test_batch = jnp.ones((4, 32, 32, 3))
        encode = nnx.jit(lambda model, x: model.encode(x))

        start_time = time.perf_counter()
        encode.lower(model, test_batch).compile()
        compilation_time = time.perf_counter() - start_time

        # Should compile within reasonable time
        max_compilation_time = 10.0  # 10 seconds max
        assert compilation_time < max_compilation_time, (
            f"Model compilation too slow: {compilation_time:.3f}s"
        )
