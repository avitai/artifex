# Advanced Flow Examples

This guide demonstrates advanced normalizing flow architectures using Artifex and Flax NNX, including continuous normalizing flows (CNF), FFJORD, custom coupling flows, and conditional flows.

## Overview

<div class="grid cards" markdown>

- :material-chart-timeline-variant:{ .lg .middle } **Continuous Normalizing Flows**

    ---

    Neural ODEs for flexible continuous-time transformations

    [:octicons-arrow-right-24: CNF](#continuous-normalizing-flows-cnf)

- :material-lightning-bolt:{ .lg .middle } **FFJORD**

    ---

    Free-form Jacobian of Reversible Dynamics with efficient trace estimation

    [:octicons-arrow-right-24: FFJORD](#ffjord)

- :material-link-variant:{ .lg .middle } **Advanced Coupling Flows**

    ---

    Custom coupling architectures with attention and residual connections

    [:octicons-arrow-right-24: Coupling Flows](#advanced-coupling-flows)

- :material-label-variant:{ .lg .middle } **Conditional Flows**

    ---

    Conditional generation with class, text, or image inputs

    [:octicons-arrow-right-24: Conditional Flows](#conditional-flows)

</div>

## Prerequisites

```bash
# Install Artifex with all dependencies
uv pip install "artifex[cuda]"  # With GPU support
# or
uv pip install artifex  # CPU only
```

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from artifex.generative_models.core import DeviceManager
from artifex.generative_models.models.flow import FlowModel
```

## Continuous Normalizing Flows (CNF)

CNFs use neural ODEs to learn continuous-time transformations, providing more flexibility than discrete flows.

### Architecture

```mermaid
graph LR
    A[z ~ N(0,I)] --> B[ODE Solver]
    B --> C[x = z_T]
    C --> D[Data]

    style A fill:#e1f5ff
    style B fill:#f3e5f5
    style C fill:#e8f5e9
    style D fill:#fff3e0
```

### CNF Implementation

```python
from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial

class ContinuousNormalizingFlow(nnx.Module):
    """Continuous Normalizing Flow using Neural ODEs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Dynamics network f(z, t)
        # Maps (z, t) -> dz/dt
        layers = []
        for i in range(num_layers):
            in_dim = input_dim + 1 if i == 0 else hidden_dim  # +1 for time
            out_dim = input_dim if i == num_layers - 1 else hidden_dim

            layers.append(nnx.Linear(in_dim, out_dim, rngs=rngs))

            if i < num_layers - 1:
                layers.append(nnx.tanh)

        self.dynamics_net = nnx.Sequential(*layers)

    def dynamics(self, t: float, z: jax.Array) -> jax.Array:
        """
        Compute dz/dt at time t.

        Args:
            t: Current time (scalar)
            z: Current state [batch, input_dim]

        Returns:
            Time derivative dz/dt [batch, input_dim]
        """
        batch_size = z.shape[0]

        # Concatenate time to input
        t_expanded = jnp.full((batch_size, 1), t)
        z_t = jnp.concatenate([z, t_expanded], axis=-1)

        # Compute dynamics
        return self.dynamics_net(z_t)

    def forward(
        self,
        z0: jax.Array,
        t_span: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 100,
    ) -> jax.Array:
        """
        Integrate from z0 at t=t0 to t=t1.

        Args:
            z0: Initial state [batch, input_dim]
            t_span: Time interval (t0, t1)
            num_steps: Number of integration steps

        Returns:
            Final state z1 [batch, input_dim]
        """
        from jax.experimental.ode import odeint

        # Time points
        t_eval = jnp.linspace(t_span[0], t_span[1], num_steps)

        # Solve ODE: dz/dt = f(z, t)
        def ode_func(z, t):
            return self.dynamics(t, z)

        # Integrate (returns [num_steps, batch, input_dim])
        z_trajectory = odeint(ode_func, z0, t_eval)

        # Return final state
        return z_trajectory[-1]

    def inverse(
        self,
        z1: jax.Array,
        t_span: tuple[float, float] = (1.0, 0.0),
        num_steps: int = 100,
    ) -> jax.Array:
        """
        Integrate backwards from z1 at t=t1 to t=t0.

        Args:
            z1: Final state [batch, input_dim]
            t_span: Time interval (t1, t0) - reversed
            num_steps: Number of integration steps

        Returns:
            Initial state z0 [batch, input_dim]
        """
        return self.forward(z1, t_span, num_steps)

    def log_prob(
        self,
        x: jax.Array,
        base_log_prob_fn,
        num_steps: int = 100,
    ) -> jax.Array:
        """
        Compute log probability using instantaneous change of variables.

        Args:
            x: Data samples [batch, input_dim]
            base_log_prob_fn: Log probability function for base distribution
            num_steps: Integration steps

        Returns:
            Log probabilities [batch]
        """
        from jax.experimental.ode import odeint

        batch_size = x.shape[0]

        # Integrate backwards to get z0 and log determinant
        def augmented_dynamics(augmented_state, t):
            z, _ = augmented_state
            dz_dt = self.dynamics(t, z)

            # Trace of Jacobian (Hutchinson's trace estimator in FFJORD)
            # For exact computation (expensive):
            def dynamics_fn(z_single):
                return self.dynamics(t, z_single[None, :])[0]

            jacobian = jax.jacfwd(dynamics_fn)(z)
            trace = jnp.trace(jacobian)

            return dz_dt, -trace  # Negative for inverse

        # Initial augmented state
        initial_state = (x, jnp.zeros(batch_size))

        # Integrate
        t_eval = jnp.linspace(1.0, 0.0, num_steps)

        def ode_func(state, t):
            return augmented_dynamics(state, t)

        trajectory = odeint(ode_func, initial_state, t_eval)

        z0, log_det_jacobian = trajectory[0], trajectory[1]

        # Compute log probability
        base_log_prob = base_log_prob_fn(z0)
        return base_log_prob + log_det_jacobian


def train_cnf(
    model: ContinuousNormalizingFlow,
    train_data: jnp.ndarray,
    num_epochs: int = 100,
    batch_size: int = 128,
):
    """Train continuous normalizing flow."""

    rngs = nnx.Rngs(42)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    # Base distribution (standard normal)
    def base_log_prob(z):
        return -0.5 * jnp.sum(z ** 2, axis=-1) - 0.5 * z.shape[-1] * jnp.log(2 * jnp.pi)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx in range(0, len(train_data), batch_size):
            batch = train_data[batch_idx:batch_idx + batch_size]

            def loss_fn(model):
                # Compute negative log likelihood
                log_probs = model.log_prob(batch, base_log_prob, num_steps=50)
                return -jnp.mean(log_probs)

            # Compute loss and gradients, then update
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


def sample_from_cnf(
    model: ContinuousNormalizingFlow,
    num_samples: int,
    *,
    rngs: nnx.Rngs,
) -> jax.Array:
    """Sample from learned distribution."""

    # Sample from base distribution
    z0 = jax.random.normal(rngs.sample(), (num_samples, model.input_dim))

    # Transform to data space
    x = model.forward(z0, t_span=(0.0, 1.0), num_steps=100)

    return x
```

## FFJORD

FFJORD (Free-Form Jacobian of Reversible Dynamics) uses Hutchinson's trace estimator for efficient computation of log determinants.

### Hutchinson's Trace Estimator

```python
class FFJORD(nnx.Module):
    """
    FFJORD: Scalable Continuous Normalizing Flow.

    Uses Hutchinson's trace estimator for O(1) memory complexity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Time-conditioned dynamics network
        self.dynamics_net = self._build_dynamics_net(
            input_dim,
            hidden_dim,
            num_layers,
            rngs,
        )

    def _build_dynamics_net(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        rngs: nnx.Rngs,
    ) -> nnx.Module:
        """Build dynamics network with time conditioning."""

        class TimeConditionedMLP(nnx.Module):
            def __init__(self, in_dim, hidden, n_layers, *, rngs):
                super().__init__()

                layers = []
                for i in range(n_layers):
                    layer_in = in_dim + 1 if i == 0 else hidden  # +1 for time
                    layer_out = in_dim if i == n_layers - 1 else hidden

                    layers.append(nnx.Linear(layer_in, layer_out, rngs=rngs))

                    if i < n_layers - 1:
                        layers.append(nnx.softplus)

                self.net = nnx.Sequential(*layers)

            def __call__(self, z, t):
                batch_size = z.shape[0]
                t_expanded = jnp.full((batch_size, 1), t)
                z_t = jnp.concatenate([z, t_expanded], axis=-1)
                return self.net(z_t)

        return TimeConditionedMLP(input_dim, hidden_dim, num_layers, rngs=rngs)

    def dynamics(self, t: float, z: jax.Array) -> jax.Array:
        """Compute dz/dt."""
        return self.dynamics_net(z, t)

    def divergence_approx(
        self,
        t: float,
        z: jax.Array,
        epsilon: jax.Array,
    ) -> jax.Array:
        """
        Approximate divergence using Hutchinson's trace estimator.

        Tr(J) ≈ E[ε^T J ε] where ε ~ N(0, I)

        Args:
            t: Time
            z: State [batch, dim]
            epsilon: Random vector [batch, dim]

        Returns:
            Trace estimate [batch]
        """

        def dynamics_fn(z_single):
            return self.dynamics(t, z_single[None, :])[0]

        # Compute Jacobian-vector product efficiently
        _, jvp_result = jax.jvp(dynamics_fn, (z,), (epsilon,))

        # Hutchinson estimator: ε^T J ε
        trace_estimate = jnp.sum(epsilon * jvp_result, axis=-1)

        return trace_estimate

    def ode_with_log_prob(
        self,
        z: jax.Array,
        t_span: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 100,
        *,
        rngs: nnx.Rngs,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Integrate ODE and compute log probability.

        Args:
            z: Initial/final state [batch, dim]
            t_span: Time interval
            num_steps: Integration steps
            rngs: For trace estimation

        Returns:
            Tuple of (final_state, log_determinant)
        """
        from jax.experimental.ode import odeint

        batch_size = z.shape[0]

        # Sample noise for trace estimation (reuse across time)
        epsilon = jax.random.normal(rngs.sample(), (batch_size, self.input_dim))

        def augmented_dynamics(augmented_state, t):
            z_current, _ = augmented_state

            # Dynamics
            dz_dt = self.dynamics(t, z_current)

            # Trace estimate
            trace = self.divergence_approx(t, z_current, epsilon)

            # For forward: positive trace, for inverse: negative
            sign = 1.0 if t_span[1] > t_span[0] else -1.0

            return dz_dt, sign * trace

        # Initial state
        initial_augmented = (z, jnp.zeros(batch_size))

        # Time points
        t_eval = jnp.linspace(t_span[0], t_span[1], num_steps)

        # Integrate
        def ode_func(state, t):
            return augmented_dynamics(state, t)

        trajectory = odeint(ode_func, initial_augmented, t_eval)

        # Extract final values
        z_final = trajectory[0][-1]
        log_det_jacobian = trajectory[1][-1]

        return z_final, log_det_jacobian

    def forward_and_log_det(
        self,
        z0: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward transformation with log determinant."""
        return self.ode_with_log_prob(z0, t_span=(0.0, 1.0), rngs=rngs)

    def inverse_and_log_det(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation with log determinant."""
        z0, log_det = self.ode_with_log_prob(x, t_span=(1.0, 0.0), rngs=rngs)
        return z0, -log_det  # Negate for inverse

    def log_prob(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Compute log probability."""

        # Transform to base space
        z0, log_det = self.inverse_and_log_det(x, rngs=rngs)

        # Base distribution log prob (standard normal)
        base_log_prob = -0.5 * jnp.sum(z0 ** 2, axis=-1) - 0.5 * self.input_dim * jnp.log(2 * jnp.pi)

        return base_log_prob + log_det


def train_ffjord(
    model: FFJORD,
    train_data: jnp.ndarray,
    num_epochs: int = 100,
    batch_size: int = 128,
):
    """Train FFJORD model."""

    rngs = nnx.Rngs(42)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx in range(0, len(train_data), batch_size):
            batch = train_data[batch_idx:batch_idx + batch_size]

            def loss_fn(model):
                # Compute negative log likelihood
                log_probs = model.log_prob(batch, rngs=rngs)
                return -jnp.mean(log_probs)

            # Compute loss and gradients, then update
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

            epoch_loss += loss
            num_batches += 1

        if epoch % 10 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}/{num_epochs}, NLL: {avg_loss:.4f}")

    return model
```

## Advanced Coupling Flows

Custom coupling architectures with attention mechanisms and residual connections for improved expressiveness.

### Attention Coupling Layer

```python
class AttentionCouplingLayer(nnx.Module):
    """Coupling layer with self-attention in the transformation network."""

    def __init__(
        self,
        features: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.features = features
        self.hidden_dim = hidden_dim
        self.split_dim = features // 2

        # Transformation layers (applied manually for reshape flexibility)
        self.linear1 = nnx.Linear(self.split_dim, hidden_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            decode=False,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(hidden_dim, self.split_dim * 2, rngs=rngs)

    def _transform(self, x1: jax.Array) -> jax.Array:
        """Apply transformation network with attention."""
        h = nnx.relu(self.linear1(x1))
        # Reshape for attention: [batch, features] -> [batch, 1, features]
        h = h[:, None, :]
        h = self.attention(h)
        # Reshape back: [batch, 1, features] -> [batch, features]
        h = h[:, 0, :]
        return self.linear2(h)

    def __call__(
        self,
        x: jax.Array,
        reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Forward or inverse transformation.

        Args:
            x: Input [batch, features]
            reverse: If True, compute inverse

        Returns:
            Tuple of (output, log_det_jacobian)
        """
        # Split input
        x1, x2 = jnp.split(x, [self.split_dim], axis=-1)

        if not reverse:
            # Forward: x2' = x2 * exp(s(x1)) + t(x1)
            transform_params = self._transform(x1)
            log_scale, shift = jnp.split(transform_params, 2, axis=-1)

            # Bound log scale for stability
            log_scale = jnp.tanh(log_scale)

            x2_transformed = x2 * jnp.exp(log_scale) + shift

            output = jnp.concatenate([x1, x2_transformed], axis=-1)
            log_det = jnp.sum(log_scale, axis=-1)

        else:
            # Inverse: x2 = (x2' - t(x1)) / exp(s(x1))
            transform_params = self._transform(x1)
            log_scale, shift = jnp.split(transform_params, 2, axis=-1)

            log_scale = jnp.tanh(log_scale)

            x2_original = (x2 - shift) * jnp.exp(-log_scale)

            output = jnp.concatenate([x1, x2_original], axis=-1)
            log_det = -jnp.sum(log_scale, axis=-1)

        return output, log_det


class ResidualCouplingFlow(nnx.Module):
    """Coupling flow with residual connections."""

    def __init__(
        self,
        features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.features = features
        self.split_dim = features // 2

        # Residual transformation blocks
        blocks = []
        for _ in range(num_blocks):
            block = nnx.Sequential(
                nnx.Linear(self.split_dim, hidden_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(hidden_dim, self.split_dim, rngs=rngs),
            )
            blocks.append(block)
        self.blocks = nnx.List(blocks)

        # Final projection
        self.final_proj = nnx.Linear(self.split_dim, self.split_dim * 2, rngs=rngs)

    def transform_network(self, x1: jax.Array) -> jax.Array:
        """Apply residual blocks."""
        h = x1

        for block in self.blocks:
            h = h + block(h)  # Residual connection

        return self.final_proj(h)

    def __call__(
        self,
        x: jax.Array,
        reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward or inverse transformation."""

        x1, x2 = jnp.split(x, [self.split_dim], axis=-1)

        # Get transformation parameters
        params = self.transform_network(x1)
        log_scale, shift = jnp.split(params, 2, axis=-1)

        # Stabilize log scale
        log_scale = 2.0 * jnp.tanh(log_scale / 2.0)

        if not reverse:
            x2_new = x2 * jnp.exp(log_scale) + shift
            log_det = jnp.sum(log_scale, axis=-1)
        else:
            x2_new = (x2 - shift) * jnp.exp(-log_scale)
            log_det = -jnp.sum(log_scale, axis=-1)

        output = jnp.concatenate([x1, x2_new], axis=-1)
        return output, log_det


class AdvancedCouplingFlow(nnx.Module):
    """Multi-scale coupling flow with attention and residual connections."""

    def __init__(
        self,
        features: int,
        num_layers: int = 8,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.features = features
        self.num_layers = num_layers

        # Build coupling layers (stored in nnx.List)
        layers = []
        for i in range(num_layers):
            if i % 3 == 0:
                # Use attention every 3 layers
                layer = AttentionCouplingLayer(
                    features=features,
                    hidden_dim=hidden_dim,
                    rngs=rngs,
                )
            else:
                # Use residual coupling
                layer = ResidualCouplingFlow(
                    features=features,
                    hidden_dim=hidden_dim,
                    num_blocks=2,
                    rngs=rngs,
                )
            layers.append(layer)
        self.coupling_layers = nnx.List(layers)

        # Pre-compute permutation indices (static data)
        self.permutations = []
        for i in range(num_layers):
            if i % 2 == 0:
                perm = jnp.arange(features)[::-1]  # Reverse
            else:
                perm = jnp.roll(jnp.arange(features), features // 2)  # Roll
            self.permutations.append(perm)

    def __call__(
        self,
        x: jax.Array,
        reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward or inverse pass through all layers."""

        log_det_total = jnp.zeros(x.shape[0])

        indices = range(self.num_layers) if not reverse else reversed(range(self.num_layers))

        for i in indices:
            # Apply coupling layer
            x, log_det = self.coupling_layers[i](x, reverse=reverse)
            log_det_total += log_det

            # Apply permutation
            x = x[:, self.permutations[i]]

        return x, log_det_total

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log probability."""

        # Transform to base space
        z, log_det = self(x, reverse=True)

        # Base distribution log prob
        base_log_prob = -0.5 * jnp.sum(z ** 2, axis=-1) - 0.5 * self.features * jnp.log(2 * jnp.pi)

        return base_log_prob + log_det

    def sample(self, num_samples: int, *, rngs: nnx.Rngs) -> jax.Array:
        """Sample from the flow."""

        # Sample from base
        z = jax.random.normal(rngs.sample(), (num_samples, self.features))

        # Transform to data space
        x, _ = self(z, reverse=False)

        return x
```

## Conditional Flows

Flows can be conditioned on additional information for controlled generation.

### Class-Conditional Flow

```python
class ConditionalCouplingLayer(nnx.Module):
    """Coupling layer with class conditioning."""

    def __init__(
        self,
        features: int,
        num_classes: int,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.features = features
        self.split_dim = features // 2

        # Class embedding
        self.class_embedding = nnx.Embed(
            num_embeddings=num_classes,
            features=hidden_dim,
            rngs=rngs,
        )

        # Conditioned transformation network
        self.transform_net = nnx.Sequential(
            nnx.Linear(self.split_dim + hidden_dim, hidden_dim * 2, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, self.split_dim * 2, rngs=rngs),
        )

    def __call__(
        self,
        x: jax.Array,
        class_labels: jax.Array,
        reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Conditional transformation.

        Args:
            x: Input [batch, features]
            class_labels: Class indices [batch]
            reverse: Forward or inverse

        Returns:
            Tuple of (output, log_det)
        """
        # Split
        x1, x2 = jnp.split(x, [self.split_dim], axis=-1)

        # Embed class
        class_embed = self.class_embedding(class_labels)

        # Concatenate x1 and class embedding
        x1_conditioned = jnp.concatenate([x1, class_embed], axis=-1)

        # Get transformation parameters
        params = self.transform_net(x1_conditioned)
        log_scale, shift = jnp.split(params, 2, axis=-1)

        log_scale = jnp.tanh(log_scale)

        if not reverse:
            x2_new = x2 * jnp.exp(log_scale) + shift
            log_det = jnp.sum(log_scale, axis=-1)
        else:
            x2_new = (x2 - shift) * jnp.exp(-log_scale)
            log_det = -jnp.sum(log_scale, axis=-1)

        output = jnp.concatenate([x1, x2_new], axis=-1)
        return output, log_det


class ConditionalNormalizingFlow(nnx.Module):
    """Full conditional normalizing flow model."""

    def __init__(
        self,
        features: int,
        num_classes: int,
        num_layers: int = 8,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.features = features
        self.num_classes = num_classes

        # Stack of conditional coupling layers
        layers = []
        for i in range(num_layers):
            layer = ConditionalCouplingLayer(
                features=features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                rngs=rngs,
            )
            layers.append(layer)
        self.layers = nnx.List(layers)

    def __call__(
        self,
        x: jax.Array,
        class_labels: jax.Array,
        reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward or inverse pass."""

        log_det_total = jnp.zeros(x.shape[0])

        layers = self.layers if not reverse else reversed(self.layers)

        for layer in layers:
            x, log_det = layer(x, class_labels, reverse=reverse)
            log_det_total += log_det

        return x, log_det_total

    def log_prob(self, x: jax.Array, class_labels: jax.Array) -> jax.Array:
        """Compute conditional log probability."""

        z, log_det = self(x, class_labels, reverse=True)

        base_log_prob = -0.5 * jnp.sum(z ** 2, axis=-1) - 0.5 * self.features * jnp.log(2 * jnp.pi)

        return base_log_prob + log_det

    def sample(
        self,
        num_samples: int,
        class_labels: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Sample conditioned on classes."""

        z = jax.random.normal(rngs.sample(), (num_samples, self.features))

        x, _ = self(z, class_labels, reverse=False)

        return x


def train_conditional_flow(
    model: ConditionalNormalizingFlow,
    train_data: jnp.ndarray,
    train_labels: jnp.ndarray,
    num_epochs: int = 100,
):
    """Train conditional flow."""

    rngs = nnx.Rngs(42)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    for epoch in range(num_epochs):
        for batch_idx in range(0, len(train_data), 128):
            batch = train_data[batch_idx:batch_idx + 128]
            labels = train_labels[batch_idx:batch_idx + 128]

            def loss_fn(model):
                # Negative log likelihood
                log_probs = model.log_prob(batch, labels)
                return -jnp.mean(log_probs)

            # Compute loss and gradients, then update
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

    return model


# Generate samples for specific class
def generate_class_samples(
    model: ConditionalNormalizingFlow,
    class_id: int,
    num_samples: int = 16,
    *,
    rngs: nnx.Rngs,
) -> jax.Array:
    """Generate samples for a specific class."""

    class_labels = jnp.full(num_samples, class_id)
    return model.sample(num_samples, class_labels, rngs=rngs)
```

## Best Practices

!!! success "DO"
    - Use FFJORD for high-dimensional data (more efficient)
    - Add residual connections in coupling networks
    - Use attention for long-range dependencies
    - Monitor both NLL and sample quality
    - Use adaptive ODE solvers for CNF
    - Implement gradient clipping for training stability

!!! danger "DON'T"
    - Don't use too few ODE steps (<20 for CNF)
    - Don't forget to alternate coupling directions
    - Don't use unbounded activation in scale networks
    - Don't skip permutation layers between couplings
    - Don't train with learning rates >1e-3
    - Don't use batch norm in flow transformations

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Training instability | Unbounded scales | Use tanh or bounded activations for log_scale |
| Slow ODE integration | Too many steps | Use adaptive solvers, reduce steps |
| Poor sample quality | Insufficient coupling | Add more layers, use attention |
| NaN in training | Exploding gradients | Add gradient clipping, reduce learning rate |
| High memory usage | Full Jacobian computation | Use FFJORD with Hutchinson estimator |

## Summary

We covered four advanced normalizing flow techniques:

1. **Continuous Normalizing Flows**: Flexible continuous-time transformations with Neural ODEs
2. **FFJORD**: Efficient CNF with Hutchinson's trace estimator
3. **Advanced Coupling**: Attention and residual connections for expressiveness
4. **Conditional Flows**: Class or context-conditional generation

**Key Takeaways**:

- CNF provides more flexibility than discrete flows
- FFJORD makes CNF scalable to high dimensions
- Attention and residual connections improve coupling flows
- Conditional flows enable controlled generation

## Next Steps

<div class="grid cards" markdown>

- :material-book-open-variant:{ .lg .middle } **Flow Concepts**

    ---

    Deep dive into normalizing flow theory

    [:octicons-arrow-right-24: Flow Explained](../../user-guide/concepts/flow-explained.md)

- :material-laptop:{ .lg .middle } **Training Guide**

    ---

    Scale flow training efficiently

    [:octicons-arrow-right-24: Training Guide](../../user-guide/training/training-guide.md)

- :material-chart-box:{ .lg .middle } **Benchmarks**

    ---

    Evaluate flow models

    [:octicons-arrow-right-24: Evaluation](../../user-guide/training/overview.md#evaluation)

- :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Complete flow API documentation

    [:octicons-arrow-right-24: Flow API](../../api/models/flow.md)

</div>
