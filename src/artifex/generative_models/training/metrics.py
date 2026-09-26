"""Record-weighted metric means accumulated on the device.

A step, or an evaluation batch, reports the mean of each metric over its records.
:class:`MeanAccumulator` adds each mean weighted by the batch's record count, so the mean it
computes over any run of batches -- a short last batch included -- is the mean over every
record. Its totals are ``nnx`` Variables, so it updates inside a compiled step and reading it
is the only host transfer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MeanAccumulator(nnx.Module):
    """Running record-weighted means of named metrics.

    Attributes:
        totals: Each metric's sum of batch means weighted by record count (``float32``).
        records: The records counted since the last reset (``int32``).
    """

    def __init__(self, names: Sequence[str]) -> None:
        """Accumulate the metrics ``names``.

        Args:
            names: The metric names, fixed for the accumulator's life.
        """
        self.totals = nnx.Dict({name: nnx.Variable(jnp.zeros((), jnp.float32)) for name in names})
        self.records = nnx.Variable(jnp.zeros((), jnp.int32))

    def update(self, metrics: Mapping[str, jax.Array], records: jax.Array) -> None:
        """Add a batch's mean metrics, weighted by the records they average over.

        Args:
            metrics: Each metric's mean over the batch; every name must be one the accumulator
                was built for.
            records: The batch's record count.

        Raises:
            KeyError: If a metric is not one the accumulator was built for.
        """
        unknown = sorted(set(metrics) - set(self.totals))
        if unknown:
            raise KeyError(
                f"metrics {unknown} are not accumulated; built for {sorted(self.totals)}"
            )
        weight = records.astype(jnp.float32)
        for name, value in metrics.items():
            total = self.totals[name]
            total[...] = total[...] + weight * value.astype(jnp.float32)
        self.records[...] = self.records[...] + records

    def compute(self) -> dict[str, jax.Array]:
        """Each metric's mean over the records counted; ``nan`` before any.

        Returns:
            The means, by name, as device scalars.
        """
        count = self.records[...].astype(jnp.float32)
        return {name: total[...] / count for name, total in self.totals.items()}

    def reset(self) -> None:
        """Start a new window: zero every total and the record count."""
        for total in self.totals.values():
            total[...] = jnp.zeros_like(total[...])
        self.records[...] = jnp.zeros_like(self.records[...])
