"""Metric learning loss functions.

Re-exports calibrax's metric learning losses for use in artifex training
pipelines. These losses operate on embedding spaces and are differentiable
for end-to-end training.

Stateless losses (inherit from MetricLearningLoss):
    ContrastiveLoss, TripletMarginLoss, NTXentLoss

Trainable losses (nnx.Module with learnable parameters):
    ArcFaceLoss, CosFaceLoss, ProxyNCALoss, ProxyAnchorLoss
"""

from calibrax.metrics.learning import (
    ArcFaceLoss,
    ContrastiveLoss,
    CosFaceLoss,
    MetricLearningLoss,
    NTXentLoss,
    ProxyAnchorLoss,
    ProxyNCALoss,
    TripletMarginLoss,
)


__all__ = [
    "MetricLearningLoss",
    "ContrastiveLoss",
    "TripletMarginLoss",
    "NTXentLoss",
    "ArcFaceLoss",
    "CosFaceLoss",
    "ProxyNCALoss",
    "ProxyAnchorLoss",
]
