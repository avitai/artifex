"""Conformance checks a case is known to fail, each with the defect that makes it fail.

Keys are ``(qualified class name, check)``. ``test_conformance.py`` marks each as a strict
expected failure, so the commit that fixes a defect must delete its entries. Every entry was
diagnosed, and a check failing for a reason other than its own property names the defect it
inherits. Each defect names its row in the audit register.
"""

from __future__ import annotations


_CORE = "artifex.generative_models.core"
_LAYERS = f"{_CORE}.layers"
_BACKBONES = "artifex.generative_models.models.backbones"

_EVAL_OVERRIDE = "a deterministic=False default forwarded to {target} overrides .eval() ({where})"

TRANSFORMERS = "R005 " + _EVAL_OVERRIDE.format(
    target="nnx.MultiHeadAttention", where="transformers.py:228-241, 292-297, 475-500"
)
POSITIONAL = "R010 " + _EVAL_OVERRIDE.format(target="nnx.Dropout", where="positional.py:{line}")
RESNET_BATCH_NORM = (
    "R011 BatchNorm is built with use_running_average=True and called with deterministic=True by"
    " default, so .train() never updates its statistics (layers/_utils.py:60, 94; resnet.py:{line})"
)
EGNN = "R013 " + _EVAL_OVERRIDE.format(target="nnx.Dropout", where="egnn.py:85, 104")
CLIFFORD_BATCH_NORM = (
    "R017 CliffordBatchNorm takes its mode from a call-time deterministic=False and has no"
    " set_view, so .eval() still normalises with batch statistics and updates them"
    " (clifford/norm.py:74, 88)"
)
CLIFFORD_GROUP_NORM = (
    "R018 CliffordGroupNorm reduces over the batch axis, so a sample's output depends on the"
    " others (clifford/norm.py:198-214)"
)
HYPERPARAMETERS_IN_VARIABLES = (
    "R056 {names} are stored in nnx.Variable rather than as static attributes (base.py:{lines})"
)
DIT = "R154 " + _EVAL_OVERRIDE.format(
    target="nnx.Dropout and nnx.MultiHeadAttention", where="backbones/dit.py:168, 248, 461"
)
SHARED_RNGS = (
    "R027 GenerativeModule stores self.rngs (core/base.py:122), so child blocks built from their"
    " parent's Rngs share one RngCount (.block1.rngs, .block2.rngs) and tree mode refuses the"
    " module"
)
JAX_ATTENTION_SOFTMAX = (
    "R517 nnx.MultiHeadAttention with dropout_rate=0 delegates to jax.nn.dot_product_attention,"
    " whose softmax runs in float32 even for float64 inputs (jax/_src/nn/functions.py:955 at"
    " jax 0.11.1; jax#34107, #34112), so a float64 gradient check cannot resolve it"
)
EMBED_DTYPE = (
    " With R154 fixed this check still fails: R518 nnx.Embed fixes its compute dtype when built"
    " (flax nnx Embed: self.dtype = dtype or self.embedding.dtype), so its float64 copy looks up"
    " in float32 (backbones/dit.py:155)"
)

_EVAL_CASCADE = ("gradient_finite", "gradient_matches_finite_differences", "rows_independent")
"""Checks that run in ``.eval()``: a module still drawing dropout keys there fails each."""


def _eval_override(name: str, defect: str) -> dict[tuple[str, str], str]:
    inherited = ".eval() still draws dropout keys: " + defect
    return {
        (name, "eval_deterministic"): defect,
        **{(name, check): inherited for check in _EVAL_CASCADE},
    }


def _transformers() -> dict[tuple[str, str], str]:
    failures: dict[tuple[str, str], str] = {}
    for block in (
        "TransformerEncoderBlock",
        "TransformerEncoder",
        "TransformerDecoderBlock",
        "TransformerDecoder",
    ):
        failures |= _eval_override(f"{_LAYERS}.transformers.{block}", TRANSFORMERS)
    return failures


def _positional() -> dict[tuple[str, str], str]:
    failures: dict[tuple[str, str], str] = {}
    for encoding, line in (
        ("SinusoidalPositionalEncoding", 159),
        ("LearnedPositionalEncoding", 238),
        ("RotaryPositionalEncoding", 353),
    ):
        failures |= _eval_override(f"{_LAYERS}.positional.{encoding}", POSITIONAL.format(line=line))
    return failures


def _egnn() -> dict[tuple[str, str], str]:
    return _eval_override(f"{_LAYERS}.egnn.EGNNLayer", EGNN) | _eval_override(
        f"{_LAYERS}.egnn.EGNNBlock", EGNN
    )


def _clifford() -> dict[tuple[str, str], str]:
    batch_norm = f"{_LAYERS}.clifford.norm.CliffordBatchNorm"
    return {
        (batch_norm, check): CLIFFORD_BATCH_NORM for check in ("batch_stats", *_EVAL_CASCADE)
    } | {(f"{_LAYERS}.clifford.norm.CliffordGroupNorm", "rows_independent"): CLIFFORD_GROUP_NORM}


def _resnet() -> dict[tuple[str, str], str]:
    return {
        (f"{_LAYERS}.resnet.ResNetBlock", "batch_stats"): RESNET_BATCH_NORM.format(line=180),
        (f"{_LAYERS}.resnet.BottleneckBlock", "batch_stats"): RESNET_BATCH_NORM.format(line=429),
    }


def _base() -> dict[tuple[str, str], str]:
    mlp = HYPERPARAMETERS_IN_VARIABLES.format(
        names="hidden_dims, dropout_rate, in_features and use_batch_norm", lines="448-451"
    )
    cnn = HYPERPARAMETERS_IN_VARIABLES.format(
        names="dropout_rate, groups and other hyperparameters", lines="715-720"
    )
    return {
        (f"{_CORE}.base.MLP", "state_holds_arrays"): mlp,
        (f"{_CORE}.base.CNN", "state_holds_arrays"): cnn,
    }


def _backbones() -> dict[tuple[str, str], str]:
    failures: dict[tuple[str, str], str] = {}
    for module in ("DiTBlock", "LabelEmbedder", "DiffusionTransformer"):
        failures |= _eval_override(f"{_BACKBONES}.dit.{module}", DIT)
    for module in ("LabelEmbedder", "DiffusionTransformer"):
        key = (f"{_BACKBONES}.dit.{module}", "gradient_matches_finite_differences")
        failures[key] += EMBED_DTYPE
    for module in (
        "unet.DownBlock",
        "unet.UpBlock",
        "unet.UNet",
        "unet_cross_attention.DownBlockWithCrossAttention",
        "unet_cross_attention.UpBlockWithCrossAttention",
        "unet_cross_attention.UNet2DCondition",
    ):
        for check in ("tree_mode_split", "gradient_matches_finite_differences"):
            failures[(f"{_BACKBONES}.{module}", check)] = SHARED_RNGS
    key = (
        f"{_BACKBONES}.unet_cross_attention.CrossAttentionBlock",
        "gradient_matches_finite_differences",
    )
    failures[key] = JAX_ATTENTION_SOFTMAX
    return failures


KNOWN_FAILURES: dict[tuple[str, str], str] = (
    _transformers() | _positional() | _egnn() | _clifford() | _resnet() | _base() | _backbones()
)
