"""Cases for the text modality, ``artifex.generative_models.modalities.text``."""

from __future__ import annotations

import jax
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal


_TEXT = "artifex.generative_models.modalities.text"
_VOCAB, _LENGTH, _EMBED = 50, 8, 6


def _tokens(key: jax.Array) -> tuple[jax.Array, ...]:
    """Token ids above the special tokens (pad 0, unk 1, bos 2, eos 3)."""
    return (jax.random.randint(key, (BATCH, _LENGTH), 4, _VOCAB),)


def _token_pairs(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return _tokens(k1)[0], _tokens(k2)[0]


def _config() -> object:
    from artifex.generative_models.core.configuration.modality_config import (  # noqa: PLC0415
        ModalityConfig,
    )

    return ModalityConfig(
        name="text",
        modality_name="text",
        metadata={"text_params": {"vocab_size": _VOCAB, "max_length": _LENGTH}},
    )


def _sequences(module: nnx.Module, tokens: jax.Array) -> PyTree:
    return module.process_sequences(tokens), module.pad_sequences(list(tokens))


def cases() -> dict[str, Case]:
    """Return the text-modality cases, by the qualified name of the class each builds."""

    from artifex.generative_models.modalities.text import (  # noqa: PLC0415
        base,
        evaluation,
        representations,
    )

    config = _config()
    return {
        f"{_TEXT}.base.TextModality": Case(
            build=lambda r: base.TextModality(config, rngs=r),
            inputs=_tokens,
            call=lambda module, tokens: module.process(tokens),
            is_differentiable=False,
        ),
        f"{_TEXT}.representations.TextProcessor": Case(
            build=lambda r: representations.TextProcessor(config, rngs=r),
            inputs=_tokens,
            call=_sequences,
            is_differentiable=False,
            # process_sequences also reports the batch size and padded length.
            is_rowwise=False,
        ),
        f"{_TEXT}.representations.PositionEncodingProcessor": Case(
            build=lambda r: representations.PositionEncodingProcessor(
                config, embedding_dim=_EMBED, encoding_type="learned", rngs=r
            ),
            inputs=normal(_LENGTH, _EMBED),
            call=lambda module, embeddings: module.apply_position_encoding(embeddings),
        ),
        f"{_TEXT}.representations.SequenceAugmentationProcessor": Case(
            build=lambda r: representations.SequenceAugmentationProcessor(
                config, dropout_rate=0.5, rngs=r
            ),
            inputs=_tokens,
            call=lambda module, tokens: module.apply_token_dropout(tokens),
            is_differentiable=False,
            has_dropout=True,
        ),
        f"{_TEXT}.evaluation.TextEvaluationSuite": Case(
            build=lambda r: evaluation.TextEvaluationSuite(config, rngs=r),
            inputs=_token_pairs,
            call=lambda module, generated, reference: module.evaluate_batch(generated, reference),
            is_differentiable=False,
            is_rowwise=False,
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_TEXT}.representations.TokenizationProcessor": (
        "its methods map Python strings to token ids and back on the host (tokenize_text,"
        " encode_batch, decode_batch), so no check can pass an array through it; its defects are"
        " R377 (6 of 9 real words map to UNK) and R297 (salted hash token ids)"
    ),
}
