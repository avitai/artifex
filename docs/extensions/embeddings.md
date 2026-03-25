# Embeddings

**Status:** `Supported runtime extension owner`

**Module:** `artifex.generative_models.extensions.nlp.embeddings`

**Source:** `src/artifex/generative_models/extensions/nlp/embeddings.py`

Text embedding owners for the NLP extension family.

## Top-Level Module Exports

- `precompute_rope_freqs()`
- `apply_rope()`
- `create_sinusoidal_positions()`
- `TextEmbeddings`

## Class APIs

### `TextEmbeddings`

- `embed()`
- `get_token_embeddings()`
- `compute_similarity()`
- `create_contextual_embeddings()`
- `project_to_vocabulary()`
- `extract_sentence_embedding()`
- `compute_attention_weights()`
- `interpolate_embeddings()`
- `apply_rope_embeddings()`
- `get_sinusoidal_embeddings()`
- `embed_with_sinusoidal_positions()`
- `embed_with_rope()`
- `get_embedding_statistics()`
