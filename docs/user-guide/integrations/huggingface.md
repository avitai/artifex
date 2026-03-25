# HuggingFace Integration

Status: Coming soon

HuggingFace remains a relevant distribution target, but Artifex does not currently ship built-in HuggingFace Hub upload/download helpers, Spaces deployment wrappers, or an Artifex-owned model-hub API.

## What Exists Today

- checkpoint artifacts through `artifex.generative_models.core.checkpointing`
- family-owned model construction from typed configs
- the low-level deployment flow documented in [deployment.md](deployment.md)

## Current Guidance

If you need HuggingFace Hub today, keep the integration in your application
layer:

1. Save checkpoints and sidecar metadata with `core.checkpointing`.
2. Upload those artifacts with your own `huggingface_hub` client code.
3. Restore the correct family model locally by rebuilding its typed config and
   loading the checkpointed state.

## Why This Page Is Not A Runtime API Guide

The current runtime does not own one generic model uploader, downloader, or
Spaces launcher. This page stays published only to mark the relevant future
integration area without presenting a shared integration framework as current
supported API.
