# Inference Configs

Inference uses a small hierarchy of frozen dataclasses.

## Public Imports

```python
from artifex.configs import DiffusionInferenceConfig, InferenceConfig, ProteinDiffusionInferenceConfig

base_config = InferenceConfig(
    name="base_inference",
    checkpoint_path="{ARTIFEX_CHECKPOINT_ROOT}/latest",
    batch_size=4,
    num_samples=8,
    device="cpu",
)
```

## Available Types

- `InferenceConfig`
- `DiffusionInferenceConfig`
- `ProteinDiffusionInferenceConfig`

`get_inference_config()` selects the narrowest supported type that matches the
YAML shape. The retained `protein_diffusion_inference` asset therefore loads as
`ProteinDiffusionInferenceConfig`.

## Key Fields

Base inference:

- `checkpoint_path`
- `output_dir`
- `batch_size`
- `num_samples`
- `device`

Diffusion-specific additions:

- `sampler`
- `timesteps`
- `temperature`
- `guidance_scale`
- `save_intermediate_steps`

Protein-specific additions:

- `target_seq_length`
- `backbone_atom_indices`
- `calculate_metrics`
- `visualize_structures`
- `save_as_pdb`
