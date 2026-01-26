# HuggingFace Integration

Complete guide to integrating Artifex with HuggingFace Hub for model sharing, dataset loading, and deployment to Spaces.

## Overview

HuggingFace provides a powerful ecosystem for sharing models, datasets, and demos. Artifex integrates seamlessly with HuggingFace Hub, enabling you to share your generative models with the community and leverage thousands of existing datasets.

!!! tip "HuggingFace Benefits"
    - **Model Hub**: Share and discover pretrained models
    - **Datasets**: Access 50,000+ datasets
    - **Spaces**: Deploy interactive demos
    - **Community**: Connect with researchers worldwide

<div class="grid cards" markdown>

- :material-cloud-upload:{ .lg .middle } **Model Hub**

    ---

    Upload and download models from HuggingFace Hub

    [:octicons-arrow-right-24: Model Hub Guide](#model-hub-integration)

- :material-database:{ .lg .middle } **Datasets**

    ---

    Load and preprocess HuggingFace datasets

    [:octicons-arrow-right-24: Datasets Guide](#datasets-integration)

- :material-rocket-launch:{ .lg .middle } **Spaces**

    ---

    Deploy interactive demos with Gradio or Streamlit

    [:octicons-arrow-right-24: Spaces Guide](#spaces-integration)

- :material-tag:{ .lg .middle } **Model Cards**

    ---

    Document models with metadata and examples

    [:octicons-arrow-right-24: Model Cards](#model-cards-and-metadata)

</div>

---

## Prerequisites

Install HuggingFace libraries:

```bash
# Install HuggingFace ecosystem
pip install huggingface_hub datasets gradio

# Or using uv
uv pip install huggingface_hub datasets gradio
```

Authenticate with HuggingFace Hub:

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in code
from huggingface_hub import login
login(token="hf_...")
```

---

## Model Hub Integration

### Uploading Models to Hub

Share your trained models with the community.

```python
from huggingface_hub import HfApi, create_repo
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
import json

class ModelUploader:
    """Upload Artifex models to HuggingFace Hub."""

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.api = HfApi()

    def upload_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = False,
    ):
        """Upload model to HuggingFace Hub.

        Args:
            repo_id: Repository ID (e.g., "username/model-name")
            commit_message: Commit message
            private: Make repository private

        Returns:
            URL to uploaded model
        """
        # Create repository
        repo_url = create_repo(
            repo_id=repo_id,
            exist_ok=True,
            private=private,
        )

        # Save model locally
        save_dir = Path(f"./tmp/{self.model_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Export model parameters
        self._save_model(save_dir)

        # Create model card
        self._create_model_card(save_dir, repo_id)

        # Upload to Hub
        self.api.upload_folder(
            folder_path=str(save_dir),
            repo_id=repo_id,
            commit_message=commit_message,
        )

        print(f"Model uploaded to: {repo_url}")
        return repo_url

    def _save_model(self, save_dir: Path):
        """Save model parameters and config."""
        # Extract and save parameters
        state = nnx.state(self.model)

        # Convert to serializable format
        serialized_state = {
            k: v.tolist() if isinstance(v, jnp.ndarray) else v
            for k, v in state.items()
        }

        # Save parameters
        with open(save_dir / "model_params.json", "w") as f:
            json.dump(serialized_state, f)

        # Save configuration
        config = {
            "model_type": self.model.__class__.__name__,
            "latent_dim": getattr(self.model, "latent_dim", None),
            "image_shape": getattr(self.model, "image_shape", None),
            "framework": "artifex",
            "backend": "flax-nnx",
        }

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def _create_model_card(self, save_dir: Path, repo_id: str):
        """Create README.md model card."""
        model_card = self._generate_model_card_content(repo_id)

        # Write model card to file
        with open(save_dir / "README.md", "w") as f:
            f.write(model_card)

    def _generate_model_card_content(self, repo_id: str) -> str:
        """Generate model card markdown content."""
        return f"""---
library_name: artifex
tags: [generative-models, jax, flax]
license: apache-2.0
---

# {repo_id}

Artifex generative model trained with JAX/Flax NNX.

## Model Details
- **Type**: {self.model.__class__.__name__}
- **Framework**: Artifex (JAX/Flax NNX)

## Usage
See [Model Cards section](#model-cards-and-metadata) for complete example.
"""
```

### Downloading Models from Hub

Load pretrained models shared by others.

```python
from huggingface_hub import hf_hub_download
import json

class ModelDownloader:
    """Download Artifex models from HuggingFace Hub."""

    @staticmethod
    def download_model(
        repo_id: str,
        revision: str = "main",
    ):
        """Download model from HuggingFace Hub.

        Args:
            repo_id: Repository ID
            revision: Git revision (branch, tag, or commit)

        Returns:
            Loaded model
        """
        # Download config
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
        )

        with open(config_path) as f:
            config = json.load(f)

        # Download parameters
        params_path = hf_hub_download(
            repo_id=repo_id,
            filename="model_params.json",
            revision=revision,
        )

        with open(params_path) as f:
            params = json.load(f)

        # Reconstruct model
        model = ModelDownloader._build_model(config, params)

        return model

    @staticmethod
    def _build_model(config: dict, params: dict):
        """Build model from config and parameters."""
        from artifex.generative_models.models.vae import VAE
        from artifex.generative_models.models.gan import GAN

        model_type = config["model_type"]

        if model_type == "VAE":
            model = VAE(
                latent_dim=config["latent_dim"],
                image_shape=config["image_shape"],
                rngs=nnx.Rngs(0),
            )
        elif model_type == "GAN":
            model = GAN(
                latent_dim=config["latent_dim"],
                image_shape=config["image_shape"],
                rngs=nnx.Rngs(0),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load parameters
        # (Simplified - actual implementation needs proper parameter loading)

        return model
```

---

## Datasets Integration

### Loading HuggingFace Datasets

Access thousands of datasets for training.

```python
from datasets import load_dataset
import jax.numpy as jnp
import numpy as np

class HFDatasetLoader:
    """Load HuggingFace datasets for Artifex."""

    @staticmethod
    def load_image_dataset(
        dataset_name: str,
        split: str = "train",
        image_key: str = "image",
    ):
        """Load image dataset from HuggingFace.

        Args:
            dataset_name: Dataset name (e.g., "mnist", "cifar10")
            split: Dataset split
            image_key: Key for image column

        Returns:
            JAX-compatible dataset
        """
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        # Convert to JAX arrays
        def process_example(example):
            image = np.array(example[image_key])

            # Normalize to [-1, 1]
            image = (image.astype(np.float32) / 255.0) * 2 - 1

            return {"image": image}

        # Process dataset
        dataset = dataset.map(process_example)
        dataset.set_format(type="numpy")

        return dataset

    @staticmethod
    def load_text_dataset(
        dataset_name: str,
        split: str = "train",
        text_key: str = "text",
        max_length: int = 512,
    ):
        """Load text dataset.

        Args:
            dataset_name: Dataset name
            split: Dataset split
            text_key: Key for text column
            max_length: Maximum sequence length

        Returns:
            Processed text dataset
        """
        from transformers import AutoTokenizer

        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        def tokenize(example):
            tokens = tokenizer(
                example[text_key],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="np",
            )
            return {"input_ids": tokens["input_ids"][0]}

        dataset = dataset.map(tokenize)
        dataset.set_format(type="numpy")

        return dataset
```

### Custom Dataset Creation

Create and upload your own datasets.

```python
from datasets import Dataset, Features, Image, Value
from pathlib import Path

class CustomDatasetCreator:
    """Create custom HuggingFace datasets."""

    @staticmethod
    def create_image_dataset(
        image_dir: Path,
        save_path: str = "my_dataset",
    ):
        """Create dataset from image directory.

        Args:
            image_dir: Directory containing images
            save_path: Path to save dataset

        Returns:
            Created dataset
        """
        # Collect image paths
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        # Create dataset dictionary
        data = {
            "image": [str(p) for p in image_paths],
            "filename": [p.name for p in image_paths],
        }

        # Define features
        features = Features({
            "image": Image(),
            "filename": Value("string"),
        })

        # Create dataset
        dataset = Dataset.from_dict(data, features=features)

        # Save dataset
        dataset.save_to_disk(save_path)

        return dataset

    @staticmethod
    def upload_dataset(
        dataset_path: str,
        repo_id: str,
        private: bool = False,
    ):
        """Upload dataset to HuggingFace Hub.

        Args:
            dataset_path: Path to saved dataset
            repo_id: Repository ID
            private: Make repository private
        """
        from datasets import load_from_disk

        # Load dataset
        dataset = load_from_disk(dataset_path)

        # Upload to Hub
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
        )

        print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
```

### Streaming Large Datasets

Efficiently handle datasets too large for memory.

```python
from datasets import load_dataset

def stream_large_dataset(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
):
    """Stream large dataset in batches.

    Args:
        dataset_name: Dataset name
        split: Dataset split
        batch_size: Batch size for streaming

    Yields:
        Batches of data
    """
    # Load in streaming mode
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )

    # Batch and iterate
    batch = []
    for example in dataset:
        batch.append(example)

        if len(batch) >= batch_size:
            yield jnp.array([ex["image"] for ex in batch])
            batch = []

    # Yield final batch
    if batch:
        yield jnp.array([ex["image"] for ex in batch])
```

---

## Spaces Integration

### Creating Gradio Demos

Deploy interactive demos with Gradio.

```python
import gradio as gr
import jax
import jax.numpy as jnp
from flax import nnx

class GradioDemo:
    """Create Gradio demo for generative model."""

    def __init__(self, model):
        self.model = model

    def create_interface(self):
        """Create Gradio interface.

        Returns:
            Gradio interface
        """
        def generate(
            num_samples: int,
            temperature: float,
            seed: int,
        ):
            """Generate samples."""
            rngs = nnx.Rngs(seed)

            # Sample latent codes
            z = jax.random.normal(
                rngs.sample(),
                (num_samples, self.model.latent_dim)
            ) * temperature

            # Generate images
            images = self.model.decode(z)

            # Convert to numpy and denormalize
            images = np.array(images)
            images = ((images + 1) / 2 * 255).astype(np.uint8)

            return images

        # Create interface
        interface = gr.Interface(
            fn=generate,
            inputs=[
                gr.Slider(1, 16, value=4, step=1, label="Number of Samples"),
                gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature"),
                gr.Slider(0, 1000, value=42, step=1, label="Random Seed"),
            ],
            outputs=gr.Gallery(label="Generated Images"),
            title="Artifex Generative Model",
            description="Generate images using a trained generative model",
        )

        return interface

    def launch(self, share: bool = True):
        """Launch demo.

        Args:
            share: Create public link
        """
        interface = self.create_interface()
        interface.launch(share=share)
```

### Deploying to Spaces

Deploy your demo to HuggingFace Spaces.

```python
# app.py - Main Gradio app file
from huggingface_hub import hf_hub_download
from artifex.generative_models.models import load_model
import gradio as gr

# Download model
model_path = hf_hub_download(
    repo_id="username/my-model",
    filename="model_params.json"
)

# Load model
model = load_model(model_path)

# Create demo
demo = GradioDemo(model)
interface = demo.create_interface()

# Launch
if __name__ == "__main__":
    interface.launch()
```

Create `requirements.txt`:

```txt
artifex
gradio
jax[cpu]
flax
huggingface_hub
```

Deploy to Spaces:

```bash
# Create Space
huggingface-cli repo create my-demo --type space --space_sdk gradio

# Clone and setup
git clone https://huggingface.co/spaces/username/my-demo
cd my-demo

# Add files
cp app.py .
cp requirements.txt .

# Commit and push
git add .
git commit -m "Initial demo"
git push
```

### Advanced Gradio Features

Create more interactive demos.

```python
class AdvancedGradioDemo:
    """Advanced Gradio demo with multiple features."""

    def __init__(self, model):
        self.model = model

    def create_interface(self):
        """Create advanced interface."""

        with gr.Blocks() as demo:
            gr.Markdown("# Artifex Generative Model Demo")

            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column():
                        num_samples = gr.Slider(1, 16, value=4, label="Samples")
                        temperature = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
                        seed = gr.Number(value=42, label="Seed")
                        generate_btn = gr.Button("Generate")

                    with gr.Column():
                        output_gallery = gr.Gallery(label="Generated Images")

                generate_btn.click(
                    fn=self.generate,
                    inputs=[num_samples, temperature, seed],
                    outputs=output_gallery,
                )

            with gr.Tab("Interpolate"):
                with gr.Row():
                    seed1 = gr.Number(value=42, label="Start Seed")
                    seed2 = gr.Number(value=123, label="End Seed")
                    steps = gr.Slider(5, 20, value=10, label="Steps")
                    interpolate_btn = gr.Button("Interpolate")

                interpolation_output = gr.Gallery(label="Interpolation")

                interpolate_btn.click(
                    fn=self.interpolate,
                    inputs=[seed1, seed2, steps],
                    outputs=interpolation_output,
                )

        return demo

    def generate(self, num_samples, temperature, seed):
        """Generate samples."""
        # Implementation from previous example
        pass

    def interpolate(self, seed1, seed2, steps):
        """Interpolate between two latent codes."""
        rngs1 = nnx.Rngs(int(seed1))
        rngs2 = nnx.Rngs(int(seed2))

        z1 = jax.random.normal(rngs1.sample(), (1, self.model.latent_dim))
        z2 = jax.random.normal(rngs2.sample(), (1, self.model.latent_dim))

        alphas = jnp.linspace(0, 1, steps)[:, None]
        z_interp = z1 * (1 - alphas) + z2 * alphas

        images = self.model.decode(z_interp)
        images = np.array(images)
        images = ((images + 1) / 2 * 255).astype(np.uint8)

        return images
```

---

## Model Cards and Metadata

### Complete Model Card Template

```markdown
---
library_name: artifex
tags:
  - generative-models
  - vae
  - image-generation
  - jax
  - flax
datasets:
  - mnist
metrics:
  - fid
  - inception_score
license: apache-2.0
---

# VAE for MNIST Generation

Variational Autoencoder trained on MNIST for digit generation.

## Model Details

- **Model Type**: Variational Autoencoder (VAE)
- **Architecture**: Convolutional encoder/decoder
- **Latent Dimension**: 128
- **Framework**: Artifex (JAX/Flax NNX)
- **Training Data**: MNIST (60,000 images)
- **Parameters**: 2.4M

## Intended Use

Generate realistic handwritten digits or encode images to latent space.

## Training Details

- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 128
- **Training Steps**: 50,000
- **Hardware**: NVIDIA A100 (1x)
- **Training Time**: 2 hours

## Evaluation

| Metric | Value |
|--------|-------|
| FID | 12.3 |
| Reconstruction MSE | 0.012 |
| KL Divergence | 8.5 |

## Usage

```python
from huggingface_hub import hf_hub_download
from artifex.generative_models.models import load_model

model = load_model("username/vae-mnist")
samples = model.sample(num_samples=16)
```

## Limitations

- Trained only on MNIST (28×28 grayscale)
- May not generalize to other digit styles
- Limited to digit generation (0-9)

## Citation

```bibtex
@software{vae_mnist,
  author = {Your Name},
  title = {VAE for MNIST},
  year = {2025},
  url = {https://huggingface.co/username/vae-mnist}
}
```

```

---

## Best Practices

### DO

!!! success "Recommended Practices"
    ✅ **Create detailed model cards** with usage examples

    ✅ **Version your models** with Git tags

    ✅ **Include evaluation metrics** (FID, IS, etc.)

    ✅ **Provide example outputs** in model card

    ✅ **Document limitations** and intended use

    ✅ **Use meaningful tags** for discoverability

### DON'T

!!! danger "Avoid These Mistakes"
    ❌ **Don't upload without model card** (required for Hub)

    ❌ **Don't hardcode API tokens** (use environment variables)

    ❌ **Don't upload large files directly** (use Git LFS)

    ❌ **Don't skip dataset licensing** information

    ❌ **Don't ignore versioning** (use semantic versioning)

---

## Summary

HuggingFace integration enables:

- **Model Sharing**: Upload and discover pretrained models
- **Dataset Access**: Leverage 50,000+ datasets
- **Demo Deployment**: Create interactive Gradio demos
- **Community**: Connect with researchers worldwide

Start sharing your models and building demos today!

---

## Next Steps

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } __Weights & Biases__

    ---

    Track experiments and hyperparameter sweeps

    [:octicons-arrow-right-24: W&B Integration](wandb.md)

-   :material-monitor-dashboard:{ .lg .middle } __TensorBoard__

    ---

    Visualize training metrics and samples

    [:octicons-arrow-right-24: TensorBoard Guide](tensorboard.md)

-   :material-cloud:{ .lg .middle } __Deployment__

    ---

    Deploy models to production servers

    [:octicons-arrow-right-24: Deployment Guide](deployment.md)

-   :material-code-braces:{ .lg .middle } __API Reference__

    ---

    Explore complete API documentation

    [:octicons-arrow-right-24: API Docs](../../api/core/base.md)

</div>
