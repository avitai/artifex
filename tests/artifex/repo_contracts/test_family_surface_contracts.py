"""Contracts for the first resumed G3 family helper and shape slice."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_vae_docs_do_not_publish_removed_unified_helpers_or_resnet_surface() -> None:
    """Curated VAE docs should only teach the retained dense/cnn helper surface."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "api" / "models" / "vae.md": (
            "create_encoder_unified",
            "create_decoder_unified",
            '"resnet"',
        ),
        PROJECT_ROOT / "docs" / "models" / "encoders.md": ("create_encoder_unified",),
        PROJECT_ROOT / "docs" / "models" / "decoders.md": ("create_decoder_unified",),
        PROJECT_ROOT / "docs" / "getting-started" / "quickstart.md": ('"resnet"',),
        PROJECT_ROOT / "examples" / "generative_models" / "image" / "vae" / "vae_mnist.py": (
            '"resnet"',
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not publish {term}"


def test_transformer_and_wavenet_docs_do_not_teach_dead_cached_helpers() -> None:
    """Autoregressive docs should stop teaching unimplemented cache helpers."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "models" / "transformer.md": (
            "generate_with_cache",
            "get_attention_weights",
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "models" / "autoregressive-guide.md": (
            "generate_with_cache",
            "built-in generate_with_cache",
        ),
        PROJECT_ROOT / "docs" / "models" / "wavenet.md": ("generate_fast",),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_dit_direct_forward_docs_do_not_teach_cfg_scale() -> None:
    """Direct DiT forward docs/examples should keep CFG on the generation path only."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "api" / "models" / "diffusion.md": (
            "__call__(x, t, y=None, *, deterministic=False, cfg_scale=None)",
            "cfg_scale (`float | None`): Classifier-free guidance scale",
        ),
        PROJECT_ROOT / "docs" / "examples" / "diffusion" / "dit-demo.md": (
            "output = model(x, t, y, deterministic=True, cfg_scale=3.0)",
        ),
        PROJECT_ROOT / "examples" / "generative_models" / "diffusion" / "dit_demo.py": (
            "output = model(x, t, y, deterministic=True, cfg_scale=3.0)",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach dead direct-forward CFG"


def test_point_cloud_examples_do_not_teach_dead_temperature_control() -> None:
    """Point-cloud example surfaces should not teach the removed temperature knob."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "examples" / "geometric" / "simple-point-cloud-example.md": (
            "temperature=0.8",
            "temperature=0.6",
            "temperature=1.2",
            "Temperature effects",
            "Adjust Temperature",
            "Control generation diversity with temperature",
        ),
        PROJECT_ROOT
        / "examples"
        / "generative_models"
        / "geometric"
        / "simple_point_cloud_example.py": (
            "temperature=0.8",
            "temperature control",
            "Experiment with `temperature`",
            "Control sampling diversity with temperature",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_geometric_mesh_surfaces_do_not_teach_removed_num_faces_knob() -> None:
    """Mesh example surfaces should stay on the retained sphere-template contract."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "examples" / "geometric" / "geometric-models-demo.md": (
            "num_faces=1024",
            "- `num_faces`:",
            '"num_faces": 2048',
        ),
        PROJECT_ROOT
        / "examples"
        / "generative_models"
        / "geometric"
        / "geometric_models_demo.py": (
            "num_faces=1024",
            "- `num_faces`:",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_voxel_surfaces_do_not_teach_removed_conditioning_aliases_or_generated_dump() -> None:
    """Voxel docs/examples should stay on the retained typed decoder surface."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "models" / "voxel.md": (
            "class Reshape",
            "Module Statistics",
            "use_conditioning",
            "conditioning_dim",
            "model_type",
            "`resolution`:",
        ),
        PROJECT_ROOT / "docs" / "examples" / "geometric" / "geometric-models-demo.md": (
            "Voxel Model with Conditioning",
            "Creating voxel model with conditioning",
            "resolution=16",
            "use_conditioning=True",
            "conditioning_dim=10",
            "- `resolution`: Grid resolution",
            "- `channels`: Multi-scale architecture layers",
            '"resolution": 32',
            '"resolution": 8',
        ),
        PROJECT_ROOT
        / "examples"
        / "generative_models"
        / "geometric"
        / "geometric_models_demo.py": (
            "Regular 3D grids with optional conditioning",
            "Voxel Model with Conditioning",
            "Creating voxel model with conditioning",
            "class-conditioning-friendly",
            "use_conditioning",
            "conditioning_dim",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_energy_docs_and_examples_do_not_teach_removed_spectral_norm_or_internal_buffer_access() -> (
    None
):
    """Curated energy docs/examples should stay on the surviving DeepEBM and buffer surface."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "user-guide" / "models" / "ebm-guide.md": (
            "use_spectral_norm=",
            "len(model.sample_buffer.buffer)",
            "model.sample_buffer.buffer = []",
        ),
        PROJECT_ROOT / "docs" / "examples" / "energy" / "simple-ebm.md": (
            "use_spectral_norm=",
            "buffer.add(",
            "Spectral normalization",
        ),
        PROJECT_ROOT / "examples" / "generative_models" / "energy" / "simple_ebm_example.py": (
            "use_spectral_norm=",
            "len(buffer.buffer)",
            "Spectral norm:",
            "spectral normalization",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_gan_docs_do_not_teach_removed_spectral_norm() -> None:
    """GAN docs should not teach the removed discriminator spectral-norm knob."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "api" / "models" / "gan.md": ("use_spectral_norm",),
        PROJECT_ROOT / "docs" / "user-guide" / "models" / "gan-guide.md": (
            "use_spectral_norm",
            "Use spectral normalization",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_audio_diffusion_docs_do_not_publish_phantom_module_symbols() -> None:
    """Audio diffusion docs should only publish the surviving module surface."""
    files_and_banned_terms = {
        PROJECT_ROOT / "docs" / "models" / "diffusion.md": (
            "AudioUNet1D",
            "ConvBlock1D",
            "DownBlock1D",
            "TimeEmbedding1D",
            "UpBlock1D",
            "create_audio_unet_backbone",
            "get_num_groups",
            "setup_noise_schedule",
        ),
        PROJECT_ROOT / "docs" / "models" / "index.md": ("[Diffusion API Reference](diffusion.md)",),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not publish {term}"


def test_se3_molecular_docs_do_not_teach_unverified_symmetry_claims() -> None:
    """SE3 molecular surfaces should not teach unverified symmetry guarantees."""
    files_and_banned_terms = {
        PROJECT_ROOT
        / "src"
        / "artifex"
        / "generative_models"
        / "models"
        / "flow"
        / "se3_molecular.py": (
            "SE(3)-Equivariant Molecular Flow",
            "rotational and translational symmetries",
            "SE(3)-equivariant coupling layer",
            "For true SE(3) equivariance",
        ),
        PROJECT_ROOT / "docs" / "models" / "se3_molecular.md": (
            "SE(3)-Equivariant Molecular Flow",
        ),
        PROJECT_ROOT / "docs" / "models" / "index.md": (
            "SE(3) equivariant flows for molecular generation",
            "SE(3) molecular flows",
            "| **SE(3) Molecular Flow** | Equivariant molecular flows | SE(3) symmetry, molecular generation | Drug design, molecular modeling |",
            "[SE(3) Molecular Flows](se3_molecular.md) - Equivariant flows for molecules",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_glow_surfaces_do_not_teach_removed_multi_scale_contract() -> None:
    """Glow surfaces should stay on the retained single-scale runtime story."""
    files_and_banned_terms = {
        PROJECT_ROOT / "src" / "artifex" / "generative_models" / "models" / "flow" / "glow.py": (
            "A multi-scale architecture as described in the Glow paper.",
            "Simulate channel increase after squeezing",
        ),
        PROJECT_ROOT / "docs" / "models" / "glow.md": (
            "Module Statistics",
            "num_scales",
        ),
        PROJECT_ROOT / "docs" / "api" / "models" / "flow.md": (
            "num_scales",
            "Multi-scale flow with ActNorm and invertible convolutions",
            "Number of multi-scale levels",
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "models" / "flow-guide.md": (
            "Glow uses a multi-scale architecture",
            "Use Glow with multi-scale architecture",
            "More scales for higher resolution",
            "num_scales",
            "Glow for high-quality images",
            "Best for high-quality image generation",
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "concepts" / "flow-explained.md": (
            "High-resolution image generation (256×256 and above)",
            "Need state-of-the-art sample quality",
            "Want to leverage multi-scale processing",
            '"num_scales": 3',
            "| **Glow** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High-res images, quality |",
        ),
        PROJECT_ROOT / "docs" / "models" / "index.md": (
            "| **Glow** | Generative Flow | Invertible 1x1 convolutions, ActNorm | High-quality image generation |",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"


def test_stylegan3_docs_do_not_teach_alias_free_equivariance() -> None:
    """StyleGAN3 public docs should stay on the simplified surviving contract."""
    files_and_banned_terms = {
        PROJECT_ROOT
        / "src"
        / "artifex"
        / "generative_models"
        / "models"
        / "gan"
        / "stylegan3.py": (
            "Translation and Rotation Equivariance",
            "alias-free",
            "translation/rotation equivariance",
        ),
        PROJECT_ROOT / "docs" / "models" / "stylegan3.md": (
            "Translation and Rotation Equivariance",
            "alias-free",
            "translation/rotation equivariance",
        ),
        PROJECT_ROOT / "docs" / "models" / "index.md": (
            "Alias-free StyleGAN",
            "Translation/rotation equivariance",
            "Alias-free high-quality generation",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not teach {term}"
