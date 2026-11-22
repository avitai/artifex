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
_VAE = "artifex.generative_models.models.vae"
_FLOW = "artifex.generative_models.models.flow"
_DIFFUSION = "artifex.generative_models.models.diffusion"

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
EMBED_DTYPE = (
    " With R154 fixed this check still fails: R518 flax's nnx.Embed fixes its dtype when built"
    " (flax/nnx/nn/linear.py:1278 at 204a819c, self.dtype = dtype or self.embedding.dtype), so"
    " the float64 copy of LabelEmbedder looks up in float32. An upstream defect: linen's Embed"
    " resolves the same default at call time, and an nnx.Embed cast to bfloat16 after build"
    " returns float32 where linen returns bfloat16 (backbones/dit.py:155)"
)
VAE_SAMPLING = (
    "R520 VAE.__call__ draws its reparameterisation noise from the Rngs the module stores"
    " (vae/base.py:127, reparameterize), so .eval() calls draw keys and differ; flax's VAE takes"
    " the key as a call argument (examples/vae/models.py:56)"
)
INHERITED_HYPERPARAMETERS = (
    "R056 inherited from core.base.{backbone}, which stores its hyperparameters in nnx.Variable"
    " (base.py:{lines}); this module builds one"
)
VQ_AUX = (
    "R141 VQVAE.quantize replaces the nnx.Dict that __init__ built (vq_vae.py:44) with a plain"
    " dict (vq_vae.py:102), so a forward pass adds four state leaves under _last_quantize_aux,"
    " changes the graph definition, and nnx.jit traces again"
)
ACTNORM = (
    "R112 ActNormLayer initialises bias and logs from the first batch it sees (glow.py:36, 60-77),"
    " reshaping them from (1, 1, C) to (1, 1, 1, C), so the forward pass changes the state,"
    " nnx.jit traces again, and the output does not depend on the parameters it was built with"
)
GLOW_COUPLING = (
    "R113 AffineCouplingLayer builds its network on the first call (glow.py:306, 316-317, 372-373),"
    " so it has no nnx.Param at construction and the forward pass adds four"
)
GLOW_LAZY = "{} inherited: its ActNormLayer and AffineCouplingLayer create state on the first call"
MADE_DEAD_UNITS = (
    "R102 MADE gives hidden units degrees 0..D-2 under a strict < (made.py:121), so a degree-0"
    " unit has no input and sits at its zero bias on ReLU's kink; the parameter gradient there"
    " is not the derivative (0.268 from finite differences, MAFLayer(4, [8, 8]))"
)
SPLINE_SHARED_RNGS = (
    "R166 NeuralSplineFlow shares its Rngs with its spline transform (neural_spline.py:53), so"
    " tree mode refuses the module (Duplicate RngCount)"
)
EMBED_LOOKUP = (
    "R518 flax's nnx.Embed fixes its dtype when built (flax/nnx/nn/linear.py:1278 at 204a819c), so"
    " the float64 copy of {embed} looks up in float32 and its float32 rounding swamps the finite"
    " differences"
)
SE3_EMBED_DTYPE = EMBED_LOOKUP.format(
    embed="SE3CouplingLayer's atom_embedding (se3_molecular.py:221)"
)
STORED_RNGS_TWICE = (
    "R027 {model} stores self.rngs ({where}) and so does {holder}, built from the same Rngs, so"
    " the two share one RngCount and tree mode refuses the module"
)
DIFFUSION_SHARED_RNGS = STORED_RNGS_TWICE.format(
    model="DiffusionModel, a GenerativeModel,",
    where="core/base.py:122",
    holder="its UNet1D's TimeEmbedding, a GenerativeModule (backbones/unet.py:18)",
)
STABLE_DIFFUSION_SHARED_RNGS = STORED_RNGS_TWICE.format(
    model="StableDiffusionModel",
    where="stable_diffusion.py:62",
    holder="each GenerativeModule block of its UNet2DCondition (.unet.down_blocks[0].block1)",
)
CLIP_TRANSFORMER = (
    "R005 inherited: CLIPTextEncoder calls its TransformerEncoder with no mode"
    " (clip_text_encoder.py:161), whose deterministic=False default overrides .eval() (R197)"
)
LDM_SAMPLING = (
    "R527 LDMModel.__call__ falls back to the Rngs the module stores (latent.py:443-444) for its"
    " reparameterisation, timestep and forward-diffusion draws (latent.py:400, 456; ddpm.py:85),"
    " so .eval() calls draw keys and differ, as R520 does for the VAE"
)
SE3_JIT_METHOD = (
    "R523 SE3MolecularFlow.log_prob is decorated with nnx.jit (se3_molecular.py:442), whose"
    " write-back of the module's stored Rngs raises TraceContextError under an outer jax.grad; the"
    " undecorated method differentiates"
)

VIEW_STORED_RNGS = (
    "R529 an nnx.Rngs is stored at .{path}, and nnx.view calls set_view on every node answering"
    " hasattr(node, 'set_view'), which an Rngs with a default stream does by returning that stream"
    " (flax/nnx/module.py, view); so a view draws a key and advances an RNG stream .eval() does"
    " not"
)
_GENERATIVE = "artifex.generative_models"
_VIEW_RNGS_PATHS = {
    "models.audio.diffusion.AudioDiffusionModel": "backbone.time_embedding.rngs",
    "models.backbones.unet.ConvBlock": "rngs",
    "models.backbones.unet.DownBlock": "block1.rngs",
    "models.backbones.unet.TimeEmbedding": "rngs",
    "models.backbones.unet.UNet": "down_blocks[0].block1.rngs",
    "models.backbones.unet.UpBlock": "block1.rngs",
    "models.backbones.unet_1d.UNet1D": "time_embedding.rngs",
    "models.backbones.unet_cross_attention.CrossAttentionBlock": "rngs",
    "models.backbones.unet_cross_attention.DownBlockWithCrossAttention": "attn_blocks[0].rngs",
    "models.backbones.unet_cross_attention.UNet2DCondition": "down_blocks[0].block1.rngs",
    "models.backbones.unet_cross_attention.UpBlockWithCrossAttention": "attn_blocks[0].rngs",
    "models.diffusion.base.DiffusionModel": "backbone.time_embedding.rngs",
    "models.diffusion.ddim.DDIMModel": "backbone.time_embedding.rngs",
    "models.diffusion.ddpm.DDPMModel": "backbone.time_embedding.rngs",
    "models.diffusion.dit.DiTModel": "rngs",
    "models.diffusion.guidance.GuidedDiffusionModel": "backbone.time_embedding.rngs",
    "models.diffusion.latent.LDMModel": "backbone.time_embedding.rngs",
    "models.diffusion.score.ScoreDiffusionModel": "backbone.time_embedding.rngs",
    "models.diffusion.stable_diffusion.StableDiffusionModel": "rngs",
    "models.diffusion.stable_diffusion_pipeline.StableDiffusionPipeline": "model.rngs",
    "models.flow.conditional.ConditionalNormalizingFlow": "rngs",
    "models.flow.conditional.ConditionalRealNVP": "rngs",
    "models.flow.glow.Glow": "rngs",
    "models.flow.iaf.IAF": "rngs",
    "models.flow.maf.MAF": "rngs",
    "models.flow.neural_spline.NeuralSplineFlow": "coupling_layers[0].spline_transform.rngs",
    "models.flow.neural_spline.RationalQuadraticSplineTransform": "rngs",
    "models.flow.neural_spline.SplineCouplingLayer": "spline_transform.rngs",
    "models.flow.real_nvp.RealNVP": "rngs",
    "models.flow.se3_molecular.SE3MolecularFlow": "rngs",
    "models.gan.base.GAN": "rngs",
    "models.gan.conditional.ConditionalGAN": "rngs",
    "models.gan.cyclegan.CycleGAN": "rngs",
    "models.gan.dcgan.DCGAN": "rngs",
    "models.gan.wgan.WGAN": "rngs",
    "models.vae.base.VAE": "rngs",
    "models.vae.beta_vae.BetaVAE": "rngs",
    "models.vae.beta_vae.BetaVAEWithCapacity": "rngs",
    "models.vae.conditional.ConditionalVAE": "rngs",
    "models.vae.vq_vae.VQVAE": "rngs",
}
"""Where each module that fails ``view_matches_modes`` holds an ``nnx.Rngs``, the first path."""
LDM_VIEW_OUTPUTS = (
    "; LDMModel's own .rngs is that same Rngs, from which .eval() draws its sampling noise"
    " (R527), so the view's outputs differ too"
)

_EVAL_CASCADE = ("gradient_finite", "gradient_matches_finite_differences", "rows_independent")
"""Checks that run in ``.eval()``: a module still drawing keys there fails each."""


def _eval_override(
    name: str, defect: str, draws: str = "dropout keys"
) -> dict[tuple[str, str], str]:
    inherited = f".eval() still draws {draws}: " + defect
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
    return failures


def _vae() -> dict[tuple[str, str], str]:
    mlp = INHERITED_HYPERPARAMETERS.format(backbone="MLP", lines="448-451")
    cnn = INHERITED_HYPERPARAMETERS.format(backbone="CNN", lines="715-720")
    failures: dict[tuple[str, str], str] = {
        (f"{_VAE}.{module}", "state_holds_arrays"): backbone
        for module, backbone in (
            ("encoders.MLPEncoder", mlp),
            ("encoders.ConditionalEncoder", mlp),
            ("encoders.CNNEncoder", cnn),
            ("encoders.ResNetEncoder", cnn),
            ("decoders.MLPDecoder", mlp),
            ("decoders.ConditionalDecoder", mlp),
            ("decoders.CNNDecoder", cnn),
            ("decoders.ResNetDecoder", cnn),
            ("base.VAE", mlp),
            ("beta_vae.BetaVAE", mlp),
            ("beta_vae.BetaVAEWithCapacity", mlp),
            ("conditional.ConditionalVAE", mlp),
            ("vq_vae.VQVAE", mlp),
        )
    }
    for module in (
        "base.VAE",
        "beta_vae.BetaVAE",
        "beta_vae.BetaVAEWithCapacity",
        "conditional.ConditionalVAE",
    ):
        failures |= _eval_override(f"{_VAE}.{module}", VAE_SAMPLING, draws="sampling keys")
    for check in ("state_structure", "jit_matches_eager"):
        failures[(f"{_VAE}.vq_vae.VQVAE", check)] = VQ_AUX
    return failures


def _flow() -> dict[tuple[str, str], str]:
    glow = f"{_FLOW}.glow"
    lazy = ("state_structure", "jit_matches_eager")
    gradients = ("gradient_finite", "gradient_matches_finite_differences")
    failures: dict[tuple[str, str], str] = {
        **{(f"{glow}.ActNormLayer", check): ACTNORM for check in (*lazy, *gradients)},
        **{(f"{glow}.AffineCouplingLayer", check): GLOW_COUPLING for check in (*lazy, *gradients)},
        **{(f"{glow}.GlowBlock", check): GLOW_LAZY.format("R112, R113") for check in lazy},
        **{(f"{glow}.Glow", check): GLOW_LAZY.format("R112, R113") for check in lazy},
        **{
            (f"{_FLOW}.neural_spline.NeuralSplineFlow", check): SPLINE_SHARED_RNGS
            for check in ("tree_mode_split", "gradient_matches_finite_differences")
        },
        **{
            (f"{_FLOW}.se3_molecular.SE3MolecularFlow", check): SE3_JIT_METHOD
            for check in gradients
        },
        (f"{_FLOW}.se3_molecular.SE3CouplingLayer", "gradient_matches_finite_differences"): (
            SE3_EMBED_DTYPE
        ),
    }
    for module in ("maf.MAFLayer", "maf.MAF", "iaf.IAFLayer", "iaf.IAF"):
        failures[(f"{_FLOW}.{module}", "gradient_matches_finite_differences")] = MADE_DEAD_UNITS
    return failures


def _diffusion() -> dict[tuple[str, str], str]:
    shared = ("tree_mode_split", "gradient_matches_finite_differences")
    failures: dict[tuple[str, str], str] = {}
    for model in (
        f"{_DIFFUSION}.base.DiffusionModel",
        f"{_DIFFUSION}.ddpm.DDPMModel",
        f"{_DIFFUSION}.ddim.DDIMModel",
        f"{_DIFFUSION}.guidance.GuidedDiffusionModel",
        f"{_DIFFUSION}.score.ScoreDiffusionModel",
        "artifex.generative_models.models.audio.diffusion.AudioDiffusionModel",
    ):
        failures |= {(model, check): DIFFUSION_SHARED_RNGS for check in shared}
    pipeline = f"{_DIFFUSION}.stable_diffusion_pipeline.StableDiffusionPipeline"
    for model, defect in (
        (f"{_DIFFUSION}.stable_diffusion.StableDiffusionModel", STABLE_DIFFUSION_SHARED_RNGS),
        (pipeline, f"inherited from its StableDiffusionModel: {STABLE_DIFFUSION_SHARED_RNGS}"),
    ):
        failures |= {(model, check): defect for check in shared}

    clip = f"{_DIFFUSION}.clip_text_encoder.CLIPTextEncoder"
    failures |= _eval_override(clip, CLIP_TRANSFORMER)
    failures[(clip, "gradient_matches_finite_differences")] += (
        ". With R005 fixed this check still fails: "
        + EMBED_LOOKUP.format(embed="its token_embedding (clip_text_encoder.py:88)")
    )

    ldm = f"{_DIFFUSION}.latent.LDMModel"
    failures |= _eval_override(ldm, LDM_SAMPLING, draws="sampling keys")
    failures |= {(ldm, check): DIFFUSION_SHARED_RNGS for check in shared}
    failures[(ldm, "state_holds_arrays")] = INHERITED_HYPERPARAMETERS.format(
        backbone="MLP", lines="448-451"
    ).replace("this module builds one", "its MLPEncoder and MLPDecoder each build one")

    failures[(f"{_DIFFUSION}.dit.DiTModel", "gradient_matches_finite_differences")] = (
        EMBED_LOOKUP.format(embed="its DiffusionTransformer's LabelEmbedder (backbones/dit.py:155)")
    )
    return failures


def _views() -> dict[tuple[str, str], str]:
    failures = {
        (f"{_GENERATIVE}.{module}", "view_matches_modes"): VIEW_STORED_RNGS.format(path=path)
        for module, path in _VIEW_RNGS_PATHS.items()
    }
    failures[(f"{_DIFFUSION}.latent.LDMModel", "view_matches_modes")] += LDM_VIEW_OUTPUTS
    return failures


KNOWN_FAILURES: dict[tuple[str, str], str] = (
    _transformers()
    | _positional()
    | _egnn()
    | _clifford()
    | _resnet()
    | _base()
    | _backbones()
    | _vae()
    | _flow()
    | _diffusion()
    | _views()
)
