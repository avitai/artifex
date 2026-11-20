"""Cases for the GAN family, ``artifex.generative_models.models.gan``."""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import arrays, labels, normal


_GAN = "artifex.generative_models.models.gan"
_GAN_LATENT, _GAN_CLASSES, _GAN_DROPOUT = 3, 3, 0.5
_GAN_FLAT = (1, 4, 4)
_GAN_IMAGE = (3, 8, 8)
_CRITIC_IMAGE = (3, 16, 16)
_MULTISCALE_IMAGE = (3, 16, 16)
_CYCLE_IMAGE = (8, 8, 3)
_STYLE_DIM, _STYLE_RESOLUTION = 8, 8


def _scores_and_samples(module: nnx.Module, x: jax.Array, z: jax.Array) -> PyTree:
    """Run both networks of a GAN: the discriminator on data and the generator on latents."""
    return module.discriminator(x), module.generator(z)


def _gan_forward_and_samples(module: nnx.Module, x: jax.Array, z: jax.Array) -> PyTree:
    """The GAN's own forward pass, which only discriminates, and its generator on latents."""
    return module(x), module.generator(z)


def _conditioned_scores_and_samples(
    module: nnx.Module, x: jax.Array, z: jax.Array, labels: jax.Array
) -> PyTree:
    return module.discriminator(x, labels), module.generator(z, labels)


def _translations(module: nnx.Module, a: jax.Array, b: jax.Array) -> PyTree:
    """Both CycleGAN translations and both discriminators."""
    return (
        module.generator_a_to_b(a),
        module.generator_b_to_a(b),
        module.discriminator_a(a),
        module.discriminator_b(b),
    )


def _data_and_latents(image: tuple[int, ...]) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    return arrays(image, (_GAN_LATENT,))


def _labelled_data_and_latents(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    x, z = _data_and_latents(_GAN_IMAGE)(k1)
    return x, z, jax.nn.one_hot(labels(k2, _GAN_CLASSES), _GAN_CLASSES)


def _dense_generator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        GeneratorConfig,
    )

    return GeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=_GAN_FLAT,
        hidden_dims=(8, 6),
        activation="silu",
        batch_norm=True,
        dropout_rate=_GAN_DROPOUT,
    )


def _dense_discriminator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        DiscriminatorConfig,
    )

    return DiscriminatorConfig(
        name="discriminator",
        input_shape=_GAN_FLAT,
        hidden_dims=(8, 6),
        activation="silu",
        batch_norm=True,
        dropout_rate=_GAN_DROPOUT,
    )


def _conv_generator(
    output_shape: tuple[int, ...] = _GAN_IMAGE, hidden_dims: tuple[int, ...] = (8, 4)
) -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConvGeneratorConfig,
    )

    return ConvGeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=output_shape,
        hidden_dims=hidden_dims,
        activation="silu",
        batch_norm=True,
    )


def _conv_discriminator(input_shape: tuple[int, ...] = _GAN_IMAGE, **fields: object) -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConvDiscriminatorConfig,
    )

    return ConvDiscriminatorConfig(
        name="discriminator",
        input_shape=input_shape,
        hidden_dims=(4, 8),
        activation="silu",
        **{"batch_norm": True, **fields},
    )


def _conditional_generator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConditionalGeneratorConfig,
        ConditionalParams,
    )

    return ConditionalGeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=_GAN_IMAGE,
        hidden_dims=(8, 4),
        activation="silu",
        batch_norm=True,
        conditional=ConditionalParams(num_classes=_GAN_CLASSES, embedding_dim=4),
    )


def _conditional_discriminator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConditionalDiscriminatorConfig,
        ConditionalParams,
    )

    return ConditionalDiscriminatorConfig(
        name="discriminator",
        input_shape=_GAN_IMAGE,
        hidden_dims=(4, 8),
        activation="silu",
        dropout_rate=_GAN_DROPOUT,
        conditional=ConditionalParams(num_classes=_GAN_CLASSES, embedding_dim=4),
        kernel_size=(3, 3),
    )


def _cycle_generator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        CycleGANGeneratorConfig,
    )

    # No dropout: any dropout_rate > 0 raises at construction (R453), as it does for the
    # CycleGAN discriminator.
    return CycleGANGeneratorConfig(
        name="generator",
        input_shape=_CYCLE_IMAGE,
        output_shape=_CYCLE_IMAGE,
        hidden_dims=(4, 8),
        n_residual_blocks=1,
        activation="silu",
        batch_norm=True,
    )


def _patch_discriminator(
    input_shape: tuple[int, ...], dropout_rate: float = _GAN_DROPOUT
) -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        PatchGANDiscriminatorConfig,
    )

    return PatchGANDiscriminatorConfig(
        name="discriminator",
        input_shape=input_shape,
        hidden_dims=(4, 8),
        activation="silu",
        batch_norm=True,
        dropout_rate=dropout_rate,
        num_filters=4,
        num_layers=2,
    )


def _gan_config(config_class: type, generator: object, discriminator: object) -> object:
    return config_class(name="gan", generator=generator, discriminator=discriminator)


def cases() -> dict[str, Case]:
    """Return the GAN cases, by the qualified name of the class each builds."""

    from artifex.generative_models.core.configuration.gan_config import (  # noqa: PLC0415
        ConditionalGANConfig,
        CycleGANConfig,
        DCGANConfig,
        GANConfig,
        LSGANConfig,
        WGANConfig,
    )
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        MultiScalePatchGANConfig,
        StyleGAN3DiscriminatorConfig,
        StyleGAN3GeneratorConfig,
    )
    from artifex.generative_models.models.gan import (  # noqa: PLC0415
        base,
        conditional,
        cyclegan,
        dcgan,
        lsgan,
        patchgan,
        stylegan3,
        wgan,
    )

    style_channels = 4
    cycle_config = CycleGANConfig(
        name="cyclegan",
        generator={"a_to_b": _cycle_generator(), "b_to_a": _cycle_generator()},
        discriminator={
            "disc_a": _patch_discriminator(_CYCLE_IMAGE, dropout_rate=0.0),
            "disc_b": _patch_discriminator(_CYCLE_IMAGE, dropout_rate=0.0),
        },
        input_shape_a=_CYCLE_IMAGE,
        input_shape_b=_CYCLE_IMAGE,
    )
    style_image = (_STYLE_RESOLUTION, _STYLE_RESOLUTION, 3)
    # The base configs' validation requires hidden_dims, activation and a shape, which the
    # StyleGAN3 modules never read (their widths follow the resolution).
    unread = {"hidden_dims": (8,), "activation": "leaky_relu"}
    style_generator = StyleGAN3GeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=style_image,
        style_dim=_STYLE_DIM,
        mapping_layers=2,
        img_resolution=_STYLE_RESOLUTION,
        img_channels=3,
        **unread,
    )
    style_discriminator = StyleGAN3DiscriminatorConfig(
        name="discriminator",
        input_shape=style_image,
        img_resolution=_STYLE_RESOLUTION,
        img_channels=3,
        base_channels=4,
        max_channels=8,
        **unread,
    )
    return {
        f"{_GAN}.base.Generator": Case(
            build=lambda r: base.Generator(_dense_generator(), rngs=r),
            inputs=normal(_GAN_LATENT),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.base.Discriminator": Case(
            build=lambda r: base.Discriminator(_dense_discriminator(), rngs=r),
            inputs=normal(*_GAN_FLAT[1:]),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.base.GAN": Case(
            build=lambda r: base.GAN(
                _gan_config(GANConfig, _dense_generator(), _dense_discriminator()), rngs=r
            ),
            inputs=_data_and_latents(_GAN_FLAT[1:]),
            call=_gan_forward_and_samples,
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.dcgan.DCGANGenerator": Case(
            build=lambda r: dcgan.DCGANGenerator(_conv_generator(hidden_dims=(8, 6, 4)), rngs=r),
            inputs=normal(_GAN_LATENT),
            has_batch_stats=True,
        ),
        f"{_GAN}.dcgan.DCGANDiscriminator": Case(
            build=lambda r: dcgan.DCGANDiscriminator(_conv_discriminator(), rngs=r),
            inputs=normal(*_GAN_IMAGE),
            has_batch_stats=True,
        ),
        f"{_GAN}.dcgan.DCGAN": Case(
            build=lambda r: dcgan.DCGAN(
                _gan_config(
                    DCGANConfig,
                    _conv_generator(hidden_dims=(8, 6, 4)),
                    _conv_discriminator(),
                ),
                rngs=r,
            ),
            inputs=_data_and_latents(_GAN_IMAGE),
            call=_gan_forward_and_samples,
            has_batch_stats=True,
        ),
        f"{_GAN}.wgan.WGANGenerator": Case(
            build=lambda r: wgan.WGANGenerator(_conv_generator(_CRITIC_IMAGE), rngs=r),
            inputs=normal(_GAN_LATENT),
            has_batch_stats=True,
        ),
        f"{_GAN}.wgan.WGANDiscriminator": Case(
            build=lambda r: wgan.WGANDiscriminator(
                _conv_discriminator(_CRITIC_IMAGE, batch_norm=False, use_instance_norm=True),
                rngs=r,
            ),
            inputs=normal(*_CRITIC_IMAGE),
        ),
        f"{_GAN}.wgan.WGAN": Case(
            build=lambda r: wgan.WGAN(
                _gan_config(
                    WGANConfig,
                    _conv_generator(_CRITIC_IMAGE),
                    _conv_discriminator(_CRITIC_IMAGE, batch_norm=False, use_instance_norm=True),
                ),
                rngs=r,
            ),
            inputs=_data_and_latents(_CRITIC_IMAGE),
            call=_scores_and_samples,
            has_batch_stats=True,
        ),
        f"{_GAN}.lsgan.LSGANGenerator": Case(
            build=lambda r: lsgan.LSGANGenerator(_conv_generator(), rngs=r),
            inputs=normal(_GAN_LATENT),
            has_batch_stats=True,
        ),
        f"{_GAN}.lsgan.LSGANDiscriminator": Case(
            build=lambda r: lsgan.LSGANDiscriminator(
                _conv_discriminator(dropout_rate=_GAN_DROPOUT), rngs=r
            ),
            inputs=normal(*_GAN_IMAGE),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.lsgan.LSGAN": Case(
            build=lambda r: lsgan.LSGAN(
                _gan_config(
                    LSGANConfig,
                    _conv_generator(),
                    _conv_discriminator(dropout_rate=_GAN_DROPOUT),
                ),
                rngs=r,
            ),
            inputs=_data_and_latents(_GAN_IMAGE),
            call=_scores_and_samples,
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.conditional.ConditionalGenerator": Case(
            build=lambda r: conditional.ConditionalGenerator(_conditional_generator(), rngs=r),
            inputs=lambda key: _labelled_data_and_latents(key)[1:],
            has_batch_stats=True,
        ),
        f"{_GAN}.conditional.ConditionalDiscriminator": Case(
            build=lambda r: conditional.ConditionalDiscriminator(
                _conditional_discriminator(), rngs=r
            ),
            inputs=lambda key: _labelled_data_and_latents(key)[::2],
            has_dropout=True,
        ),
        f"{_GAN}.conditional.ConditionalGAN": Case(
            build=lambda r: conditional.ConditionalGAN(
                _gan_config(
                    ConditionalGANConfig, _conditional_generator(), _conditional_discriminator()
                ),
                rngs=r,
            ),
            inputs=_labelled_data_and_latents,
            call=_conditioned_scores_and_samples,
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.cyclegan.CycleGANGenerator": Case(
            build=lambda r: cyclegan.CycleGANGenerator(_cycle_generator(), rngs=r),
            inputs=normal(*_CYCLE_IMAGE),
            has_batch_stats=True,
        ),
        f"{_GAN}.cyclegan.CycleGANDiscriminator": Case(
            build=lambda r: cyclegan.CycleGANDiscriminator(
                _patch_discriminator(_CYCLE_IMAGE, dropout_rate=0.0), rngs=r
            ),
            inputs=normal(*_CYCLE_IMAGE),
            has_batch_stats=True,
        ),
        f"{_GAN}.cyclegan.CycleGAN": Case(
            build=lambda r: cyclegan.CycleGAN(cycle_config, rngs=r),
            inputs=arrays(_CYCLE_IMAGE, _CYCLE_IMAGE),
            call=_translations,
            has_batch_stats=True,
        ),
        f"{_GAN}.patchgan.PatchGANDiscriminator": Case(
            build=lambda r: patchgan.PatchGANDiscriminator(
                _patch_discriminator(_GAN_IMAGE), rngs=r
            ),
            inputs=normal(*_GAN_IMAGE),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.patchgan.MultiScalePatchGANDiscriminator": Case(
            build=lambda r: patchgan.MultiScalePatchGANDiscriminator(
                MultiScalePatchGANConfig(
                    discriminator=_patch_discriminator(_MULTISCALE_IMAGE), num_discriminators=2
                ),
                rngs=r,
            ),
            inputs=normal(*_MULTISCALE_IMAGE),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.stylegan3.MappingNetwork": Case(
            build=lambda r: stylegan3.MappingNetwork(
                _GAN_LATENT, _STYLE_DIM, num_layers=2, num_ws=2, rngs=r
            ),
            inputs=normal(_GAN_LATENT),
        ),
        f"{_GAN}.stylegan3.StyleModulatedConv": Case(
            build=lambda r: stylegan3.StyleModulatedConv(
                style_channels, style_channels, 3, _STYLE_DIM, rngs=r
            ),
            inputs=arrays((4, 4, style_channels), (_STYLE_DIM,)),
        ),
        f"{_GAN}.stylegan3.SynthesisBlock": Case(
            build=lambda r: stylegan3.SynthesisBlock(
                style_channels, style_channels, _STYLE_DIM, upsample=True, rngs=r
            ),
            inputs=arrays((4, 4, style_channels), (_STYLE_DIM,), (_STYLE_DIM,)),
        ),
        f"{_GAN}.stylegan3.SynthesisNetwork": Case(
            build=lambda r: stylegan3.SynthesisNetwork(_STYLE_DIM, _STYLE_RESOLUTION, 3, rngs=r),
            inputs=normal(2, _STYLE_DIM),
        ),
        f"{_GAN}.stylegan3.StyleGAN3Generator": Case(
            build=lambda r: stylegan3.StyleGAN3Generator(style_generator, rngs=r),
            inputs=normal(_GAN_LATENT),
        ),
        f"{_GAN}.stylegan3.StyleGAN3Discriminator": Case(
            build=lambda r: stylegan3.StyleGAN3Discriminator(style_discriminator, rngs=r),
            inputs=normal(*style_image),
        ),
    }
