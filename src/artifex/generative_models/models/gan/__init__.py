"""GAN models package."""

from .base import Discriminator, GAN, Generator
from .conditional import (
    ConditionalDiscriminator,
    ConditionalGAN,
    ConditionalGenerator,
)
from .cyclegan import CycleGAN, CycleGANDiscriminator, CycleGANGenerator
from .dcgan import DCGAN, DCGANDiscriminator, DCGANGenerator
from .lsgan import LSGAN, LSGANDiscriminator, LSGANGenerator
from .patchgan import MultiScalePatchGANDiscriminator, PatchGANDiscriminator
from .wgan import compute_gradient_penalty, WGAN, WGANDiscriminator, WGANGenerator


__all__ = [
    "Discriminator",
    "GAN",
    "Generator",
    "ConditionalDiscriminator",
    "ConditionalGAN",
    "ConditionalGenerator",
    "CycleGAN",
    "CycleGANDiscriminator",
    "CycleGANGenerator",
    "DCGAN",
    "DCGANDiscriminator",
    "DCGANGenerator",
    "LSGAN",
    "LSGANDiscriminator",
    "LSGANGenerator",
    "MultiScalePatchGANDiscriminator",
    "PatchGANDiscriminator",
    "WGAN",
    "WGANDiscriminator",
    "WGANGenerator",
    "compute_gradient_penalty",
]
