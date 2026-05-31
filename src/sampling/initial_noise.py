"""Colored initial-noise object.

Duck-typed to ComfyUI's NOISE contract (``.seed`` + ``.generate_noise(input_latent)``).
Mirrors ``comfy.sample.prepare_noise_inner``'s generator-consumption order so that
``batch_index`` subsets are bit-for-bit consistent with a full-batch draw, then applies
spectral coloring to each per-item white tensor (which consumes no extra randomness).
"""
import numpy as np
import torch

from ..engine.colored_noise import ColorParams, scaling_for_progress
from ..engine.spectral import color_tensor
from .. import _log


class Noise_ColoredInitial:
    def __init__(self, seed, params: ColorParams):
        self.seed = seed
        self.params = params

    def _color(self, white_f32, out_dtype):
        # Initial noise has no trajectory; evaluate the (constant-alpha) profile at progress 0.
        # Colored per batch item (batch size 1 here) so each sample gets unit variance and
        # batch_index subsets stay consistent with the per-item draw regime.
        scaling = scaling_for_progress(self.params, 0.0, device=torch.device("cpu"))
        shaped = color_tensor(white_f32, scaling, energy_scale=self.params.energy_scale)
        return shaped.to(out_dtype)

    def _inner(self, latent, generator, noise_inds):
        if noise_inds is None:
            white = torch.randn(
                latent.size(), dtype=torch.float32, layout=latent.layout,
                generator=generator, device="cpu",
            )
            # color each batch item independently (per-item unit variance)
            items = [self._color(white[i:i + 1], latent.dtype) for i in range(white.shape[0])]
            return torch.cat(items, axis=0)

        unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
        noises = []
        for i in range(unique_inds[-1] + 1):
            white = torch.randn(
                [1] + list(latent.size())[1:], dtype=torch.float32, layout=latent.layout,
                generator=generator, device="cpu",
            )
            if i in unique_inds:
                noises.append(self._color(white, latent.dtype))
        noises = [noises[i] for i in inverse]
        return torch.cat(noises, axis=0)

    def generate_noise(self, input_latent):
        latent = input_latent["samples"]
        _log.info("colored initial noise: alpha=%.2f | energy=%.2f | seed=%s",
                  self.params.alpha_start, self.params.energy_scale, self.seed)
        noise_inds = input_latent.get("batch_index", None)
        generator = torch.manual_seed(self.seed)
        if latent.is_nested:  # mirror comfy.sample.prepare_noise: same generator + noise_inds per sub-tensor
            import comfy.nested_tensor as nt
            return nt.NestedTensor([self._inner(t, generator, noise_inds) for t in latent.unbind()])
        return self._inner(latent, generator, noise_inds)
