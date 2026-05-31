"""Wrap a base k-diffusion sampler so its per-step noise is colored.

The colored ``noise_sampler`` is built lazily from ``x`` inside the sampler_function (the
only place x's shape/dtype/device are known), then passed to the base sampler, overriding
its default white/Brownian noise. The result is a ``comfy.samplers.KSAMPLER`` — a drop-in
SAMPLER for ``SamplerCustom`` / ``SamplerCustomAdvanced`` / ``sample_custom``.
"""
import inspect

import comfy.samplers

from ..engine.colored_noise import make_colored_noise_sampler


def build_colored_sampler(base_fn, base_opts, params):
    """Return a KSAMPLER that runs ``base_fn`` with a colored noise_sampler.

    ``base_opts`` may contain extra kwargs (eta, s_noise, r, solver_type); only those the
    chosen base actually accepts are forwarded.
    """
    accepted = inspect.signature(base_fn).parameters
    opts = {k: v for k, v in base_opts.items() if k in accepted}

    def sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
        seed = (extra_args or {}).get("seed", None)
        noise_sampler = make_colored_noise_sampler(x, sigmas, params, seed)
        return base_fn(
            model, x, sigmas,
            extra_args=extra_args, callback=callback, disable=disable,
            noise_sampler=noise_sampler, **opts,
        )

    return comfy.samplers.KSAMPLER(sampler_function)
