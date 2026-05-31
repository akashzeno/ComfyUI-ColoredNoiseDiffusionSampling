"""Introspect k-diffusion for stochastic samplers.

A sampler is "stochastic" (and thus colorable) iff its function exposes a ``noise_sampler``
parameter — that is the seam where per-step noise is injected. Deterministic samplers never
call it, so coloring them would be a silent no-op and they are excluded.
"""
import inspect

from comfy.k_diffusion import sampling as kds


def stochastic_samplers():
    """Return ``{base_name: sample_fn}`` for every stochastic sampler, sorted by name.

    ``_gpu`` variants are deduped: they only differ in the noise device, which the node's
    own ``noise_device`` control already governs (we always supply our own noise_sampler).

    Samplers are required to use the standard ``(model, x, sigmas, ...)`` signature; this
    excludes ``dpm_fast``/``dpm_adaptive``, which take ``(model, x, sigma_min, sigma_max, n)``
    and are special-cased by core ``comfy.samplers.ksampler`` for the same reason.
    """
    out = {}
    for name in dir(kds):
        if not name.startswith("sample_"):
            continue
        fn = getattr(kds, name)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "noise_sampler" not in params or "sigmas" not in params:
            continue
        base = name[len("sample_"):]
        if base.endswith("_gpu"):
            continue
        out[base] = fn
    return dict(sorted(out.items()))
