"""Introspect k-diffusion for stochastic samplers.

A sampler is "stochastic" (and thus colorable) iff its function exposes a ``noise_sampler``
parameter — that is the seam where per-step noise is injected. Deterministic samplers never
call it, so coloring them would be a silent no-op and they are excluded.
"""
import inspect

from comfy.k_diffusion import sampling as kds

# Samplers that expose ``noise_sampler`` but inject ZERO stochastic noise: they call the
# underlying solver with eta=0 hardcoded and have no eta knob of their own, so the injected
# noise term is multiplied by sigma_up==0. Coloring them would be a silent no-op, so they are
# excluded. (The ancestral res_multistep variants DO expose eta and are genuinely colorable.)
_NOOP_NOISE = {"res_multistep", "res_multistep_cfg_pp"}


def stochastic_samplers():
    """Return ``{base_name: sample_fn}`` for every colorable stochastic sampler, sorted by name.

    A name qualifies when its function exposes both ``noise_sampler`` and ``sigmas`` parameters
    (the standard ``(model, x, sigmas, ...)`` signature) and is not a known no-op. The ``sigmas``
    requirement excludes ``dpm_fast``/``dpm_adaptive`` (which take ``sigma_min, sigma_max, n``);
    ``_gpu`` variants are deduped because the node always supplies its own noise_sampler (the
    noise device is taken from the latent ``x``, so the gpu/cpu split is irrelevant here).
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
        if base.endswith("_gpu") or base in _NOOP_NOISE:
            continue
        out[base] = fn
    return dict(sorted(out.items()))
