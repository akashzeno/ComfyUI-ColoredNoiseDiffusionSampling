# ComfyUI-ColoredNoiseDiffusionSampling

Frequency-shaped (**colored**) noise for ComfyUI's stochastic diffusion sampling — a clean,
native port of the *transferable mechanism* behind
[Colored Noise Diffusion Sampling (CNS)](https://github.com/hadardavidson/colored-noise-sampling)
(Davidson, Issachar & Benaim, 2026, arXiv:2605.30332).

Instead of injecting plain white Gaussian noise at each stochastic step, this pack injects noise
whose **power spectrum is shaped** — and whose shape can **vary across the sampling trajectory**
(broadband early → high-frequency detail late). It plugs into ComfyUI's standard
`noise_sampler` seam, so it works with the existing custom-sampling graph and the all-in-one node.

> **Honest scope.** The CNS *technique* ports cleanly and is genuinely useful as a quality/character
> knob. The paper's quantitative results (FID 6.27 vs 8.26) are specific to **SiT on ImageNet-256**
> and will **not** reproduce on SD/SDXL/Flux. The optional γ-matrix mode ships the paper's original
> matrices, but on non-SiT models they are a documented heuristic, not a calibrated schedule. For
> general use, prefer the **parametric** mode.

---

## What it does (in one paragraph)

At each stochastic step it draws white noise, FFTs it, multiplies the spectrum by a
radially-symmetric per-frequency profile, inverse-FFTs, and **renormalizes to unit variance** —
then hands that to the sampler, which applies its own per-step amplitude (`sigma_up · s_noise`).
The per-frequency profile is either a **power-law** `amplitude(f) ∝ f^(−α/2)` (α: 0 white, +1 pink,
+2 brown/red, −1 blue, −2 violet) with a time-varying exponent, or a **γ-matrix residual**
`(1 − γ(progress, f))` reproducing the paper's algorithm.

## Nodes

All three live under the **`sampling/colored_noise`** category.

### 1. Colored Noise Sampler — `ColoredNoise_SamplerSelect`  → `SAMPLER`
The core node. Outputs a `SAMPLER` for `SamplerCustom` / `SamplerCustomAdvanced`. Wraps a chosen
**stochastic** base sampler and colors its per-step noise.

Key inputs: `base_sampler` (all stochastic ancestral/SDE samplers, incl. the `_RF` rectified-flow
variants for Flux/SD3), `eta`, `s_noise`, `mode` (`parametric` | `gamma_matrix`),
`alpha_start`/`alpha_end` + `interpolation` (the time-varying spectral tilt), the γ-matrix controls,
and `energy_scale`.

### 2. Colored Noise (Initial) — `ColoredNoise_Noise`  → `NOISE`
Outputs a `NOISE` object producing **colored initial latent noise** (a single constant color:
white/pink/brown/blue/violet or a custom α). Use it on the `noise` input of `SamplerCustomAdvanced`.
Honors `batch_index` determinism and nested (video) latents.

### 3. Colored Noise KSampler — `ColoredNoise_KSampler`  → `LATENT`
All-in-one (model + conditioning + latent → latent), for when you don't want to wire the custom
sampling graph. Picks a stochastic base sampler + scheduler + steps + cfg + the coloring controls,
plus an optional `color_initial_noise` toggle.

## Two modes

| Mode | What it does | When to use |
|---|---|---|
| **parametric** (default) | Power-law spectrum `f^(−α/2)`, α interpolated `alpha_start → alpha_end` across the trajectory (linear or exponential). Model-agnostic. | Everything. Start here. |
| **gamma_matrix** | Loads a `[steps, bins]` γ matrix and injects noise into the *unresolved* bands per the paper. Bundled SiT/ImageNet matrices included. | Reproducing/experimenting with the paper; SiT-like setups. |

### Quick parametric recipes
- **White** (baseline / sanity): `alpha_start = 0`, `alpha_end = 0`.
- **CNS-like** (broadband → high-freq late): `alpha_start = 0`, `alpha_end = -1` … `-2`.
- **Constant pink/brown** (softer, low-freq emphasis): `alpha_start = alpha_end = 1` … `2`.

`energy_scale` (default `1.0`) scales the noise std *after* unit-variance renorm — a deliberate
"heat" knob; values ≠ 1 intentionally change the effective noise level.

## γ-matrix files

A model folder **`colored_noise_gamma`** is registered automatically. The two bundled matrices
(`gamma_matrix_scaled.pt`, `gamma_matrix_scaled_cfg_1.5.pt`) appear in the dropdown. To add your
own, drop a raw `[steps, bins]` float32 tensor (`torch.save`, `.pt`/`.pth`) into
`models/colored_noise_gamma/`. *(New files may require a ComfyUI restart to appear.)* See
[`gamma_matrices/SOURCE.md`](gamma_matrices/SOURCE.md) for provenance.

## Compatibility

- **Attention backends** — fully compatible with SageAttention 2/3 and FlashAttention 2/3 (and
  xformers / pytorch / split / sub-quad) **by construction**. This pack only shapes the stochastic
  noise term; it never touches the model forward pass or `transformer_options`, where attention is
  selected. Your attention choice is untouched.
- **Memory** — uses ComfyUI's native memory management only. Noise/FFT tensors are transient,
  latent-sized, hold no model references, and do not impede offloading.
- **Precision** — the FFT runs in float32 with a CPU fallback (half-precision FFT is unsupported on
  most backends), then casts back; output matches the latent dtype.
- **Latent ranks** — 4D image `[B,C,H,W]` and 5D video `[B,C,T,H,W]` get a 2D spatial FFT; 3D audio
  `[B,C,L]` gets a 1D time FFT; nested (video) latents are colored per sub-tensor.
- **Samplers** — only **stochastic** samplers are offered (deterministic ones never inject per-step
  noise, so coloring them would be a silent no-op).

## Logging

The pack logs through Python's standard `logging` (named logger `ComfyUI-ColoredNoiseDiffusionSampling`),
so you can see in the ComfyUI console that it's actually running. All messages are tagged
`[ColoredNoiseDiffusionSampling]`.

- **On startup** (once): `[ColoredNoiseDiffusionSampling] loaded: 3 nodes | 22 stochastic base samplers | gamma folder 'colored_noise_gamma' (2 matrix file(s))`
- **Each generation** (per sample): `[ColoredNoiseDiffusionSampling] sampling: base=dpmpp_2m_sde | parametric alpha 0.00->-1.50 (linear) | energy=1.00 | 25 steps | colored noise ACTIVE`
- **Colored initial noise** (when the NOISE node is used): `[ColoredNoiseDiffusionSampling] colored initial noise: alpha=-1.00 | energy=1.00 | seed=...`
- **Warnings** (deduped): e.g. a one-time notice if `torch.fft` falls back to CPU on your backend.
- **Per-step detail** (`DEBUG`): launch ComfyUI with `--verbose DEBUG` to also see per-step
  `progress` and the low/mid/high spectral scaling — useful for tuning, off by default.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. No extra dependencies (torch comes with
ComfyUI). The nodes appear under `sampling/colored_noise`.

## Usage

**All-in-one:** `Load Checkpoint → Colored Noise KSampler → VAE Decode`. Set `base_sampler`
(e.g. `dpmpp_2m_sde`), `mode = parametric`, `alpha_start = 0`, `alpha_end = -1`. Render.

**Custom-sampling graph:**
```
Colored Noise (Initial) ─┐
Colored Noise Sampler ───┤
BasicScheduler ──────────┼─► SamplerCustomAdvanced ─► VAE Decode
BasicGuider ─────────────┘
```

## Development / tests

Pure-engine unit tests (no model weights), run in the `comfyenv` environment:
```
cd custom_nodes/ComfyUI-ColoredNoiseDiffusionSampling
python -m pytest tests/ -q          # spectral shaping, schedule, batch determinism, every-sampler integration, nodes, logging
ruff check .
```

## Credits & license

Method and bundled γ matrices: **Hadar Davidson, Roy Issachar, Sagie Benaim**,
[colored-noise-sampling](https://github.com/hadardavidson/colored-noise-sampling) (MIT),
arXiv:2605.30332. This pack is MIT-licensed; see [LICENSE](LICENSE).
