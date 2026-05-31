# Bundled gamma matrices — provenance

`gamma_matrix_scaled.pt` and `gamma_matrix_scaled_cfg_1.5.pt` are copied verbatim from
[hadardavidson/colored-noise-sampling](https://github.com/hadardavidson/colored-noise-sampling)
(MIT License), directory `gamma_matrix/`.

Each is a float32 tensor of shape **[250 steps, 32 radial frequency bins]**, values in `[0, 1]`,
encoding empirical per-band *structural completion* `γ(step, freq)` for **SiT on ImageNet-256**.

These values are **model- and dataset-specific**. The *shape* of the schedule (low frequencies
resolve early, high frequencies late) is a general property of diffusion models, so the matrices
behave sensibly as a heuristic on other models — but they are **not calibrated** for SD/SDXL/Flux
and will not reproduce the paper's FID figures there. Prefer the **parametric** mode for general
use; use these for experimentation or SiT-like setups.

Paper: Davidson, Issachar & Benaim (2026), *Colored Noise Diffusion Sampling*, arXiv:2605.30332.

To add your own matrices: drop a raw `[steps, bins]` float32 tensor saved with `torch.save`
(`.pt`/`.pth`) into `models/colored_noise_gamma/`; it will appear in the node's dropdown.
