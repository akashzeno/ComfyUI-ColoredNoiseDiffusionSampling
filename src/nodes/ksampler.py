"""ColoredNoise_KSampler — all-in-one colored-noise sampler (model+conds -> latent).

Mirrors comfy_extras.nodes_custom_sampler.SamplerCustom: computes sigmas internally, builds
the colored sampler, optionally swaps in colored initial noise, runs comfy.sample.sample_custom.
"""
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from comfy_api.latest import io

from ..sampling.registry import stochastic_samplers
from ..sampling.colored_ksampler import build_colored_sampler
from ..sampling.initial_noise import Noise_ColoredInitial
from ..engine.colored_noise import ColorParams
from .sampler_select import resolve_color_params, gamma_choices, default_sampler, NONE


class ColoredNoise_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        samplers = list(stochastic_samplers().keys())
        return io.Schema(
            node_id="ColoredNoise_KSampler",
            display_name="Colored Noise KSampler",
            category="sampling/colored_noise",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("base_sampler", options=samplers, default=default_sampler(samplers)),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("color_initial_noise", default=False,
                                 tooltip="Also color the initial latent noise (uses alpha_start as a constant color)."),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Combo.Input("mode", options=["parametric", "gamma_matrix"], default="parametric"),
                io.Float.Input("alpha_start", default=0.0, min=-8.0, max=8.0, step=0.05, round=False),
                io.Float.Input("alpha_end", default=-1.0, min=-8.0, max=8.0, step=0.05, round=False),
                io.Combo.Input("interpolation", options=["linear", "exponential"], default="linear", advanced=True),
                io.Float.Input("exp_sharpness", default=4.0, min=0.1, max=16.0, step=0.1, round=False, advanced=True),
                io.Combo.Input("gamma_matrix", options=gamma_choices(), default=NONE, advanced=True),
                io.Float.Input("gamma_divider", default=1.0, min=0.1, max=10.0, step=0.01, round=False, advanced=True),
                io.Combo.Input("gamma_shaping", options=["none", "sqrt", "power"], default="none", advanced=True),
                io.Float.Input("power_gamma", default=1.0, min=0.1, max=8.0, step=0.05, round=False, advanced=True),
                io.Float.Input("alpha_tilting", default=0.0, min=-8.0, max=8.0, step=0.05, round=False, advanced=True),
                io.Float.Input("energy_scale", default=1.0, min=0.0, max=4.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent_image, seed, steps, cfg, base_sampler,
                scheduler, denoise, color_initial_noise, eta, s_noise, mode, alpha_start, alpha_end,
                interpolation, exp_sharpness, gamma_matrix, gamma_divider, gamma_shaping,
                power_gamma, alpha_tilting, energy_scale) -> io.NodeOutput:
        params = resolve_color_params(mode, alpha_start, alpha_end, interpolation, exp_sharpness,
                                      gamma_matrix, gamma_divider, gamma_shaping, power_gamma,
                                      alpha_tilting, energy_scale)
        base_fn = stochastic_samplers()[base_sampler]
        sampler = build_colored_sampler(
            base_fn, {"eta": eta, "s_noise": s_noise, "r": 0.5, "solver_type": "midpoint"}, params)

        # sigmas (mirror BasicScheduler denoise handling)
        total_steps = steps if denoise >= 1.0 else max(1, int(steps / max(denoise, 1e-6)))
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]

        latent = latent_image.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
        latent["samples"] = latent_samples

        if color_initial_noise:
            init_params = ColorParams(mode="parametric", alpha_start=alpha_start,
                                      alpha_end=alpha_start, energy_scale=energy_scale)
            noise = Noise_ColoredInitial(seed, init_params).generate_noise(latent)
        else:
            noise = comfy.sample.prepare_noise(latent_samples, seed, latent.get("batch_index", None))

        noise_mask = latent.get("noise_mask", None)
        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative, latent_samples,
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return io.NodeOutput(out, out_denoised)

    sample = execute
