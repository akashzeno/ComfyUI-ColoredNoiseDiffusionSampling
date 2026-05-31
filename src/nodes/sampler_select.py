"""ColoredNoise_SamplerSelect — outputs a SAMPLER that injects per-step colored noise."""
import folder_paths
from comfy_api.latest import io

from ..sampling.registry import stochastic_samplers
from ..sampling.colored_ksampler import build_colored_sampler
from ..engine.colored_noise import ColorParams
from ..engine.schedule import load_gamma_matrix

GAMMA_FOLDER = "colored_noise_gamma"
NONE = "none (parametric)"


def gamma_choices():
    try:
        return [NONE] + folder_paths.get_filename_list(GAMMA_FOLDER)
    except Exception:
        return [NONE]


def resolve_color_params(mode, alpha_start, alpha_end, interpolation, exp_sharpness,
                         gamma_matrix, gamma_divider, gamma_shaping, power_gamma,
                         alpha_tilting, energy_scale):
    """Build a ColorParams, loading the gamma matrix only when gamma mode is active."""
    gm = None
    if mode == "gamma_matrix" and gamma_matrix != NONE:
        path = folder_paths.get_full_path(GAMMA_FOLDER, gamma_matrix)
        if path is None:
            raise FileNotFoundError(f"gamma matrix '{gamma_matrix}' not found in {GAMMA_FOLDER}")
        gm = load_gamma_matrix(path)
    return ColorParams(
        mode=("gamma_matrix" if (mode == "gamma_matrix" and gm is not None) else "parametric"),
        alpha_start=alpha_start, alpha_end=alpha_end, interpolation=interpolation,
        exp_sharpness=exp_sharpness, gamma_matrix=gm, gamma_divider=gamma_divider,
        gamma_shaping=gamma_shaping, power_gamma=power_gamma, alpha_tilting=alpha_tilting,
        energy_scale=energy_scale,
    )


def default_sampler(samplers):
    return "dpmpp_2m_sde" if "dpmpp_2m_sde" in samplers else samplers[0]


class ColoredNoise_SamplerSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        samplers = list(stochastic_samplers().keys())
        return io.Schema(
            node_id="ColoredNoise_SamplerSelect",
            display_name="Colored Noise Sampler",
            category="sampling/colored_noise",
            inputs=[
                io.Combo.Input("base_sampler", options=samplers, default=default_sampler(samplers),
                               tooltip="Stochastic base sampler whose per-step noise is colored."),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False,
                               tooltip="Stochasticity. eta=0 makes the SDE/ancestral step deterministic, "
                                       "which disables colored noise entirely."),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False),
                io.Combo.Input("mode", options=["parametric", "gamma_matrix"], default="parametric"),
                io.Float.Input("alpha_start", default=0.0, min=-8.0, max=8.0, step=0.05, round=False,
                               tooltip="Spectral exponent at the start (high sigma). 0=white, +red/pink, -blue/violet."),
                io.Float.Input("alpha_end", default=-1.0, min=-8.0, max=8.0, step=0.05, round=False,
                               tooltip="Spectral exponent at the end (low sigma). Interpolated over the trajectory."),
                io.Combo.Input("interpolation", options=["linear", "exponential"], default="linear"),
                io.Float.Input("exp_sharpness", default=4.0, min=0.1, max=16.0, step=0.1, round=False, advanced=True),
                io.Combo.Input("gamma_matrix", options=gamma_choices(), default=NONE, advanced=True,
                               tooltip="gamma_matrix mode only. Drop .pt matrices into models/colored_noise_gamma."),
                io.Float.Input("gamma_divider", default=1.0, min=0.1, max=10.0, step=0.01, round=False, advanced=True),
                io.Combo.Input("gamma_shaping", options=["none", "sqrt", "power"], default="none", advanced=True),
                io.Float.Input("power_gamma", default=1.0, min=0.1, max=8.0, step=0.05, round=False, advanced=True),
                io.Float.Input("alpha_tilting", default=0.0, min=-8.0, max=8.0, step=0.05, round=False, advanced=True),
                io.Float.Input("energy_scale", default=1.0, min=0.0, max=4.0, step=0.01, round=False, advanced=True,
                               tooltip="Scales noise std after unit-variance renorm. 1.0 = neutral; >1 injects more 'heat'."),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, base_sampler, eta, s_noise, mode, alpha_start, alpha_end, interpolation,
                exp_sharpness, gamma_matrix, gamma_divider, gamma_shaping, power_gamma,
                alpha_tilting, energy_scale) -> io.NodeOutput:
        base_fn = stochastic_samplers()[base_sampler]
        params = resolve_color_params(mode, alpha_start, alpha_end, interpolation, exp_sharpness,
                                      gamma_matrix, gamma_divider, gamma_shaping, power_gamma,
                                      alpha_tilting, energy_scale)
        # Only the universal SDE knobs are forwarded; each base keeps its own valid default for
        # solver_type/r (forcing solver_type crashed phi_* bases and downgraded the _heun variant).
        # build_colored_sampler filters these to what the chosen base actually accepts.
        sampler = build_colored_sampler(base_fn, {"eta": eta, "s_noise": s_noise}, params)
        return io.NodeOutput(sampler)

    get_sampler = execute
