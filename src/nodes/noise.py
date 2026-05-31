"""ColoredNoise_Noise — outputs a NOISE object that produces colored initial noise."""
from comfy_api.latest import io

from ..sampling.initial_noise import Noise_ColoredInitial
from ..engine.colored_noise import ColorParams

COLOR_ALPHA = {"white": 0.0, "pink": 1.0, "brown": 2.0, "blue": -1.0, "violet": -2.0}


class ColoredNoise_Noise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColoredNoise_Noise",
            display_name="Colored Noise (Initial)",
            category="sampling/colored_noise",
            inputs=[
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Combo.Input("color", options=list(COLOR_ALPHA.keys()) + ["custom"], default="white"),
                io.Float.Input("alpha", default=0.0, min=-8.0, max=8.0, step=0.05, round=False,
                               tooltip="Used when color = custom. PSD ~ 1/f^alpha; +red/pink, -blue/violet."),
                io.Float.Input("energy_scale", default=1.0, min=0.0, max=4.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Noise.Output()],
        )

    @classmethod
    def execute(cls, noise_seed, color, alpha, energy_scale) -> io.NodeOutput:
        a = alpha if color == "custom" else COLOR_ALPHA[color]
        params = ColorParams(mode="parametric", alpha_start=a, alpha_end=a, energy_scale=energy_scale)
        return io.NodeOutput(Noise_ColoredInitial(noise_seed, params))

    get_noise = execute
