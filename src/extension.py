"""V3 extension entrypoint + gamma-matrix folder registration."""
import os

import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

GAMMA_FOLDER = "colored_noise_gamma"


def _register_gamma_folder():
    """Register `colored_noise_gamma` so the bundled and user gamma matrices are listed."""
    package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    bundled = os.path.join(package_root, "gamma_matrices")
    user = os.path.join(folder_paths.models_dir, GAMMA_FOLDER)
    paths = [p for p in (bundled, user) if p]
    exts = {".pt", ".pth"}
    if GAMMA_FOLDER in folder_paths.folder_names_and_paths:
        existing_paths, existing_exts = folder_paths.folder_names_and_paths[GAMMA_FOLDER]
        for p in paths:
            if p not in existing_paths:
                existing_paths.append(p)
        existing_exts.update(exts)
    else:
        folder_paths.folder_names_and_paths[GAMMA_FOLDER] = (paths, exts)


_register_gamma_folder()

# Import nodes AFTER folder registration so their Combo option lists populate correctly.
from .nodes.sampler_select import ColoredNoise_SamplerSelect  # noqa: E402
from .nodes.noise import ColoredNoise_Noise  # noqa: E402
from .nodes.ksampler import ColoredNoise_KSampler  # noqa: E402


class ColoredNoiseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColoredNoise_SamplerSelect, ColoredNoise_Noise, ColoredNoise_KSampler]


async def comfy_entrypoint() -> ColoredNoiseExtension:
    return ColoredNoiseExtension()
