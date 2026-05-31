"""ComfyUI-ColoredNoiseSampling — V3 custom-node pack.

This package registers exclusively via the V3 ``comfy_entrypoint``. It deliberately does NOT
define ``NODE_CLASS_MAPPINGS`` — doing so would make ComfyUI's loader take the V1 branch and
never call ``comfy_entrypoint`` (silent failure).
"""
from .src.extension import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
