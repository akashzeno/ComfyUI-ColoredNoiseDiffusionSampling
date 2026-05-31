import os
import sys

HERE = os.path.dirname(__file__)
PKG = os.path.dirname(HERE)                          # ComfyUI-ColoredNoiseSampling
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))   # ComfyUI repo root

for p in (COMFY_ROOT, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)
