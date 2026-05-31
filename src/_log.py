"""Package logging.

A named logger (best practice — the app owns handlers/levels). ComfyUI's root formatter shows
only ``%(message)s``, so messages are tagged ``[ColoredNoiseDiffusionSampling]`` to stay identifiable.
INFO = lifecycle (load, each sample, initial noise); WARNING = fallbacks (deduped);
DEBUG = per-step diagnostics (visible with ``--verbose DEBUG``).
"""
import logging

LOGGER = logging.getLogger("ComfyUI-ColoredNoiseDiffusionSampling")
_TAG = "[ColoredNoiseDiffusionSampling]"
_seen_once = set()


def info(msg, *args):
    LOGGER.info(_TAG + " " + msg, *args)


def warning(msg, *args):
    LOGGER.warning(_TAG + " " + msg, *args)


def debug(msg, *args):
    LOGGER.debug(_TAG + " " + msg, *args)


def debug_enabled():
    return LOGGER.isEnabledFor(logging.DEBUG)


def warn_once(key, msg, *args):
    """Emit a WARNING only the first time a given key is seen (avoids per-step spam)."""
    if key in _seen_once:
        return
    _seen_once.add(key)
    LOGGER.warning(_TAG + " " + msg, *args)
