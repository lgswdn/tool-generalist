"""Disabled legacy 7D momentum environment.

The config-driven RL path uses ``env_tool.py`` with the SDF policy.  The old
momentum observation layout is intentionally not adapted.
"""

raise RuntimeError(
    "Legacy tool-momentum env is disabled. Use the config-driven tool-sdf-v0 "
    "entrypoint instead."
)
