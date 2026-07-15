"""NavDP safety critic: diffusion trajectory generator + ESDF-fused safety scorer.

Subpackages are intentionally not imported eagerly here, so that e.g.
``navdp_safety.data`` stays usable without the heavier model dependencies.
"""

__all__ = ["data", "engine", "models"]
