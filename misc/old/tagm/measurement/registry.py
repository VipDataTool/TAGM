"""Minimal measurement registry stub.

Per-prompt measurements are now computed by tagm.engine.analyzer directly.
This registry provides list_measurements() for the modules UI, returning
the engine's built-in measurements as metadata.
"""

_BUILTIN_MEASUREMENTS = [
    {"name": "stress_score", "display_name": "Stress Score",
     "description": "Per-token attention-side correction stress."},
    {"name": "last_position_attribution", "display_name": "Signed Attribution",
     "description": "Per-token signed correction attribution to last position."},
    {"name": "amplitude_trajectory", "display_name": "Amplitude Trajectory",
     "description": "Correction amplitude across all sublayers."},
    {"name": "lateral_tension_profile", "display_name": "Lateral Tension Profile",
     "description": "Directional structure of the alignment correction field."},
    {"name": "spectral_field_density", "display_name": "Spectral Field Density",
     "description": "Per-token subspace engagement in the QK routing topology."},
    {"name": "rank_displacement", "display_name": "Rank Displacement",
     "description": "Base vs instruct counterfactual ordering divergence."},
]


def list_measurements() -> list[dict]:
    return list(_BUILTIN_MEASUREMENTS)


def find_measurement(name: str):
    match = [m for m in _BUILTIN_MEASUREMENTS if m["name"] == name]
    if not match:
        raise KeyError(f"Unknown measurement: {name}")
    return match[0]
