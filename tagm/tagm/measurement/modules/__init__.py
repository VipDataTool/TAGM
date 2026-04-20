"""Measurement module implementations.

Importing this package triggers side-effect registration of every measurement
via `register_measurement` decorators in each module's source file.
"""
# Import ordering here is not significant — registration is idempotent and
# measurements have no runtime dependency on each other (they share data
# only through the ActivationStore and DeltaStore, not through direct calls).

from tagm.measurement.modules import last_position_attribution  # noqa: F401
from tagm.measurement.modules import stress_score                # noqa: F401
from tagm.measurement.modules import amplitude_trajectory        # noqa: F401
from tagm.measurement.modules import amplitude_derived_metrics   # noqa: F401
from tagm.measurement.modules import lateral_tension_profile     # noqa: F401
from tagm.measurement.modules import spectral_field_density      # noqa: F401
from tagm.measurement.modules import rank_displacement           # noqa: F401
from tagm.measurement.modules import probe_projection            # noqa: F401
from tagm.measurement.modules import per_token_embedding         # noqa: F401
from tagm.measurement.modules import backscatter_projection      # noqa: F401
