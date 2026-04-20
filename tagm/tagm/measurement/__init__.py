"""TAGM measurement layer.

Defines the contract for per-prompt measurement modules. Individual
measurement implementations live in `modules/`; each one declares what
it needs captured and what probe embeddings (if any) it consumes, and
implements `compute()` to produce a MeasurementResult from a RunResult.

Contract summary: the pipeline's CaptureConfig is user-controlled.
Measurements declare a CaptureExpectation (what the capture must provide
for this measurement to run) and consume data from the ActivationStore.
They do not drive capture.
"""
from tagm.measurement.base import MeasurementModule
from tagm.measurement.requirements import (
    CaptureExpectation,
    ExpectationViolation,
    ProbeRequirement,
    ProbeNotAvailableError,
    validate_expectation,
)
from tagm.measurement.result import MeasurementResult, FieldSpec
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import (
    register_measurement,
    find_measurement,
    list_measurements,
)

__all__ = [
    "MeasurementModule",
    "CaptureExpectation",
    "ExpectationViolation",
    "ProbeRequirement",
    "ProbeNotAvailableError",
    "validate_expectation",
    "MeasurementResult",
    "FieldSpec",
    "ModuleParameter",
    "register_measurement",
    "find_measurement",
    "list_measurements",
]
