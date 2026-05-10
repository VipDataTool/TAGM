"""ModuleParameter: declarative schema for user-settable measurement parameters.

Each measurement declares its parameters as a class attribute. The framework:
  - Renders them as UI controls (number input, checkbox, dropdown, multi-select)
  - Validates user-submitted values against declared types and ranges
  - Fills in defaults for any parameter the user didn't set
  - Records the resolved parameter values in the MeasurementResult so the
    "no hidden parameters" contract survives across export/reload
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Parameter kinds. "layer_list" is the multi-select whose options are
# dynamically populated from adapter.n_layers(model) at UI-render time.
VALID_KINDS = frozenset({
    "int", "float", "bool", "string",
    "select", "multi_select",
    "layer_list",
})


@dataclass(frozen=True)
class ModuleParameter:
    """A user-settable parameter for a measurement or analysis module.

    Declared as a class-attribute list on the module. The framework uses
    the declarations to (1) render UI, (2) validate, (3) fill defaults,
    (4) record resolved values on the result.
    """
    name: str
    display_name: str
    description: str
    kind: str                        # one of VALID_KINDS
    default: Any = None
    options: tuple = ()              # for "select" / "multi_select"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    advanced: bool = False           # UI hint: hide behind an "advanced" toggle

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"ModuleParameter kind '{self.kind}' not valid. "
                f"Options: {sorted(VALID_KINDS)}")
        if self.kind in ("select", "multi_select") and not self.options:
            raise ValueError(
                f"Parameter '{self.name}' of kind '{self.kind}' requires options")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "kind": self.kind,
            "default": self.default,
            "options": list(self.options),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "advanced": self.advanced,
        }

    def validate(self, value: Any) -> list[str]:
        """Check a user-provided value against this parameter's declaration.

        Returns a list of error strings; empty list means valid.
        """
        errors: list[str] = []
        if self.kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"'{self.name}' must be int, got {type(value).__name__}")
                return errors
            if self.min_value is not None and value < self.min_value:
                errors.append(f"'{self.name}' must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"'{self.name}' must be <= {self.max_value}")
        elif self.kind == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"'{self.name}' must be number")
                return errors
            if self.min_value is not None and value < self.min_value:
                errors.append(f"'{self.name}' must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"'{self.name}' must be <= {self.max_value}")
        elif self.kind == "bool":
            if not isinstance(value, bool):
                errors.append(f"'{self.name}' must be bool")
        elif self.kind == "string":
            if not isinstance(value, str):
                errors.append(f"'{self.name}' must be string")
        elif self.kind == "select":
            if value not in self.options:
                errors.append(f"'{self.name}' must be one of {list(self.options)}, "
                              f"got {value!r}")
        elif self.kind == "multi_select":
            if not isinstance(value, (list, tuple)):
                errors.append(f"'{self.name}' must be list")
                return errors
            for v in value:
                if v not in self.options:
                    errors.append(f"'{self.name}' contains {v!r} not in "
                                  f"{list(self.options)}")
        elif self.kind == "layer_list":
            if not isinstance(value, (list, tuple)):
                errors.append(f"'{self.name}' must be list of layer indices")
                return errors
            for v in value:
                if not isinstance(v, int) or v < 0:
                    errors.append(f"'{self.name}' contains non-layer-index {v!r}")
        return errors


def resolve_parameters(declared: list[ModuleParameter],
                        user_supplied: Optional[dict] = None) -> dict:
    """Merge user-supplied parameter values with declared defaults.

    For each declared parameter, use the user-supplied value if present,
    otherwise the default. Unknown user-supplied keys raise ValueError
    (they indicate a stale UI or a typo).
    """
    user = dict(user_supplied or {})
    declared_names = {p.name for p in declared}

    unknown = set(user) - declared_names
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for this module: {sorted(unknown)}. "
            f"Declared: {sorted(declared_names)}")

    resolved: dict = {}
    errors: list[str] = []
    for p in declared:
        value = user.get(p.name, p.default)
        if value is not None:
            for e in p.validate(value):
                errors.append(e)
        resolved[p.name] = value

    if errors:
        raise ValueError(f"Parameter validation errors: {errors}")

    return resolved
