"""Per-run overrides for every user-editable 3D Video reconstruction knob.

The tracked ``CONFIGS/replay.yaml`` values stay the source of truth for defaults.
A :class:`ReplayOverrides` is the boundary object the UI passes in: it is parsed
once by :meth:`ReplayOverrides.validate`, then applied to the frozen ``Settings``.
Workers rebuild the same settings from :func:`settings_json`, so a direct worker
invocation without ``--settings-json`` still reproduces the tracked defaults.
"""

from dataclasses import dataclass, fields, replace
from hashlib import sha256
import json

from .settings import Settings, load_settings


@dataclass(frozen=True)
class ReplayOverrides:
    """Every replay knob the UI may change without editing tracked config.

    Field names mirror ``Settings`` exactly so :meth:`apply` is a single
    ``dataclasses.replace`` and workers can rebuild settings from JSON. Defaults
    are the current effective values (see ``CONFIGS/replay.yaml``); the test
    suite asserts they match to catch drift.
    """

    # Capture
    fps: int = 12
    max_side: int = 576
    # Reconstruction
    anyview_steps: int = 20
    depth_input_size: int = 392
    gaussian_stride: int = 6
    gaussian_steps: int = 260
    # Gaussian training
    gaussian_lr_xyz: float = 0.003
    gaussian_lr_rotation: float = 0.003
    gaussian_lr_scale: float = 0.003
    gaussian_lr_color: float = 0.015
    gaussian_lr_opacity: float = 0.01
    gaussian_lr_motion: float = 0.002
    gaussian_loss_depth: float = 0.002
    gaussian_loss_smoothness: float = 0.01
    gaussian_init_opacity_logit: float = 2.2
    gaussian_background_r: float = 0.5
    gaussian_background_g: float = 0.68
    gaussian_background_b: float = 0.86
    # Implicit
    anyview_guidance_scale: float = 0.0

    def validate(self) -> None:
        """Fail loudly when a knob cannot be honored by the workers.

        All type and range checks live in :meth:`_normalized_dict`, so the value
        proven here is exactly the value :meth:`apply` writes and :meth:`as_dict`
        serializes. ``depth_input_size`` must additionally be a multiple of the
        Video Depth Anything patch size (14); any other value silently changes
        the encoder grid.
        """
        self._normalized_dict()

    def apply(self, settings: Settings) -> Settings:
        """Return a copy of tracked settings with every validated override applied."""
        return replace(settings, **self._normalized_dict())

    def as_dict(self) -> dict:
        """Return the validated, normalized overrides as a plain mapping."""
        return self._normalized_dict()

    def _normalized_dict(self) -> dict:
        """Return every field proven and normalized at the override boundary.

        The Gradio controls hand raw values straight to ``ReplayOverrides``, so
        this is the single place that decides what a knob's value must be.
        Integer fields reject ``bool``, non-numeric values, and fractional floats
        such as ``12.5``; an integral float such as ``12.0`` is converted
        losslessly to ``int`` 12. Range checks run on those normalized values, so
        the worker can never receive a knob validated as one type but applied as
        another.
        """
        values = {
            field.name: _require_typed(field.name, getattr(self, field.name), field.type)
            for field in fields(self)
        }
        _require_range("fps", values["fps"], 1, 60)
        _require_range("max_side", values["max_side"], 64, 2160)
        _require_range("anyview_steps", values["anyview_steps"], 1, 200)
        if not 112 <= values["depth_input_size"] <= 1568 or values["depth_input_size"] % 14:
            raise ValueError(
                "depth_input_size must be a multiple of 14 between 112 and 1568 "
                "(Video Depth Anything patch size)."
            )
        _require_range("gaussian_stride", values["gaussian_stride"], 1, 64)
        _require_range("gaussian_steps", values["gaussian_steps"], 1, 100_000)
        for name in (
            "gaussian_lr_xyz", "gaussian_lr_rotation", "gaussian_lr_scale",
            "gaussian_lr_color", "gaussian_lr_opacity", "gaussian_lr_motion",
        ):
            _require_range(name, values[name], 1e-6, 1.0)
        for name in ("gaussian_loss_depth", "gaussian_loss_smoothness"):
            _require_range(name, values[name], 0.0, 1000.0)
        _require_range("gaussian_init_opacity_logit", values["gaussian_init_opacity_logit"], -20.0, 20.0)
        for name in ("gaussian_background_r", "gaussian_background_g", "gaussian_background_b"):
            _require_range(name, values[name], 0.0, 1.0)
        _require_range("anyview_guidance_scale", values["anyview_guidance_scale"], 0.0, 100.0)
        return values

    def to_json(self) -> str:
        """Serialize with sorted keys so equal overrides produce equal payloads."""
        return json.dumps(self.as_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str, base: "ReplayOverrides | None" = None) -> "ReplayOverrides":
        """Strictly parse a ``--settings-json`` payload into validated overrides.

        A payload is the untrusted boundary where a malformed UI call or a hand-
        typed worker command could otherwise inject a half-broken state. We fail
        loudly here so nothing invalid ever reaches a GPU worker: the payload must
        be a JSON object, every key must name a real override field, and every
        value must match the numeric type the field declares (an integral float
        such as ``12.0`` is normalized to ``int``; a fractional one is rejected).
        ``base`` supplies
        the values for omitted keys -- callers pass the worker's tracked settings
        so a partial payload resolves against ``CONFIGS/replay.yaml`` instead of
        the literal class defaults. Calling :meth:`validate` before returning
        means an out-of-range knob fails during parsing, never mid-reconstruction.
        """
        payload = _parse_settings_object(raw)
        expected = {field.name: field.type for field in fields(cls)}
        unknown = sorted(set(payload) - set(expected))
        if unknown:
            raise ValueError("--settings-json contains unknown key(s): " + ", ".join(unknown) + ".")
        values = {
            name: _require_typed(name, value, expected[name])
            for name, value in payload.items()
        }
        if base is not None:
            # Omitted keys inherit from the base (the worker's tracked settings).
            # Only a base-less direct parse falls back to the literal defaults.
            for name, value in base.as_dict().items():
                values.setdefault(name, value)
        parsed = cls(**values)
        parsed.validate()
        return parsed

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReplayOverrides":
        """Read the exposed knobs back out of a (possibly already overridden) Settings."""
        return cls(**{field.name: getattr(settings, field.name) for field in fields(cls)})


# Fields that define an explicit Gaussian scene. Kept in one place so the cache
# key and the fingerprint can never silently disagree about what matters. The
# implicit step count is included conservatively: it is part of the "reconstruction"
# group, and a scene built with different worker settings must not be reused.
_RECONSTRUCTION_FIELDS = (
    "anyview_steps",
    "depth_input_size",
    "gaussian_stride",
    "gaussian_steps",
    "gaussian_lr_xyz",
    "gaussian_lr_rotation",
    "gaussian_lr_scale",
    "gaussian_lr_color",
    "gaussian_lr_opacity",
    "gaussian_lr_motion",
    "gaussian_loss_depth",
    "gaussian_loss_smoothness",
    "gaussian_init_opacity_logit",
    "gaussian_background_r",
    "gaussian_background_g",
    "gaussian_background_b",
)


def reconstruction_fingerprint(settings: Settings) -> str:
    """Hash every reconstruction value that a cached explicit scene depends on.

    Sorted keys make the fingerprint independent of dict/field ordering, and the
    digest is trimmed for use as a cache-key component.
    """
    payload = {name: getattr(settings, name) for name in _RECONSTRUCTION_FIELDS}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def settings_json(settings: Settings) -> str:
    """Serialize the effective exposed values for a worker subprocess.

    Reads from ``settings`` rather than the literal defaults so a worker always
    receives exactly the effective run configuration, including tracked-config
    values that a future edit may change. Serialization goes through the same
    ``ReplayOverrides.from_settings(...).to_json()`` boundary as :meth:`apply`
    and :meth:`as_dict`, so a hand-built ``Settings`` with an integral float
    (e.g. ``fps=12.0``) is normalized to ``12`` before the worker sees it.
    """
    return ReplayOverrides.from_settings(settings).to_json()


def apply_settings_json(raw: str | None) -> Settings:
    """Resolve a worker payload into validated settings, defaulting to tracked config.

    A worker invoked directly (no ``--settings-json``) must reproduce tracked
    defaults exactly, so ``None`` short-circuits to ``load_settings()``. A JSON
    payload is parsed and validated against the same tracked settings it is
    applied to, so omitted keys and range checks always agree with
    ``CONFIGS/replay.yaml``.
    """
    tracked = load_settings()
    if raw is None:
        return tracked
    return ReplayOverrides.from_json(raw, base=ReplayOverrides.from_settings(tracked)).apply(tracked)


def _parse_settings_object(raw: str) -> dict:
    """Decode a payload and reject anything that is not a JSON object mapping."""
    if not isinstance(raw, str):
        raise ValueError("--settings-json must be a JSON object string.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"--settings-json is not valid JSON: {error}.") from error
    if not isinstance(payload, dict):
        raise ValueError("--settings-json must be a JSON object of override keys and values.")
    return payload


def _require_typed(name: str, value: object, expected: type) -> int | float:
    """Prove ``value`` matches a field's declared numeric type, never coercing lossily.

    Integer fields accept a ``bool``-free ``int`` or an integral ``float`` (e.g.
    ``12.0`` becomes ``int`` 12). A fractional float (``12.5``), a ``bool``, or a
    non-numeric value raises with the field name. Float fields accept any real
    number and normalize to ``float``. This is the shared type rule for both the
    in-process UI boundary and the ``--settings-json`` worker boundary.
    """
    if expected is int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a whole number, not a boolean (got {value!r}).")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"{name} must be a whole number (got {value!r}).")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number (got {value!r}).")
    return float(value)


def _require_range(name: str, value: float, low: float, high: float) -> None:
    """Guard helper shared by validate() so each range reads as one sentence."""
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g} (got {value!r}).")
