"""Unit coverage for per-run replay overrides and the explicit scene fingerprint."""

from dataclasses import replace
import json

import pytest

from GREENFIELD.replay.overrides import (
    ReplayOverrides,
    apply_settings_json,
    reconstruction_fingerprint,
    settings_json,
)
from GREENFIELD.replay.settings import load_settings


def test_default_overrides_reproduce_tracked_settings_exactly():
    """An empty ReplayOverrides must be a no-op against the tracked config."""
    tracked = load_settings()

    assert ReplayOverrides().apply(tracked) == tracked


def test_override_defaults_match_tracked_config_values():
    """Literal defaults and CONFIGS/replay.yaml must not drift apart."""
    tracked = load_settings()

    assert ReplayOverrides.from_settings(tracked) == ReplayOverrides()


def test_apply_only_changes_the_named_knobs():
    """Applying overrides must leave paths and unmentioned knobs untouched."""
    tracked = load_settings()

    effective = ReplayOverrides(gaussian_steps=99, anyview_steps=7).apply(tracked)

    assert effective.gaussian_steps == 99
    assert effective.anyview_steps == 7
    assert effective.fps == tracked.fps
    assert effective.depth_checkpoint == tracked.depth_checkpoint


def test_json_round_trip_preserves_values():
    """Workers rebuild the same overrides from the serialized payload."""
    overrides = ReplayOverrides(gaussian_steps=42, gaussian_background_r=0.25)

    assert ReplayOverrides.from_json(overrides.to_json()) == overrides


@pytest.mark.parametrize(
    "bad",
    [
        ReplayOverrides(fps=0),
        ReplayOverrides(max_side=8),
        ReplayOverrides(anyview_steps=0),
        ReplayOverrides(depth_input_size=393),  # not a multiple of the 14px patch size
        ReplayOverrides(depth_input_size=111),
        ReplayOverrides(gaussian_stride=0),
        ReplayOverrides(gaussian_steps=0),
        ReplayOverrides(gaussian_lr_xyz=0.0),
        ReplayOverrides(gaussian_lr_color=2.0),
        ReplayOverrides(gaussian_loss_depth=-1.0),
        ReplayOverrides(gaussian_init_opacity_logit=99.0),
        ReplayOverrides(gaussian_background_r=1.5),
        ReplayOverrides(anyview_guidance_scale=-0.1),
    ],
)
def test_validation_rejects_out_of_range_knobs(bad):
    """Out-of-range values fail loudly instead of reaching a GPU worker."""
    with pytest.raises(ValueError):
        bad.validate()


# Every integer-typed knob with an in-range integral value, used to prove the
# boundary rejects fractional/bool/string input without lowering each case by hand.
_INTEGER_FIELDS = {
    "fps": 12.0,
    "max_side": 576.0,
    "anyview_steps": 20.0,
    "depth_input_size": 392.0,
    "gaussian_stride": 6.0,
    "gaussian_steps": 260.0,
}


@pytest.mark.parametrize("field, integral", _INTEGER_FIELDS.items())
def test_validate_rejects_fractional_integer_field(field, integral):
    """A raw UI float like 12.5 must fail, not silently truncate to 12."""
    with pytest.raises(ValueError, match=field):
        ReplayOverrides(**{field: integral + 0.5}).validate()


@pytest.mark.parametrize("field", list(_INTEGER_FIELDS))
@pytest.mark.parametrize("bad", [True, "12"])
def test_validate_rejects_bool_and_string_integer_fields(field, bad):
    """Bools and numeric strings are not integers and must name the field."""
    with pytest.raises(ValueError, match=field):
        ReplayOverrides(**{field: bad}).validate()


@pytest.mark.parametrize("field, integral", _INTEGER_FIELDS.items())
def test_validate_normalizes_integral_float_to_int(field, integral):
    """12.0 is accepted and applied as int 12 everywhere, with no loss."""
    tracked = load_settings()
    overrides = ReplayOverrides(**{field: integral})

    overrides.validate()
    normalized = overrides.as_dict()
    assert normalized[field] == int(integral)
    assert type(normalized[field]) is int

    applied = overrides.apply(tracked)
    assert getattr(applied, field) == int(integral)
    assert type(getattr(applied, field)) is int


def test_validate_still_accepts_fractional_float_fields():
    """Genuinely float-typed knobs keep accepting fractional values."""
    overrides = ReplayOverrides(gaussian_lr_xyz=0.5, gaussian_background_g=0.5)

    overrides.validate()
    applied = overrides.apply(load_settings())

    assert applied.gaussian_lr_xyz == 0.5
    assert applied.gaussian_background_g == 0.5


def test_validate_rejects_non_numeric_float_field():
    """A numeric string in a float field fails instead of being coerced."""
    with pytest.raises(ValueError, match="gaussian_lr_xyz"):
        ReplayOverrides(gaussian_lr_xyz="0.003").validate()


def test_settings_json_serializes_effective_values():
    """The worker payload carries the fallback values for omitted knobs."""
    tracked = load_settings()
    effective = ReplayOverrides(gaussian_steps=77).apply(tracked)

    payload = json.loads(settings_json(effective))

    assert payload["gaussian_steps"] == 77
    assert payload["gaussian_lr_color"] == tracked.gaussian_lr_color
    assert payload["fps"] == tracked.fps


def test_settings_json_normalizes_integral_float_in_hand_built_settings():
    """A hand-built Settings with an integral float serializes as the normalized int."""
    hand_built = replace(load_settings(), fps=12.0)

    payload = json.loads(settings_json(hand_built))

    assert payload["fps"] == 12
    assert type(payload["fps"]) is int


def test_settings_json_rejects_unknown_keys():
    """A key that is not a ReplayOverrides field must fail loudly and name itself."""
    with pytest.raises(ValueError, match="unknown key.*not_a_knob"):
        apply_settings_json(json.dumps({"gaussian_steps": 10, "not_a_knob": 1}))


@pytest.mark.parametrize("payload", ["[]", "42", '"fps"', "null", "true"])
def test_settings_json_requires_a_json_object(payload):
    """Lists, scalars, and null are not override objects and must be rejected."""
    with pytest.raises(ValueError, match="JSON object"):
        apply_settings_json(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"fps": "12"},          # string would silently int()-coerce
        {"fps": True},          # bool is an int subclass
        {"gaussian_steps": 12.5},   # float would silently truncate to an int
        {"gaussian_lr_xyz": "0.003"},  # string would silently float()-coerce
        {"gaussian_lr_xyz": True},  # bool is a number subclass
    ],
)
def test_settings_json_rejects_silent_type_coercion(payload):
    """Only the field's exact numeric type is accepted; nothing coerces silently."""
    with pytest.raises(ValueError):
        apply_settings_json(json.dumps(payload))


def test_settings_json_rejects_out_of_range_before_apply():
    """A bad payload fails during parsing, before any settings are applied."""
    with pytest.raises(ValueError, match="gaussian_steps"):
        apply_settings_json(json.dumps({"gaussian_steps": 0}))


def test_settings_json_missing_keys_fall_back_to_tracked_settings():
    """A partial payload resolves omitted keys against CONFIGS/replay.yaml."""
    tracked = load_settings()

    effective = apply_settings_json(json.dumps({"gaussian_steps": 123}))

    assert effective.gaussian_steps == 123
    assert effective.fps == tracked.fps
    assert effective.gaussian_lr_color == tracked.gaussian_lr_color


def test_settings_json_missing_keys_use_the_supplied_base_not_literal_defaults():
    """The parser inherits a supplied base rather than the dataclass defaults."""
    base = ReplayOverrides(fps=5, gaussian_steps=50)

    parsed = ReplayOverrides.from_json(json.dumps({"max_side": 100}), base=base)

    assert parsed.fps == 5
    assert parsed.gaussian_steps == 50
    assert parsed.max_side == 100
    # Literal defaults would have produced fps=12/gaussian_steps=260.
    assert parsed.fps != ReplayOverrides().fps


def test_reconstruction_fingerprint_tracks_every_reconstruction_knob():
    """Changing any reconstruction value must change the scene fingerprint."""
    tracked = load_settings()
    baseline = reconstruction_fingerprint(tracked)

    assert reconstruction_fingerprint(tracked) == baseline
    for name, value in (
        ("anyview_steps", 7),
        ("depth_input_size", 336),
        ("gaussian_stride", 8),
        ("gaussian_steps", 99),
        ("gaussian_lr_xyz", 0.01),
        ("gaussian_lr_rotation", 0.01),
        ("gaussian_lr_scale", 0.01),
        ("gaussian_lr_color", 0.02),
        ("gaussian_lr_opacity", 0.02),
        ("gaussian_loss_depth", 0.5),
        ("gaussian_loss_smoothness", 0.5),
        ("gaussian_init_opacity_logit", 1.0),
        ("gaussian_background_r", 0.1),
        ("gaussian_background_g", 0.1),
        ("gaussian_background_b", 0.1),
    ):
        changed = reconstruction_fingerprint(replace(tracked, **{name: value}))
        assert changed != baseline, name


def test_reconstruction_fingerprint_is_order_independent():
    """Equal knob values must hash equally regardless of construction order."""
    tracked = load_settings()
    first = ReplayOverrides(gaussian_steps=100, gaussian_lr_xyz=0.01).apply(tracked)
    second = ReplayOverrides(gaussian_lr_xyz=0.01, gaussian_steps=100).apply(tracked)

    assert reconstruction_fingerprint(first) == reconstruction_fingerprint(second)
