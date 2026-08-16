"""Coordinate resolution tools: ``geometry.py``.

Small, pure, and the highest consequence per line in the package: every one of
these functions turns a model's number into a pixel a click lands on, so an error
here does not raise -- it silently aims every action at the wrong place.

``DisplayGeometry``, ``scale_normalized_coordinate`` and
``anthropic_scale_coordinates`` are byte-equal copies from Harbor, kept that way
deliberately so an upstream fix stays a diff.  The tests below pin their
behaviour so "byte-equal" can be checked against something.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from desktop.geometry import (
    ANTHROPIC_MAX_LONG_EDGE,
    ANTHROPIC_MAX_TOTAL_PIXELS,
    DisplayGeometry,
    anthropic_scale_coordinates,
    clamp_to_desktop,
    geometry_from_screen_size,
    resolve_relative,
    scale_normalized_coordinate,
)

FULL_HD = DisplayGeometry(desktop_width=1920, desktop_height=1080)


def _geometry_source() -> str:
    from desktop import geometry

    return Path(geometry.__file__).read_text()


def test_a_bare_vm_has_no_window_inset():
    geometry = geometry_from_screen_size(1920, 1080)
    assert (geometry.desktop_width, geometry.desktop_height) == (1920, 1080)
    assert (geometry.window_x, geometry.window_y) == (0, 0)
    assert (geometry.window_width, geometry.window_height) == (0, 0)


def test_screen_size_is_coerced_to_int():
    geometry = geometry_from_screen_size("1920", 1080.0)  # type: ignore[arg-type]
    assert geometry.desktop_width == 1920 and geometry.desktop_height == 1080


def test_the_geometry_uses_slots_so_a_typo_cannot_add_a_field():
    with pytest.raises(AttributeError):
        FULL_HD.desktop_with = 1920  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ((0, 0), (0, 0)),
        ((999, 999), (1919, 1079)),
        ((500, 500), (960, 540)),
        ((1, 1), (2, 1)),
    ],
)
def test_the_normalized_grid_spans_the_whole_desktop(model, expected):
    """0 maps to the first pixel and 999 to the LAST pixel, not to width."""
    assert scale_normalized_coordinate(model[0], model[1], FULL_HD) == expected


def test_the_normalized_grid_is_monotonic():
    previous = -1
    for value in range(0, 1000, 7):
        x, _ = scale_normalized_coordinate(value, 0, FULL_HD)
        assert x >= previous
        previous = x


def test_a_normalized_coordinate_out_of_range_is_clamped_not_wrapped():
    assert scale_normalized_coordinate(5000, 5000, FULL_HD) == (1919, 1079)
    assert scale_normalized_coordinate(-100, -100, FULL_HD) == (0, 0)


def test_the_normalized_grid_never_leaves_the_framebuffer():
    for geometry in (FULL_HD, DisplayGeometry(1280, 720), DisplayGeometry(3840, 2160)):
        for value in (0, 1, 499, 998, 999):
            x, y = scale_normalized_coordinate(value, value, geometry)
            assert 0 <= x < geometry.desktop_width
            assert 0 <= y < geometry.desktop_height


def test_a_small_desktop_is_a_no_op():
    """For sizes under both limits nothing is rescaled, so the identity must hold
    exactly -- an off-by-one here would shift every click on a small guest."""
    for width, height in ((1024, 768), (1280, 800), (1366, 768)):
        assert width * height <= ANTHROPIC_MAX_TOTAL_PIXELS
        assert max(width, height) <= ANTHROPIC_MAX_LONG_EDGE
        assert anthropic_scale_coordinates(11, 22, width, height) == (11, 22)


def test_a_long_edge_over_the_limit_is_reversed():
    """A size that breaches ONLY the long edge, so that limit is the one applied.

    1920x1080 does not isolate it: 2.07M pixels breaches the total-pixel limit
    too, and that one is stricter, so a test written against 1568/1920 would be
    asserting the wrong scale.
    """
    width, height = 1600, 700
    assert max(width, height) > ANTHROPIC_MAX_LONG_EDGE
    assert width * height <= ANTHROPIC_MAX_TOTAL_PIXELS
    scale = ANTHROPIC_MAX_LONG_EDGE / width
    x, y = anthropic_scale_coordinates(784, 100, width, height)
    assert (x, y) == (int(784 / scale), int(100 / scale))
    assert x > 784, "the reversal must scale UP into desktop space"


def test_a_total_pixel_count_over_the_limit_is_reversed():
    width, height = 1400, 1000  # long edge under 1568, but 1.4M pixels
    assert max(width, height) <= ANTHROPIC_MAX_LONG_EDGE
    assert width * height > ANTHROPIC_MAX_TOTAL_PIXELS
    scale = math.sqrt(ANTHROPIC_MAX_TOTAL_PIXELS / (width * height))
    assert anthropic_scale_coordinates(100, 100, width, height) == (
        int(100 / scale),
        int(100 / scale),
    )


def test_the_stricter_of_the_two_limits_wins():
    """A 4K desktop breaches both; the smaller scale must be the one applied."""
    width, height = 3840, 2160
    long_edge_scale = ANTHROPIC_MAX_LONG_EDGE / width
    total_scale = math.sqrt(ANTHROPIC_MAX_TOTAL_PIXELS / (width * height))
    expected = min(long_edge_scale, total_scale)
    x, _ = anthropic_scale_coordinates(500, 500, width, height)
    assert x == int(500 / expected)


def test_a_portrait_desktop_uses_its_own_long_edge():
    """The long edge is the HEIGHT here; the same scale must come out."""
    landscape = anthropic_scale_coordinates(100, 100, 1600, 700)
    portrait = anthropic_scale_coordinates(100, 100, 700, 1600)
    assert landscape == portrait
    scale = ANTHROPIC_MAX_LONG_EDGE / 1600
    assert portrait == (int(100 / scale), int(100 / scale))


def test_the_reversal_is_the_identity_exactly_at_the_limits():
    assert anthropic_scale_coordinates(5, 5, ANTHROPIC_MAX_LONG_EDGE, 100) == (5, 5)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0, 0), (0, 0)),
        ((1919, 1079), (1919, 1079)),
        ((1920, 1080), (1919, 1079)),
        ((5000, 5000), (1919, 1079)),
        ((-1, -1), (0, 0)),
    ],
)
def test_clamping_keeps_a_point_addressable(point, expected):
    assert clamp_to_desktop(point[0], point[1], FULL_HD) == expected


def test_clamping_coerces_floats_to_int():
    assert clamp_to_desktop(10.7, 20.2, FULL_HD) == (10, 20)


def test_a_relative_delta_resolves_against_the_cursor():
    assert resolve_relative((100, 200), 10, -20, FULL_HD) == (110, 180)
    assert resolve_relative((100, 200), 0, 0, FULL_HD) == (100, 200)


def test_a_relative_delta_is_clamped_at_the_edges():
    assert resolve_relative((10, 10), -100, -100, FULL_HD) == (0, 0)
    assert resolve_relative((1900, 1070), 500, 500, FULL_HD) == (1919, 1079)


def test_a_relative_resolution_from_a_clamped_position_does_not_drift_further():
    """Repeated clamped moves must sit still rather than accumulating."""
    cursor = (1919, 1079)
    for _ in range(5):
        cursor = resolve_relative(cursor, 10, 10, FULL_HD)
    assert cursor == (1919, 1079)


def test_relative_resolution_is_reversible_away_from_the_edges():
    cursor = resolve_relative((500, 500), 37, -21, FULL_HD)
    assert resolve_relative(cursor, -37, 21, FULL_HD) == (500, 500)


def test_the_module_declares_no_coordinate_space_enumeration():
    """Harbor's ``CoordinateSpace`` StrEnum is deliberately NOT copied: which
    convention a model emits is a fact about a codec, consumed before an
    ``Operation`` exists.  A closed enum here would put it back in this package."""
    import ast

    source = _geometry_source()
    assert "CoordinateSpace" in source, "the omission should stay documented"
    tree = ast.parse(source)
    assert [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)] == [
        "DisplayGeometry"
    ]
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "enum" not in imported


def test_the_verbatim_region_is_still_marked():
    """The copy's boundaries are load-bearing for future upstream diffs."""
    source = _geometry_source()
    assert source.count("BEGIN verbatim copy") == 1
    assert source.count("END verbatim copy") == 1
    begin = source.index("BEGIN verbatim copy")
    end = source.index("END verbatim copy")
    verbatim = source[begin:end]
    for name in (
        "class DisplayGeometry",
        "def scale_normalized_coordinate",
        "def anthropic_scale_coordinates",
    ):
        assert name in verbatim, f"{name} drifted out of the verbatim region"
    # ... and the package's own additions stay OUT of it.
    for name in (
        "def clamp_to_desktop",
        "def resolve_relative",
        "def geometry_from_screen_size",
    ):
        assert name not in verbatim
