"""Execute one versioned desktop action from JSON on stdin."""

import json
import math
import sys
import time as _de_time
import traceback

from Xlib import X
from Xlib.display import Display
from Xlib.ext import xtest

_de_display = Display()
if _de_display.query_extension("XTEST") is None:
    raise RuntimeError("XTEST extension unavailable")
_de_root = _de_display.screen().root
_de_trace = []
_de_primitives = []
_de_events = []
_de_keymap_restorations = []
_de_cleanup = False
_de_error = None
_de_failure_kind = None
_de_observed_mask = -1
_de_observed_keycodes = []
_de_cursor_before = [-1, -1]
_de_cursor_after = [-1, -1]
_de_event_sequence = 0


def _de_pointer_state():
    _de_display.sync()
    _pointer = _de_root.query_pointer()
    return int(_pointer.root_x), int(_pointer.root_y), int(_pointer.mask) & _de_all_button_mask


def _de_key_is_down(bitmap, keycode):
    return bool((bitmap[keycode >> 3] >> (keycode & 7)) & 1)


def _de_down_keycodes(keycodes):
    _bitmap = _de_display.query_keymap()
    return sorted(code for code in set(keycodes) if _de_key_is_down(_bitmap, code))


def _de_resolve_keysym(keysym):
    _matches = sorted(
        (int(index), int(code))
        for code, index in _de_display.keysym_to_keycodes(int(keysym))
        if int(index) in (0, 1)
    )
    if not _matches:
        raise RuntimeError(f"keysym 0x{int(keysym):x} is not mapped at level 0 or 1")
    _index, _code = _matches[0]
    return _code, _index


_de_keycodes = {}
_de_touched_keycodes = []


def _de_initialize_keycodes():
    global _de_keycodes, _de_touched_keycodes
    _de_keycodes = {name: _de_resolve_keysym(keysym) for name, keysym in _de_keysyms.items()}
    _de_touched_keycodes = sorted({code for code, _index in _de_keycodes.values()})


def _de_fake_input(event_type, detail=0, *, x=0, y=0, phase):
    global _de_event_sequence
    _started = _de_time.monotonic_ns()
    xtest.fake_input(_de_display, event_type, detail, x=int(x), y=int(y))
    _de_display.sync()
    _de_event_sequence += 1
    _names = {
        X.KeyPress: "key_press",
        X.KeyRelease: "key_release",
        X.ButtonPress: "button_press",
        X.ButtonRelease: "button_release",
        X.MotionNotify: "motion_notify",
    }
    _de_events.append(
        {
            "sequence": _de_event_sequence,
            "event": _names[event_type],
            "event_type": int(event_type),
            "detail": int(detail),
            "x": int(x) if event_type == X.MotionNotify else None,
            "y": int(y) if event_type == X.MotionNotify else None,
            "phase": phase,
            "guest_monotonic_ns": _started,
        }
    )


def _de_key_event(name, pressed, *, phase):
    _code, _index = _de_keycodes[name]
    _shift_code = _de_keycodes.get("shiftleft", (0, 0))[0]
    _bitmap = _de_display.query_keymap()
    _temporary_shift = _index == 1 and not _de_key_is_down(_bitmap, _shift_code)
    if _temporary_shift:
        _de_fake_input(X.KeyPress, _shift_code, phase=phase + "_shift_down")
    _de_fake_input(X.KeyPress if pressed else X.KeyRelease, _code, phase=phase)
    if _temporary_shift:
        _de_fake_input(X.KeyRelease, _shift_code, phase=phase + "_shift_up")


def _de_text_keysym(character):
    if character in ("\n", "\r"):
        return 0xFF0D
    if character == "\t":
        return 0xFF09
    if character == "\b":
        return 0xFF08
    _codepoint = ord(character)
    return _codepoint if _codepoint <= 0xFF else 0x01000000 | _codepoint


def _de_tap_keysym(keysym, *, phase):
    _code, _index = _de_resolve_keysym(keysym)
    _shift_code = _de_keycodes.get("shiftleft", (0, 0))[0]
    _temporary_shift = _index == 1 and not _de_key_is_down(
        _de_display.query_keymap(), _shift_code
    )
    if _temporary_shift:
        _de_fake_input(X.KeyPress, _shift_code, phase=phase + "_shift_down")
    _de_fake_input(X.KeyPress, _code, phase=phase + "_down")
    _de_fake_input(X.KeyRelease, _code, phase=phase + "_up")
    if _temporary_shift:
        _de_fake_input(X.KeyRelease, _shift_code, phase=phase + "_shift_up")


def _de_temp_keycode():
    _held = set(_de_down_keycodes(range(256)))
    _modifiers = {
        int(code) for group in _de_display.get_modifier_mapping() for code in group if code
    }
    _info = _de_display.display.info
    for _code in range(int(_info.max_keycode), int(_info.min_keycode) - 1, -1):
        _mapping = _de_display.get_keyboard_mapping(_code, 1)[0]
        if (
            not any(int(value) for value in _mapping)
            and _code not in _held
            and _code not in _modifiers
            and _code not in _de_touched_keycodes
        ):
            return _code
    raise RuntimeError("no temporary non-modifier keycode is available")


def _de_type_text(text):
    _active_modifiers = []
    _bitmap = _de_display.query_keymap()
    for _group in _de_display.get_modifier_mapping():
        for _code in _group:
            if (
                _code
                and _de_key_is_down(_bitmap, int(_code))
                and int(_code) not in _active_modifiers
            ):
                _active_modifiers.append(int(_code))
    _keycode = None
    _original = None
    try:
        for _code in reversed(_active_modifiers):
            _de_fake_input(X.KeyRelease, _code, phase="type_modifier_release")
        for _character in text:
            _keysym = _de_text_keysym(_character)
            if ord(_character) < 0x100:
                try:
                    _de_tap_keysym(_keysym, phase="type")
                    continue
                except RuntimeError:
                    pass
            if _keycode is None:
                _keycode = _de_temp_keycode()
                _original = tuple(
                    int(value) for value in _de_display.get_keyboard_mapping(_keycode, 1)[0]
                )
                if not _original:
                    raise RuntimeError("temporary keycode has an empty mapping tuple")
            _replacement = tuple([int(_keysym)] * len(_original))
            _de_display.change_keyboard_mapping(_keycode, [_replacement])
            _de_display.sync()
            _observed = tuple(
                int(value) for value in _de_display.get_keyboard_mapping(_keycode, 1)[0]
            )
            if not _observed or _observed[0] != int(_keysym):
                raise RuntimeError("temporary keysym mapping did not take effect")
            _de_fake_input(X.KeyPress, _keycode, phase="unicode_down")
            _de_fake_input(X.KeyRelease, _keycode, phase="unicode_up")
            _de_time.sleep(0.005)
    finally:
        _restore_errors = []
        if _keycode is not None:
            try:
                if _de_key_is_down(_de_display.query_keymap(), _keycode):
                    _de_fake_input(X.KeyRelease, _keycode, phase="unicode_cleanup")
            except BaseException as _de_exc:
                _restore_errors.append(
                    "temporary key release failed: "
                    + "".join(traceback.format_exception_only(type(_de_exc), _de_exc)).strip()
                )
            _restored = None
            try:
                _de_display.change_keyboard_mapping(_keycode, [_original])
                _de_display.sync()
                _restored = tuple(
                    int(value) for value in _de_display.get_keyboard_mapping(_keycode, 1)[0]
                )
            except BaseException as _de_exc:
                _restore_errors.append(
                    "temporary keysym mapping restore failed: "
                    + "".join(traceback.format_exception_only(type(_de_exc), _de_exc)).strip()
                )
            _de_keymap_restorations.append(
                {
                    "keycode": _keycode,
                    "original": list(_original),
                    "restored": None if _restored is None else list(_restored),
                    "exact": _restored == _original,
                }
            )
        for _code in _active_modifiers:
            try:
                if not _de_key_is_down(_de_display.query_keymap(), _code):
                    _de_fake_input(X.KeyPress, _code, phase="type_modifier_restore")
            except BaseException as _de_exc:
                _restore_errors.append(
                    "modifier restore failed: "
                    + "".join(traceback.format_exception_only(type(_de_exc), _de_exc)).strip()
                )
        if _keycode is not None and _restored != _original:
            _restore_errors.append("temporary keysym mapping restoration drifted")
        if _restore_errors:
            raise RuntimeError("; ".join(_restore_errors))


def _de_move(x, y, *, phase):
    _geometry = _de_root.get_geometry()
    _tx = max(0, min(int(_geometry.width) - 1, int(x)))
    _ty = max(0, min(int(_geometry.height) - 1, int(y)))
    _de_fake_input(X.MotionNotify, x=_tx, y=_ty, phase=phase)
    return _tx, _ty


def _de_glide(x, y, seconds):
    _before_x, _before_y, _mask = _de_pointer_state()
    _geometry = _de_root.get_geometry()
    _tx = max(0, min(int(_geometry.width) - 1, int(x)))
    _ty = max(0, min(int(_geometry.height) - 1, int(y)))
    _steps = max(2, int(math.ceil(float(seconds) * _de_motion_hz)))
    _started = _de_time.monotonic_ns()
    for _step in range(1, _steps + 1):
        _deadline = _started + int(float(seconds) * 1000000000 * _step / _steps)
        _remaining = (_deadline - _de_time.monotonic_ns()) / 1000000000
        if _remaining > 0:
            _de_time.sleep(_remaining)
        _x = round(_before_x + (_tx - _before_x) * _step / _steps)
        _y = round(_before_y + (_ty - _before_y) * _step / _steps)
        _de_fake_input(X.MotionNotify, x=_x, y=_y, phase="glide")
    return _tx, _ty, _steps, (_de_time.monotonic_ns() - _started) / 1000000000


def _de_button(number, pressed, *, phase):
    _de_fake_input(X.ButtonPress if pressed else X.ButtonRelease, number, phase=phase)


def _de_click(number):
    _de_button(number, True, phase="click_press")
    _de_time.sleep(_de_click_dwell)
    _de_button(number, False, phase="click_release")


def _de_scroll(dx, dy):
    for _number, _count in (
        (7, max(0, dx)),
        (6, max(0, -dx)),
        (4, max(0, dy)),
        (5, max(0, -dy)),
    ):
        for _unused in range(_count):
            _de_button(_number, True, phase="scroll_press")
            _de_button(_number, False, phase="scroll_release")


def _de_apply(row):
    _kind = row["kind"]
    _args = row["args"]
    if _kind == "move_to":
        _before = _de_pointer_state()[:2]
        _tx, _ty = _de_move(*_args, phase="move")
        _de_primitives.append(
            {
                "kind": _kind,
                "backend": "python-xlib XTEST",
                "requested_position": _args,
                "cursor_before": list(_before),
                "cursor_after": [_tx, _ty],
                "clamped": [_tx, _ty] != _args,
            }
        )
        _de_trace.append({"kind": _kind, "args": [_tx, _ty]})
    elif _kind == "glide_to":
        _before = _de_pointer_state()[:2]
        _tx, _ty, _steps, _elapsed = _de_glide(*_args)
        _de_primitives.append(
            {
                "kind": _kind,
                "backend": "python-xlib XTEST",
                "requested_position": _args[:2],
                "seconds": _args[2],
                "elapsed_seconds": _elapsed,
                "motion_events": _steps,
                "cursor_before": list(_before),
                "cursor_after": [_tx, _ty],
                "clamped": [_tx, _ty] != _args[:2],
            }
        )
        _de_trace.append({"kind": _kind, "args": [_tx, _ty, _args[2]]})
    elif _kind == "drag":
        _start = _de_move(_args[0], _args[1], phase="drag_start")
        _de_button(1, True, phase="drag_press")
        _end = _de_move(_args[2], _args[3], phase="drag_end")
        _de_button(1, False, phase="drag_release")
        _de_primitives.append(
            {
                "kind": _kind,
                "backend": "python-xlib XTEST",
                "start": list(_start),
                "end": list(_end),
                "zero_extent": _start == _end,
            }
        )
        _de_trace.append({"kind": _kind, "args": [*_start, *_end]})
    elif _kind == "click":
        _de_click(row["button_number"])
        _de_primitives.append(
            {
                "kind": _kind,
                "backend": "python-xlib XTEST",
                "button": _args[0],
                "dwell_ms": int(_de_click_dwell * 1000),
                "event_shape": ["button_press", "button_release"],
            }
        )
        _de_trace.append({"kind": _kind, "args": _args})
    elif _kind in ("mouse_down", "mouse_up"):
        _de_button(row["button_number"], _kind == "mouse_down", phase=_kind)
        _de_primitives.append(
            {"kind": _kind, "backend": "python-xlib XTEST", "button": _args[0]}
        )
        _de_trace.append({"kind": _kind, "args": _args})
    elif _kind in ("key_down", "key_up"):
        _de_key_event(row["key"], _kind == "key_down", phase=_kind)
        _de_primitives.append(
            {
                "kind": _kind,
                "backend": "python-xlib XTEST",
                "key": _args[0],
                "mapped_key": row["key"],
                "keysym": row["keysym"],
            }
        )
        _de_trace.append({"kind": _kind, "args": _args})
    elif _kind == "scroll":
        _de_scroll(*_args)
        _de_primitives.append(
            {
                "kind": _kind,
                "backend": "python-xlib XTEST",
                "dx": _args[0],
                "dy": _args[1],
            }
        )
        _de_trace.append({"kind": _kind, "args": _args})
    elif _kind in ("coalesced_type", "ascii_type"):
        _de_type_text(_args[0])
        _de_primitives.append(
            {
                "kind": _kind,
                "backend": "python-xlib XTEST",
                "utf8_bytes": len(_args[0].encode("utf-8")),
            }
        )
        _de_trace.append({"kind": _kind, "args": _args})
    elif _kind == "wait":
        _de_time.sleep(_args[0])
        _de_primitives.append({"kind": _kind, "seconds": _args[0]})
        _de_trace.append({"kind": _kind, "args": _args})
    elif _kind == "raise_for_test":
        raise RuntimeError(_args[0])


_ACTION_CONTRACT = "desktop_actions_v1"
_REQUEST_KEYS = {
    "contract",
    "schema_version",
    "rows",
    "operations",
    "lowered_operations",
    "keysyms",
    "expected_keys",
    "expected_initial_keys",
    "expected_mask",
    "expected_initial_mask",
    "touched_button_numbers",
    "all_button_mask",
    "result_schema_version",
    "click_dwell_seconds",
    "motion_hz",
}
try:
    _de_request = json.load(sys.stdin)
except (UnicodeDecodeError, json.JSONDecodeError) as _de_exc:
    raise ValueError("action request is not valid JSON") from _de_exc
if not isinstance(_de_request, dict) or set(_de_request) != _REQUEST_KEYS:
    raise ValueError("action request has an unexpected shape")
if _de_request["contract"] != _ACTION_CONTRACT or _de_request["schema_version"] != 1:
    raise ValueError("unsupported action protocol")
if (
    not isinstance(_de_request["rows"], list)
    or not all(isinstance(row, dict) for row in _de_request["rows"])
    or not isinstance(_de_request["operations"], list)
    or not isinstance(_de_request["lowered_operations"], list)
    or not isinstance(_de_request["keysyms"], dict)
    or not all(
        isinstance(name, str) and type(keysym) is int
        for name, keysym in _de_request["keysyms"].items()
    )
    or not isinstance(_de_request["expected_keys"], list)
    or not all(isinstance(key, str) for key in _de_request["expected_keys"])
    or not isinstance(_de_request["expected_initial_keys"], list)
    or not all(isinstance(key, str) for key in _de_request["expected_initial_keys"])
    or not isinstance(_de_request["touched_button_numbers"], list)
    or not all(type(number) is int for number in _de_request["touched_button_numbers"])
):
    raise ValueError("action request contains invalid values")
for _de_integer_name in (
    "expected_mask",
    "expected_initial_mask",
    "all_button_mask",
    "result_schema_version",
    "motion_hz",
):
    if type(_de_request[_de_integer_name]) is not int:
        raise ValueError(f"action request {_de_integer_name} must be an integer")
if isinstance(_de_request["click_dwell_seconds"], bool) or not isinstance(
    _de_request["click_dwell_seconds"], (int, float)
):
    raise ValueError("action request click_dwell_seconds must be numeric")

_de_rows = _de_request["rows"]
_de_semantic_operations = _de_request["operations"]
_de_lowered_operations = _de_request["lowered_operations"]
_de_keysyms = _de_request["keysyms"]
_de_expected_keys = _de_request["expected_keys"]
_de_expected_initial_keys = _de_request["expected_initial_keys"]
_de_expected_mask = _de_request["expected_mask"]
_de_expected_initial_mask = _de_request["expected_initial_mask"]
_de_touched_button_numbers = _de_request["touched_button_numbers"]
_de_all_button_mask = _de_request["all_button_mask"]
_de_schema = _de_request["result_schema_version"]
_de_click_dwell = float(_de_request["click_dwell_seconds"])
_de_motion_hz = _de_request["motion_hz"]
_de_initialize_keycodes()
_de_expected_initial_keycodes = sorted(
    {code for name, (code, index) in _de_keycodes.items() if name in _de_expected_initial_keys}
)
_de_expected_keycodes = sorted(
    {code for name, (code, index) in _de_keycodes.items() if name in _de_expected_keys}
)

try:
    _de_bx, _de_by, _de_initial_mask = _de_pointer_state()
    _de_cursor_before = [_de_bx, _de_by]
    if _de_initial_mask != _de_expected_initial_mask:
        _de_failure_kind = "verification"
        raise RuntimeError(
            f"initial pointer button mask {_de_initial_mask} "
            f"!= expected {_de_expected_initial_mask}"
        )
    _de_initial_keycodes = _de_down_keycodes(_de_touched_keycodes)
    if _de_initial_keycodes != _de_expected_initial_keycodes:
        _de_failure_kind = "verification"
        raise RuntimeError(
            f"initial held keycodes {_de_initial_keycodes} "
            f"!= expected {_de_expected_initial_keycodes}"
        )
    for _de_row in _de_rows:
        _de_apply(_de_row)
    _de_cx, _de_cy, _de_observed_mask = _de_pointer_state()
    _de_cursor_after = [_de_cx, _de_cy]
    _de_observed_keycodes = _de_down_keycodes(_de_touched_keycodes)
    if _de_observed_mask != _de_expected_mask:
        _de_failure_kind = "verification"
        raise RuntimeError(
            f"pointer button mask {_de_observed_mask} != expected {_de_expected_mask}"
        )
    if _de_observed_keycodes != _de_expected_keycodes:
        _de_failure_kind = "verification"
        raise RuntimeError(
            f"held keycodes {_de_observed_keycodes} != expected {_de_expected_keycodes}"
        )
except BaseException as _de_exc:
    _de_error = "".join(traceback.format_exception_only(type(_de_exc), _de_exc)).strip()
    if _de_failure_kind is None:
        _de_failure_kind = (
            "injected"
            if any(row["kind"] == "raise_for_test" for row in _de_rows)
            else "infrastructure"
        )
    _de_cleanup = True
    for _de_code in reversed(_de_touched_keycodes):
        try:
            _de_fake_input(X.KeyRelease, _de_code, phase="cleanup_key")
        except BaseException:
            pass
    for _de_number in sorted(_de_touched_button_numbers, reverse=True):
        try:
            _de_button(_de_number, False, phase="cleanup_button")
        except BaseException:
            pass

_de_final_mask = -1
_de_final_keys = []
_de_final_readback_error = None
try:
    _de_cx, _de_cy, _de_final_mask = _de_pointer_state()
    _de_cursor_after = [_de_cx, _de_cy]
    _de_final_codes = _de_down_keycodes(_de_touched_keycodes)
    _de_final_keys = sorted(
        name
        for name, (code, _index) in _de_keycodes.items()
        if name in _de_expected_keys and code in _de_final_codes
    )
except BaseException as _de_exc:
    _de_final_readback_error = "".join(
        traceback.format_exception_only(type(_de_exc), _de_exc)
    ).strip()
    if _de_error is None:
        _de_error = "final input readback failed: " + _de_final_readback_error
        _de_failure_kind = "infrastructure"

_de_payload = {
    "ok": _de_error is None,
    "cursor": _de_cursor_after,
    "cursor_before": _de_cursor_before,
    "cursor_after": _de_cursor_after,
    "schema_version": _de_schema,
    "pointer_button_mask": _de_final_mask,
    "observed_pointer_button_mask": _de_observed_mask,
    "expected_pointer_button_mask": _de_expected_mask,
    "held_keys": _de_final_keys,
    "executor_process_count": 1,
    "cleanup_attempted": _de_cleanup,
    "error": _de_error,
    "failure_kind": _de_failure_kind,
    "operations": _de_trace,
    "lowered_operations": _de_lowered_operations,
    "backend_primitives": _de_primitives,
    "x_injection_evidence": _de_events,
    "keymap_restorations": _de_keymap_restorations,
    "final_pointer_readback": {
        "success": _de_final_mask >= 0,
        "mask": _de_final_mask,
        "error": _de_final_readback_error,
    },
}
print(json.dumps(_de_payload, separators=(",", ":")), flush=True)
_de_display.close()
sys.exit(0 if _de_error is None else 1)
