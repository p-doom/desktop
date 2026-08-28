"""Compile one action into one direct XTEST guest process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import Operation, glide_seconds, scroll_deltas
from .keymap import KEYSYMS, guest_button, guest_key


class ExecutionError(RuntimeError):
    """A guest action could not be compiled, dispatched, or verified."""

    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence


class HeldStateError(ExecutionError):
    """The caller supplied an impossible held-state transition."""


@dataclass
class InputAudit:
    """Host-side record of executed operations and observed held state."""

    operations: list[Operation] = field(default_factory=list)
    held_buttons: set[str] = field(default_factory=set)
    held_keys: set[str] = field(default_factory=set)
    scroll_total: int = 0
    typed_texts: list[str] = field(default_factory=list)


BUTTON_MASKS = {"left": 1 << 8, "middle": 1 << 9, "right": 1 << 10}
BUTTON_NUMBERS = {"left": 1, "middle": 2, "right": 3}
ALL_POINTER_BUTTON_MASK = sum(BUTTON_MASKS.values())
ATOMIC_RESULT_PREFIX = "DESKTOP_ENV_ATOMIC_RESULT="
ATOMIC_SCHEMA_VERSION = 2
CLICK_DWELL_S = 0.05
MOTION_HZ = 60


@dataclass(frozen=True)
class AtomicExecutionResult:
    """Everything one guest process reported about one action."""

    ok: bool
    cursor: tuple[int, int]
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    pointer_button_mask: int
    observed_pointer_button_mask: int
    expected_pointer_button_mask: int
    guest_process_count: int
    guest_returncode: int
    raw_result_marker: str
    cleanup_attempted: bool
    error: str | None
    failure_kind: str | None
    operations: tuple[Operation, ...]
    semantic_operations: tuple[Operation, ...]
    lowered_operations: tuple[Operation, ...]
    held_keys: tuple[str, ...] = ()
    backend_primitives: tuple[dict[str, Any], ...] = ()
    x_injection_evidence: tuple[dict[str, Any], ...] = ()
    keymap_restorations: tuple[dict[str, Any], ...] = ()
    final_pointer_readback: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "cursor": list(self.cursor),
            "cursor_before": list(self.cursor_before),
            "cursor_after": list(self.cursor_after),
            "pointer_button_mask": self.pointer_button_mask,
            "observed_pointer_button_mask": self.observed_pointer_button_mask,
            "expected_pointer_button_mask": self.expected_pointer_button_mask,
            "guest_process_count": self.guest_process_count,
            "guest_returncode": self.guest_returncode,
            "raw_result_marker": self.raw_result_marker,
            "cleanup_attempted": self.cleanup_attempted,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "operations": [item.as_dict() for item in self.operations],
            "semantic_operations": [item.as_dict() for item in self.semantic_operations],
            "lowered_operations": [item.as_dict() for item in self.lowered_operations],
            "held_keys": list(self.held_keys),
            "backend_primitives": list(self.backend_primitives),
            "x_injection_evidence": list(self.x_injection_evidence),
            "keymap_restorations": list(self.keymap_restorations),
            "final_pointer_readback": dict(self.final_pointer_readback),
        }


def expected_atomic_input_state(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
) -> tuple[set[str], set[str]]:
    """Simulate held-state transitions before dispatch."""
    buttons = {guest_button(button) for button in initial_buttons}
    keys = {guest_key(key) for key in initial_keys}
    for operation in operations:
        if operation.kind == "drag":
            if "left" in buttons:
                raise HeldStateError("button already held: left")
        elif operation.kind == "mouse_down":
            button = guest_button(operation.args[0])
            if button in buttons:
                raise HeldStateError(f"button already held: {button}")
            buttons.add(button)
        elif operation.kind == "mouse_up":
            button = guest_button(operation.args[0])
            if button not in buttons:
                raise HeldStateError(f"button not held: {button}")
            buttons.remove(button)
        elif operation.kind == "key_down":
            key = guest_key(operation.args[0])
            if key in keys:
                raise HeldStateError(f"key already held: {key}")
            keys.add(key)
        elif operation.kind == "key_up":
            key = guest_key(operation.args[0])
            if key not in keys:
                raise HeldStateError(f"key not held: {key}")
            keys.remove(key)
    return buttons, keys


def pointer_mask_for_buttons(buttons: set[str]) -> int:
    normalized = {guest_button(button) for button in buttons}
    return sum(BUTTON_MASKS[button] for button in normalized)


def lower_guest_operations(operations: tuple[Operation, ...]) -> tuple[Operation, ...]:
    """Lower only an adjacent same-button press/release to one click."""
    lowered: list[Operation] = []
    index = 0
    while index < len(operations):
        operation = operations[index]
        if operation.kind == "mouse_down" and index + 1 < len(operations):
            following = operations[index + 1]
            if following.kind == "mouse_up" and guest_button(following.args[0]) == guest_button(
                operation.args[0]
            ):
                lowered.append(Operation("click", (guest_button(operation.args[0]),)))
                index += 2
                continue
        lowered.append(operation)
        index += 1
    return tuple(lowered)


def _require_args(operation: Operation, count: int) -> tuple:
    if len(operation.args) != count:
        raise ExecutionError(
            f"{operation.kind} requires {count} arguments, got {operation.args!r}"
        )
    return operation.args


def _validate_text(text: Any) -> str:
    if not isinstance(text, str):
        raise ExecutionError("typing text must be a string")
    for character in text:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ExecutionError("typing text contains a surrogate code point")
        if (codepoint < 0x20 and character not in "\b\t\n\r") or 0x7F <= codepoint < 0xA0:
            raise ExecutionError(f"typing text contains unsupported U+{codepoint:04X}")
    return text


def _compile_rows(operations: tuple[Operation, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operation in operations:
        kind = operation.kind
        if kind in {"move_to", "drag"}:
            count = 2 if kind == "move_to" else 4
            args = _require_args(operation, count)
            rows.append({"kind": kind, "args": [int(value) for value in args]})
        elif kind == "glide_to":
            x, y, seconds = _require_args(operation, 3)
            rows.append({"kind": kind, "args": [int(x), int(y), glide_seconds(seconds)]})
        elif kind in {"click", "mouse_down", "mouse_up"}:
            (raw_button,) = _require_args(operation, 1)
            button = guest_button(raw_button)
            rows.append(
                {"kind": kind, "args": [button], "button_number": BUTTON_NUMBERS[button]}
            )
        elif kind in {"key_down", "key_up"}:
            (raw_key,) = _require_args(operation, 1)
            key = guest_key(raw_key)
            rows.append(
                {"kind": kind, "args": [str(raw_key)], "key": key, "keysym": KEYSYMS[key]}
            )
        elif kind == "scroll":
            dx, dy = scroll_deltas(operation.args)
            rows.append({"kind": kind, "args": [dx, dy]})
        elif kind in {"coalesced_type", "ascii_type"}:
            (raw_text,) = _require_args(operation, 1)
            text = _validate_text(raw_text)
            if kind == "ascii_type":
                try:
                    text.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise ExecutionError("ascii_type received non-ASCII text") from exc
                if "\n" in text or "\r" in text:
                    raise ExecutionError("ascii_type cannot embed Enter; emit a key event")
            rows.append({"kind": kind, "args": [text]})
        elif kind == "wait":
            (seconds,) = _require_args(operation, 1)
            rows.append({"kind": kind, "args": [max(0.0, min(10.0, float(seconds)))]})
        elif kind == "raise_for_test":
            (message,) = _require_args(operation, 1)
            rows.append({"kind": kind, "args": [str(message)]})
        else:
            raise ExecutionError(f"unsupported atomic operation: {kind}")
    return rows


_GUEST_RUNTIME = r"""
import json, math, sys, time as _de_time, traceback
from Xlib import X
from Xlib.display import Display
from Xlib.ext import xtest

_de_display=Display()
if _de_display.query_extension('XTEST') is None:
    raise RuntimeError('XTEST extension unavailable')
_de_root=_de_display.screen().root
_de_trace=[]
_de_primitives=[]
_de_events=[]
_de_keymap_restorations=[]
_de_cleanup=False
_de_error=None
_de_failure_kind=None
_de_observed_mask=-1
_de_observed_keycodes=[]
_de_cursor_before=[-1,-1]
_de_cursor_after=[-1,-1]
_de_event_sequence=0

def _de_pointer_state():
    _de_display.sync()
    _pointer=_de_root.query_pointer()
    return int(_pointer.root_x),int(_pointer.root_y),int(_pointer.mask)&_de_all_button_mask

def _de_key_is_down(bitmap,keycode):
    return bool((bitmap[keycode>>3]>>(keycode&7))&1)

def _de_down_keycodes(keycodes):
    _bitmap=_de_display.query_keymap()
    return sorted(code for code in set(keycodes) if _de_key_is_down(_bitmap,code))

def _de_resolve_keysym(keysym):
    _matches=sorted(
        (int(index),int(code))
        for code,index in _de_display.keysym_to_keycodes(int(keysym))
        if int(index) in (0,1)
    )
    if not _matches:
        raise RuntimeError(f'keysym 0x{int(keysym):x} is not mapped at level 0 or 1')
    _index,_code=_matches[0]
    return _code,_index

_de_keycodes={}
_de_touched_keycodes=[]

def _de_initialize_keycodes():
    global _de_keycodes,_de_touched_keycodes
    _de_keycodes={name:_de_resolve_keysym(keysym) for name,keysym in _de_keysyms.items()}
    _de_touched_keycodes=sorted({code for code,_index in _de_keycodes.values()})

def _de_fake_input(event_type,detail=0,*,x=0,y=0,phase):
    global _de_event_sequence
    _started=_de_time.monotonic_ns()
    xtest.fake_input(_de_display,event_type,detail,x=int(x),y=int(y))
    _de_display.sync()
    _de_event_sequence+=1
    _names={
        X.KeyPress:'key_press',X.KeyRelease:'key_release',
        X.ButtonPress:'button_press',X.ButtonRelease:'button_release',
        X.MotionNotify:'motion_notify',
    }
    _de_events.append({
        'sequence':_de_event_sequence,'event':_names[event_type],
        'event_type':int(event_type),'detail':int(detail),
        'x':int(x) if event_type==X.MotionNotify else None,
        'y':int(y) if event_type==X.MotionNotify else None,
        'phase':phase,'guest_monotonic_ns':_started,
    })

def _de_key_event(name,pressed,*,phase):
    _code,_index=_de_keycodes[name]
    _shift_code=_de_keycodes.get('shiftleft',(0,0))[0]
    _bitmap=_de_display.query_keymap()
    _temporary_shift=_index==1 and not _de_key_is_down(_bitmap,_shift_code)
    if _temporary_shift:
        _de_fake_input(X.KeyPress,_shift_code,phase=phase+'_shift_down')
    _de_fake_input(X.KeyPress if pressed else X.KeyRelease,_code,phase=phase)
    if _temporary_shift:
        _de_fake_input(X.KeyRelease,_shift_code,phase=phase+'_shift_up')

def _de_text_keysym(character):
    if character in ('\n','\r'):
        return 0xff0d
    if character=='\t':
        return 0xff09
    if character=='\b':
        return 0xff08
    _codepoint=ord(character)
    return _codepoint if _codepoint<=0xff else 0x01000000|_codepoint

def _de_tap_keysym(keysym,*,phase):
    _code,_index=_de_resolve_keysym(keysym)
    _shift_code=_de_keycodes.get('shiftleft',(0,0))[0]
    _temporary_shift=_index==1 and not _de_key_is_down(_de_display.query_keymap(),_shift_code)
    if _temporary_shift:
        _de_fake_input(X.KeyPress,_shift_code,phase=phase+'_shift_down')
    _de_fake_input(X.KeyPress,_code,phase=phase+'_down')
    _de_fake_input(X.KeyRelease,_code,phase=phase+'_up')
    if _temporary_shift:
        _de_fake_input(X.KeyRelease,_shift_code,phase=phase+'_shift_up')

def _de_temp_keycode():
    _held=set(_de_down_keycodes(range(256)))
    _modifiers={
        int(code)
        for group in _de_display.get_modifier_mapping()
        for code in group
        if code
    }
    _info=_de_display.display.info
    for _code in range(int(_info.max_keycode),int(_info.min_keycode)-1,-1):
        if _code not in _held and _code not in _modifiers and _code not in _de_touched_keycodes:
            return _code
    raise RuntimeError('no temporary non-modifier keycode is available')

def _de_type_text(text):
    _active_modifiers=[]
    _bitmap=_de_display.query_keymap()
    for _group in _de_display.get_modifier_mapping():
        for _code in _group:
            if (
                _code
                and _de_key_is_down(_bitmap,int(_code))
                and int(_code) not in _active_modifiers
            ):
                _active_modifiers.append(int(_code))
    _keycode=None
    _original=None
    try:
        for _code in reversed(_active_modifiers):
            _de_fake_input(X.KeyRelease,_code,phase='type_modifier_release')
        for _character in text:
            _keysym=_de_text_keysym(_character)
            if ord(_character)<0x100:
                try:
                    _de_tap_keysym(_keysym,phase='type')
                    continue
                except RuntimeError:
                    pass
            if _keycode is None:
                _keycode=_de_temp_keycode()
                _original=tuple(
                    int(value)
                    for value in _de_display.get_keyboard_mapping(_keycode,1)[0]
                )
                if not _original:
                    raise RuntimeError('temporary keycode has an empty mapping tuple')
            _replacement=tuple([int(_keysym)]*len(_original))
            _de_display.change_keyboard_mapping(_keycode,[_replacement])
            _de_display.sync()
            _observed=tuple(
                int(value)
                for value in _de_display.get_keyboard_mapping(_keycode,1)[0]
            )
            if _observed!=_replacement:
                raise RuntimeError('temporary keysym mapping did not take effect')
            _de_fake_input(X.KeyPress,_keycode,phase='unicode_down')
            _de_fake_input(X.KeyRelease,_keycode,phase='unicode_up')
            _de_time.sleep(0.005)
    finally:
        _restore_errors=[]
        if _keycode is not None:
            try:
                if _de_key_is_down(_de_display.query_keymap(),_keycode):
                    _de_fake_input(X.KeyRelease,_keycode,phase='unicode_cleanup')
            except BaseException as _de_exc:
                _restore_errors.append(
                    'temporary key release failed: '
                    +''.join(traceback.format_exception_only(type(_de_exc),_de_exc)).strip()
                )
            _restored=None
            try:
                _de_display.change_keyboard_mapping(_keycode,[_original])
                _de_display.sync()
                _restored=tuple(
                    int(value)
                    for value in _de_display.get_keyboard_mapping(_keycode,1)[0]
                )
            except BaseException as _de_exc:
                _restore_errors.append(
                    'temporary keysym mapping restore failed: '
                    +''.join(traceback.format_exception_only(type(_de_exc),_de_exc)).strip()
                )
            _de_keymap_restorations.append({
                'keycode':_keycode,'original':list(_original),
                'restored':None if _restored is None else list(_restored),
                'exact':_restored==_original,
            })
        for _code in _active_modifiers:
            try:
                if not _de_key_is_down(_de_display.query_keymap(),_code):
                    _de_fake_input(X.KeyPress,_code,phase='type_modifier_restore')
            except BaseException as _de_exc:
                _restore_errors.append(
                    'modifier restore failed: '
                    +''.join(traceback.format_exception_only(type(_de_exc),_de_exc)).strip()
                )
        if _keycode is not None and _restored!=_original:
            _restore_errors.append('temporary keysym mapping restoration drifted')
        if _restore_errors:
            raise RuntimeError('; '.join(_restore_errors))

def _de_move(x,y,*,phase):
    _geometry=_de_root.get_geometry()
    _tx=max(0,min(int(_geometry.width)-1,int(x)))
    _ty=max(0,min(int(_geometry.height)-1,int(y)))
    _de_fake_input(X.MotionNotify,x=_tx,y=_ty,phase=phase)
    return _tx,_ty

def _de_glide(x,y,seconds):
    _before_x,_before_y,_mask=_de_pointer_state()
    _geometry=_de_root.get_geometry()
    _tx=max(0,min(int(_geometry.width)-1,int(x)))
    _ty=max(0,min(int(_geometry.height)-1,int(y)))
    _steps=max(2,int(math.ceil(float(seconds)*_de_motion_hz)))
    _started=_de_time.monotonic_ns()
    for _step in range(1,_steps+1):
        _deadline=_started+int(float(seconds)*1000000000*_step/_steps)
        _remaining=(_deadline-_de_time.monotonic_ns())/1000000000
        if _remaining>0:
            _de_time.sleep(_remaining)
        _x=round(_before_x+(_tx-_before_x)*_step/_steps)
        _y=round(_before_y+(_ty-_before_y)*_step/_steps)
        _de_fake_input(X.MotionNotify,x=_x,y=_y,phase='glide')
    return _tx,_ty,_steps,(_de_time.monotonic_ns()-_started)/1000000000

def _de_button(number,pressed,*,phase):
    _de_fake_input(X.ButtonPress if pressed else X.ButtonRelease,number,phase=phase)

def _de_click(number):
    _de_button(number,True,phase='click_press')
    _de_time.sleep(_de_click_dwell)
    _de_button(number,False,phase='click_release')

def _de_scroll(dx,dy):
    for _number,_count in ((7,max(0,dx)),(6,max(0,-dx)),(4,max(0,dy)),(5,max(0,-dy))):
        for _unused in range(_count):
            _de_button(_number,True,phase='scroll_press')
            _de_button(_number,False,phase='scroll_release')

def _de_apply(row):
    _kind=row['kind']
    _args=row['args']
    if _kind=='move_to':
        _before=_de_pointer_state()[:2]
        _tx,_ty=_de_move(*_args,phase='move')
        _de_primitives.append({
            'kind':_kind,'backend':'python-xlib XTEST',
            'requested_position':_args,'cursor_before':list(_before),
            'cursor_after':[_tx,_ty],'clamped':[_tx,_ty]!=_args,
        })
        _de_trace.append({'kind':_kind,'args':[_tx,_ty]})
    elif _kind=='glide_to':
        _before=_de_pointer_state()[:2]
        _tx,_ty,_steps,_elapsed=_de_glide(*_args)
        _de_primitives.append({
            'kind':_kind,'backend':'python-xlib XTEST',
            'requested_position':_args[:2],'seconds':_args[2],
            'elapsed_seconds':_elapsed,'motion_events':_steps,
            'cursor_before':list(_before),'cursor_after':[_tx,_ty],
            'clamped':[_tx,_ty]!=_args[:2],
        })
        _de_trace.append({'kind':_kind,'args':[_tx,_ty,_args[2]]})
    elif _kind=='drag':
        _start=_de_move(_args[0],_args[1],phase='drag_start')
        _de_button(1,True,phase='drag_press')
        _end=_de_move(_args[2],_args[3],phase='drag_end')
        _de_button(1,False,phase='drag_release')
        _de_primitives.append({
            'kind':_kind,'backend':'python-xlib XTEST',
            'start':list(_start),'end':list(_end),'zero_extent':_start==_end,
        })
        _de_trace.append({'kind':_kind,'args':[*_start,*_end]})
    elif _kind=='click':
        _de_click(row['button_number'])
        _de_primitives.append({
            'kind':_kind,'backend':'python-xlib XTEST','button':_args[0],
            'dwell_ms':int(_de_click_dwell*1000),
            'event_shape':['button_press','button_release'],
        })
        _de_trace.append({'kind':_kind,'args':_args})
    elif _kind in ('mouse_down','mouse_up'):
        _de_button(row['button_number'],_kind=='mouse_down',phase=_kind)
        _de_primitives.append({'kind':_kind,'backend':'python-xlib XTEST','button':_args[0]})
        _de_trace.append({'kind':_kind,'args':_args})
    elif _kind in ('key_down','key_up'):
        _de_key_event(row['key'],_kind=='key_down',phase=_kind)
        _de_primitives.append({
            'kind':_kind,'backend':'python-xlib XTEST','key':_args[0],
            'mapped_key':row['key'],'keysym':row['keysym'],
        })
        _de_trace.append({'kind':_kind,'args':_args})
    elif _kind=='scroll':
        _de_scroll(*_args)
        _de_primitives.append({
            'kind':_kind,'backend':'python-xlib XTEST',
            'dx':_args[0],'dy':_args[1],
        })
        _de_trace.append({'kind':_kind,'args':_args})
    elif _kind in ('coalesced_type','ascii_type'):
        _de_type_text(_args[0])
        _de_primitives.append({
            'kind':_kind,'backend':'python-xlib XTEST',
            'utf8_bytes':len(_args[0].encode('utf-8')),
        })
        _de_trace.append({'kind':_kind,'args':_args})
    elif _kind=='wait':
        _de_time.sleep(_args[0])
        _de_primitives.append({'kind':_kind,'seconds':_args[0]})
        _de_trace.append({'kind':_kind,'args':_args})
    elif _kind=='raise_for_test':
        raise RuntimeError(_args[0])
"""


_GUEST_EPILOGUE = r"""
try:
    _de_bx,_de_by,_de_initial_mask=_de_pointer_state()
    _de_cursor_before=[_de_bx,_de_by]
    if _de_initial_mask!=_de_expected_initial_mask:
        _de_failure_kind='verification'
        raise RuntimeError(
            f'initial pointer button mask {_de_initial_mask} '
            f'!= expected {_de_expected_initial_mask}'
        )
    _de_initial_keycodes=_de_down_keycodes(_de_touched_keycodes)
    if _de_initial_keycodes!=_de_expected_initial_keycodes:
        _de_failure_kind='verification'
        raise RuntimeError(
            f'initial held keycodes {_de_initial_keycodes} '
            f'!= expected {_de_expected_initial_keycodes}'
        )
    for _de_row in _de_rows:
        _de_apply(_de_row)
    _de_cx,_de_cy,_de_observed_mask=_de_pointer_state()
    _de_cursor_after=[_de_cx,_de_cy]
    _de_observed_keycodes=_de_down_keycodes(_de_touched_keycodes)
    if _de_observed_mask!=_de_expected_mask:
        _de_failure_kind='verification'
        raise RuntimeError(
            f'pointer button mask {_de_observed_mask} '
            f'!= expected {_de_expected_mask}'
        )
    if _de_observed_keycodes!=_de_expected_keycodes:
        _de_failure_kind='verification'
        raise RuntimeError(
            f'held keycodes {_de_observed_keycodes} '
            f'!= expected {_de_expected_keycodes}'
        )
except BaseException as _de_exc:
    _de_error=''.join(traceback.format_exception_only(type(_de_exc),_de_exc)).strip()
    if _de_failure_kind is None:
        _de_failure_kind=(
            'injected'
            if any(row['kind']=='raise_for_test' for row in _de_rows)
            else 'infrastructure'
        )
    _de_cleanup=True
    for _de_code in reversed(_de_touched_keycodes):
        try:
            _de_fake_input(X.KeyRelease,_de_code,phase='cleanup_key')
        except BaseException:
            pass
    for _de_number in sorted(_de_touched_button_numbers,reverse=True):
        try:
            _de_button(_de_number,False,phase='cleanup_button')
        except BaseException:
            pass

_de_final_mask=-1
_de_final_keys=[]
_de_final_readback_error=None
try:
    _de_cx,_de_cy,_de_final_mask=_de_pointer_state()
    _de_cursor_after=[_de_cx,_de_cy]
    _de_final_codes=_de_down_keycodes(_de_touched_keycodes)
    _de_final_keys=sorted(
        name
        for name,(code,_index) in _de_keycodes.items()
        if name in _de_expected_keys and code in _de_final_codes
    )
except BaseException as _de_exc:
    _de_final_readback_error=''.join(traceback.format_exception_only(type(_de_exc),_de_exc)).strip()
    if _de_error is None:
        _de_error='final input readback failed: '+_de_final_readback_error
        _de_failure_kind='infrastructure'

_de_payload={
    'ok':_de_error is None,
    'cursor':_de_cursor_after,
    'cursor_before':_de_cursor_before,
    'cursor_after':_de_cursor_after,
    '_de_schema':_de_schema,
    'pointer_button_mask':_de_final_mask,
    'observed_pointer_button_mask':_de_observed_mask,
    'expected_pointer_button_mask':_de_expected_mask,
    'held_keys':_de_final_keys,
    'guest_process_count':1,
    'cleanup_attempted':_de_cleanup,
    'error':_de_error,
    'failure_kind':_de_failure_kind,
    'operations':_de_trace,
    'semantic_operations':_de_semantic_operations,
    'lowered_operations':_de_lowered_operations,
    'backend_primitives':_de_primitives,
    'x_injection_evidence':_de_events,
    'keymap_restorations':_de_keymap_restorations,
    'final_pointer_readback':{'success':_de_final_mask>=0,'mask':_de_final_mask,'error':_de_final_readback_error},
}
print(_de_result_prefix+json.dumps(_de_payload,separators=(',',':')),flush=True)
_de_display.close()
sys.exit(0 if _de_error is None else 1)
"""


def compile_atomic_guest_program(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
) -> tuple[str, int]:
    """Compile one action to one direct python-xlib XTEST process."""
    normalized_initial_buttons = {guest_button(button) for button in initial_buttons}
    normalized_initial_keys = {guest_key(key) for key in initial_keys}
    final_buttons, final_keys = expected_atomic_input_state(
        operations,
        initial_buttons=normalized_initial_buttons,
        initial_keys=normalized_initial_keys,
    )
    lowered = lower_guest_operations(operations)
    rows = _compile_rows(lowered)
    touched_keys = set(normalized_initial_keys)
    touched_buttons = set(normalized_initial_buttons)
    for row in rows:
        if row["kind"] in {"key_down", "key_up"}:
            touched_keys.add(str(row["key"]))
        elif row["kind"] in {"click", "mouse_down", "mouse_up"}:
            touched_buttons.add(str(row["args"][0]))
        elif row["kind"] == "drag":
            touched_buttons.add("left")
    keysyms = {key: KEYSYMS[key] for key in sorted(touched_keys | {"shiftleft"})}
    expected_mask = pointer_mask_for_buttons(final_buttons)
    prelude = "\n".join(
        [
            f"_de_rows={rows!r}",
            f"_de_semantic_operations={[item.as_dict() for item in operations]!r}",
            f"_de_lowered_operations={[item.as_dict() for item in lowered]!r}",
            f"_de_keysyms={keysyms!r}",
            f"_de_expected_keys={sorted(final_keys)!r}",
            f"_de_expected_initial_keys={sorted(normalized_initial_keys)!r}",
            f"_de_expected_mask={expected_mask}",
            f"_de_expected_initial_mask={pointer_mask_for_buttons(normalized_initial_buttons)}",
            "_de_touched_button_numbers="
            f"{sorted(BUTTON_NUMBERS[button] for button in touched_buttons)!r}",
            f"_de_all_button_mask={ALL_POINTER_BUTTON_MASK}",
            f"_de_schema={ATOMIC_SCHEMA_VERSION}",
            f"_de_result_prefix={ATOMIC_RESULT_PREFIX!r}",
            f"_de_click_dwell={CLICK_DWELL_S!r}",
            f"_de_motion_hz={MOTION_HZ}",
            "_de_initialize_keycodes()",
            "_de_expected_initial_keycodes=sorted({"
            "code for name,(code,index) in _de_keycodes.items() "
            "if name in _de_expected_initial_keys})",
            "_de_expected_keycodes=sorted({"
            "code for name,(code,index) in _de_keycodes.items() "
            "if name in _de_expected_keys})",
        ]
    )
    return _GUEST_RUNTIME + "\n" + prelude + "\n" + _GUEST_EPILOGUE, expected_mask
