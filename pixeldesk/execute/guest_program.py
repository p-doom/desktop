"""Compile a tuple of ``Operation``s into exactly ONE ordered guest process.

What one process per action buys:

  * **One process per action.**  Not one per event.  A press, a move, and a
    release in one action cannot be interleaved with anything else, cannot pay
    three interpreter startups, and cannot half-apply if the middle event fails.
  * **Held state is verified, not assumed.**  The program reads the X11 pointer
    mask before and after and refuses to report success if the observed mask
    disagrees with the mask its own operation list implies.
  * **Failure always cleans up.**  Any exception releases every key the program
    touched and every pointer button, so a crashed action cannot leave the guest
    with ``ctrl`` stuck down for the rest of the episode.
  * **Unicode typing actually works.**  ``pyautogui.write`` drops every
    character outside printable ASCII on the pinned image, so ``coalesced_type``
    falls back to a GTK clipboard paste for those payloads -- and only those,
    because that paste is itself a silent no-op in a terminal.  One predicate,
    ``coalesced_type_mechanism``, decides; ``compile_unicode_coalesced_type``'s
    docstring has the reason.
  * **The click primitive is switchable.**  ``CLICK_BACKENDS`` exists because the
    release-side ``MotionNotify`` that PyAutoGUI emits is an *observable*
    difference to some toolkits; the two backends emit an identical press-side
    stream and differ only in that event.

THERE IS NO RELATIVE MOVE KIND.  Resolving a delta against
``pyautogui.position()`` inside the guest would put a coordinate convention in
the executor, which this package forbids; relative grammars resolve host-side in
``Codec.compile(text, geometry, cursor)`` and emit ``move_to``.  The resolution
stays checkable because the receipt reports ``cursor_before``, so a host cursor
read that disagreed with the guest's is visible after the fact rather than
silently absorbed.

``drag`` is lowered to press / move / release inside the single process, so a
zero-extent drag survives resolution.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from ..ir import Operation, scroll_deltas
from .keymap import guest_button, guest_key


class ExecutionError(RuntimeError):
    """A guest action could not be compiled, dispatched, or verified."""

    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence


@dataclass
class InputAudit:
    """Host-side record of what has been sent to one guest, and what is held."""

    operations: list[Operation] = field(default_factory=list)
    held_buttons: set[str] = field(default_factory=set)
    held_keys: set[str] = field(default_factory=set)
    scroll_total: int = 0
    typed_texts: list[str] = field(default_factory=list)


BUTTON_MASKS = {"left": 1 << 8, "middle": 1 << 9, "right": 1 << 10}
ALL_POINTER_BUTTON_MASK = sum(BUTTON_MASKS.values())

# Coalesced-typing clipboard timings.  The guest clipboard owner must outlive
# the paste by enough for the target application to request the selection
# contents.
CLIPBOARD_PASTE_DELAY_MS = 150
CLIPBOARD_OWNER_LIFETIME_MS = 750
# The two realisations of ``coalesced_type``.  See
# ``coalesced_type_mechanism`` -- the ONE place that chooses between them.
PYAUTOGUI_WRITE_TYPING_MECHANISM = "pyautogui_write"
GTK_CLIPBOARD_TYPING_MECHANISM = "gtk_clipboard_ctrl_v"
ATOMIC_RESULT_PREFIX = "DESKTOP_ENV_ATOMIC_RESULT="
PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND = "pyautogui_release_motion"
DIRECT_XTEST_CLICK_BACKEND = "direct_xtest_no_release_motion"
CLICK_BACKENDS = frozenset(
    {PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND, DIRECT_XTEST_CLICK_BACKEND}
)
CLICK_DWELL_S = 0.05
PASSIVE_X_OBSERVER_LIMITATION = (
    "not installed: a same-process XRecord/XI2 observer requires a second X "
    "connection and concurrent event consumption, which is not demonstrably "
    "non-perturbing for this timing experiment"
)
ATOMIC_SCHEMA_VERSION = 1


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
    backend_primitives: tuple[dict[str, Any], ...] = ()
    x_event_sync_evidence: tuple[dict[str, Any], ...] = ()
    x_sync_attempt_evidence: tuple[dict[str, Any], ...] = ()
    click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND
    x_injection_evidence: tuple[dict[str, Any], ...] = ()
    x_injection_timestamps: tuple[dict[str, Any], ...] = ()
    final_pointer_readback: dict[str, Any] = field(default_factory=dict)
    passive_x_observer: dict[str, Any] = field(default_factory=dict)

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
            "backend_primitives": list(self.backend_primitives),
            "x_event_sync_evidence": list(self.x_event_sync_evidence),
            "x_sync_attempt_evidence": list(self.x_sync_attempt_evidence),
            "click_backend": self.click_backend,
            "x_injection_evidence": list(self.x_injection_evidence),
            "x_injection_timestamps": list(self.x_injection_timestamps),
            "final_pointer_readback": dict(self.final_pointer_readback),
            "passive_x_observer": dict(self.passive_x_observer),
        }


def coalesced_type_mechanism(text: str) -> str:
    """Which substrate mechanism realises this text as guest input.

    THE one decision point.  ``coalesced_type`` is an intent -- *this text
    becomes input* -- and the Operation vocabulary deliberately says nothing
    about how.  Everything that needs to know the answer (the compiler below,
    the receipt's ``backend_primitives``) asks here rather than re-deciding.

    The predicate is "can ``pyautogui.write`` express this text": printable
    ASCII, ``U+0020``-``U+007E``.  Those characters are exactly the ones in the
    pinned guest's ``pyautogui.KEYBOARD_MAPPING``; anything outside it --
    accents, CJK, emoji, and also the control characters, for which a keystroke
    is the wrong encoding of the intent -- is silently DROPPED by
    ``pyautogui.write`` and must take the clipboard route.
    """
    if not isinstance(text, str):
        raise TypeError("coalesced type text must be a string")
    if all(" " <= character <= "~" for character in text):
        return PYAUTOGUI_WRITE_TYPING_MECHANISM
    return GTK_CLIPBOARD_TYPING_MECHANISM


def compile_unicode_coalesced_type(text: str) -> str:
    """Compile text to one guest process that makes it input, by either route.

    ``coalesced_type_mechanism`` picks the route from the payload:
    ``pyautogui.write(text, interval=0)`` for printable ASCII, the GTK clipboard
    below for everything else.  Callers do not choose and cannot; the Operation
    vocabulary has one typing intent and this is where it is realised.

    WHY THE SPLIT EXISTS, because the code cannot show it.  The clipboard route
    pastes with ``Ctrl-A`` + ``Ctrl-V``, and **in gnome-terminal ``Ctrl-V`` is
    readline's quoted-insert, not paste** (the terminal's paste chord is
    ``Ctrl-Shift-V``).  So in a terminal the whole program runs, owns the
    selection, proves its round trip, injects both chords and reports ``ok`` --
    while nothing is typed, and the *following* ``Return`` is swallowed as a
    literal ``^M`` by the pending quoted-insert.  A silent no-op with a clean
    receipt.  ``pyautogui.write`` has no such chord, which is why it carries
    every payload it can express.  ``Ctrl-Shift-V`` is not the fix: it is right
    for gnome-terminal and wrong nearly everywhere else, trading one
    context-dependent bug for another.

    The pinned image has no ``xclip``, ``xsel`` or ``wl-copy``, so ``pyperclip``
    imports but raises at runtime, and Tk owns the X11 selection only while the
    interpreter pumps its event loop -- and its ``clipboard_clear`` collapses the
    editor selection in VS Code/LibreOffice.  So the paste is driven from a real
    GLib main loop: GTK owns the selection, proves the round trip before pasting,
    and re-asserts select-all immediately before its single paste, because taking
    clipboard ownership drops the target widget's selection on that image.

    Focus/type trajectories already emit their own Ctrl-A; the re-assertion
    inside the clipboard owner is deliberately redundant with it so the paste
    replaces rather than appends even when ownership stole the selection.
    """
    if coalesced_type_mechanism(text) == PYAUTOGUI_WRITE_TYPING_MECHANISM:
        return f"pyautogui.write({text!r},interval=0)"
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    program = f"""
import base64,gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk,Gdk,GLib
value=base64.b64decode({encoded!r}).decode('utf-8')
clipboard=Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
clipboard.set_text(value,-1)
if clipboard.wait_for_text()!=value:
 raise RuntimeError('clipboard round-trip failed')
_de_pasted=[]
def paste():
 pyautogui.hotkey('ctrl','a')
 pyautogui.hotkey('ctrl','v')
 _de_pasted.append(True)
 return False
GLib.timeout_add({CLIPBOARD_PASTE_DELAY_MS},paste)
GLib.timeout_add({CLIPBOARD_OWNER_LIFETIME_MS},Gtk.main_quit)
Gtk.main()
if not _de_pasted:
 raise RuntimeError('clipboard owner expired before the paste callback ran')
""".strip()
    return f"exec({program!r})"


def expected_atomic_input_state(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
) -> tuple[set[str], set[str]]:
    """Simulate held-state transitions, rejecting impossible sequences.

    Runs host-side before anything is sent, so a double press or an unmatched
    release is a compile error instead of a guest-side surprise.

    BOTH halves of a transition are normalised through ``guest_key`` /
    ``guest_button``, i.e. through the SAME table the lowering below presses with.
    Tracking held keys by the raw operation argument instead made the held-key set
    and the lowering disagree whenever a trajectory spelled the two halves of one
    key differently -- ``key_down("Return")`` then ``key_up("Enter")``, or
    ``key_down("KeyA")`` then ``key_up("a")`` -- which the guest would have
    executed as a matched pair on one key while this rejected it as "key not
    held".  That is reachable from any grammar that passes rdev names through
    verbatim, which several deliberately do: normalising the NAMES at the grammar
    boundary would erase the distinction between ``Alt`` and ``AltLeft``, so the
    agreement has to be established here, where held state is tracked.
    """
    buttons = set(initial_buttons)
    keys = {guest_key(key) for key in initial_keys}
    for operation in operations:
        if operation.kind == "drag":
            # Self-balancing: the press and the release both happen inside the
            # one operation, so the held set is unchanged -- but a drag still
            # cannot start with the button already down.
            if "left" in buttons:
                raise ExecutionError("button already held: left")
        elif operation.kind == "mouse_down":
            button = guest_button(operation.args[0])
            if button in buttons:
                raise ExecutionError(f"button already held: {button}")
            buttons.add(button)
        elif operation.kind == "mouse_up":
            button = guest_button(operation.args[0])
            if button not in buttons:
                raise ExecutionError(f"button not held: {button}")
            buttons.remove(button)
        elif operation.kind == "key_down":
            key = guest_key(operation.args[0])
            if key in keys:
                raise ExecutionError(f"key already held: {key}")
            keys.add(key)
        elif operation.kind == "key_up":
            key = guest_key(operation.args[0])
            if key not in keys:
                raise ExecutionError(f"key not held: {key}")
            keys.remove(key)
    return buttons, keys


def pointer_mask_for_buttons(buttons: set[str]) -> int:
    unknown = buttons - set(BUTTON_MASKS)
    if unknown:
        raise ExecutionError(f"unsupported pointer buttons: {sorted(unknown)}")
    return sum(BUTTON_MASKS[button] for button in buttons)


def lower_guest_operations(
    operations: tuple[Operation, ...],
) -> tuple[Operation, ...]:
    """Lower only an adjacent same-button press/release to one click primitive.

    The input remains the canonical semantic event stream.  In particular, a
    move, key event, type, or any other operation between the transitions
    prevents coalescing, which keeps drag/hold trajectories explicit.
    """
    lowered: list[Operation] = []
    index = 0
    while index < len(operations):
        operation = operations[index]
        if operation.kind == "mouse_down" and index + 1 < len(operations):
            following = operations[index + 1]
            if following.kind == "mouse_up" and following.args == operation.args:
                lowered.append(Operation("click", operation.args))
                index += 2
                continue
        lowered.append(operation)
        index += 1
    return tuple(lowered)


def compile_atomic_guest_program(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
    click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
) -> tuple[str, int]:
    """Compile one action to exactly one ordered guest process.

    Returns the guest program source and the pointer-button mask the program
    expects to observe when it finishes.
    """
    if click_backend not in CLICK_BACKENDS:
        raise ExecutionError(f"unsupported click backend: {click_backend}")
    final_buttons, _ = expected_atomic_input_state(
        operations,
        initial_buttons=initial_buttons,
        initial_keys=initial_keys,
    )
    expected_mask = pointer_mask_for_buttons(final_buttons)
    guest_operations = lower_guest_operations(operations)
    semantic_payload = [item.as_dict() for item in operations]
    lowered_payload = [item.as_dict() for item in guest_operations]
    lines = [
        "import json, sys, traceback, time as _de_time, pyautogui",
        "pyautogui.FAILSAFE=False",
        "pyautogui.PAUSE=0",
        "_de_trace=[]",
        f"_de_expected_mask={expected_mask}",
        f"_de_expected_initial_mask={pointer_mask_for_buttons(initial_buttons)}",
        f"_de_touched_buttons=set({sorted(initial_buttons)!r})",
        f"_de_touched_keys=set({[guest_key(k) for k in sorted(initial_keys)]!r})",
        "_de_error=None",
        "_de_failure_kind=None",
        "_de_cleanup=False",
        "_de_observed_mask=-1",
        "_de_cursor_before=[-1,-1]",
        "_de_cursor_after=[-1,-1]",
        f"_de_semantic_operations={semantic_payload!r}",
        f"_de_lowered_operations={lowered_payload!r}",
        f"_de_click_backend={click_backend!r}",
        "_de_backend_primitives=[]",
        "_de_x_event_sync=[]",
        "_de_x_sync_attempts=[]",
        "_de_x_injections=[]",
        "_de_click_timings=[]",
        "_de_x_injection_sequence=0",
        "_de_x_sync_sequence=0",
        "_de_x_phase='setup'",
        "_de_attempt_hooks_installed=False",
        "_de_original_fake_input=None",
        "_de_original_display_sync=None",
        "_de_passive_x_observer={'installed':False,'observer_process_count':0,'additional_x_connection_count':0,'assessment':'omitted_not_demonstrably_non_perturbing','limitation':"
        f"{PASSIVE_X_OBSERVER_LIMITATION!r}" "}",
        "def _de_install_attempt_hooks():",
        "    global _de_attempt_hooks_installed,_de_original_fake_input,_de_original_display_sync",
        "    _backend=pyautogui.platformModule",
        "    _display=getattr(_backend,'_display',None)",
        "    _fake_input=getattr(_backend,'fake_input',None)",
        "    _sync=getattr(_display,'sync',None)",
        "    _x11=getattr(_backend,'X',None)",
        "    if not callable(_fake_input) or not callable(_sync) or _x11 is None: raise RuntimeError('global XTest/sync attempt hooks unavailable')",
        "    _event_names={2:'key_press',3:'key_release',int(_x11.MotionNotify):'motion_notify',int(_x11.ButtonPress):'button_press',int(_x11.ButtonRelease):'button_release'}",
        "    _de_original_fake_input=_fake_input",
        "    _de_original_display_sync=_sync",
        "    def _de_traced_fake_input(*_args,**_kwargs):",
        "        global _de_x_injection_sequence",
        "        _event_type=int(_args[1] if len(_args)>1 else _kwargs.get('event_type',-1))",
        "        _detail=int(_args[2] if len(_args)>2 else _kwargs.get('detail',0))",
        "        _started=_de_time.monotonic_ns()",
        "        _success=False",
        "        _error=None",
        "        try:",
        "            _result=_fake_input(*_args,**_kwargs)",
        "            _success=True",
        "        except BaseException as _exc:",
        "            _error=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
        "            raise",
        "        finally:",
        "            _completed=_de_time.monotonic_ns()",
        "            _de_x_injection_sequence+=1",
        "            _de_x_injections.append({'sequence':_de_x_injection_sequence,'phase':_de_x_phase,'event':_event_names.get(_event_type,'event_'+str(_event_type)),'event_type':_event_type,'detail':_detail,'x':_kwargs.get('x'),'y':_kwargs.get('y'),'attempted':True,'success':_success,'error':_error,'started_guest_monotonic_ns':_started,'completed_guest_monotonic_ns':_completed,'duration_ns':_completed-_started})",
        "        return _result",
        "    def _de_traced_sync(*_args,**_kwargs):",
        "        global _de_x_sync_sequence",
        "        _started=_de_time.monotonic_ns()",
        "        _success=False",
        "        _error=None",
        "        try:",
        "            _result=_sync(*_args,**_kwargs)",
        "            _success=True",
        "        except BaseException as _exc:",
        "            _error=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
        "            raise",
        "        finally:",
        "            _completed=_de_time.monotonic_ns()",
        "            _de_x_sync_sequence+=1",
        "            _de_x_sync_attempts.append({'sequence':_de_x_sync_sequence,'phase':_de_x_phase,'attempted':True,'success':_success,'error':_error,'started_guest_monotonic_ns':_started,'completed_guest_monotonic_ns':_completed,'duration_ns':_completed-_started})",
        "        return _result",
        "    _backend.fake_input=_de_traced_fake_input",
        "    try: _display.sync=_de_traced_sync",
        "    except BaseException:",
        "        _backend.fake_input=_de_original_fake_input",
        "        raise",
        "    _de_attempt_hooks_installed=True",
        "def _de_restore_attempt_hooks():",
        "    global _de_attempt_hooks_installed",
        "    _errors=[]",
        "    if _de_attempt_hooks_installed:",
        "        _backend=pyautogui.platformModule",
        "        _display=getattr(_backend,'_display',None)",
        "        try: _backend.fake_input=_de_original_fake_input",
        "        except BaseException as _exc: _errors.append('fake_input restore: '+''.join(traceback.format_exception_only(type(_exc),_exc)).strip())",
        "        try: _display.sync=_de_original_display_sync",
        "        except BaseException as _exc: _errors.append('display sync restore: '+''.join(traceback.format_exception_only(type(_exc),_exc)).strip())",
        "        _de_attempt_hooks_installed=False",
        "    return _errors",
        "def _de_sync_after_x_event(_event):",
        "    _backend=pyautogui.platformModule",
        "    _display=getattr(_backend,'_display',None)",
        "    _flush=getattr(_display,'flush',None)",
        "    _sync=getattr(_display,'sync',None)",
        "    _supported=callable(_flush) and callable(_sync)",
        "    _started=_de_time.monotonic_ns()",
        "    _flush_attempted=False",
        "    _flush_success=False",
        "    _sync_attempted=False",
        "    _sync_success=False",
        "    _error=None",
        "    try:",
        "        if not _supported: raise RuntimeError('X11 flush/sync unavailable after '+_event)",
        "        _flush_attempted=True",
        "        _flush()",
        "        _flush_success=True",
        "        _sync_attempted=True",
        "        _sync()",
        "        _sync_success=True",
        "    except BaseException as _exc:",
        "        _error=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
        "        raise",
        "    finally:",
        "        _completed=_de_time.monotonic_ns()",
        "        _de_x_event_sync.append({'event':_event,'backend':getattr(_backend,'__name__',type(_backend).__name__),'supported':_supported,'flush_attempted':_flush_attempted,'flush':_flush_success,'sync_attempted':_sync_attempted,'sync':_sync_success,'success':_flush_success and _sync_success,'error':_error,'started_guest_monotonic_ns':_started,'completed_guest_monotonic_ns':_completed,'duration_ns':_completed-_started})",
        "def _de_click(_button):",
        "    global _de_x_injection_sequence,_de_x_phase",
        "    _backend=pyautogui.platformModule",
        "    _display=getattr(_backend,'_display',None)",
        "    _down=getattr(_backend,'_mouseDown',None)",
        "    _up=getattr(_backend,'_mouseUp',None)",
        "    _move=getattr(_backend,'_moveTo',None)",
        "    _fake_input=getattr(_backend,'fake_input',None)",
        "    _x11=getattr(_backend,'X',None)",
        "    _button_map=getattr(_backend,'BUTTON_NAME_MAPPING',None)",
        "    _hooked=callable(_down) and callable(_up) and callable(_move) and callable(_fake_input) and _x11 is not None and isinstance(_button_map,dict) and callable(getattr(_display,'flush',None)) and callable(getattr(_display,'sync',None))",
        "    _release_motion=_de_click_backend=="
        f"{PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND!r}",
        "    _primitive={'kind':'click','button':_button,'call':'pyautogui.click(clicks=1, interval=0.05)','click_backend':_de_click_backend,'x11_per_event_sync_hooked':_hooked,'dwell_ms':50,'click_premove_same_coordinate_motion_notify':True,'release_side_motion_notify':_release_motion,'injection_attempt_count':1,'retry_count':0,'ordering':['click_premove_motion','mouse_down','flush','sync','dwell','mouse_up','flush','sync']}",
        "    _de_backend_primitives.append(_primitive)",
        "    if not _hooked:",
        "        raise RuntimeError('X11 click primitive hooks unavailable')",
        "    def _de_direct_down(*_args,**_kwargs):",
        "        _x,_y,_raw_button=_args[:3]",
        # Preserve the current backend's press-side XTest stream exactly.  The
        # experimental delta is only the release-side MotionNotify below.
        "        _move(_x,_y)",
        "        _backend.fake_input(_display,_x11.ButtonPress,_button_map[_raw_button])",
        "        _display.sync()",
        "    def _de_direct_up(*_args,**_kwargs):",
        "        _raw_button=_args[2]",
        # Match PyAutoGUI's release-side _moveTo sync without emitting its
        # MotionNotify, so the sole injected-event delta remains the motion.
        "        _display.sync()",
        "        _backend.fake_input(_display,_x11.ButtonRelease,_button_map[_raw_button])",
        "        _display.sync()",
        "    _timing={'click_backend':_de_click_backend,'backend_identity':getattr(_backend,'__name__',type(_backend).__name__),'release_side_motion_notify':_release_motion,'clock':'time.monotonic_ns','dwell_requested_ns':50000000,'x_injection_start_sequence':_de_x_injection_sequence}",
        "    def _de_down(*_args,**_kwargs):",
        "        global _de_x_phase",
        "        _prior_phase=_de_x_phase",
        "        _de_x_phase='press'",
        "        _timing['press_call_before_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "        _timing['press_call_success']=False",
        "        _timing['press_call_error']=None",
        "        try:",
        "            _result=(_down if _release_motion else _de_direct_down)(*_args,**_kwargs)",
        "            _timing['press_call_success']=True",
        "        except BaseException as _exc:",
        "            _timing['press_call_error']=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
        "            raise",
        "        finally:",
        "            _timing['press_call_after_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "            _de_x_phase=_prior_phase",
        "        _de_x_phase='press_sync'",
        "        try: _de_sync_after_x_event('mouse_down')",
        "        finally: _de_x_phase=_prior_phase",
        "        _timing['press_sync_completed_guest_monotonic_ns']=_de_time.monotonic_ns()",
        # pyautogui.click's interval is between repeated clicks, not between
        # press and release.  Chromium intermittently observed only pointerdown
        # when the two XTest events were adjacent even though X11 reported a
        # released final mask.  A bounded dwell after the synced press keeps
        # the fixed click primitive while making browser receipt causal.
        "        _timing['dwell_started_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "        _timing['dwell_success']=False",
        "        _timing['dwell_error']=None",
        "        try:",
        f"            _de_time.sleep({CLICK_DWELL_S!r})",
        "            _timing['dwell_success']=True",
        "        except BaseException as _exc:",
        "            _timing['dwell_error']=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
        "            raise",
        "        finally:",
        "            _timing['dwell_completed_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "            _timing['dwell_duration_ns']=_timing['dwell_completed_guest_monotonic_ns']-_timing['dwell_started_guest_monotonic_ns']",
        "        _de_trace.append({'kind':'mouse_down','args':[_button]})",
        "        return _result",
        "    def _de_up(*_args,**_kwargs):",
        "        global _de_x_phase",
        "        _prior_phase=_de_x_phase",
        "        _de_x_phase='release'",
        "        _timing['release_call_before_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "        _timing['release_call_success']=False",
        "        _timing['release_call_error']=None",
        "        try:",
        "            _result=(_up if _release_motion else _de_direct_up)(*_args,**_kwargs)",
        "            _timing['release_call_success']=True",
        "        except BaseException as _exc:",
        "            _timing['release_call_error']=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
        "            raise",
        "        finally:",
        "            _timing['release_call_after_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "            _de_x_phase=_prior_phase",
        "        _de_x_phase='release_sync'",
        "        try: _de_sync_after_x_event('mouse_up')",
        "        finally: _de_x_phase=_prior_phase",
        "        _timing['release_sync_completed_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "        _de_trace.append({'kind':'mouse_up','args':[_button]})",
        "        return _result",
        "    _backend._mouseDown=_de_down",
        "    _backend._mouseUp=_de_up",
        "    _click_prior_phase=_de_x_phase",
        "    _de_x_phase='click_premove'",
        "    _timing['click_started_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "    try:",
        "        pyautogui.click(clicks=1,interval=0.05,button=_button)",
        "    finally:",
        "        _timing['click_completed_guest_monotonic_ns']=_de_time.monotonic_ns()",
        "        _timing['x_injection_end_sequence']=_de_x_injection_sequence",
        "        _timing['click_premove_xtest_sequence']=[_item['event'] for _item in _de_x_injections if _item['sequence']>_timing['x_injection_start_sequence'] and _item['sequence']<=_timing['x_injection_end_sequence'] and _item['phase']=='click_premove']",
        "        _timing['press_xtest_sequence']=[_item['event'] for _item in _de_x_injections if _item['sequence']>_timing['x_injection_start_sequence'] and _item['sequence']<=_timing['x_injection_end_sequence'] and _item['phase']=='press']",
        "        _timing['release_xtest_sequence']=[_item['event'] for _item in _de_x_injections if _item['sequence']>_timing['x_injection_start_sequence'] and _item['sequence']<=_timing['x_injection_end_sequence'] and _item['phase']=='release']",
        "        _primitive['click_premove_xtest_sequence']=list(_timing['click_premove_xtest_sequence'])",
        "        _primitive['press_xtest_sequence']=list(_timing['press_xtest_sequence'])",
        "        _primitive['release_xtest_sequence']=list(_timing['release_xtest_sequence'])",
        "        _de_click_timings.append(_timing)",
        "        _backend._mouseDown=_down",
        "        _backend._mouseUp=_up",
        "        _de_x_phase=_click_prior_phase",
        "def _de_pointer_state():",
        "    _backend=pyautogui.platformModule",
        "    _backend._display.sync()",
        "    _pointer=_backend._display.screen().root.query_pointer()",
        f"    return int(_pointer.root_x),int(_pointer.root_y),int(_pointer.mask)&{ALL_POINTER_BUTTON_MASK}",
        "try:",
        "    _de_install_attempt_hooks()",
        "    _de_x_phase='initial_readback'",
        "    _de_bx,_de_by,_de_initial_mask=_de_pointer_state()",
        "    _de_cursor_before=[_de_bx,_de_by]",
        "    _de_x_phase='outside_action'",
        "    if _de_initial_mask != _de_expected_initial_mask:",
        "        _de_failure_kind='verification'",
        "        raise RuntimeError(f'initial pointer button mask {_de_initial_mask} != expected {_de_expected_initial_mask}')",
    ]
    indent = "    "
    for index, operation in enumerate(guest_operations):
        kind, args = operation.kind, operation.args
        lines.append(f"{indent}# DESKTOP_ENV_ATOMIC_STEP_{index}:{kind}")
        if kind == "move_to":
            x, y = int(args[0]), int(args[1])
            lines.extend(_move_to_lines(indent, x, y))
        elif kind == "drag":
            # A press-move-release inside the one process.  Emitted as separate
            # primitives rather than reusing the click path so a zero-extent
            # drag still produces a real ButtonPress and ButtonRelease, and so
            # the intermediate motion is observable in the evidence.
            x0, y0, x1, y1 = (int(value) for value in args[:4])
            lines.extend(_move_to_lines(indent, x0, y0))
            lines.extend(
                [
                    f"{indent}_de_touched_buttons.add('left')",
                    f"{indent}pyautogui.mouseDown(button='left')",
                    f"{indent}_de_sync_after_x_event('mouse_down')",
                    f"{indent}_de_backend_primitives.append({{'kind':'mouse_down','button':'left','call':'pyautogui.mouseDown','drag':True}})",
                    f"{indent}_de_trace.append({{'kind':'mouse_down','args':['left']}})",
                ]
            )
            lines.extend(_move_to_lines(indent, x1, y1))
            lines.extend(
                [
                    f"{indent}pyautogui.mouseUp(button='left')",
                    f"{indent}_de_sync_after_x_event('mouse_up')",
                    f"{indent}_de_backend_primitives.append({{'kind':'mouse_up','button':'left','call':'pyautogui.mouseUp','drag':True,'zero_extent':{(x0, y0) == (x1, y1)!r}}})",
                    f"{indent}_de_trace.append({{'kind':'mouse_up','args':['left']}})",
                ]
            )
        elif kind == "glide_to":
            # A timed move.  pyautogui interpolates, so the app sees a sweep of
            # MotionNotify events rather than one teleport -- which is the whole
            # difference between a drag stroke and a jump.
            x, y, seconds = int(args[0]), int(args[1]), max(0.0, min(10.0, float(args[2])))
            lines.extend(
                [
                    f"{indent}_x,_y=pyautogui.position()",
                    f"{indent}_w,_h=pyautogui.size()",
                    f"{indent}_tx=max(0,min(int(_w)-1,{x}))",
                    f"{indent}_ty=max(0,min(int(_h)-1,{y}))",
                    f"{indent}_de_x_phase='canonical_move'",
                    f"{indent}pyautogui.moveTo(_tx,_ty,duration={seconds!r})",
                    f"{indent}_de_x_phase='outside_action'",
                    f"{indent}_de_backend_primitives.append({{'kind':'glide_to','call':'pyautogui.moveTo(duration=)','requested_position':[{x},{y}],'seconds':{seconds!r},'cursor_before':[int(_x),int(_y)],'cursor_after':[_tx,_ty],'clamped':(_tx,_ty)!=({x},{y})}})",
                    f"{indent}_de_trace.append({{'kind':'glide_to','args':[_tx,_ty,{seconds!r}]}})",
                ]
            )
        elif kind == "scroll":
            dx, dy = scroll_deltas(args)
            block = [
                f"{indent}_de_backend_primitives.append({{'kind':'scroll','call':'pyautogui.scroll/hscroll','dx':{dx},'dy':{dy}}})",
                f"{indent}_de_trace.append({{'kind':'scroll','args':[{dx},{dy}]}})",
            ]
            if dy:
                block.insert(0, f"{indent}pyautogui.scroll({dy})")
            if dx:
                # hscroll exists on the pinned guest's backend; a guest without it
                # raises inside the program, which the receipt reports rather than
                # silently dropping the axis.
                block.insert(0, f"{indent}pyautogui.hscroll({dx})")
            lines.extend(block)
        elif kind == "click":
            button = guest_button(args[0])
            lines.extend(
                [
                    f"{indent}_de_touched_buttons.add({button!r})",
                    f"{indent}_de_click({button!r})",
                ]
            )
        elif kind in {"mouse_down", "mouse_up"}:
            button = guest_button(args[0])
            method = "mouseDown" if kind == "mouse_down" else "mouseUp"
            lines.extend(
                [
                    f"{indent}_de_touched_buttons.add({button!r})",
                    f"{indent}pyautogui.{method}(button={button!r})",
                    f"{indent}_de_sync_after_x_event({kind!r})",
                    f"{indent}_de_backend_primitives.append({{'kind':{kind!r},'button':{button!r},'call':'pyautogui.{method}'}})",
                    f"{indent}_de_trace.append({{'kind':{kind!r},'args':[{button!r}]}})",
                ]
            )
        elif kind in {"key_down", "key_up"}:
            key = str(args[0])
            mapped = guest_key(key)
            method = "keyDown" if kind == "key_down" else "keyUp"
            lines.extend(
                [
                    f"{indent}_de_touched_keys.add({mapped!r})",
                    f"{indent}pyautogui.{method}({mapped!r})",
                    f"{indent}_de_backend_primitives.append({{'kind':{kind!r},'key':{key!r},'mapped_key':{mapped!r},'call':'pyautogui.{method}'}})",
                    f"{indent}_de_trace.append({{'kind':{kind!r},'args':[{key!r}]}})",
                ]
            )
        elif kind == "coalesced_type":
            text = str(args[0])
            lines.extend(
                [
                    f"{indent}{compile_unicode_coalesced_type(text)}",
                    f"{indent}_de_backend_primitives.append({{'kind':'coalesced_type','call':{coalesced_type_mechanism(text)!r},'utf8_bytes':{len(text.encode('utf-8'))}}})",
                    f"{indent}_de_trace.append({{'kind':'coalesced_type','args':[{text!r}]}})",
                ]
            )
        elif kind == "ascii_type":
            text = str(args[0])
            try:
                text.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ExecutionError("ascii_type received non-ASCII text") from exc
            if "\n" in text or "\r" in text:
                raise ExecutionError("ascii_type cannot embed Enter; emit a key event")
            lines.extend(
                [
                    f"{indent}pyautogui.write({text!r},interval=0)",
                    f"{indent}_de_backend_primitives.append({{'kind':'ascii_type','call':'pyautogui.write','ascii_bytes':{len(text)}}})",
                    f"{indent}_de_trace.append({{'kind':'ascii_type','args':[{text!r}]}})",
                ]
            )
        elif kind == "wait":
            seconds = max(0.0, min(10.0, float(args[0])))
            lines.extend(
                [
                    f"{indent}_de_time.sleep({seconds!r})",
                    f"{indent}_de_backend_primitives.append({{'kind':'wait','call':'time.sleep','seconds':{seconds!r}}})",
                    f"{indent}_de_trace.append({{'kind':'wait','args':[{seconds!r}]}})",
                ]
            )
        elif kind == "raise_for_test":
            lines.extend(
                [
                    f"{indent}_de_failure_kind='injected'",
                    f"{indent}raise RuntimeError({str(args[0])!r})",
                ]
            )
        else:
            raise ExecutionError(f"unsupported atomic operation: {kind}")
    lines.extend(
        [
            f"{indent}_de_x_phase='verification_readback'",
            f"{indent}_cx,_cy,_de_observed_mask=_de_pointer_state()",
            f"{indent}_de_cursor_after=[_cx,_cy]",
            f"{indent}_de_x_phase='outside_action'",
            f"{indent}if _de_observed_mask != _de_expected_mask:",
            f"{indent}    _de_failure_kind='verification'",
            f"{indent}    raise RuntimeError(f'pointer button mask {{_de_observed_mask}} != expected {{_de_expected_mask}}')",
            "except BaseException as _exc:",
            "    _de_error=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
            "    if _de_failure_kind is None: _de_failure_kind='infrastructure'",
            "    _de_cleanup=True",
            "    _de_x_phase='cleanup'",
            "    for _key in sorted(_de_touched_keys,reverse=True):",
            "        try: pyautogui.keyUp(_key)",
            "        except BaseException: pass",
            "    for _button in ('left','middle','right'):",
            "        try: pyautogui.mouseUp(button=_button)",
            "        except BaseException: pass",
            "_de_final_mask=-1",
            "_de_final_readback_error=None",
            "_de_final_readback_success=False",
            "_de_x_phase='final_readback'",
            "try:",
            "    _de_cx,_de_cy,_de_final_mask=_de_pointer_state()",
            "    _de_cursor_after=[_de_cx,_de_cy]",
            "    _de_final_readback_success=True",
            "except BaseException as _exc:",
            "    _de_final_readback_error=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
            "    _readback_message='final pointer readback failed: '+_de_final_readback_error",
            "    _de_error=_readback_message if _de_error is None else _de_error+'; '+_readback_message",
            "    _de_failure_kind='infrastructure'",
            "_de_final_pointer_readback={'attempted':True,'success':_de_final_readback_success,'error':_de_final_readback_error,'cursor':list(_de_cursor_after),'pointer_button_mask':_de_final_mask}",
            "_de_attempt_hook_restore_errors=_de_restore_attempt_hooks()",
            "if _de_attempt_hook_restore_errors:",
            "    _restore_message='; '.join(_de_attempt_hook_restore_errors)",
            "    _de_error=_restore_message if _de_error is None else _de_error+'; '+_restore_message",
            "    _de_failure_kind='infrastructure'",
            "_de_payload={'ok':_de_error is None,'cursor':list(_de_cursor_after),",
            " 'cursor_before':_de_cursor_before,'cursor_after':_de_cursor_after,",
            f" '_de_schema':{ATOMIC_SCHEMA_VERSION},'pointer_button_mask':_de_final_mask,",
            " 'observed_pointer_button_mask':_de_observed_mask,",
            " 'expected_pointer_button_mask':_de_expected_mask,",
            " 'guest_process_count':1,'cleanup_attempted':_de_cleanup,",
            " 'error':_de_error,'failure_kind':_de_failure_kind,",
            " 'operations':_de_trace,'semantic_operations':_de_semantic_operations,",
            " 'lowered_operations':_de_lowered_operations,",
            " 'backend_primitives':_de_backend_primitives,",
            " 'x_event_sync_evidence':_de_x_event_sync,",
            " 'x_sync_attempt_evidence':_de_x_sync_attempts,",
            " 'click_backend':_de_click_backend,",
            " 'x_injection_evidence':_de_x_injections,",
            " 'x_injection_timestamps':_de_click_timings,",
            " 'final_pointer_readback':_de_final_pointer_readback,",
            " 'attempt_hook_restore_errors':_de_attempt_hook_restore_errors,",
            " 'passive_x_observer':_de_passive_x_observer}",
            f"print({ATOMIC_RESULT_PREFIX!r}+json.dumps(_de_payload,separators=(',',':'),ensure_ascii=False))",
            "if _de_error is not None: sys.exit(1)",
        ]
    )
    return "\n".join(lines), expected_mask


def _move_to_lines(indent: str, x: int, y: int) -> list[str]:
    """Guest lines for one absolute move, recording the clamp if one happens."""
    return [
        f"{indent}_x,_y=pyautogui.position()",
        f"{indent}_w,_h=pyautogui.size()",
        f"{indent}_tx=max(0,min(int(_w)-1,{x}))",
        f"{indent}_ty=max(0,min(int(_h)-1,{y}))",
        f"{indent}_de_x_phase='canonical_move'",
        f"{indent}pyautogui.moveTo(_tx,_ty)",
        f"{indent}_de_x_phase='outside_action'",
        f"{indent}_de_backend_primitives.append({{'kind':'move_to','call':'pyautogui.moveTo','requested_position':[{x},{y}],'cursor_before':[int(_x),int(_y)],'cursor_after':[_tx,_ty],'clamped':(_tx,_ty)!=({x},{y})}})",
        f"{indent}_de_trace.append({{'kind':'move_to','args':[_tx,_ty]}})",
    ]
