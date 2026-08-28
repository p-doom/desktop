"""Applying resolved operations to a guest desktop.

Nothing in this subpackage knows what an action grammar is.  Operations arrive
already resolved to absolute screen pixels; this code turns them into guest input
and reports what happened.
"""

from .engine import Engine, StepReceipt
from .guest_program import (
    ALL_POINTER_BUTTON_MASK,
    BUTTON_MASKS,
    AtomicExecutionResult,
    ExecutionError,
    InputAudit,
    compile_atomic_guest_program,
    lower_guest_operations,
)
from .keymap import (
    BUTTON_NUMBERS,
    KEY_ALIASES,
    KEY_NAMES,
    KEYSYMS,
    POINTER_BUTTONS,
    PRESSABLE_KEYS,
    KeymapError,
    guest_button,
    guest_key,
    key_chord,
    key_press,
)
from .transport import GuiTransport, HttpGuiTransport, RecordingTransport

__all__ = [
    "ALL_POINTER_BUTTON_MASK",
    "BUTTON_MASKS",
    "BUTTON_NUMBERS",
    "KEY_ALIASES",
    "KEY_NAMES",
    "KEYSYMS",
    "POINTER_BUTTONS",
    "PRESSABLE_KEYS",
    "AtomicExecutionResult",
    "Engine",
    "ExecutionError",
    "GuiTransport",
    "HttpGuiTransport",
    "InputAudit",
    "KeymapError",
    "RecordingTransport",
    "StepReceipt",
    "compile_atomic_guest_program",
    "guest_button",
    "guest_key",
    "key_chord",
    "key_press",
    "lower_guest_operations",
]
