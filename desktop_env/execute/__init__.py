"""Applying resolved operations to a guest desktop.

Nothing in this subpackage knows what an action grammar is.  Operations arrive
already resolved to absolute screen pixels; this code turns them into guest input
and reports what happened.
"""

from .engine import Engine, StepReceipt
from .guest_program import (
    ALL_POINTER_BUTTON_MASK,
    BUTTON_MASKS,
    CLICK_BACKENDS,
    CLIPBOARD_OWNER_LIFETIME_MS,
    CLIPBOARD_PASTE_DELAY_MS,
    DIRECT_XTEST_CLICK_BACKEND,
    GTK_CLIPBOARD_TYPING_MECHANISM,
    PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    PYAUTOGUI_WRITE_TYPING_MECHANISM,
    AtomicExecutionResult,
    ExecutionError,
    InputAudit,
    coalesced_type_mechanism,
    compile_atomic_guest_program,
    compile_unicode_coalesced_type,
    lower_guest_operations,
)
from .keymap import (
    BUTTON_NUMBERS,
    KEY_ALIASES,
    KEY_NAMES,
    POINTER_BUTTONS,
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
    "CLICK_BACKENDS",
    "CLIPBOARD_OWNER_LIFETIME_MS",
    "CLIPBOARD_PASTE_DELAY_MS",
    "DIRECT_XTEST_CLICK_BACKEND",
    "GTK_CLIPBOARD_TYPING_MECHANISM",
    "KEY_ALIASES",
    "KEY_NAMES",
    "POINTER_BUTTONS",
    "PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND",
    "PYAUTOGUI_WRITE_TYPING_MECHANISM",
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
    "coalesced_type_mechanism",
    "compile_unicode_coalesced_type",
    "guest_button",
    "guest_key",
    "key_chord",
    "key_press",
    "lower_guest_operations",
]
