"""Direct-XTEST text input and temporary Unicode keymap restoration."""

from __future__ import annotations

import pytest

from desktop import ir
from desktop.execute.guest_program import ExecutionError, compile_atomic_guest_program

from .support.guest_runner import run_guest_program

ASCII_TEXT = "printf SOLV2_COMPOUND_OK > /tmp/solv2_compound.txt"
UNICODE_TEXT = "héllo ✓"


@pytest.mark.parametrize(
    "text",
    ["ls", ASCII_TEXT, "", " ", "~", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}"],
)
def test_printable_ascii_uses_the_direct_xtest_program(text):
    source, _ = compile_atomic_guest_program(
        (ir.coalesced_type(text),), initial_buttons=set(), initial_keys=set()
    )
    assert "from Xlib import X" in source
    assert "xtest.fake_input" in source
    assert "pyautogui" not in source
    assert "Gtk" not in source
    assert "clipboard" not in source.lower()


@pytest.mark.parametrize("text", [UNICODE_TEXT, "é", "日本語", "🙂", "a\nb", "a\tb"])
def test_unicode_and_supported_controls_use_the_direct_xtest_program(text):
    run = run_guest_program((ir.coalesced_type(text),))
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    assert run.trace() == [("coalesced_type", [text])]
    assert {row["backend"] for row in run.primitives()} == {"python-xlib XTEST"}


def test_typing_rejects_a_non_string_payload_at_compile_time():
    with pytest.raises(ExecutionError, match="typing text must be a string"):
        compile_atomic_guest_program(
            (ir.Operation("coalesced_type", (b"ls",)),),
            initial_buttons=set(),
            initial_keys=set(),
        )


def test_ascii_typing_emits_only_key_events():
    run = run_guest_program((ir.coalesced_type(ASCII_TEXT),))
    assert run.returncode == 0, run.stderr
    events = run.payload["x_injection_evidence"]
    assert events
    assert {event["event"] for event in events} == {"key_press", "key_release"}
    assert run.keymap_restored is True


def test_unicode_typing_restores_the_exact_temporary_keysym_row():
    run = run_guest_program((ir.coalesced_type(UNICODE_TEXT),))
    assert run.returncode == 0, run.stderr
    assert run.keymap_restored is True
    (restoration,) = run.payload["keymap_restorations"]
    assert restoration["exact"] is True
    assert restoration["restored"] == restoration["original"]


def test_mixed_ascii_and_unicode_preserves_character_order():
    run = run_guest_program((ir.coalesced_type("a✓b"),))
    phases = [event["phase"] for event in run.payload["x_injection_evidence"]]
    assert phases == [
        "type_down",
        "type_up",
        "unicode_down",
        "unicode_up",
        "type_down",
        "type_up",
    ]


def test_typing_temporarily_clears_and_restores_a_held_modifier():
    run = run_guest_program((ir.coalesced_type("✓"),), initial_keys={"ControlLeft"})
    phases = [event["phase"] for event in run.payload["x_injection_evidence"]]
    assert phases == [
        "type_modifier_release",
        "unicode_down",
        "unicode_up",
        "type_modifier_restore",
    ]
    assert run.payload["held_keys"] == ["ctrlleft"]


@pytest.mark.parametrize("text", ["\x00", "\x1f", "\x7f", "\ud800"])
def test_untypable_codepoints_fail_at_compile_time(text):
    with pytest.raises(ExecutionError, match="typing text"):
        compile_atomic_guest_program(
            (ir.coalesced_type(text),), initial_buttons=set(), initial_keys=set()
        )


def test_ascii_type_rejects_unicode_and_embedded_enter_at_compile_time():
    for text in ("é", "echo\n"):
        with pytest.raises(ExecutionError):
            compile_atomic_guest_program(
                (ir.ascii_type(text),), initial_buttons=set(), initial_keys=set()
            )


@pytest.mark.parametrize("command", ["ls", ASCII_TEXT])
def test_typed_command_and_return_share_one_xtest_stream(command):
    run = run_guest_program(
        (ir.coalesced_type(command), ir.key_down("Return"), ir.key_up("Return"))
    )
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    phases = [event["phase"] for event in run.payload["x_injection_evidence"]]
    assert "type_down" in phases
    assert phases[-2:] == ["key_down", "key_up"]
    assert run.payload["held_keys"] == []


def test_unicode_mapping_and_key_state_are_restored_after_xtest_failure():
    run = run_guest_program((ir.coalesced_type("✓"),), fail_xtest_at=2)
    assert run.returncode == 1
    assert run.payload["ok"] is False
    assert "injected XTEST failure" in run.payload["error"]
    (restoration,) = run.payload["keymap_restorations"]
    assert restoration["exact"] is True
    assert restoration["restored"] == restoration["original"]
    assert run.keymap_restored is True
    assert run.held_keycodes == []
