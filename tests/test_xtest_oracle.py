from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from desktop import ir
from desktop.execute.protocol import build_action_request
from desktop.vm.client import ACTION_EXECUTOR_PATH


@pytest.fixture(scope="module")
def x_display():
    missing = [binary for binary in ("Xvfb", "xdotool") if shutil.which(binary) is None]
    if missing:
        pytest.skip("xdotool differential oracle unavailable: missing " + ", ".join(missing))
    from Xlib.display import Display

    process = None
    display_name = None
    for number in range(90, 120):
        candidate = f":{number}"
        process = subprocess.Popen(
            ["Xvfb", candidate, "-screen", "0", "800x600x24", "-nolisten", "tcp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(50):
            if process.poll() is not None:
                break
            try:
                probe = Display(candidate)
            except Exception:
                time.sleep(0.02)
                continue
            probe.close()
            display_name = candidate
            break
        if display_name is not None:
            break
    if process is None or display_name is None:
        pytest.fail("Xvfb is installed but no test display could be started")
    try:
        yield display_name
    finally:
        process.terminate()
        process.wait(timeout=5)


def _run_while_updating(
    root, argv: list[str], env: dict[str, str], *, stdin: str | None = None
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if stdin is not None:
        assert process.stdin is not None
        process.stdin.write(stdin)
        process.stdin.close()
        process.stdin = None
    while process.poll() is None:
        root.update()
        time.sleep(0.002)
    for _ in range(5):
        root.update()
    stdout, stderr = process.communicate()
    assert process.returncode == 0, stderr or stdout
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


@pytest.fixture
def focused_tk(x_display, monkeypatch):
    import tkinter

    monkeypatch.setenv("DISPLAY", x_display)
    root = tkinter.Tk()
    root.overrideredirect(True)
    root.geometry("400x200+0+0")
    entry = tkinter.Entry(root)
    entry.pack(fill="both", expand=True)
    root.update()
    entry.focus_force()
    root.update()
    try:
        yield root, entry
    finally:
        root.destroy()


def _environment(display_name: str) -> dict[str, str]:
    return {**os.environ, "DISPLAY": display_name, "LANG": "C.UTF-8"}


def test_click_event_shape_matches_xdotool(x_display, focused_tk):
    root, entry = focused_tk
    env = _environment(x_display)
    observed: list[str] = []
    entry.bind("<ButtonPress-1>", lambda event: observed.append("button_press"))
    entry.bind("<ButtonRelease-1>", lambda event: observed.append("button_release"))
    entry.bind("<Motion>", lambda event: observed.append("motion_notify"))
    _run_while_updating(root, ["xdotool", "mousemove", "100", "100"], env)
    observed.clear()
    _run_while_updating(root, ["xdotool", "click", "1"], env)
    oracle = list(observed)
    observed.clear()
    request, _, _ = build_action_request(
        (ir.click("left"),), initial_buttons=set(), initial_keys=set()
    )
    _run_while_updating(
        root,
        [sys.executable, str(ACTION_EXECUTOR_PATH)],
        env,
        stdin=json.dumps(request),
    )
    assert oracle == ["button_press", "button_release"]
    assert observed == oracle


def test_unicode_keysym_and_keymap_restoration_extend_the_xdotool_ascii_oracle(
    x_display, focused_tk
):
    from Xlib.display import Display

    root, entry = focused_tk
    env = _environment(x_display)
    keysyms = []
    entry.bind(
        "<KeyPress>",
        lambda event: keysyms.append(event.keysym_num),
    )
    display = Display(x_display)
    info = display.display.info
    first = int(info.min_keycode)
    count = int(info.max_keycode) - first + 1
    before = display.get_keyboard_mapping(first, count)
    _run_while_updating(root, ["xdotool", "windowfocus", str(entry.winfo_id())], env)
    ascii_text = "ab"
    _run_while_updating(root, ["xdotool", "type", "--delay", "0", "--", ascii_text], env)
    oracle = entry.get()
    after_oracle = display.get_keyboard_mapping(first, count)
    entry.delete(0, "end")
    keysyms.clear()
    text = "a✓b"
    request, _, _ = build_action_request(
        (ir.coalesced_type(text),), initial_buttons=set(), initial_keys=set()
    )
    _run_while_updating(
        root,
        [sys.executable, str(ACTION_EXECUTOR_PATH)],
        env,
        stdin=json.dumps(request),
    )
    after_executor = display.get_keyboard_mapping(first, count)
    display.close()
    assert oracle == ascii_text
    assert keysyms == [ord("a"), 0x01002713, ord("b")]
    assert after_oracle == before
    assert after_executor == before
