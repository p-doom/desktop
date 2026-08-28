"""The in-VM agent client, driven against a real HTTP server on localhost.

Not on the original risk list, but it was the least-exercised module in the
package and it is the only path by which anything reaches a guest.  A
``http.server`` in a thread is enough to exercise every branch, so none of this
needs a VM.

The client is also where the "no runtime dependencies" rule is felt:
``urllib.request`` has no session object and raises for HTTP errors, so each
method's error handling is hand-written and therefore worth checking.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from desktop.vm.osworld_client import (
    GuestAgentError,
    OSWorldClient,
    _decode_detached_result,
)

#: What the fake agent should answer, per path.  Mutated per test.
ROUTES: dict = {}


class _Handler(BaseHTTPRequestHandler):
    # Deliberately NOT HTTP/1.1: keep-alive made every server teardown wait for
    # an idle connection, which cost ~0.5s per test (21s across this module).
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):  # keep the test output clean
        pass

    def _respond(self):
        route = ROUTES.get(self.path)
        if route is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status, body, content_type = route() if callable(route) else route
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        self._respond()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._respond()


@pytest.fixture(scope="module")
def agent_server():
    """One fake in-VM agent for the whole module; ROUTES is swapped per test."""

    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def agent(agent_server):
    """A client pointed at the fake agent, with a clean route table."""
    ROUTES.clear()
    return OSWorldClient(f"http://127.0.0.1:{agent_server.server_port}", timeout_s=5.0)


PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(64))


def _json_route(payload, status=200):
    return (status, json.dumps(payload), "application/json")


def test_a_trailing_slash_in_the_base_url_is_normalised():
    assert OSWorldClient("http://host:5000/").base_url == "http://host:5000"
    assert OSWorldClient("http://host:5000///").base_url == "http://host:5000"


def test_a_screenshot_comes_back_as_undecoded_bytes(agent):
    ROUTES["/screenshot"] = (200, PNG, "image/png")
    body = agent.screenshot()
    assert body == PNG
    assert isinstance(body, bytes)


def test_an_empty_screenshot_body_is_an_error(agent):
    ROUTES["/screenshot"] = (200, b"", "image/png")
    with pytest.raises(GuestAgentError, match="/screenshot returned 200"):
        agent.screenshot()


def test_a_non_200_screenshot_is_an_error(agent):
    ROUTES["/screenshot"] = (500, b"boom", "text/plain")
    with pytest.raises(GuestAgentError, match="/screenshot returned 500"):
        agent.screenshot()


def test_an_unreachable_agent_raises_rather_than_hanging():
    client = OSWorldClient("http://127.0.0.1:1", timeout_s=1.0)
    with pytest.raises(GuestAgentError, match="failed"):
        client.screenshot()


def test_screen_size_is_parsed(agent):
    ROUTES["/screen_size"] = _json_route({"width": 1920, "height": 1080})
    assert agent.screen_size() == (1920, 1080)


@pytest.mark.parametrize(
    "payload", [{"width": 1920}, {"height": 1080}, [], "1920x1080", None, 42]
)
def test_a_malformed_screen_size_is_an_error(agent, payload):
    ROUTES["/screen_size"] = _json_route(payload)
    with pytest.raises(GuestAgentError, match="invalid screen size"):
        agent.screen_size()


def test_a_non_json_screen_size_is_an_error(agent):
    ROUTES["/screen_size"] = (200, b"\xff\xfe not json", "application/json")
    with pytest.raises(GuestAgentError, match="non-JSON"):
        agent.screen_size()


def test_an_http_error_status_is_reported_with_its_body(agent):
    ROUTES["/screen_size"] = (503, b"agent is restarting", "text/plain")
    with pytest.raises(GuestAgentError, match="returned 503"):
        agent.screen_size()


def test_a_successful_execute_returns_the_agents_object(agent):
    ROUTES["/execute"] = _json_route(
        {"status": "success", "returncode": 0, "output": "hello\n"}
    )
    assert agent.execute(["echo", "hello"])["output"] == "hello\n"


def test_a_failed_execute_raises_when_checked(agent):
    ROUTES["/execute"] = _json_route(
        {"status": "error", "returncode": 2, "error": "no such file"}
    )
    with pytest.raises(GuestAgentError, match="guest command failed"):
        agent.execute(["false"])


def test_a_failed_execute_is_returned_when_unchecked(agent):
    ROUTES["/execute"] = _json_route({"status": "error", "returncode": 2})
    assert agent.execute(["false"], check=False)["returncode"] == 2


def test_a_nonzero_returncode_with_success_status_still_fails(agent):
    """Both conditions are required, so a guest that reports success with a
    non-zero code cannot slip through."""
    ROUTES["/execute"] = _json_route({"status": "success", "returncode": 1})
    with pytest.raises(GuestAgentError, match="guest command failed"):
        agent.execute(["false"])


def test_a_non_object_execute_response_is_an_error(agent):
    ROUTES["/execute"] = _json_route(["not", "an", "object"])
    with pytest.raises(GuestAgentError, match="non-object"):
        agent.execute(["true"])


def test_execute_never_asks_the_agent_for_a_shell(agent):
    """``shell: false`` always: the agent runs a subprocess and does not eval."""
    seen: dict = {}

    class Recording(BaseHTTPRequestHandler):
        pass

    def route():
        return _json_route({"status": "success", "returncode": 0})

    ROUTES["/execute"] = route
    # Inspect the request body by intercepting at the client level instead.
    original = agent._request_json

    def spy(method, path, *, payload=None, timeout_s=None):
        seen["payload"] = payload
        return original(method, path, payload=payload, timeout_s=timeout_s)

    agent._request_json = spy  # type: ignore[method-assign]
    agent.execute(["python", "-c", "print(1)"])
    assert seen["payload"] == {"command": ["python", "-c", "print(1)"], "shell": False}


def test_the_accessibility_tree_is_returned_as_text(agent):
    ROUTES["/accessibility"] = (200, "<tree/>", "text/xml")
    assert agent.accessibility() == "<tree/>"


def test_a_json_wrapped_accessibility_tree_is_not_unwrapped(agent):
    """One accepted shape.  Unwrapping any of four key spellings meant a guest that
    renamed the field kept "working" while returning a different slice."""
    ROUTES["/accessibility"] = _json_route({"tree": "<tree/>"})
    assert agent.accessibility() == '{"tree": "<tree/>"}'


def test_undecodable_accessibility_bytes_are_refused(agent):
    """The tree is XML a consumer parses, so a replaced byte is a malformed tree
    that looks like a valid one."""
    ROUTES["/accessibility"] = (200, b"<tree>\xff</tree>", "text/xml")
    with pytest.raises(GuestAgentError, match="are not UTF-8"):
        agent.accessibility()


def test_a_non_200_accessibility_is_an_error(agent):
    ROUTES["/accessibility"] = (404, b"", "text/plain")
    with pytest.raises(GuestAgentError, match="/accessibility returned 404"):
        agent.accessibility()


def test_a_cursor_position_pair_is_parsed(agent):
    ROUTES["/cursor_position"] = _json_route([11, 22])
    assert agent.cursor_position() == (11, 22)


@pytest.mark.parametrize(
    "payload", [{"x": 11, "y": 22}, {"x": 1}, [1], [1, 2, 3], "11,22", None]
)
def test_a_malformed_cursor_position_is_an_error(agent, payload):
    """Including the ``{"x", "y"}`` object: ``HttpGuiTransport`` reads the same
    endpoint and has only ever accepted the array, so accepting both here made
    two readers of one endpoint disagree about its wire format."""
    ROUTES["/cursor_position"] = _json_route(payload)
    with pytest.raises(GuestAgentError, match="invalid cursor position"):
        agent.cursor_position()


def test_wait_ready_returns_once_the_agent_answers_with_a_body(agent):
    ROUTES["/screenshot"] = (200, PNG, "image/png")
    agent.wait_ready(timeout_s=5.0, poll_s=0.05)


def test_wait_ready_keeps_polling_a_200_with_an_empty_body(agent):
    """A 200 with no body is an agent that is up without an X server behind it."""
    ROUTES["/screenshot"] = (200, b"", "image/png")
    with pytest.raises(TimeoutError, match="not ready"):
        agent.wait_ready(timeout_s=0.4, poll_s=0.05)


def test_wait_ready_becomes_ready_after_a_few_failures(agent):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            return (500, b"", "text/plain")
        return (200, PNG, "image/png")

    ROUTES["/screenshot"] = flaky
    agent.wait_ready(timeout_s=5.0, poll_s=0.05)
    assert calls["n"] >= 3


def test_wait_ready_on_an_unreachable_agent_times_out():
    client = OSWorldClient("http://127.0.0.1:1", timeout_s=0.2)
    with pytest.raises(TimeoutError, match="not ready"):
        client.wait_ready(timeout_s=0.4, poll_s=0.05)


def test_settled_with_zero_delays_is_exactly_one_screenshot(agent):
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return (200, PNG, "image/png")

    ROUTES["/screenshot"] = counting
    assert agent.screenshot_settled() == PNG
    assert calls["n"] == 1


def test_settled_returns_as_soon_as_two_frames_match(agent):
    frames = [PNG, PNG + b"a", PNG + b"b", PNG + b"b", PNG + b"c"]
    calls = {"n": 0}

    def changing():
        body = frames[min(calls["n"], len(frames) - 1)]
        calls["n"] += 1
        return (200, body, "image/png")

    ROUTES["/screenshot"] = changing
    assert agent.screenshot_settled(stability_timeout_s=5.0, poll_s=0.01) == PNG + b"b"


def test_settled_returns_the_last_frame_when_the_desktop_never_stops_moving(agent):
    """A blinking caret prevents stability; waiting forever is not an option."""
    calls = {"n": 0}

    def always_changing():
        calls["n"] += 1
        return (200, PNG + str(calls["n"]).encode(), "image/png")

    ROUTES["/screenshot"] = always_changing
    body = agent.screenshot_settled(stability_timeout_s=0.3, poll_s=0.01)
    assert body.startswith(PNG)
    assert calls["n"] > 2


def test_settled_honours_the_minimum_delay(agent):
    import time

    ROUTES["/screenshot"] = (200, PNG, "image/png")
    started = time.monotonic()
    agent.screenshot_settled(min_delay_s=0.2)
    assert time.monotonic() - started >= 0.2


# ``write_file`` and ``execute_detached`` are almost entirely program text: one
# Python one-liner and three POSIX shell scripts.  An agent that recorded the
# argv would assert that this module composed the strings it meant to compose and
# nothing about whether they work, so the fake below shells out for real and the
# guest's filesystem is this host's.

#: Reject the next N ``/execute`` calls with a 503, standing in for the guest
#: agent restarting under a command.  Reset per test by the fixture.
AGENT_FAULTS: dict = {"reject_next": 0}
#: Every request body the executing agent accepted, in order.
AGENT_CALLS: list = []


class _ExecutingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        if AGENT_FAULTS["reject_next"] > 0:
            AGENT_FAULTS["reject_next"] -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        AGENT_CALLS.append(body)
        assert body["shell"] is False, "the agent is never asked for a shell"
        done = subprocess.run(body["command"], capture_output=True, text=True)
        payload = json.dumps(
            {
                "status": "success" if done.returncode == 0 else "error",
                "returncode": done.returncode,
                "output": done.stdout,
                "error": done.stderr,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def executing_agent_server():
    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _ExecutingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def executing_agent(executing_agent_server):
    AGENT_FAULTS["reject_next"] = 0
    AGENT_CALLS.clear()
    return OSWorldClient(
        f"http://127.0.0.1:{executing_agent_server.server_port}", timeout_s=30.0
    )


def test_a_pushed_file_arrives_byte_for_byte(executing_agent, tmp_path):
    """Bytes that are not text and not valid UTF-8: the payload crosses as base64
    precisely so nothing on either side interprets it."""
    target = tmp_path / "payload.bin"
    content = bytes(range(256)) + b"\n'\"\\ \x00 end"
    executing_agent.write_file(str(target), content)
    assert target.read_bytes() == content


def test_a_pushed_file_is_never_program_text_the_guest_could_run(
    executing_agent, tmp_path
):
    target = tmp_path / "quoted.txt"
    dangerous = b"#!/bin/sh\nshutdown --now\n"
    executing_agent.write_file(str(target), dangerous)
    assert target.read_bytes() == dangerous
    command = AGENT_CALLS[-1]["command"]
    assert command[:2] == ["python3", "-c"]
    # The content rides as base64 inside the program, so no byte of it is ever
    # shell or Python syntax.
    assert "shutdown" not in command[2]


def test_a_relative_guest_path_is_refused(executing_agent):
    with pytest.raises(ValueError, match="must be absolute"):
        executing_agent.write_file("relative/path.txt", b"x")
    assert AGENT_CALLS == [], "a refused path must not reach the guest"


def test_a_file_too_large_for_one_argv_element_is_refused(executing_agent, tmp_path):
    with pytest.raises(ValueError, match="do not fit in one guest argv element"):
        executing_agent.write_file(str(tmp_path / "huge.bin"), b"x" * 99_000)
    assert AGENT_CALLS == []


def test_the_largest_permitted_file_really_reaches_the_guest(executing_agent, tmp_path):
    """The cap is the kernel's own ``MAX_ARG_STRLEN``, so the size just under it
    has to work: a cap picked with margin to spare would pass the refusal test
    above while refusing files the transport can carry."""
    target = tmp_path / "big.bin"
    content = b"x" * 97_000
    executing_agent.write_file(str(target), content)
    assert target.read_bytes() == content


def test_a_missing_parent_directory_is_reported_rather_than_created(
    executing_agent, tmp_path
):
    """A caller that named a path whose directory does not exist named the wrong
    path; creating it silently is guessing."""
    with pytest.raises(GuestAgentError, match="guest command failed"):
        executing_agent.write_file(str(tmp_path / "absent" / "f.txt"), b"x")
    assert not (tmp_path / "absent").exists()


def test_a_detached_command_returns_its_code_and_both_streams(executing_agent):
    result = executing_agent.execute_detached(
        ["sh", "-c", "printf to-out; printf to-err >&2; exit 3"], timeout_s=20.0
    )
    assert result.returncode == 3
    assert result.stdout == "to-out"
    assert result.stderr == "to-err"


def test_a_detached_command_does_not_wait_for_a_survivor_it_spawned(executing_agent):
    """The whole reason this exists.  The agent runs a command as a child and
    holds its pipes, so a daemon that inherits them keeps ``/execute`` blocked
    long after the command exited -- here for the five seconds of the ``sleep``.
    Redirecting the guest's stdio to files is what breaks that hold."""
    started = time.monotonic()
    result = executing_agent.execute_detached(
        ["sh", "-c", "sleep 5 & echo started"], timeout_s=30.0
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 0
    assert result.stdout.strip() == "started"
    assert elapsed < 4.0, f"a survivor held the call for {elapsed:.1f}s"


def test_a_detached_command_that_overruns_its_timeout_raises_and_cleans_up(
    executing_agent,
):
    with pytest.raises(TimeoutError, match="did not finish within"):
        executing_agent.execute_detached(["sleep", "5"], timeout_s=0.5)
    base = AGENT_CALLS[0]["command"][4]
    for suffix in (".pid", ".rc", ".out", ".err"):
        assert not Path(base + suffix).exists(), f"{suffix} survived the timeout"
    discard = AGENT_CALLS[-1]["command"]
    assert discard[2].startswith("base=$1\nif [ -n \"$2\" ]; then kill")
    assert discard[5].isdigit(), "a timeout must discard with the pid, to kill it"


def test_a_finished_detached_command_is_discarded_without_a_kill(executing_agent):
    executing_agent.execute_detached(["true"], timeout_s=20.0)
    discard = AGENT_CALLS[-1]["command"]
    assert discard[2].startswith("base=$1\nif [ -n \"$2\" ]; then kill")
    assert discard[5] == "", "nothing is left to kill, so nothing may be killed"


def test_the_environment_reaches_the_command(executing_agent):
    result = executing_agent.execute_detached(
        ["sh", "-c", 'printf %s "$DESKTOP_TEST_TOKEN"'],
        timeout_s=20.0,
        env={"DESKTOP_TEST_TOKEN": "a value with spaces"},
    )
    assert result.stdout == "a value with spaces"


def test_an_environment_name_containing_an_equals_sign_is_refused(executing_agent):
    """``env`` splits on the FIRST ``=``, so ``{"A=B": "c"}`` would silently set
    ``A`` to ``B=c`` instead of failing."""
    with pytest.raises(ValueError, match="invalid guest command environment name"):
        executing_agent.execute_detached(["true"], timeout_s=5.0, env={"A=B": "c"})
    assert AGENT_CALLS == []


@pytest.mark.parametrize("argv", [[], [""], ["ok", ""]])
def test_an_empty_argv_element_is_refused(executing_agent, argv):
    with pytest.raises(ValueError, match="at least one non-empty string"):
        executing_agent.execute_detached(argv, timeout_s=5.0)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_a_non_positive_detached_timeout_is_refused(executing_agent, timeout_s):
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        executing_agent.execute_detached(["true"], timeout_s=timeout_s)


def test_a_detached_command_survives_a_restart_of_the_guest_agent(executing_agent):
    """A guest agent restart takes every open connection with it and answers
    again within seconds; each of the three control calls is idempotent so that
    the command underneath is unaffected."""
    AGENT_FAULTS["reject_next"] = 2
    result = executing_agent.execute_detached(["echo", "still here"], timeout_s=20.0)
    assert result.returncode == 0
    assert result.stdout.strip() == "still here"
    assert AGENT_FAULTS["reject_next"] == 0


def test_a_guest_channel_that_never_answers_gives_up(executing_agent, monkeypatch):
    """Eight attempts, not forever.  The backoff is flattened because the count
    and the give-up are under test, not the wall-clock of the sleeps."""
    import desktop.vm.osworld_client as client_module

    monkeypatch.setattr(client_module, "_DETACHED_RETRY_BACKOFF_S", 0.0)
    AGENT_FAULTS["reject_next"] = 8
    with pytest.raises(GuestAgentError, match="channel failed 8 times"):
        executing_agent.execute_detached(["true"], timeout_s=20.0)
    assert AGENT_FAULTS["reject_next"] == 0, "it must have used every attempt"


@pytest.mark.parametrize(
    "output", ["", "0\n", "0\nYQ==\n\nextra\n", "not-a-code\nYQ==\n\n"]
)
def test_a_malformed_detached_result_is_refused(output):
    """Three lines, the first an exit code.  Anything else is a guest that
    answered something other than the collect script's output."""
    with pytest.raises(GuestAgentError, match="invalid detached guest command result"):
        _decode_detached_result({"returncode": 0, "output": output})
