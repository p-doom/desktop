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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from desktop_env.vm.osworld_client import GuestAgentError, OSWorldClient

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


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_a_trailing_slash_in_the_base_url_is_normalised():
    assert OSWorldClient("http://host:5000/").base_url == "http://host:5000"
    assert OSWorldClient("http://host:5000///").base_url == "http://host:5000"


# --------------------------------------------------------------------------- #
# /screenshot
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# /screen_size
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# /execute
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# /accessibility
# --------------------------------------------------------------------------- #


def test_the_accessibility_tree_is_returned_as_text(agent):
    ROUTES["/accessibility"] = (200, "<tree/>", "text/xml")
    assert agent.accessibility() == "<tree/>"


@pytest.mark.parametrize("key", ["AT", "at", "accessibility_tree", "tree"])
def test_a_wrapped_accessibility_tree_is_unwrapped(agent, key):
    ROUTES["/accessibility"] = _json_route({key: "<tree/>"})
    assert agent.accessibility() == "<tree/>"


def test_an_unrecognised_accessibility_object_is_returned_verbatim(agent):
    ROUTES["/accessibility"] = _json_route({"unexpected": "x"})
    assert agent.accessibility() == '{"unexpected": "x"}'


def test_undecodable_accessibility_bytes_are_replaced_not_raised(agent):
    """A best-effort decode is right here: the tree is the platform's schema and
    a mangled character is better than losing the whole tree."""
    ROUTES["/accessibility"] = (200, b"<tree>\xff</tree>", "text/xml")
    assert "�" in agent.accessibility()


def test_a_non_200_accessibility_is_an_error(agent):
    ROUTES["/accessibility"] = (404, b"", "text/plain")
    with pytest.raises(GuestAgentError, match="/accessibility returned 404"):
        agent.accessibility()


# --------------------------------------------------------------------------- #
# /cursor_position
# --------------------------------------------------------------------------- #


def test_a_cursor_position_object_is_parsed(agent):
    ROUTES["/cursor_position"] = _json_route({"x": 11, "y": 22})
    assert agent.cursor_position() == (11, 22)


def test_a_cursor_position_pair_is_parsed(agent):
    ROUTES["/cursor_position"] = _json_route([11, 22])
    assert agent.cursor_position() == (11, 22)


@pytest.mark.parametrize("payload", [{"x": 1}, [1], [1, 2, 3], "11,22", None])
def test_a_malformed_cursor_position_is_an_error(agent, payload):
    ROUTES["/cursor_position"] = _json_route(payload)
    with pytest.raises(GuestAgentError, match="invalid cursor position"):
        agent.cursor_position()


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# screenshot_settled
# --------------------------------------------------------------------------- #


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
