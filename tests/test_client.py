from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from desktop import ir
from desktop.execute import ACTION_CONTRACT, ExecutionError, ExecutionReceipt
from desktop.vm.client import ACTION_EXECUTOR_SHA256, DesktopClient, GuestAgentError
from tests.support.guest_runner import run_guest_program

STATE: dict = {}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._respond()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        STATE.setdefault("requests", []).append(
            {"path": self.path, "body": json.loads(raw) if raw else None}
        )
        self._respond()

    def _respond(self):
        route = STATE.get("routes", {}).get(self.path)
        if route is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status, payload = route() if callable(route) else route
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def agent_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def client(agent_server):
    STATE.clear()
    STATE["cursor"] = [50, 50]
    STATE["requests"] = []
    STATE["routes"] = {
        "/v1/actions": (
            200,
            {
                "contract": ACTION_CONTRACT,
                "executor_sha256": ACTION_EXECUTOR_SHA256,
            },
        ),
        "/cursor_position": lambda: (200, list(STATE["cursor"])),
    }
    return DesktopClient(f"http://127.0.0.1:{agent_server.server_port}", timeout_s=5)


def _action_route(operations, **runner_kwargs):
    run = run_guest_program(operations, **runner_kwargs)
    assert run.payload is not None, run.stderr
    payload = dict(run.payload)
    payload["executor_returncode"] = run.returncode
    STATE["cursor"] = list(payload["cursor_before"])

    def respond():
        STATE["cursor"] = list(payload["cursor_after"])
        return 200, payload

    STATE["routes"]["/v1/actions"] = respond
    return payload


def test_the_client_attests_the_installed_executor(client):
    client.verify_actions_contract()


def test_an_old_guest_image_is_refused(client):
    STATE["routes"]["/v1/actions"] = (404, {"status": "missing"})
    with pytest.raises(GuestAgentError, match="returned 404"):
        client.verify_actions_contract()


def test_one_action_round_trips_as_json_and_returns_one_receipt(client):
    operations = (ir.move_to(300, 400), ir.click("left"))
    _action_route(operations)
    receipt = client.execute(operations)
    assert isinstance(receipt, ExecutionReceipt)
    assert receipt.ok
    assert receipt.cursor_before == (50, 50)
    assert receipt.cursor_after == (300, 400)
    assert receipt.cursor_readback_verified
    requests = [row for row in STATE["requests"] if row["path"] == "/v1/actions"]
    assert len(requests) == 1
    assert requests[0]["body"]["contract"] == ACTION_CONTRACT
    assert "command" not in requests[0]["body"]


def test_held_input_state_is_verified_across_requests(client):
    down = (ir.mouse_down("left"),)
    _action_route(down)
    assert client.execute(down).ok
    assert client.audit.held_buttons == {"left"}

    up = (ir.mouse_up("left"),)
    _action_route(up, initial_buttons={"left"}, initial_mask=1 << 8)
    assert client.execute(up).ok
    request = STATE["requests"][-1]["body"]
    assert request["expected_initial_mask"] == 1 << 8
    assert client.audit.held_buttons == set()


def test_a_guest_failure_is_a_receipt(client):
    operations = (ir.Operation("raise_for_test", ("boom",)),)
    _action_route(operations)
    receipt = client.execute(operations)
    assert not receipt.ok
    assert receipt.failure_kind == "injected"
    assert "boom" in receipt.error


def test_host_cursor_disagreement_fails_verification(client):
    operations = (ir.move_to(10, 20),)
    payload = _action_route(operations)

    def lie():
        return 200, [999, 999]

    action_response = STATE["routes"]["/v1/actions"]

    def respond():
        status, body = action_response()
        STATE["routes"]["/cursor_position"] = lie
        return status, body

    STATE["routes"]["/v1/actions"] = respond
    STATE["cursor"] = payload["cursor_before"]
    receipt = client.execute(operations)
    assert not receipt.ok
    assert receipt.failure_kind == "verification"
    assert not receipt.cursor_readback_verified


def test_an_invalid_result_schema_is_refused(client):
    operations = (ir.move_to(1, 2),)
    STATE["routes"]["/v1/actions"] = (200, {"schema_version": 999})
    with pytest.raises(ExecutionError, match="unexpected schema"):
        client.execute(operations)


def test_operations_fail_before_network_io(client):
    before = len(STATE["requests"])
    with pytest.raises(ExecutionError, match="non-empty tuple"):
        client.execute([])  # type: ignore[arg-type]
    assert len(STATE["requests"]) == before


def test_guest_command_results_have_one_typed_shape(client):
    STATE["routes"]["/execute"] = (
        200,
        {"status": "error", "returncode": 3, "output": "out", "error": "err"},
    )
    result = client.run_guest(["false"])
    assert (result.returncode, result.stdout, result.stderr) == (3, "out", "err")
