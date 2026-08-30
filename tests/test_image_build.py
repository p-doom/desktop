"""The guest-image builder, exercised without ever booting a VM.

Nothing here has been run against a real image build: no qcow2 is created, no
guest boots, no apt or pip runs.  What IS checked is everything a build gets
wrong silently -- the QEMU argv, the overlay argv, the provision and verification
scripts, the manifest, and the parsing of what the guest sends back -- because
those are the parts whose failure is only visible hours later, in a published
image, as a grader that scores the wrong thing.

The verification probe is not merely string-matched: its Python body is extracted
from the heredoc and EXECUTED here, with ``open`` shadowed to stand in for the
guest's ``main.py``.  That is the only way to know the generated program parses,
runs, and really does put one JSON object on its last stdout line.

The guest-facing paths run against a real ``http.server`` speaking the in-VM
agent's ``/execute`` and ``/screenshot``, so ``OSWorldClient`` and the module's
own error handling are in the loop rather than mocked away.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import desktop.vm.image_build as image_build
from desktop.vm.image_build import (
    AGENT_INSTALLED_MODULES,
    DEBUG_SERVER_CALL,
    GUEST_MODULE_PROBE,
    GUEST_SCREENSHOT_PATCH,
    GUEST_SERVER_SOURCE,
    GUEST_TOOL_PROBE,
    PRODUCTION_SERVER_CALL,
    XCFTOOLS_DEB,
    BuildReport,
    DebArtifact,
    DesktopImageBuildConfig,
    DesktopImageBuilder,
    GuestCommandError,
    build_manifest,
    image_convert_argv,
    overlay_create_argv,
    qemu_argv,
    render_provision_script,
    render_verification_script,
)
from desktop.vm.observation import OBSERVATION_CONTRACT, OBSERVATION_SIZE
from desktop.vm.osworld_client import SECRET_STDIN_EXECUTE_CONTRACT


@pytest.fixture
def config(tmp_path) -> DesktopImageBuildConfig:
    upstream = tmp_path / "upstream.qcow2"
    upstream.write_bytes(b"QFI\xfb" + b"\x00" * 64)
    return DesktopImageBuildConfig(
        upstream=upstream,
        output=tmp_path / "provisioned.qcow2",
        runtime_dir=tmp_path / "runtime",
    )


def _flag(argv: list[str], name: str) -> str:
    """The value that follows a QEMU flag, so a test names the flag not an index."""
    return argv[argv.index(name) + 1]


# --------------------------------------------------------------------------
# configuration


@pytest.mark.parametrize("field", ["upstream", "output", "runtime_dir"])
def test_a_relative_path_is_refused_before_anything_boots(tmp_path, field):
    paths = {
        "upstream": tmp_path / "a.qcow2",
        "output": tmp_path / "b.qcow2",
        "runtime_dir": tmp_path / "run",
    }
    paths[field] = Path("relative/path")
    with pytest.raises(ValueError, match="must be an absolute path"):
        DesktopImageBuildConfig(**paths)


def test_the_build_may_not_overwrite_the_image_it_provisions_from(tmp_path):
    with pytest.raises(ValueError, match="must not overwrite its upstream"):
        DesktopImageBuildConfig(
            upstream=tmp_path / "a.qcow2",
            output=tmp_path / "a.qcow2",
            runtime_dir=tmp_path / "run",
        )


def test_a_zero_cpu_build_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cpu_cores must be positive"):
        DesktopImageBuildConfig(
            upstream=tmp_path / "a.qcow2",
            output=tmp_path / "b.qcow2",
            runtime_dir=tmp_path / "run",
            cpu_cores=0,
        )


def test_the_partial_and_manifest_paths_sit_beside_the_output(config):
    assert config.partial_path.name == "provisioned.qcow2.partial"
    assert config.manifest_path.name == "provisioned.qcow2.build.json"
    assert config.partial_path.parent == config.output.parent


# --------------------------------------------------------------------------
# argv


def test_the_build_boot_always_enables_kvm(config):
    argv = qemu_argv(config, Path("/img.qcow2"), 20000, Path("/tmp/qmp.sock"))
    assert "-enable-kvm" in argv
    assert _flag(argv, "-cpu") == "host"
    # There is no accelerator knob to turn: a TCG build would run apt and pip for
    # hours to produce the same bytes, so the argv must not be reachable without
    # /dev/kvm.
    assert "tcg" not in " ".join(argv)


def test_the_image_under_build_is_not_opened_with_snapshot_on(config):
    """``QemuRuntime`` boots ``snapshot=on``; a build must do the opposite."""
    argv = qemu_argv(config, Path("/images/work.qcow2"), 20000, Path("/tmp/qmp.sock"))
    drive = _flag(argv, "-drive")
    assert drive.startswith("file=/images/work.qcow2,")
    assert "snapshot" not in drive


def test_the_guest_agent_port_is_forwarded_from_the_leased_port(config):
    argv = qemu_argv(config, Path("/img.qcow2"), 20450, Path("/tmp/qmp.sock"))
    assert _flag(argv, "-netdev") == (
        "user,id=net0,hostfwd=tcp:127.0.0.1:20450-:5000"
    )


def test_a_qmp_monitor_is_requested_so_the_guest_can_be_powered_down(config):
    argv = qemu_argv(config, Path("/img.qcow2"), 20000, Path("/tmp/deqmp.sock"))
    assert _flag(argv, "-qmp") == "unix:/tmp/deqmp.sock,server=on,wait=off"


def test_the_configured_qemu_binary_is_the_one_executed(tmp_path):
    config = DesktopImageBuildConfig(
        upstream=tmp_path / "a.qcow2",
        output=tmp_path / "b.qcow2",
        runtime_dir=tmp_path / "run",
        qemu_binary="/opt/kvm/bin/qemu-system-x86_64",
    )
    argv = qemu_argv(config, Path("/img.qcow2"), 20000, Path("/tmp/qmp.sock"))
    assert argv[0] == "/opt/kvm/bin/qemu-system-x86_64"


def test_no_container_wrapper_survives_into_the_boot(config):
    """The port's one deliberate deletion: QEMU runs on the host, not in a SIF."""
    argv = qemu_argv(config, Path("/img.qcow2"), 20000, Path("/tmp/qmp.sock"))
    assert "apptainer" not in " ".join(argv)
    assert "--bind" not in argv
    assert "apptainer" not in Path(image_build.__file__).read_text().lower()


def test_the_overlay_command_names_the_backing_format_explicitly():
    """Modern ``qemu-img`` refuses to probe a backing format; ``-F`` is required."""
    argv = overlay_create_argv("qemu-img", Path("/base.qcow2"), Path("/child.qcow2"))
    assert argv == [
        "qemu-img",
        "create",
        "-q",
        "-f",
        "qcow2",
        "-F",
        "qcow2",
        "-b",
        "/base.qcow2",
        "/child.qcow2",
    ]


def test_the_published_image_command_flattens_its_source():
    argv = image_convert_argv(
        "qemu-img", Path("/base.qcow2"), Path("/published.qcow2")
    )
    assert argv == [
        "qemu-img",
        "convert",
        "-f",
        "qcow2",
        "-O",
        "qcow2",
        "/base.qcow2",
        "/published.qcow2",
    ]


def test_a_failing_qemu_img_surfaces_its_output(config, monkeypatch):
    monkeypatch.setattr(
        image_build.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "out", "boom"),
    )
    builder = DesktopImageBuilder(config, log=lambda _: None)
    with pytest.raises(RuntimeError, match="qemu-img create failed \\(1\\)") as raised:
        builder._create_overlay(config.upstream, config.partial_path)
    assert "boom" in str(raised.value)


# --------------------------------------------------------------------------
# provision script


def test_the_provision_script_is_valid_bash(config):
    result = subprocess.run(
        ["bash", "-n"],
        input=render_provision_script(config),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_provision_script_rewrites_the_debug_server_call_and_proves_it(config):
    """A Flask reloader in the guest drops requests held across a site-packages
    write, which is why the rewrite is verified in the same script rather than
    trusted."""
    script = render_provision_script(config)
    assert f"sed -i 's|{DEBUG_SERVER_CALL}|{PRODUCTION_SERVER_CALL}|'" in script
    assert f"grep -qF '{PRODUCTION_SERVER_CALL}' {GUEST_SERVER_SOURCE}" in script
    assert script.splitlines()[0] == "set -eux"


def _patch_preimage() -> str:
    lines: list[str] = []
    in_hunk = False
    for line in GUEST_SCREENSHOT_PATCH.read_text().splitlines(keepends=True):
        if line.startswith("@@ "):
            in_hunk = True
            continue
        if in_hunk and line[:1] in {" ", "-"}:
            lines.append(line[1:])
    return "".join(lines)


def _patch_check(tmp_path, source: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "main.py").write_text(source)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return subprocess.run(
        ["git", "-C", str(tmp_path), "apply", "--check", str(GUEST_SCREENSHOT_PATCH)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_guest_patch_applies_to_its_exact_expected_source_shape(tmp_path):
    checked = _patch_check(tmp_path, _patch_preimage())
    assert checked.returncode == 0, checked.stderr
    applied = subprocess.run(
        ["git", "-C", str(tmp_path), "apply", str(GUEST_SCREENSHOT_PATCH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    source = (tmp_path / "main.py").read_text()
    assert OBSERVATION_CONTRACT in source
    assert 'format="JPEG", quality=92, subsampling=2, optimize=False' in source
    assert 'mimetype="image/jpeg"' in source
    assert source.count("screenshot.save(") == 1
    assert source.index("screenshot.paste(cursor_img") < source.index("screenshot.save(")
    assert "screenshot.size != (1920, 1080)" in source
    assert "screenshot.png" not in source
    assert "image/png" not in source
    assert f'SECRET_STDIN_EXECUTE_CONTRACT = "{SECRET_STDIN_EXECUTE_CONTRACT}"' in source
    assert "input=secret_stdin" in source
    assert "stdout=subprocess.DEVNULL" in source
    assert "stderr=subprocess.DEVNULL" in source
    assert "'contract': SECRET_STDIN_EXECUTE_CONTRACT" in source
    assert "'output': result.stdout" in source
    assert "'error': result.stderr" in source


def test_the_guest_patch_refuses_a_changed_source_shape(tmp_path):
    source = _patch_preimage().replace(
        "screenshot.save(file_path)", "screenshot.save(file_path, format='PNG')"
    )
    checked = _patch_check(tmp_path, source)
    assert checked.returncode != 0


def test_the_provision_script_applies_and_checks_the_canonical_guest_patch(config):
    script = render_provision_script(config)
    assert "patch --batch --forward --fuzz=0 -p1 -d /home/user/server" in script
    assert GUEST_SCREENSHOT_PATCH.read_text() in script
    assert f"grep -qF '{OBSERVATION_CONTRACT}' {GUEST_SERVER_SOURCE}" in script
    assert "quality=92, subsampling=2, optimize=False" in script
    assert 'mimetype="image/jpeg"' in script


def test_every_declared_deb_is_checksummed_before_it_is_installed(tmp_path):
    config = DesktopImageBuildConfig(
        upstream=tmp_path / "a.qcow2",
        output=tmp_path / "b.qcow2",
        runtime_dir=tmp_path / "run",
        deb_artifacts=(
            XCFTOOLS_DEB,
            DebArtifact(url="http://example/two.deb", sha256="beef" * 16),
        ),
    )
    lines = render_provision_script(config).splitlines()
    for index, artifact in enumerate(config.deb_artifacts):
        local = f"/tmp/desktop_image_{index}.deb"
        fetch = lines.index(f"retry curl -fsSL -o {local} {artifact.url}")
        check = lines.index(f"echo '{artifact.sha256}  {local}' | sha256sum -c -")
        install = lines.index(f"retry dpkg -i {local} || retry apt-get -f install -y")
        assert fetch < check < install


def test_every_declared_package_reaches_the_installer(config):
    script = render_provision_script(config)
    for package in config.apt_packages:
        assert package in script
    for package in config.pip_packages:
        assert package in script


# --------------------------------------------------------------------------
# the verification probe, actually executed


_PROBE_PREFIX = "python3 - <<'PY'\n"
_PROBE_SUFFIX = "\nPY"


def _probe_body(script: str) -> str:
    assert script.startswith(_PROBE_PREFIX) and script.endswith(_PROBE_SUFFIX)
    return script[len(_PROBE_PREFIX) : -len(_PROBE_SUFFIX)]


def _run_probe(guest_source: str) -> dict:
    """Execute the generated probe here, with ``open`` standing in for the guest.

    ``open`` is the probe's only unsatisfiable dependency off a guest; everything
    else -- the imports, ``pgrep``, ``shutil.which`` -- runs for real.
    """
    program = compile(_probe_body(render_verification_script()), "<probe>", "exec")
    namespace: dict = {"open": lambda *a, **k: io.StringIO(guest_source)}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(program, namespace)
    return json.loads(captured.getvalue().strip().splitlines()[-1])


def test_the_probe_runs_and_puts_one_json_object_on_its_last_stdout_line():
    payload = _run_probe(f"    {PRODUCTION_SERVER_CALL}\n")
    assert set(payload) == {
        "missing_modules",
        "tools",
        "server_pids",
        "agent_installed_modules",
        "production_server_call",
    }
    assert set(payload["tools"]) == set(GUEST_TOOL_PROBE)


def test_the_probe_reports_exactly_the_modules_it_could_not_import():
    expected = [
        name
        for name in GUEST_MODULE_PROBE
        if importlib.util.find_spec(name) is None
    ]
    assert _run_probe("")["missing_modules"] == expected


def test_the_probe_detects_an_agent_installed_module_that_leaked_in():
    """``pytest`` is one of the modules a grader scores by its ABSENCE, and it is
    installed here -- so this asserts the detection branch, not just the empty
    case."""
    assert "pytest" in AGENT_INSTALLED_MODULES
    assert "pytest" in _run_probe("")["agent_installed_modules"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [(PRODUCTION_SERVER_CALL, True), (DEBUG_SERVER_CALL, False)],
)
def test_the_probe_reads_the_server_call_out_of_the_guests_own_source(
    source, expected
):
    assert _run_probe(f"if __name__ == '__main__':\n    {source}\n")[
        "production_server_call"
    ] is expected


# --------------------------------------------------------------------------
# the guest-facing paths, against a real HTTP server


ROUTES: dict = {}


class _Handler(BaseHTTPRequestHandler):
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
        if self.path == "/execute":
            body = json.dumps(
                {"contract": SECRET_STDIN_EXECUTE_CONTRACT}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._respond()

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
        )
        if "secret_stdin_base64" in body:
            assert set(body) == {
                "command",
                "shell",
                "contract",
                "secret_stdin_base64",
            }
            assert body["shell"] is False
            assert body["contract"] == SECRET_STDIN_EXECUTE_CONTRACT
            secret = base64.b64decode(body["secret_stdin_base64"], validate=True)
            done = subprocess.run(
                body["command"],
                input=secret,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            response = json.dumps(
                {
                    "contract": SECRET_STDIN_EXECUTE_CONTRACT,
                    "status": "success",
                    "returncode": done.returncode,
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return
        self._respond()


@pytest.fixture(scope="module")
def agent_server():
    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def guest(config, agent_server):
    """A builder wired to the fake in-VM agent, with no QEMU behind it."""
    ROUTES.clear()
    builder = DesktopImageBuilder(config, log=lambda _: None)
    builder._client = image_build.OSWorldClient(agent_server)
    return builder


def _execute(output: str, *, returncode: int = 0, error: str = "") -> tuple:
    body = json.dumps(
        {
            "status": "success" if returncode == 0 else "error",
            "returncode": returncode,
            "output": output,
            "error": error,
        }
    )
    return (200, body, "application/json")


def _jpeg(fill: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", OBSERVATION_SIZE, (fill, fill, fill)).save(
        buffer, format="JPEG", quality=92, subsampling=2, optimize=False
    )
    return buffer.getvalue()


def _structured_jpeg() -> bytes:
    image = Image.new("RGB", OBSERVATION_SIZE, (0, 0, 0))
    ImageDraw.Draw(image).rectangle((0, 0, 1919, 539), fill=(200, 200, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, subsampling=2, optimize=False)
    return buffer.getvalue()


VERIFICATION_OUTPUT = {
    "missing_modules": [],
    "tools": {name: f"/usr/bin/{name}" for name in GUEST_TOOL_PROBE},
    "server_pids": ["1234"],
    "agent_installed_modules": [],
    "production_server_call": True,
}

RELOADER_OUTPUT = {
    "site_package_rewritten": "/usr/lib/python3/dist-packages/flask/__init__.py",
    "held_request_returncode": 0,
    "held_request_error": None,
    "site_packages_write_seconds": 0.4,
    "concurrent": True,
    "pids_before": ["1234"],
    "pids_after": ["1234"],
    "pids_stable": True,
}

VERIFIED_CHECKS = {
    **VERIFICATION_OUTPUT,
    "secret_stdin_execute_contract": SECRET_STDIN_EXECUTE_CONTRACT,
    "reloader_disabled": RELOADER_OUTPUT,
}


def test_verify_reads_the_last_stdout_line_of_the_probe_as_json(guest, monkeypatch):
    """The guest prints apt/pip noise before the object; only the last line is it."""
    monkeypatch.setattr(DesktopImageBuilder, "_probe_reloader", lambda self: {})
    ROUTES["/execute"] = _execute(
        "Reading package lists...\nWARNING: noise\n"
        + json.dumps(VERIFICATION_OUTPUT)
        + "\n"
    )
    checks = guest._verify()
    assert checks["server_pids"] == ["1234"]
    assert checks["production_server_call"] is True
    assert checks["secret_stdin_execute_contract"] == SECRET_STDIN_EXECUTE_CONTRACT
    assert checks["tools"]["xcf2png"] == "/usr/bin/xcf2png"
    assert guest.report.steps["verify"] >= 0


def test_verify_surfaces_a_grader_package_the_guest_could_not_import(
    guest, monkeypatch
):
    monkeypatch.setattr(DesktopImageBuilder, "_probe_reloader", lambda self: {})
    ROUTES["/execute"] = _execute(
        json.dumps({**VERIFICATION_OUTPUT, "missing_modules": ["pandas", "pikepdf"]})
    )
    assert guest._verify()["missing_modules"] == ["pandas", "pikepdf"]


def test_a_probe_that_returns_nothing_parseable_fails_the_verification(
    guest, monkeypatch
):
    monkeypatch.setattr(DesktopImageBuilder, "_probe_reloader", lambda self: {})
    ROUTES["/execute"] = _execute("Traceback (most recent call last):\n")
    with pytest.raises(json.JSONDecodeError):
        guest._verify()


def test_a_guest_script_that_exits_non_zero_raises(guest):
    ROUTES["/execute"] = _execute("", returncode=3, error="No such file")
    with pytest.raises(GuestCommandError, match="Guest script failed \\(3\\).*No such"):
        guest.guest_bash("false")


def test_a_root_script_is_handed_to_sudo_as_one_quoted_argument(guest, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        DesktopImageBuilder,
        "guest_bash",
        lambda self, script, *, timeout_s=120.0: (seen.append(script), "")[1],
    )
    guest.guest_root_bash("echo hello")
    assert seen == ["echo password | sudo -S -k bash -c 'echo hello'"]


def test_the_whole_provision_script_survives_nesting_inside_sudo_bash_c(config):
    """The load-bearing case: the provision script itself is full of single
    quotes (``sed -i 's|..|..|'``, ``grep -qF '..'``, the sha256 echo), and it is
    handed to the guest through TWO layers of ``bash -c``."""
    script = render_provision_script(config)
    result = subprocess.run(
        ["bash", "-c", f"printf '%s' {image_build._shell_quote(script)}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == script


def _script(*responses):
    """Answer ``/execute`` with a different response per call, last one repeating."""
    queue = list(responses)
    return lambda: queue.pop(0) if len(queue) > 1 else queue[0]


def test_a_provision_that_reports_a_non_zero_marker_raises_with_the_guests_log(
    guest, monkeypatch
):
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    ROUTES["/execute"] = _script(
        _execute(""),  # the setsid launch
        _execute("100\n"),  # the marker
        _execute("E: Unable to locate package qpdf\n"),  # the log tail
    )
    with pytest.raises(GuestCommandError, match="Guest provisioning failed") as raised:
        guest._provision()
    assert "Unable to locate package qpdf" in str(raised.value)
    assert "provision" not in guest.report.steps


def test_a_provision_that_never_finishes_times_out_carrying_the_log(
    config, agent_server, monkeypatch
):
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    ROUTES.clear()
    ROUTES["/execute"] = _script(_execute(""), _execute("still installing pandas\n"))
    builder = DesktopImageBuilder(
        DesktopImageBuildConfig(
            upstream=config.upstream,
            output=config.output,
            runtime_dir=config.runtime_dir,
            provision_timeout_s=0.0,
        ),
        log=lambda _: None,
    )
    builder._client = image_build.OSWorldClient(agent_server)
    with pytest.raises(GuestCommandError, match="did not finish within 0.0s") as raised:
        builder._provision()
    assert "still installing pandas" in str(raised.value)


def test_a_provision_that_reports_zero_is_recorded_as_a_timed_step(guest, monkeypatch):
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    ROUTES["/execute"] = _script(_execute(""), _execute("0\n"))
    guest._provision()
    assert guest.report.steps["provision"] >= 0


def test_the_provision_marker_is_read_through_a_guest_that_is_still_restarting(guest):
    """``_tolerant_bash`` must swallow the debug reloader's dropped requests --
    the marker poll runs while the guest server is being rewritten under it."""
    ROUTES["/execute"] = (500, "gateway closed", "text/plain")
    assert guest._provision_result() is None
    ROUTES["/execute"] = _execute("0\n")
    assert guest._provision_result() == 0


# --------------------------------------------------------------------------
# the boot wait


class _FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


def test_the_initial_agent_wait_uses_geometry_without_accepting_upstream_png(
    guest, monkeypatch
):
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    guest._process = _FakeProcess()
    ROUTES["/screen_size"] = (
        200,
        json.dumps({"width": 1920, "height": 1080}),
        "application/json",
    )
    ROUTES["/screenshot"] = (200, b"upstream PNG", "image/png")
    guest._wait_for_agent(1.0)


def test_a_qemu_that_exited_fails_the_boot_wait_instead_of_polling_it_out(guest):
    guest._process = _FakeProcess(returncode=1)
    with pytest.raises(GuestCommandError, match="QEMU exited with 1"):
        guest._wait_for_guest(900.0)


def test_a_black_framebuffer_is_not_a_ready_guest(guest, monkeypatch):
    """A 200 from the agent arrives tens of seconds before the desktop paints."""
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    guest._process = _FakeProcess()
    ROUTES["/screenshot"] = (200, _jpeg(0), "image/jpeg")
    with pytest.raises(GuestCommandError, match="non_dark_ratio=0.000"):
        guest._wait_for_guest(1.0)


def test_a_painted_desktop_ends_the_boot_wait(guest, monkeypatch):
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    guest._process = _FakeProcess()
    ROUTES["/screenshot"] = (200, _structured_jpeg(), "image/jpeg")
    guest._wait_for_guest(1.0)


def test_an_agent_that_never_answers_times_out_naming_what_it_last_saw(
    guest, monkeypatch
):
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    guest._process = _FakeProcess()
    with pytest.raises(GuestCommandError, match="GuestAgentError|not ready"):
        guest._wait_for_guest(1.0)


# --------------------------------------------------------------------------
# the reboot

_OLD_BOOT = "11111111-1111-1111-1111-111111111111"
_NEW_BOOT = "22222222-2222-2222-2222-222222222222"


def test_the_boot_wait_starts_only_once_the_guest_reports_a_new_boot_id(guest, monkeypatch):
    monkeypatch.setattr(image_build.time, "sleep", lambda _: None)
    answers = _script(
        _execute(f"{_OLD_BOOT}\n"),
        _execute(""),
        _execute(f"{_OLD_BOOT}\n"),
        (500, "gateway closed", "text/plain"),
        _execute(f"{_NEW_BOOT}\n"),
    )
    served: list[int] = []

    def _route():
        served.append(1)
        return answers()

    ROUTES["/execute"] = _route
    at_boot_wait: list[int] = []
    monkeypatch.setattr(
        DesktopImageBuilder,
        "_wait_for_guest",
        lambda self, timeout_s: at_boot_wait.append(len(served)),
    )
    guest._reboot()
    assert at_boot_wait == [5]


def test_a_guest_that_never_reboots_fails_instead_of_being_verified(config, agent_server):
    ROUTES.clear()
    ROUTES["/execute"] = _execute(f"{_OLD_BOOT}\n")
    builder = DesktopImageBuilder(
        DesktopImageBuildConfig(
            upstream=config.upstream,
            output=config.output,
            runtime_dir=config.runtime_dir,
            boot_timeout_s=0.0,
        ),
        log=lambda _: None,
    )
    builder._client = image_build.OSWorldClient(agent_server)
    with pytest.raises(GuestCommandError, match="did not reboot before its deadline"):
        builder._reboot()


def test_a_guest_that_reports_no_boot_id_fails_rather_than_passing_vacuously(guest):
    ROUTES["/execute"] = _execute(f"{_OLD_BOOT}\n")
    with pytest.raises(GuestCommandError, match="no boot id"):
        guest._wait_for_new_boot("", image_build.time.monotonic() + 900.0)


def test_reboot_transition_and_readiness_share_one_timeout(guest, monkeypatch):
    times = iter((10.0, 20.0, 25.0, 30.0))
    monkeypatch.setattr(image_build.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(guest, "guest_bash", lambda *args, **kwargs: _OLD_BOOT)
    monkeypatch.setattr(guest, "guest_root_bash", lambda *args, **kwargs: "")
    deadlines: list[float] = []
    readiness: list[float] = []
    monkeypatch.setattr(
        guest, "_wait_for_new_boot", lambda before, deadline: deadlines.append(deadline)
    )
    monkeypatch.setattr(guest, "_wait_for_guest", lambda timeout_s: readiness.append(timeout_s))

    guest._reboot()

    assert deadlines == [920.0]
    assert readiness == [895.0]


# --------------------------------------------------------------------------
# the manifest


def test_the_manifest_records_the_upstream_it_was_built_from(config):
    manifest = build_manifest(config, BuildReport())
    assert manifest["upstream"]["path"] == str(config.upstream)
    assert manifest["upstream"]["bytes"] == config.upstream.stat().st_size
    assert manifest["output"] == str(config.output)


def test_the_manifest_records_every_declared_package_and_artifact(config):
    manifest = build_manifest(config, BuildReport())
    assert manifest["apt_packages"] == list(config.apt_packages)
    assert manifest["pip_packages"] == list(config.pip_packages)
    assert manifest["deb_artifacts"] == [
        {"url": XCFTOOLS_DEB.url, "sha256": XCFTOOLS_DEB.sha256}
    ]
    assert manifest["deliberately_absent"] == list(AGENT_INSTALLED_MODULES)
    assert manifest["guest_server_call"] == PRODUCTION_SERVER_CALL
    assert (
        manifest["secret_stdin_execute_contract"]
        == SECRET_STDIN_EXECUTE_CONTRACT
    )
    assert manifest["image_domain"] == OBSERVATION_CONTRACT
    assert manifest["guest_server_patch_sha256"] == hashlib.sha256(
        GUEST_SCREENSHOT_PATCH.read_bytes()
    ).hexdigest()


def test_the_manifest_carries_the_guest_checks_and_stays_json_serialisable(config):
    report = BuildReport()
    report.record("boot", 41.27)
    report.checks.update(VERIFICATION_OUTPUT)
    manifest = build_manifest(config, report)
    assert manifest["steps_seconds"]["boot"] == 41.3
    assert json.loads(json.dumps(manifest))["checks"]["server_pids"] == ["1234"]


# --------------------------------------------------------------------------
# publication


@contextmanager
def _no_boot(self, target):
    yield


def _stub_build(monkeypatch, checks: dict) -> None:
    monkeypatch.setattr(
        DesktopImageBuilder,
        "_create_image",
        lambda self, target: target.write_bytes(b"QFI\xfb provisioned"),
    )
    monkeypatch.setattr(DesktopImageBuilder, "_booted", _no_boot)
    monkeypatch.setattr(DesktopImageBuilder, "_provision", lambda self: None)
    monkeypatch.setattr(DesktopImageBuilder, "_reboot", lambda self: None)
    monkeypatch.setattr(DesktopImageBuilder, "_verify", lambda self: checks)


def test_a_verified_build_publishes_the_image_and_a_manifest_beside_it(
    config, monkeypatch
):
    _stub_build(monkeypatch, dict(VERIFIED_CHECKS))
    manifest = DesktopImageBuilder(config, log=lambda _: None).build()
    assert config.output.read_bytes() == b"QFI\xfb provisioned"
    assert not config.partial_path.exists()
    assert json.loads(config.manifest_path.read_text()) == manifest


def test_build_flattens_its_source_into_a_self_contained_image(config, monkeypatch):
    monkeypatch.setattr(
        DesktopImageBuilder,
        "_create_overlay",
        lambda *args: pytest.fail("published builds must not use overlays"),
    )
    commands: list[list[str]] = []

    def convert(argv, **kwargs):
        commands.append(argv)
        Path(argv[-1]).write_bytes(b"QFI\xfb flattened")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(image_build.subprocess, "run", convert)
    builder = DesktopImageBuilder(config, log=lambda _: None)
    builder._create_image(config.partial_path)
    assert commands == [
        image_convert_argv(config.qemu_img_binary, config.upstream, config.partial_path)
    ]
    assert "-b" not in commands[0]
    assert config.partial_path.read_bytes() == b"QFI\xfb flattened"


def test_a_missing_grader_package_stops_the_image_from_being_published(
    config, monkeypatch
):
    _stub_build(
        monkeypatch,
        {
            **VERIFIED_CHECKS,
            "missing_modules": ["pandas"],
            "agent_installed_modules": ["pytest"],
        },
    )
    with pytest.raises(GuestCommandError, match="failed verification") as raised:
        DesktopImageBuilder(config, log=lambda _: None).build()
    assert "pandas" in str(raised.value)
    assert "pytest" in str(raised.value)
    assert not config.output.exists()
    assert not config.manifest_path.exists()
    assert config.partial_path.is_file()


@pytest.mark.parametrize("existing", ["output", "manifest_path"])
def test_build_refuses_to_replace_an_existing_artifact(config, existing):
    path = getattr(config, existing)
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match=str(path)):
        DesktopImageBuilder(config, log=lambda _: None).build()
    assert path.read_bytes() == b"existing"


def test_verify_only_fails_on_an_invalid_published_image(config, monkeypatch):
    config.output.write_bytes(b"QFI\xfb published")
    monkeypatch.setattr(DesktopImageBuilder, "_create_overlay", lambda *args: None)
    monkeypatch.setattr(DesktopImageBuilder, "_booted", _no_boot)
    monkeypatch.setattr(DesktopImageBuilder, "_wait_for_guest", lambda *args: None)
    monkeypatch.setattr(
        DesktopImageBuilder,
        "_verify",
        lambda self: {**VERIFIED_CHECKS, "missing_modules": ["pandas"]},
    )
    with pytest.raises(GuestCommandError, match="pandas"):
        DesktopImageBuilder(config, log=lambda _: None).verify_only()


@pytest.mark.parametrize(
    ("spoiled", "named"),
    [
        ({"missing_modules": ["pandas"]}, "pandas"),
        ({"agent_installed_modules": ["pytest"]}, "pytest"),
        ({"tools": {**VERIFICATION_OUTPUT["tools"], "xcf2png": None}}, "xcf2png"),
        ({"production_server_call": False}, "debug call"),
        ({"secret_stdin_execute_contract": "old"}, "secret-stdin"),
        ({"reloader_disabled": {**RELOADER_OUTPUT, "pids_stable": False}}, "restarted"),
        (
            {"reloader_disabled": {**RELOADER_OUTPUT, "concurrent": False}},
            "site-packages write",
        ),
    ],
)
def test_every_fact_the_guest_sends_back_can_refuse_publication(spoiled, named):
    failures = image_build._verification_failures({**VERIFIED_CHECKS, **spoiled})
    assert len(failures) == 1, failures
    assert named in failures[0]


def test_a_fact_the_probe_stopped_sending_fails_closed():
    checks = dict(VERIFIED_CHECKS)
    del checks["missing_modules"]
    assert image_build._verification_failures(checks) != []


def test_the_upstream_image_is_never_touched_by_a_build(config, monkeypatch):
    before = config.upstream.read_bytes()
    _stub_build(monkeypatch, dict(VERIFIED_CHECKS))
    DesktopImageBuilder(config, log=lambda _: None).build()
    assert config.upstream.read_bytes() == before
