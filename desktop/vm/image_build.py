"""Provision the OSWorld guest image that ``desktop.vm.qemu`` boots.

Everything else in this package treats the guest qcow2 as a pinned *input*:
``vm/images/README.md`` is explicit that the desktop, its application versions
and its in-VM agent must be "an artifact a container rebuild cannot change", and
``osworld-guest-kvm.def`` therefore contains QEMU and no desktop.  That leaves
the qcow2 itself an opaque binary with no producer.  This module is the
producer: it boots the upstream image, installs what OSWorld task setups and
graders need, verifies the result from inside the guest, and only then publishes
the image next to a manifest recording exactly what went in.

Ported from the ``reinforcement-learning`` repo, which ran QEMU inside the
OSWorld ``.sif``.  Here QEMU runs directly on the host, as it does everywhere
else in this package, so there is no container argv and no bind list.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .observation import OBSERVATION_CONTRACT
from .osworld_client import GuestAgentError, OSWorldClient
from .pool import allocate_worker_ports, node_port_lock_dir
from .qemu import QemuError, QmpClient
from .readiness import ScreenshotStatus, desktop_screenshot_ready

GUEST_SERVER_SOURCE = "/home/user/server/main.py"
GUEST_SERVER_DIRECTORY = str(Path(GUEST_SERVER_SOURCE).parent)
GUEST_SCREENSHOT_PATCH = Path(__file__).with_name("images") / "osworld-cursor-jpeg.patch"
GUEST_SERVER_COMMAND = f"/usr/bin/python {GUEST_SERVER_SOURCE}"
DEBUG_SERVER_CALL = 'app.run(debug=True, host="0.0.0.0")'
PRODUCTION_SERVER_CALL = 'app.run(debug=False, host="0.0.0.0")'

APT_PACKAGES = ("patch", "pdftk-java", "qpdf", "xdotool")

PIP_PACKAGES = (
    "PyPDF2==3.0.1",
    "gimpformats==2025",
    "odfpy==1.4.1",
    "openpyxl==3.1.5",
    "pandas==2.3.3",
    "pdfplumber==0.11.10",
    "pikepdf==10.12.0",
    "pymupdf==1.25.5",
    "python-docx==1.2.0",
    "python-pptx==1.0.2",
    "python-xlib==0.33",
)

GUEST_MODULE_PROBE = (
    "PyPDF2",
    "Xlib",
    "docx",
    # PyMuPDF's grader-facing import name.
    "fitz",
    "gimpformats",
    "odf",
    "openpyxl",
    "pandas",
    "pdfplumber",
    "pikepdf",
    "pptx",
)

GUEST_TOOL_PROBE = ("pdftk", "qpdf", "xcf2png", "xdotool")

# Graders score these by asking whether the agent installed them.
AGENT_INSTALLED_MODULES = ("mypy", "pytest")

_GUEST_SUDO_PASSWORD = "password"
_PROVISION_SCRIPT = "/tmp/desktop_image_provision.sh"
_PROVISION_LOG = "/tmp/desktop_image_provision.log"
_PROVISION_MARKER = "/tmp/desktop_image_provision.rc"
_GUEST_BOOT_ID = "/proc/sys/kernel/random/boot_id"
_RELOADER_HOLD_S = 20
_RETRY_FUNCTION = (
    'retry() { for _ in $(seq 1 40); do "$@" && return 0; sleep 10; done; return 1; }'
)


@dataclass(frozen=True)
class DebArtifact:
    """A .deb fetched by URL because the guest release no longer ships it."""

    url: str
    sha256: str


XCFTOOLS_DEB = DebArtifact(
    url=(
        "http://archive.ubuntu.com/ubuntu/pool/universe/x/xcftools/"
        "xcftools_1.0.7-6build1_amd64.deb"
    ),
    sha256="74abdad0fc7a57ac91c53a3ec41ce3f40b45acb281fe4b8b187d9a0da32e8876",
)


class GuestCommandError(RuntimeError):
    """Raised when a guest command fails or the guest never answers."""


@dataclass(frozen=True)
class DesktopImageBuildConfig:
    """Inputs for one provisioned-image build."""

    upstream: Path
    output: Path
    runtime_dir: Path
    ram_size: str = "8G"
    cpu_cores: int = 4
    qemu_binary: str = "qemu-system-x86_64"
    qemu_img_binary: str = "qemu-img"
    boot_timeout_s: float = 900.0
    provision_timeout_s: float = 1800.0
    apt_packages: tuple[str, ...] = APT_PACKAGES
    pip_packages: tuple[str, ...] = PIP_PACKAGES
    deb_artifacts: tuple[DebArtifact, ...] = (XCFTOOLS_DEB,)

    def __post_init__(self) -> None:
        for name in ("upstream", "output", "runtime_dir"):
            raw = getattr(self, name)
            path = Path(raw)
            if not path.is_absolute():
                raise ValueError(
                    f"{name} must be an absolute path; got {str(raw)!r}. "
                    "Do not use '~', relative paths, or unexpanded shell variables."
                )
            object.__setattr__(self, name, path)
        if self.upstream == self.output:
            raise ValueError("The provisioned image must not overwrite its upstream")
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be positive")

    @property
    def partial_path(self) -> Path:
        return self.output.with_suffix(self.output.suffix + ".partial")

    @property
    def manifest_path(self) -> Path:
        return self.output.with_suffix(self.output.suffix + ".build.json")


@dataclass
class BuildReport:
    """Per-step timings and guest verification output for one build."""

    steps: dict[str, float] = field(default_factory=dict)
    checks: dict[str, object] = field(default_factory=dict)

    def record(self, step: str, seconds: float) -> None:
        self.steps[step] = round(seconds, 1)


def render_provision_script(config: DesktopImageBuildConfig) -> str:
    """Return the root shell script that turns the upstream image into ours."""

    lines = [
        "set -eux",
        "export DEBIAN_FRONTEND=noninteractive",
        _RETRY_FUNCTION,
        f"sed -i 's|{DEBUG_SERVER_CALL}|{PRODUCTION_SERVER_CALL}|' {GUEST_SERVER_SOURCE}",
        f"grep -qF '{PRODUCTION_SERVER_CALL}' {GUEST_SERVER_SOURCE}",
        "systemctl stop packagekit.service unattended-upgrades.service || true",
        "retry apt-get update",
        "retry apt-get install -y --no-install-recommends " + " ".join(config.apt_packages),
        f"patch --batch --forward --fuzz=0 -p1 -d {GUEST_SERVER_DIRECTORY} <<'PATCH'\n"
        + GUEST_SCREENSHOT_PATCH.read_text()
        + "PATCH",
        f"grep -qF {OBSERVATION_CONTRACT!r} {GUEST_SERVER_SOURCE}",
        f"grep -qF 'subsampling=2, optimize=False' {GUEST_SERVER_SOURCE}",
        f"grep -qF 'mimetype=\"image/jpeg\"' {GUEST_SERVER_SOURCE}",
    ]
    for index, artifact in enumerate(config.deb_artifacts):
        local = f"/tmp/desktop_image_{index}.deb"
        lines += [
            f"retry curl -fsSL -o {local} {artifact.url}",
            f"echo '{artifact.sha256}  {local}' | sha256sum -c -",
            f"retry dpkg -i {local} || retry apt-get -f install -y",
            f"rm -f {local}",
        ]
    lines += [
        "retry python3 -m pip install --no-input --disable-pip-version-check "
        + " ".join(config.pip_packages),
        "apt-get clean",
    ]
    return "\n".join(lines)


def render_verification_script() -> str:
    """Return a guest script whose last stdout line is one JSON object of facts."""

    return (
        "python3 - <<'PY'\n"
        "import importlib, importlib.util, json, shutil, subprocess\n"
        f"modules = {list(GUEST_MODULE_PROBE)!r}\n"
        f"agent_modules = {list(AGENT_INSTALLED_MODULES)!r}\n"
        f"tools = {list(GUEST_TOOL_PROBE)!r}\n"
        "missing = []\n"
        "for name in modules:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "pids = subprocess.run(\n"
        f"    ['pgrep', '-x', '-f', {GUEST_SERVER_COMMAND!r}],\n"
        "    capture_output=True, text=True,\n"
        ").stdout.split()\n"
        f"source = open({GUEST_SERVER_SOURCE!r}).read()\n"
        "print(json.dumps({\n"
        "    'missing_modules': missing,\n"
        "    'tools': {name: shutil.which(name) for name in tools},\n"
        "    'server_pids': pids,\n"
        "    'agent_installed_modules': [\n"
        "        name for name in agent_modules\n"
        "        if importlib.util.find_spec(name) is not None\n"
        "    ],\n"
        f"    'production_server_call': {PRODUCTION_SERVER_CALL!r} in source,\n"
        "}))\n"
        "PY"
    )


def _verification_failures(checks: dict[str, object]) -> list[str]:
    failures: list[str] = []
    missing = checks.get("missing_modules")
    if missing != []:
        failures.append(f"grader modules the guest cannot import: {missing!r}")
    leaked = checks.get("agent_installed_modules")
    if leaked != []:
        failures.append(f"modules a grader scores by their absence are installed: {leaked!r}")
    tools = checks.get("tools")
    if not isinstance(tools, dict):
        failures.append(f"the guest reported no tool paths: {tools!r}")
    elif absent := [name for name in GUEST_TOOL_PROBE if not tools.get(name)]:
        failures.append(f"tools absent from the guest PATH: {absent}")
    if checks.get("production_server_call") is not True:
        failures.append("the guest server source still carries the debug call")
    reloader = checks.get("reloader_disabled")
    if not isinstance(reloader, dict):
        failures.append(f"the reloader probe did not run: {reloader!r}")
    else:
        if reloader.get("pids_stable") is not True:
            failures.append(
                "a site-packages write restarted the guest server: "
                f"{reloader.get('pids_before')!r} -> {reloader.get('pids_after')!r}"
            )
        if reloader.get("concurrent") is not True:
            failures.append(
                "the guest server did not serve a request across a site-packages "
                f"write: {reloader.get('site_packages_write_seconds')!r}s"
            )
    return failures


def _require_verified(checks: dict[str, object], image: Path) -> None:
    failures = _verification_failures(checks)
    if failures:
        raise GuestCommandError(
            f"The guest image {image} failed verification:\n" + "\n".join(failures)
        )


def overlay_create_argv(binary: str, base: Path, target: Path) -> list[str]:
    return [
        binary,
        "create",
        "-q",
        "-f",
        "qcow2",
        "-F",
        "qcow2",
        "-b",
        str(base),
        str(target),
    ]


def image_convert_argv(binary: str, source: Path, target: Path) -> list[str]:
    return [
        binary,
        "convert",
        "-f",
        "qcow2",
        "-O",
        "qcow2",
        str(source),
        str(target),
    ]


def qemu_argv(
    config: DesktopImageBuildConfig,
    image: Path,
    server_port: int,
    qmp: Path,
) -> list[str]:
    return [
        config.qemu_binary,
        # Unconditional, unlike `QemuRuntime`, which lets a caller ask for TCG to
        # prove plumbing.  There is nothing to prove here: a TCG build would run
        # apt and pip for hours and produce the same bytes, so a node without
        # /dev/kvm should fail rather than start.
        "-enable-kvm",
        "-cpu",
        "host",
        "-m",
        config.ram_size,
        "-smp",
        str(config.cpu_cores),
        # NOT `snapshot=on`: this is the one boot in the package whose guest
        # writes must land in the image.
        "-drive",
        f"file={image},format=qcow2,if=virtio,cache=writeback,discard=unmap",
        "-netdev",
        f"user,id=net0,hostfwd=tcp:127.0.0.1:{server_port}-:5000",
        "-device",
        "virtio-net-pci,netdev=net0",
        "-device",
        "virtio-vga",
        "-display",
        "none",
        "-qmp",
        f"unix:{qmp},server=on,wait=off",
        "-serial",
        "null",
    ]


def build_manifest(
    config: DesktopImageBuildConfig,
    report: BuildReport,
) -> dict[str, object]:
    upstream = config.upstream.stat()
    return {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "upstream": {
            "path": str(config.upstream),
            "bytes": upstream.st_size,
            "mtime": int(upstream.st_mtime),
        },
        "output": str(config.output),
        "image_domain": OBSERVATION_CONTRACT,
        "guest_server_patch_sha256": hashlib.sha256(
            GUEST_SCREENSHOT_PATCH.read_bytes()
        ).hexdigest(),
        "guest_server_call": PRODUCTION_SERVER_CALL,
        "apt_packages": list(config.apt_packages),
        "pip_packages": list(config.pip_packages),
        "deb_artifacts": [
            {"url": artifact.url, "sha256": artifact.sha256}
            for artifact in config.deb_artifacts
        ],
        "deliberately_absent": list(AGENT_INSTALLED_MODULES),
        "steps_seconds": report.steps,
        "checks": report.checks,
    }


class DesktopImageBuilder:
    """Boot, provision, and verify one desktop image."""

    def __init__(
        self,
        config: DesktopImageBuildConfig,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.report = BuildReport()
        self._emit = log or _print_line
        self._process: subprocess.Popen[bytes] | None = None
        self._client: OSWorldClient | None = None
        self._qmp_path: Path | None = None

    def log(self, message: str) -> None:
        self._emit(f"[desktop-image] {time.strftime('%H:%M:%S')} {message}")

    def build(self) -> dict[str, object]:
        """Create the image, provision the guest, verify it, and write a manifest."""

        config = self.config
        existing = [path for path in (config.output, config.manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to replace an existing image build: "
                + ", ".join(str(path) for path in existing)
            )
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.runtime_dir.mkdir(parents=True, exist_ok=True)
        target = config.partial_path
        self._create_image(target)
        with self._booted(target):
            self._provision()
            self._reboot()
            self.report.checks.update(self._verify())
        _require_verified(self.report.checks, target)
        target.replace(config.output)
        manifest = build_manifest(config, self.report)
        config.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        self.log(f"wrote {config.output}")
        return manifest

    def verify_only(self) -> dict[str, object]:
        """Re-run every guest check on a throwaway overlay of the published image."""

        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        probe = self.config.runtime_dir / "verify-overlay.qcow2"
        self._create_overlay(self.config.output, probe)
        try:
            with self._booted(probe):
                self.report.checks.update(self._verify())
            _require_verified(self.report.checks, self.config.output)
        finally:
            probe.unlink(missing_ok=True)
        return build_manifest(self.config, self.report)

    def _create_image(self, target: Path) -> None:
        started = time.monotonic()
        target.unlink(missing_ok=True)
        argv = image_convert_argv(
            self.config.qemu_img_binary, self.config.upstream, target
        )
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"qemu-img convert failed ({result.returncode}): "
                f"{result.stdout}\n{result.stderr}"
            )
        self.report.record("create_image", time.monotonic() - started)
        self.log(f"created {target} in {self.report.steps['create_image']}s")

    def _create_overlay(self, base: Path, target: Path) -> None:
        target.unlink(missing_ok=True)
        argv = overlay_create_argv(self.config.qemu_img_binary, base, target)
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"qemu-img create failed ({result.returncode}): "
                f"{result.stdout}\n{result.stderr}"
            )

    @contextmanager
    def _booted(self, target: Path) -> Iterator[None]:
        config = self.config
        qmp = config.runtime_dir / "qemu-qmp.sock"
        qmp.unlink(missing_ok=True)
        # The package's ONE host-port allocator.  A build runs on the same nodes
        # as a pool, so taking a port outside the lock is the collision
        # `pool.allocate_worker_ports` exists to prevent.
        with allocate_worker_ports(
            lock_dir=node_port_lock_dir(),
            log_dir=config.runtime_dir,
            work_dir=config.runtime_dir,
        ) as lease:
            argv = qemu_argv(config, target, lease.ports.server, qmp)
            self.log(" ".join(argv))
            with (config.runtime_dir / "qemu.log").open("wb") as handle:
                process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT)
                self._process = process
                self._client = OSWorldClient(f"http://127.0.0.1:{lease.ports.server}")
                self._qmp_path = qmp
                try:
                    started = time.monotonic()
                    self._wait_for_guest(config.boot_timeout_s)
                    self.report.record("boot", time.monotonic() - started)
                    self.log(f"guest ready in {self.report.steps['boot']}s")
                    yield
                finally:
                    self._power_down()

    def _power_down(self) -> None:
        process = self._require_process()
        if process.poll() is not None:
            return
        self.log("powering the guest down")
        try:
            monitor = QmpClient(str(self._require_qmp_path()))
            try:
                monitor.execute("system_powerdown")
            finally:
                monitor.close()
        except (QemuError, OSError, TimeoutError) as error:
            self.log(f"powerdown request failed: {error!r}")
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            self.log("powerdown timed out; killing QEMU")
            process.kill()
            process.wait(timeout=60)

    def guest_exec(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float = 120.0,
    ) -> tuple[int, str, str]:
        payload = self._require_client().execute(
            list(argv), check=False, timeout_s=timeout_s
        )
        return (
            int(payload.get("returncode", 1)),
            str(payload.get("output", "")),
            str(payload.get("error", "")),
        )

    def guest_bash(self, script: str, *, timeout_s: float = 120.0) -> str:
        code, output, error = self.guest_exec(
            ["bash", "-c", script], timeout_s=timeout_s
        )
        if code != 0:
            raise GuestCommandError(f"Guest script failed ({code}): {error or output}")
        return output

    def guest_root_bash(self, script: str, *, timeout_s: float = 120.0) -> str:
        return self.guest_bash(
            f"echo {_GUEST_SUDO_PASSWORD} | sudo -S -k bash -c {_shell_quote(script)}",
            timeout_s=timeout_s,
        )

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise GuestCommandError("No QEMU process is running")
        return self._process

    def _require_client(self) -> OSWorldClient:
        if self._client is None:
            raise GuestCommandError("No guest agent is reachable")
        return self._client

    def _require_qmp_path(self) -> Path:
        if self._qmp_path is None:
            raise GuestCommandError("The QMP socket is not configured")
        return self._qmp_path

    def _wait_for_guest(self, timeout_s: float) -> None:
        """Wait for a guest with a DESKTOP, not merely a guest that answers 200.

        `readiness.desktop_screenshot_ready` is the rule: a 200 from the in-VM
        agent arrives tens of seconds before the framebuffer stops being black,
        and provisioning a guest whose session has not come up yet is how
        `systemctl stop packagekit` races the thing that started it.
        """
        process = self._require_process()
        client = self._require_client()
        deadline = time.monotonic() + timeout_s
        detail = "no screenshot yet"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise GuestCommandError(
                    f"QEMU exited with {process.returncode} before the guest answered"
                )
            try:
                screenshot = client.screenshot()
            except GuestAgentError as error:
                detail = repr(error)
            else:
                status, detail = desktop_screenshot_ready(screenshot)
                if status is ScreenshotStatus.READY:
                    return
            time.sleep(3)
        raise GuestCommandError(
            f"The guest desktop was not ready within {timeout_s}s ({detail})"
        )

    def _provision(self) -> None:
        config = self.config
        started = time.monotonic()
        script = render_provision_script(config)
        self.guest_root_bash(
            f"rm -f {_PROVISION_MARKER} {_PROVISION_LOG}\n"
            f"cat > {_PROVISION_SCRIPT} <<'PROVISION'\n{script}\nPROVISION\n"
            f"setsid bash -c 'bash {_PROVISION_SCRIPT} > {_PROVISION_LOG} 2>&1; "
            f"echo $? > {_PROVISION_MARKER}' </dev/null >/dev/null 2>&1 &",
            timeout_s=60.0,
        )
        self.log("provisioning the guest")
        deadline = time.monotonic() + config.provision_timeout_s
        while time.monotonic() < deadline:
            marker = self._provision_result()
            if marker is not None:
                if marker != 0:
                    raise GuestCommandError(
                        f"Guest provisioning failed ({marker}):\n{self._provision_log()}"
                    )
                self.report.record("provision", time.monotonic() - started)
                self.log(f"provisioned in {self.report.steps['provision']}s")
                return
            time.sleep(5)
        raise GuestCommandError(
            f"Guest provisioning did not finish within {config.provision_timeout_s}s:\n"
            f"{self._provision_log()}"
        )

    def _provision_result(self) -> int | None:
        text = self._tolerant_bash(f"cat {_PROVISION_MARKER} 2>/dev/null; true").strip()
        return int(text) if text.isdigit() else None

    def _provision_log(self) -> str:
        return self._tolerant_bash(f"tail -30 {_PROVISION_LOG} 2>/dev/null; true")

    def _tolerant_bash(self, script: str) -> str:
        """Poll through the restarts a debug-mode guest server still performs."""

        try:
            return self.guest_bash(script, timeout_s=60.0)
        except (GuestAgentError, GuestCommandError, ValueError):
            return ""

    def _reboot(self) -> None:
        started = time.monotonic()
        before = self.guest_bash(f"cat {_GUEST_BOOT_ID}", timeout_s=60.0).strip()
        self.guest_root_bash(
            "setsid bash -c 'sleep 1; systemctl reboot' </dev/null >/dev/null 2>&1 &",
            timeout_s=60.0,
        )
        deadline = time.monotonic() + self.config.boot_timeout_s
        self._wait_for_new_boot(before, deadline)
        self._wait_for_guest(max(0.0, deadline - time.monotonic()))
        self.report.record("reboot", time.monotonic() - started)
        self.log(f"guest rebooted in {self.report.steps['reboot']}s")

    def _wait_for_new_boot(self, before: str, deadline: float) -> None:
        if not before:
            raise GuestCommandError("The guest reported no boot id to reboot away from")
        while time.monotonic() < deadline:
            current = self._tolerant_bash(f"cat {_GUEST_BOOT_ID}").strip()
            if current and current != before:
                return
            time.sleep(3)
        raise GuestCommandError(
            f"The guest did not reboot before its deadline (boot id stayed {before})"
        )

    def _verify(self) -> dict[str, object]:
        started = time.monotonic()
        output = self.guest_bash(render_verification_script(), timeout_s=120.0)
        checks: dict[str, object] = json.loads(output.strip().splitlines()[-1])
        checks["reloader_disabled"] = self._probe_reloader()
        self.report.record("verify", time.monotonic() - started)
        self.log(f"verification: {json.dumps(checks)}")
        return checks

    def _probe_reloader(self) -> dict[str, object]:
        """Rewrite site-packages under an open request; a reloader would drop it."""

        watched = self._watched_site_package()
        before = self._server_pids()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self._hold_guest_request)
            time.sleep(3)
            write_started = time.monotonic()
            self.guest_root_bash(f"touch {watched}", timeout_s=60.0)
            write_seconds = time.monotonic() - write_started
            code, error = pending.result(timeout=240)
        time.sleep(5)
        after = self._server_pids()
        return {
            "site_package_rewritten": watched,
            "held_request_returncode": code,
            "held_request_error": error,
            "site_packages_write_seconds": round(write_seconds, 1),
            "concurrent": write_seconds < _RELOADER_HOLD_S - 5,
            "pids_before": before,
            "pids_after": after,
            "pids_stable": before == after and len(before) == 1,
        }

    def _hold_guest_request(self) -> tuple[int | None, str | None]:
        try:
            code, _, _ = self.guest_exec(
                ["sleep", str(_RELOADER_HOLD_S)], timeout_s=180.0
            )
        except (GuestAgentError, ValueError) as error:
            return None, repr(error)
        return code, None

    def _watched_site_package(self) -> str:
        """Return an installed module file the guest server has already imported."""

        for module in ("flask", "werkzeug"):
            found = self._tolerant_bash(
                f"python3 -c 'import {module}; print({module}.__file__)' 2>/dev/null"
            ).strip()
            if found:
                return found
        raise GuestCommandError("The guest server has no importable Flask install")

    def _server_pids(self) -> list[str]:
        return self.guest_bash(
            f"pgrep -x -f {_shell_quote(GUEST_SERVER_COMMAND)} | sort", timeout_s=60.0
        ).split()


def _print_line(message: str) -> None:
    print(message, flush=True)


def _shell_quote(script: str) -> str:
    return "'" + script.replace("'", "'\"'\"'") + "'"
