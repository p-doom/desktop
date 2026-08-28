"""One desktop session: an isolated runtime plus a verified reset.

A reset that silently no-ops is indistinguishable from a working one unless
something proves the guest actually rewound.  So this session plants a nonce file
in the guest *before* the reset and requires it to be gone *after*, and hashes
the runtime's own state on both sides so the generation must advance.
``reset_with_receipt`` hands that evidence back; ``consume_receipt`` verifies its
MAC, its ordering and its single use.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from ..execute.transport import HttpGuiTransport
from .osworld_client import OSWorldClient
from .runtime import Runtime

GUEST_JSON_MARKER = "DESKTOP_ENV_JSON="


class DesktopResetMode(StrEnum):
    """How far back a reset rewinds the guest before the next episode."""

    #: Restore the checkpoint.  The only mode that produces a ``ResetReceipt``,
    #: because it is the only one where there is a rewind to attest.
    SNAPSHOT = "snapshot"
    #: Leave the guest running and only drop believed input state.  For a task
    #: whose expensive setup -- a warm browser, say -- must survive the boundary.
    LOGICAL = "logical"


class SessionError(RuntimeError):
    """A session could not be isolated, started, reset, or torn down."""


_ID_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _ID_COMPONENT.sub("-", value).strip("-.")
    return (cleaned or fallback)[:40]


def task_unique_session_id() -> str:
    """An auditable id unique across jobs, tasks, and processes.

    Scheduler variables are read opportunistically; outside a scheduler the id is
    still unique because of the pid and the random suffix.
    """
    job = _safe_component(
        os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "local")),
        fallback="local",
    )
    task = _safe_component(os.environ.get("SLURM_PROCID", "0"), fallback="0")
    run = _safe_component(os.environ.get("RUN_ID", "no-run"), fallback="no-run")
    return f"{job}-{task}-{run[:16]}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write JSON through a private temp file, then rename onto the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(handle_fd, 0o600)
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GuestScript:
    """Run a Python program in the guest and read one JSON line back.

    The guest program prints ``<MARKER><json>`` on a line of its own and the host
    reads only that line, because the guest's stdout is shared with anything the
    program imports -- GTK warnings, X11 chatter, deprecation notices -- so
    parsing the whole of stdout as JSON fails intermittently and unreproducibly.
    """

    def __init__(self, client: OSWorldClient, *, marker: str = GUEST_JSON_MARKER) -> None:
        self.client = client
        self.marker = marker

    def run_json(self, program: str, *, timeout_s: float | None = None) -> Any:
        result = self.client.execute(
            ["python3", "-c", program], check=True, timeout_s=timeout_s
        )
        return self.parse(result)

    def parse(self, result: dict[str, Any]) -> Any:
        output = result.get("output")
        if not isinstance(output, str):
            raise SessionError("guest script produced no stdout")
        lines = [line for line in output.splitlines() if line.startswith(self.marker)]
        if len(lines) != 1:
            raise SessionError(
                f"guest script emitted {len(lines)} result markers, expected 1"
            )
        try:
            return json.loads(lines[0][len(self.marker) :])
        except json.JSONDecodeError as exc:
            raise SessionError(f"guest script emitted invalid JSON: {exc}") from exc

    def resolve_guest_root(self, name: str) -> PurePosixPath:
        """A writable per-session directory inside the guest.

        Tries ``$HOME`` first and falls back to the guest's temp dir, because the
        pinned image's home is not writable under every login mode.
        """
        program = f"""
import json,os,pathlib,tempfile
name={name!r}
candidates=[]
home=os.environ.get('HOME')
if home: candidates.append(pathlib.Path(home)/name)
candidates.append(pathlib.Path(tempfile.gettempdir())/name)
chosen=None
for candidate in candidates:
    try:
        candidate.mkdir(parents=True,exist_ok=True)
        probe=candidate/'.writable'
        probe.write_text('ok',encoding='utf-8')
        probe.unlink()
        chosen=str(candidate)
        break
    except OSError:
        continue
if chosen is None: raise RuntimeError('no writable guest root')
print({self.marker!r}+json.dumps({{'root':chosen}}))
""".strip()
        payload = self.run_json(program)
        if not isinstance(payload, dict) or "root" not in payload:
            raise SessionError(f"guest root resolution failed: {payload!r}")
        return PurePosixPath(str(payload["root"]))


@dataclass(frozen=True)
class ResetReceipt:
    """Evidence that one reset really rewound the guest."""

    session_id: str
    reset_id: str
    reset_sequence: int
    checkpoint_name: str
    prior_generation_id: str
    new_generation_id: str
    reset_started_monotonic_ns: int
    reset_completed_monotonic_ns: int
    guest_sentinel_path_sha256: str
    guest_sentinel_nonce_sha256: str
    runtime_state_before_sha256: str
    runtime_state_after_sha256: str
    attestor_mac: str
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DesktopSession:
    """One isolated desktop, restored to a clean checkpoint per episode.

    Isolation is enforced, not documented: one live VM per scheduler task (an
    ``flock`` on a per-task lock file), a per-VM scratch directory below the
    task's own temp root, and a ``TMPDIR`` that points at it so QEMU's unlinked
    overlay is freed with the allocation even under SIGKILL.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        scratch_root: Path | None = None,
        metadata_path: Path | None = None,
        session_id: str | None = None,
        checkpoint_name: str | None = None,
        require_single_task: bool = True,
        forbid_gpu_visibility: bool = False,
        transport_timeout_s: float = 30.0,
    ) -> None:
        self.runtime = runtime
        self.session_id = session_id or task_unique_session_id()
        self.checkpoint_name = checkpoint_name or runtime.base_checkpoint
        self.require_single_task = require_single_task
        self.forbid_gpu_visibility = forbid_gpu_visibility
        self.transport_timeout_s = float(transport_timeout_s)
        self.client: OSWorldClient | None = None
        self.transport: HttpGuiTransport | None = None
        self.scratch_dir: Path | None = None
        self.scratch_source: str | None = None
        self.metadata_path = metadata_path
        self._requested_scratch_root = scratch_root
        self._scratch_root: Path | None = None
        self._scratch_root_owned = False
        self._task_lock_handle: Any | None = None
        self._task_lock_path: Path | None = None
        self._saved_environment: dict[str, str | None] = {}
        self._reset_sequence = 0
        self._attestor_secret = secrets.token_bytes(32)
        self._outstanding_receipt_sha256: str | None = None
        self._consumed_receipts: set[str] = set()
        self._started = False

    def _set_environment(self, key: str, value: str) -> None:
        if key not in self._saved_environment:
            self._saved_environment[key] = os.environ.get(key)
        os.environ[key] = value

    def _restore_environment(self) -> None:
        for key, value in self._saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_environment.clear()

    def _prepare_isolation(self) -> None:
        if self.forbid_gpu_visibility and os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            raise SessionError("GPU visibility is forbidden for this session")
        ntasks = os.environ.get("SLURM_NTASKS")
        if self.require_single_task and ntasks is not None and ntasks != "1":
            raise SessionError(f"exactly one scheduler task is required, got {ntasks}")
        root = self._requested_scratch_root
        if root is None:
            # /tmp, not $TMPDIR: Slurm on this cluster runs
            # ``NamespaceType=namespace/tmpfs`` with ``TmpFS=/tmp``, so /tmp is the
            # node's real filesystem and the job-unique name below is what keeps two
            # array tasks apart.  There is deliberately no scheduler-supplied
            # override: Slurm exports no per-job scratch variable here.
            job = _safe_component(
                os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "local")),
                fallback="local",
            )
            task = _safe_component(os.environ.get("SLURM_PROCID", "0"), fallback="0")
            root = Path("/tmp") / f"desktop-env-job-{os.getuid()}-{job}-{task}"
            self.scratch_source = "job_unique_tmp_fallback"
            self._scratch_root_owned = True
        else:
            self.scratch_source = "explicit"
        root = root.resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_uid != os.geteuid():
            raise SessionError("VM scratch root is not owned by this user")
        if self._scratch_root_owned:
            os.chmod(root, 0o700)
        self._scratch_root = root
        self.scratch_dir = root / f"desktop-env-{self.session_id}"
        self.scratch_dir.mkdir(mode=0o700, parents=False, exist_ok=False)

        # A task may own at most one live VM.  The lock is scoped to the scratch
        # root so separate array tasks do not contend with each other.
        lock_path = root / f"desktop-env-task-{os.environ.get('SLURM_PROCID', '0')}.lock"
        self._task_lock_path = lock_path
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise SessionError("another VM is already live in this task") from exc
        self._task_lock_handle = handle

        # QEMU's -snapshot overlay is created and unlinked immediately; TMPDIR is
        # the only reliable placement boundary for it.
        self._set_environment("TMPDIR", str(self.scratch_dir))

    def start(self) -> HttpGuiTransport:
        if self._started:
            return self._require_transport()
        try:
            self._prepare_isolation()
            state = self.runtime.start()
            self.client = OSWorldClient(state.base_url, timeout_s=self.transport_timeout_s)
            self.transport = HttpGuiTransport(
                state.base_url, timeout_s=self.transport_timeout_s
            )
            self.runtime.ensure_base()
            self._started = True
            self._write_metadata(
                state_detail=state.detail,
                ports=state.ports.as_dict(),
                accelerator=state.accelerator,
            )
        except Exception:
            self.close()
            raise
        return self._require_transport()

    def _require_transport(self) -> HttpGuiTransport:
        if self.transport is None:
            raise SessionError("session is not started")
        return self.transport

    def guest(self) -> GuestScript:
        if self.client is None:
            raise SessionError("session is not started")
        return GuestScript(self.client)

    def reset(
        self, *, mode: DesktopResetMode = DesktopResetMode.SNAPSHOT
    ) -> HttpGuiTransport:
        """Prepare the guest for the next episode, per ``mode``.

        ``SNAPSHOT`` resets to the clean checkpoint and consumes the receipt for
        you.  ``LOGICAL`` does not touch the runtime at all, so there is no
        rewind, no receipt, and no reset sequence to advance -- what it still
        guarantees is the fresh transport.
        """
        if mode is DesktopResetMode.LOGICAL:
            return self._logical_reset()
        transport, receipt = self.reset_with_receipt()
        self.consume_receipt(receipt)
        return transport

    def _logical_reset(self) -> HttpGuiTransport:
        """Hand back a fresh transport over the still-running guest.

        A button believed held before the boundary must not be believed held
        after it, and a fresh transport is a fresh audit.  That is the whole of
        a logical reset: the runtime is deliberately not touched.
        """
        if not self._started or self.client is None:
            raise SessionError("session is not started")
        if self._outstanding_receipt_sha256 is not None:
            raise SessionError(
                "the previous reset receipt must be consumed before another reset"
            )
        base_url = self.transport.base_url if self.transport else ""
        self.transport = HttpGuiTransport(base_url, timeout_s=self.transport_timeout_s)
        return self._require_transport()

    def reset_with_receipt(self) -> tuple[HttpGuiTransport, ResetReceipt]:
        """Reset, and hand back proof that the guest actually rewound."""
        if not self._started or self.client is None:
            raise SessionError("session is not started")
        if self._outstanding_receipt_sha256 is not None:
            raise SessionError(
                "the previous reset receipt must be consumed before another reset"
            )
        sequence = self._reset_sequence + 1
        sentinel_path = (
            f"/tmp/desktop_env_reset_attestation_{self.session_id}_{sequence}.nonce"
        )
        nonce = secrets.token_hex(32)
        started = time.monotonic_ns()
        self._plant_sentinel(sentinel_path, nonce)
        before = self._runtime_observation()
        self.runtime.restore(self.checkpoint_name)
        after = self._runtime_observation()
        if before == after:
            raise SessionError("the reset did not change the runtime's observed state")
        prior_generation = hashlib.sha256(before).hexdigest()
        new_generation = hashlib.sha256(after).hexdigest()
        self._verify_sentinel_removed(sentinel_path)
        completed = time.monotonic_ns()
        self._reset_sequence = sequence
        # Held input state must never cross an episode boundary: a fresh transport
        # means a fresh audit, so a button believed held before the reset cannot
        # be believed held after it.
        base_url = self.transport.base_url if self.transport else ""
        self.transport = HttpGuiTransport(base_url, timeout_s=self.transport_timeout_s)
        payload = {
            "session_id": self.session_id,
            "reset_id": uuid.uuid4().hex,
            "reset_sequence": sequence,
            "checkpoint_name": self.checkpoint_name,
            "prior_generation_id": prior_generation,
            "new_generation_id": new_generation,
            "reset_started_monotonic_ns": started,
            "reset_completed_monotonic_ns": completed,
            "guest_sentinel_path_sha256": hashlib.sha256(
                sentinel_path.encode("utf-8")
            ).hexdigest(),
            "guest_sentinel_nonce_sha256": hashlib.sha256(
                nonce.encode("utf-8")
            ).hexdigest(),
            "runtime_state_before_sha256": prior_generation,
            "runtime_state_after_sha256": new_generation,
        }
        mac = hmac.new(
            self._attestor_secret, canonical_json(payload), hashlib.sha256
        ).hexdigest()
        receipt_sha256 = hashlib.sha256(
            canonical_json({**payload, "attestor_mac": mac})
        ).hexdigest()
        receipt = ResetReceipt(**payload, attestor_mac=mac, receipt_sha256=receipt_sha256)
        self._outstanding_receipt_sha256 = receipt.receipt_sha256
        return self._require_transport(), receipt

    def reset_to_checkpoint(self, name: str, *, setup: Any = None) -> HttpGuiTransport:
        """Restore ``name``, creating it from ``setup`` the first time.

        On the first call for a tag, the (expensive) per-task setup runs once and
        the result is snapshotted; every later call restores that snapshot and
        skips both the reboot and the setup.
        """
        if not self._started:
            raise SessionError("session is not started")
        if self.runtime.has_checkpoint(name):
            self.runtime.restore(name)
            base_url = self.transport.base_url if self.transport else ""
            self.transport = HttpGuiTransport(
                base_url, timeout_s=self.transport_timeout_s
            )
            return self._require_transport()
        transport = self.reset()
        if setup is not None:
            setup(transport)
        self.runtime.checkpoint(name)
        return transport

    def _runtime_observation(self) -> bytes:
        """A stable hash input describing the runtime's externally visible state.

        Deliberately *not* an introspection of a provider's private state, so the
        receipt does not depend on one implementation's internals.
        """
        checkpoints = [
            {"name": item.name, "kind": item.kind, "created": item.created_monotonic_ns}
            for item in self.runtime.list_checkpoints()
        ]
        ready = self.runtime.is_ready()
        cursor: list[int] | None = None
        with contextlib.suppress(Exception):
            if self.client is not None:
                cursor = list(self.client.cursor_position())
        screenshot_sha256: str | None = None
        with contextlib.suppress(Exception):
            if self.client is not None:
                screenshot_sha256 = hashlib.sha256(self.client.screenshot()).hexdigest()
        return canonical_json(
            {
                "checkpoints": checkpoints,
                "ready": ready,
                "cursor": cursor,
                "screenshot_sha256": screenshot_sha256,
                "reset_sequence": self._reset_sequence,
            }
        )

    def _plant_sentinel(self, path: str, nonce: str) -> None:
        if self.client is None:
            raise SessionError("session is not started")
        program = (
            "from pathlib import Path;"
            f"p=Path({path!r});v={nonce!r};"
            "p.write_text(v,encoding='utf-8');"
            "assert p.read_text(encoding='utf-8')==v"
        )
        try:
            self.client.execute(["python3", "-c", program])
        except Exception as exc:
            raise SessionError(f"could not plant the pre-reset guest sentinel: {exc}") from exc

    def _verify_sentinel_removed(self, path: str) -> None:
        if self.client is None:
            raise SessionError("session is not started")
        program = f"from pathlib import Path;assert not Path({path!r}).exists()"
        try:
            self.client.execute(["python3", "-c", program])
        except Exception as exc:
            raise SessionError(
                "the reset did not rewind the pre-reset guest sentinel"
            ) from exc

    def consume_receipt(self, receipt: ResetReceipt) -> None:
        """Verify a receipt's MAC, ordering, and single use."""
        if not isinstance(receipt, ResetReceipt):
            raise SessionError("reset receipt type mismatch")
        payload = receipt.as_dict()
        receipt_sha256 = payload.pop("receipt_sha256")
        mac = payload.pop("attestor_mac")
        expected_mac = hmac.new(
            self._attestor_secret, canonical_json(payload), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_mac, mac):
            raise SessionError("reset receipt MAC does not verify")
        expected_sha256 = hashlib.sha256(
            canonical_json({**payload, "attestor_mac": mac})
        ).hexdigest()
        if not hmac.compare_digest(expected_sha256, receipt_sha256):
            raise SessionError("reset receipt digest does not verify")
        if receipt_sha256 in self._consumed_receipts:
            raise SessionError("reset receipt was already consumed")
        if receipt.session_id != self.session_id:
            raise SessionError("reset receipt belongs to another session")
        if receipt.reset_sequence != self._reset_sequence:
            raise SessionError(
                f"reset receipt is out of order: {receipt.reset_sequence} != "
                f"{self._reset_sequence}"
            )
        if self._outstanding_receipt_sha256 != receipt_sha256:
            raise SessionError("reset receipt does not match the outstanding reset")
        self._consumed_receipts.add(receipt_sha256)
        self._outstanding_receipt_sha256 = None

    def _write_metadata(
        self, *, state_detail: dict[str, Any], ports: dict[str, int], accelerator: str
    ) -> None:
        if self.metadata_path is None:
            return
        write_json_atomic(
            self.metadata_path,
            {
                "schema_version": "desktop_env_session_v1",
                "session_id": self.session_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "runtime": self.runtime.name,
                "runtime_state": state_detail,
                # Recorded because a TCG guest is not valid for any parity number
                # and this file is what a run is audited from.  It is not in
                # ``state_detail``, so without this line the accelerator appeared
                # nowhere a reader would look.
                "accelerator": accelerator,
                "ports": ports,
                "checkpoint_name": self.checkpoint_name,
                "scratch": {
                    "directory": None if self.scratch_dir is None else str(self.scratch_dir),
                    "source": self.scratch_source,
                },
                "scheduler": {
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    "proc_id": os.environ.get("SLURM_PROCID", "0"),
                    "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
                },
                "one_vm_per_task": self.require_single_task,
                "closed": False,
            },
        )

    def _finalize_metadata(self, *, closed: bool, errors: list[str]) -> None:
        if self.metadata_path is None or not self.metadata_path.is_file():
            return
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            metadata["closed"] = closed
            metadata["cleanup_errors"] = errors
            write_json_atomic(self.metadata_path, metadata)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"metadata finalization failed: {exc}")

    def close(self) -> None:
        """Tear down, collecting every failure instead of stopping at the first.

        Raises only if something failed AND no exception is already propagating,
        so a cleanup problem cannot mask the error that caused the cleanup.
        """
        errors: list[str] = []
        stopped = True
        try:
            self.runtime.stop()
        except Exception as exc:
            stopped = False
            errors.append(f"runtime stop failed: {exc}")
        self.transport = None
        self.client = None
        scratch = self.scratch_dir
        if scratch is not None:
            try:
                shutil.rmtree(scratch)
            except OSError as exc:
                errors.append(f"VM scratch cleanup failed: {exc}")
            self.scratch_dir = None
        if scratch is not None and scratch.exists():
            errors.append("VM scratch directory still exists after cleanup")
        if self._task_lock_handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._task_lock_handle.fileno(), fcntl.LOCK_UN)
            self._task_lock_handle.close()
            self._task_lock_handle = None
        if self._task_lock_path is not None:
            try:
                self._task_lock_path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"task lock cleanup failed: {exc}")
            self._task_lock_path = None
        if self._scratch_root_owned and self._scratch_root is not None:
            with contextlib.suppress(OSError):
                self._scratch_root.rmdir()
        self._scratch_root = None
        self._scratch_root_owned = False
        self._restore_environment()
        self._started = False
        self._finalize_metadata(closed=stopped, errors=errors)
        if errors and sys.exc_info()[0] is None:
            raise SessionError("; ".join(errors))

    def __enter__(self) -> "DesktopSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "DesktopResetMode",
    "DesktopSession",
    "GuestScript",
    "ResetReceipt",
    "SessionError",
    "canonical_json",
    "sha256_file",
    "task_unique_session_id",
    "write_json_atomic",
]
