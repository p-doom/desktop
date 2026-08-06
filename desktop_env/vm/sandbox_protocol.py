"""The provider-neutral sandbox protocol, plus an Apptainer implementation.

The protocol types below -- ``SandboxStatus``, ``SandboxEndpoint``,
``SandboxResources``, ``SandboxSpec``, ``SandboxHandle``, ``SandboxExecResult``,
``SandboxProvider``, ``SupportsSandboxEndpoint``, ``ConnectableProvider`` -- are
copied from NeMo-Gym (``NVIDIA-NeMo/Gym``, Apache-2.0,
``nemo_gym/sandbox/providers/base.py``).  They import nothing outside the stdlib,
so they lift cleanly onto a zero-dependency floor.

``ConnectableProvider`` is the reason to copy rather than reinvent.
``serialize_handle`` / ``connect`` rebuild a live handle **in another process, on
another node**, from a JSON descriptor.  That is exactly the cross-node primitive
a multi-node rollout fleet otherwise hand-rolls with status files and a
convention, and getting it wrong shows up as a rollout that talks to a sandbox
that has already been recycled.

*** WHAT WAS NOT TAKEN: NeMo-Gym's OSWorld path. ***  It hard-raises unless the
provider name is literally ``"docker"``, which makes it useless on a cluster where
there is no daemon and no root.  The Apptainer provider below is ours.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import shlex
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

# --------------------------------------------------------------------------- #
# BEGIN copy -- NeMo-Gym, nemo_gym/sandbox/providers/base.py (Apache-2.0)
# --------------------------------------------------------------------------- #


class SandboxStatus(str, Enum):
    """Provider-neutral sandbox lifecycle status."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SandboxEndpoint:
    """Provider-neutral route to a long-lived service inside a sandbox.

    ``endpoint`` is an absolute URL. ``headers`` carries provider-required
    authentication or routing headers without exposing the provider's opaque
    handle to callers.
    """

    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise ValueError("Sandbox endpoint must be a non-empty absolute URL")
        endpoint = self.endpoint.strip()
        parsed = urlsplit(endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Sandbox endpoint must be a non-empty absolute URL")
        if not isinstance(self.headers, Mapping):
            raise TypeError("Sandbox endpoint headers must be a mapping")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(
            self,
            "headers",
            {str(key): str(value) for key, value in self.headers.items()},
        )


@dataclass(frozen=True)
class SandboxResources:
    """Provider-neutral resource request."""

    cpu: float | None = None
    memory_mib: int | None = None
    disk_gib: int | None = None
    gpu: int | None = None
    gpu_type: str | None = None

    @classmethod
    def from_mapping(cls, resources: Mapping[str, Any] | None) -> "SandboxResources":
        if resources is None:
            return cls()
        allowed_keys = set(cls.__dataclass_fields__)
        unknown_keys = set(resources) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            allowed = ", ".join(sorted(allowed_keys))
            raise ValueError(
                f"Unknown sandbox resource keys: {unknown}. Expected keys: {allowed}"
            )
        return cls(
            cpu=float(resources["cpu"]) if resources.get("cpu") is not None else None,
            memory_mib=(
                int(resources["memory_mib"])
                if resources.get("memory_mib") is not None
                else None
            ),
            disk_gib=(
                int(resources["disk_gib"]) if resources.get("disk_gib") is not None else None
            ),
            gpu=int(resources["gpu"]) if resources.get("gpu") is not None else None,
            gpu_type=(
                str(resources["gpu_type"]) if resources.get("gpu_type") is not None else None
            ),
        )


@dataclass(frozen=True)
class SandboxSpec:
    """Sandbox creation request."""

    image: str | None = None
    ttl_s: int | float | None = None
    ready_timeout_s: int | float | None = None
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    resources: SandboxResources | Mapping[str, Any] = field(
        default_factory=SandboxResources
    )
    entrypoint: list[str] | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    ports: tuple[int, ...] | list[int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.resources, SandboxResources):
            object.__setattr__(
                self, "resources", SandboxResources.from_mapping(self.resources)
            )
        if not isinstance(self.ports, (list, tuple)):
            raise TypeError("Sandbox ports must be a list or tuple of TCP port numbers")
        normalized_ports: list[int] = []
        for raw_port in self.ports:
            if isinstance(raw_port, bool):
                raise ValueError(f"Invalid sandbox TCP port: {raw_port!r}")
            if not isinstance(raw_port, (int, str)):
                raise ValueError(f"Invalid sandbox TCP port: {raw_port!r}")
            try:
                port = int(raw_port)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sandbox TCP port: {raw_port!r}") from exc
            if port < 1 or port > 65535:
                raise ValueError(
                    f"Sandbox TCP port must be between 1 and 65535, got {port}"
                )
            if port in normalized_ports:
                raise ValueError(f"Duplicate sandbox TCP port: {port}")
            normalized_ports.append(port)
        object.__setattr__(self, "ports", tuple(normalized_ports))


@dataclass
class SandboxHandle:
    """Provider-neutral handle to a created sandbox.

    ``raw`` is provider-owned opaque state. Public code should pass it back to
    the provider through this handle rather than inspecting or mutating it
    directly.
    """

    sandbox_id: str
    provider_name: str
    raw: Any


@dataclass(frozen=True)
class SandboxExecResult:
    """Provider-neutral process execution result.

    ``return_code`` is the process exit code when the sandbox actually ran the
    command. Providers may use a non-process sentinel with ``error_type`` set
    when the sandbox runtime reports an execution failure without a process
    exit code.
    """

    stdout: str | None
    stderr: str | None
    return_code: int
    error_type: str | None = None


class SandboxCreateError(RuntimeError):
    """Raised when a provider cannot create a sandbox."""


class SandboxCreateVerificationError(SandboxCreateError):
    """Raised when a newly-created sandbox fails provider readiness checks."""


class SandboxProvider(Protocol):
    """Runtime/infra provider contract used by the public sandbox API."""

    name: str

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a ready sandbox and return a provider-neutral handle."""
        ...

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Run a command inside a sandbox."""
        ...

    async def upload_file(
        self, handle: SandboxHandle, source_path: Path, target_path: str
    ) -> None:
        """Upload one local file into a sandbox."""
        ...

    async def download_file(
        self, handle: SandboxHandle, source_path: str, target_path: Path
    ) -> None:
        """Download one sandbox file to the local filesystem."""
        ...

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the current sandbox lifecycle status."""
        ...

    async def close(self, handle: SandboxHandle) -> None:
        """End the sandbox lifecycle and close provider resources for it."""
        ...

    async def aclose(self) -> None:
        """Close provider-scoped resources such as SDK clients."""
        ...


@runtime_checkable
class SupportsSandboxEndpoint(Protocol):
    """Optional provider capability for resolving declared service ports."""

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve a declared service port to a caller-reachable endpoint."""
        ...


@runtime_checkable
class ConnectableProvider(Protocol):
    """Optional capability: rebuild a handle in another process from a descriptor.

    Providers whose sandboxes are reachable by id implement this.  A provider that
    does not implement it can only be shared by fronting it with a sandbox server.
    Membership is checked with ``isinstance`` because the protocol is
    ``runtime_checkable``.
    """

    async def serialize_handle(
        self, handle: SandboxHandle, *, scope: str | None = None
    ) -> dict[str, Any]:
        """A JSON-serializable descriptor that ``connect`` can rebuild from."""
        ...

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Rebuild a live handle in this process from a descriptor."""
        ...


# --------------------------------------------------------------------------- #
# END copy
# --------------------------------------------------------------------------- #


#: Fence around a downloaded base64 payload.  Deliberately not base64-alphabet
#: text, so a fence can never be mistaken for payload.
_DOWNLOAD_BEGIN = "<<<DESKTOP_ENV_B64_BEGIN>>>"
_DOWNLOAD_END = "<<<DESKTOP_ENV_B64_END>>>"

#: How the guest reports an oversized file back to the host.
_OVERSIZE_MARKER = "DESKTOP_ENV_TRANSFER_TOO_LARGE"

#: Wall-clock ceiling on ONE file transfer, either direction.
#:
#: Both directions previously had no timeout at all -- ``download_file`` passed
#: ``timeout_s=None`` and ``_exec_with_stdin`` took no timeout -- so a wedged guest
#: hung the caller forever.  In a rollout fleet that presents as a stalled worker
#: rather than as an error, which is the harder failure to diagnose.
DEFAULT_TRANSFER_TIMEOUT_S = 300.0

#: Largest file this channel will move, in bytes.
#:
#: NOT a tuning knob: it is the honest limit of the mechanism.  Both directions
#: buffer the entire file AND its ~33%-larger base64 form in host memory, so the
#: peak cost is roughly 2.3x the file size -- fine for the PNG screenshots this
#: mostly carries, and untenable for the qcow2 overlays the header names, where a
#: 2.7 GB overlay would want ~6 GB of RAM. Refusing by name beats being OOM-killed
#: mid-rollout. Moving overlays needs a streaming path; this is the loud interim.
MAX_TRANSFER_BYTES = 256 * 1024 * 1024


class TransferTooLargeError(RuntimeError):
    """A file is too large for the in-memory base64 transfer channel."""


def _refuse_oversized_transfer(name: str, size: int, max_bytes: int | None) -> None:
    if max_bytes is not None and size > max_bytes:
        raise TransferTooLargeError(
            f"refusing to upload {name}: {size} bytes exceeds the {max_bytes}-byte "
            f"cap for this base64 channel, which buffers the whole file plus its "
            f"encoding in memory. A file this size needs a streaming path, not a "
            f"bigger cap."
        )


async def _run(
    argv: list[str], *, timeout_s: float | None = None, stdin: bytes | None = None
) -> SandboxExecResult:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=None if stdin is None else asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin), timeout=timeout_s
        )
    except TimeoutError:
        # Kill it, or the pipe stays open and the caller waits on a process
        # nobody is reading from.
        process.kill()
        await process.wait()
        return SandboxExecResult(None, None, -1, error_type="timeout")
    return SandboxExecResult(
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        int(process.returncode or 0),
    )


class ApptainerSandboxProvider:
    """Our own Apptainer provider: no daemon, no root, no compose.

    ``apptainer instance start`` puts a long-lived container in the user's own
    instance namespace, and ``apptainer instance list`` can find it again from any
    process on the same node -- which is what makes ``ConnectableProvider``
    implementable here without a control plane.  The descriptor therefore carries
    the node's hostname, and ``connect`` refuses a descriptor from another node
    rather than silently attaching to a same-named instance that is not the one
    the caller meant.
    """

    name = "apptainer"

    def __init__(
        self,
        *,
        binary: str = "apptainer",
        instance_prefix: str = "desktop-env",
        writable_tmpfs: bool = True,
        extra_start_args: tuple[str, ...] = (),
    ) -> None:
        self.binary = shutil.which(binary) or binary
        self.instance_prefix = instance_prefix
        # Chromium (and anything else that wants a writable profile) needs an
        # overlay: a .sif is read-only, and Chromium's failure without one is a
        # confusing crash rather than a permission error.  Left switchable because
        # some site configurations forbid it.
        self.writable_tmpfs = writable_tmpfs
        self.extra_start_args = tuple(extra_start_args)
        self._instances: set[str] = set()

    def _hostname(self) -> str:
        import socket

        return socket.gethostname()

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if not spec.image:
            raise SandboxCreateError("an Apptainer sandbox requires an image (.sif) path")
        image = Path(spec.image)
        if not image.exists():
            raise SandboxCreateError(f"Apptainer image not found: {image}")
        instance = f"{self.instance_prefix}-{uuid.uuid4().hex[:10]}"
        argv = [self.binary, "instance", "start"]
        if self.writable_tmpfs:
            argv.append("--writable-tmpfs")
        for key, value in (spec.env or {}).items():
            argv += ["--env", f"{key}={value}"]
        if spec.workdir:
            argv += ["--pwd", spec.workdir]
        argv += list(self.extra_start_args)
        argv += [str(image), instance]
        result = await _run(argv, timeout_s=spec.ready_timeout_s)
        if result.return_code != 0:
            raise SandboxCreateError(
                f"apptainer instance start failed (rc={result.return_code}): "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        self._instances.add(instance)
        handle = SandboxHandle(
            sandbox_id=instance,
            provider_name=self.name,
            raw={
                "instance": instance,
                "image": str(image.resolve()),
                "host": self._hostname(),
                "ports": list(spec.ports),
                "metadata": dict(spec.metadata),
            },
        )
        if spec.entrypoint:
            await self.exec(handle, shlex.join(spec.entrypoint))
        return handle

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        if user is not None:
            # Apptainer runs as the invoking user by definition; honouring a
            # different one would need setuid, which is the thing this provider
            # exists to avoid.
            raise ValueError("ApptainerSandboxProvider cannot change user")
        argv = [self.binary, "exec"]
        for key, value in (env or {}).items():
            argv += ["--env", f"{key}={value}"]
        if cwd:
            argv += ["--pwd", cwd]
        argv += [f"instance://{handle.sandbox_id}", "bash", "-lc", command]
        return await _run(
            argv,
            timeout_s=None if timeout_s is None else float(timeout_s),
            stdin=stdin,
        )

    # ------------------------------------------------------------- transfers
    #
    # BINARY-SAFE, and this is not incidental. The payloads that actually move
    # through here are PNG screenshots and qcow2 overlays. An earlier version
    # piped raw bytes through ``cat`` over ``apptainer exec`` and wrote the result
    # with ``write_text``, which corrupts every non-text file: stdout is decoded
    # as UTF-8 with ``errors="replace"``, so any invalid sequence becomes U+FFFD
    # and the bytes are gone. Both directions now go through base64, at a ~33%
    # size cost that is worth paying to make the channel total.

    async def upload_file(
        self,
        handle: SandboxHandle,
        source_path: Path,
        target_path: str,
        *,
        timeout_s: int | float | None = DEFAULT_TRANSFER_TIMEOUT_S,
        max_bytes: int | None = MAX_TRANSFER_BYTES,
    ) -> None:
        size = Path(source_path).stat().st_size
        _refuse_oversized_transfer(str(source_path), size, max_bytes)
        payload = base64.b64encode(Path(source_path).read_bytes()).decode("ascii")
        # base64 -d reads the encoded text on stdin, so the bytes never traverse
        # the pipe undecoded and the shell never sees them.
        result = await self.exec(
            handle,
            f"base64 -d > {shlex.quote(target_path)}",
            stdin=payload.encode("ascii"),
            timeout_s=timeout_s,
        )
        if result.error_type == "timeout":
            raise TimeoutError(
                f"upload of {source_path} ({size} bytes) timed out after {timeout_s}s"
            )
        if result.return_code != 0:
            raise RuntimeError(f"upload failed: {(result.stderr or '').strip()}")

    async def download_file(
        self,
        handle: SandboxHandle,
        source_path: str,
        target_path: Path,
        *,
        timeout_s: int | float | None = DEFAULT_TRANSFER_TIMEOUT_S,
        max_bytes: int | None = MAX_TRANSFER_BYTES,
    ) -> None:
        # The payload is fenced between two markers, for the same reason
        # ``GuestScript`` and the atomic guest program fence theirs: ``exec`` runs
        # ``bash -lc``, a LOGIN shell, so the container's profile can print a
        # banner onto the very stdout the file is travelling on.  ``validate=True``
        # alone does not close this: a banner made only of base64-alphabet
        # characters, with no newline and a length that keeps the total a multiple
        # of four, decodes without error into a SILENTLY WRONG file.
        # ``&&``, NOT ``;``: a ``;``-separated list exits with the status of the
        # LAST command, so a failing ``base64`` would be masked by the trailing
        # ``printf`` and this method would cheerfully write the empty body between
        # two fences to disk -- a missing source file persisted as a zero-byte
        # "successful" download.  Chaining also means a failure never prints the
        # closing fence, so the parse below fails too.
        # The size is checked INSIDE the guest, before any bytes are encoded, so an
        # oversized file is refused by name instead of being base64'd into host
        # memory first.
        guarded = f"base64 -w0 {shlex.quote(source_path)}"
        if max_bytes is not None:
            guarded = (
                f"_de_size=$(wc -c < {shlex.quote(source_path)}) && "
                f'if [ "$_de_size" -gt {int(max_bytes)} ]; then '
                f'echo "{_OVERSIZE_MARKER} $_de_size" >&2; exit 90; fi && ' + guarded
            )
        result = await self.exec(
            handle,
            f"printf %s {shlex.quote(_DOWNLOAD_BEGIN)} && "
            + guarded
            + f" && printf %s {shlex.quote(_DOWNLOAD_END)}",
            timeout_s=timeout_s,
        )
        if result.error_type == "timeout":
            raise TimeoutError(
                f"download of {source_path!r} timed out after {timeout_s}s"
            )
        if result.return_code != 0:
            stderr = (result.stderr or "").strip()
            if _OVERSIZE_MARKER in stderr:
                observed = stderr.split(_OVERSIZE_MARKER, 1)[1].strip().split()[0]
                raise TransferTooLargeError(
                    f"refusing to download {source_path!r}: {observed} bytes exceeds "
                    f"the {max_bytes}-byte cap for this base64 channel, which "
                    f"buffers the whole file plus its encoding in memory. A file "
                    f"this size needs a streaming path, not a bigger cap."
                )
            raise RuntimeError(f"download failed: {stderr}")
        stdout = result.stdout or ""
        begin = stdout.find(_DOWNLOAD_BEGIN)
        end = stdout.rfind(_DOWNLOAD_END)
        if begin < 0 or end < begin:
            raise RuntimeError(
                f"download of {source_path!r} returned no fenced payload; the "
                f"sandbox wrote {len(stdout)} characters of unfenced stdout"
            )
        try:
            raw = base64.b64decode(
                stdout[begin + len(_DOWNLOAD_BEGIN) : end].strip(), validate=True
            )
        except (ValueError, binascii.Error) as exc:
            # A malformed payload means the guest wrote something to stdout that
            # was not the file. Raising beats silently persisting a truncated
            # file that looks like a successful download.
            raise RuntimeError(
                f"download of {source_path!r} returned a non-base64 payload: {exc}"
            ) from exc
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(raw)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        result = await _run([self.binary, "instance", "list", "--json"])
        if result.return_code != 0:
            return SandboxStatus.UNKNOWN
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return SandboxStatus.UNKNOWN
        names = {
            str(item.get("instance"))
            for item in payload.get("instances", [])
            if isinstance(item, Mapping)
        }
        return SandboxStatus.RUNNING if handle.sandbox_id in names else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        await _run([self.binary, "instance", "stop", handle.sandbox_id])
        self._instances.discard(handle.sandbox_id)

    async def aclose(self) -> None:
        for instance in list(self._instances):
            await _run([self.binary, "instance", "stop", instance])
        self._instances.clear()

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        raw = handle.raw if isinstance(handle.raw, Mapping) else {}
        declared = [int(value) for value in raw.get("ports", [])]
        if declared and int(port) not in declared:
            raise ValueError(f"port {port} was not declared for this sandbox")
        host = str(raw.get("host") or self._hostname())
        return SandboxEndpoint(endpoint=f"http://{host}:{int(port)}")

    async def serialize_handle(
        self, handle: SandboxHandle, *, scope: str | None = None
    ) -> dict[str, Any]:
        raw = dict(handle.raw) if isinstance(handle.raw, Mapping) else {}
        return {
            "provider": self.name,
            "sandbox_id": handle.sandbox_id,
            "host": raw.get("host") or self._hostname(),
            "image": raw.get("image"),
            "ports": list(raw.get("ports", [])),
            "scope": scope,
        }

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        if descriptor.get("provider") != self.name:
            raise ValueError(
                f"descriptor is for provider {descriptor.get('provider')!r}, not {self.name!r}"
            )
        sandbox_id = str(descriptor.get("sandbox_id") or "")
        if not sandbox_id:
            raise ValueError("descriptor has no sandbox_id")
        host = str(descriptor.get("host") or "")
        if host and host != self._hostname():
            raise ValueError(
                f"Apptainer instances are node-local: descriptor is for {host!r}, "
                f"this process is on {self._hostname()!r}"
            )
        handle = SandboxHandle(
            sandbox_id=sandbox_id,
            provider_name=self.name,
            raw={
                "instance": sandbox_id,
                "image": descriptor.get("image"),
                "host": host or self._hostname(),
                "ports": list(descriptor.get("ports", [])),
            },
        )
        if await self.status(handle) is not SandboxStatus.RUNNING:
            raise ValueError(f"sandbox {sandbox_id!r} is not running on this node")
        return handle


def apptainer_available(binary: str = "apptainer") -> bool:
    return shutil.which(binary) is not None or os.path.exists(binary)
