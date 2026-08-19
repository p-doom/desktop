"""Constructor-side entry points: plain functions, explicit config.

A harness used to obtain a desktop by *naming* one -- ``provider_name="docker"``
resolving, through a third-party factory rewritten at import time, to a
local-QEMU provider.  A re-clone of that tree removed the rewrite, the name then
resolved to something else, and every dependent job failed in a way that looked
like a model regression.

So there is deliberately:

* no name registry -- nothing maps a string like ``"qemu"`` or ``"docker"`` to an
  implementation.  You pass an image path and get a runtime.
* no plugin lookup -- no entry points, no ``importlib`` by path, no scanning.
* no import-time side effects -- importing this module changes nothing anywhere.
  Every function here only ever constructs and returns objects.

Configuration is explicit arguments first.  Environment variables are a *named
fallback* for the values a scheduler legitimately owns, and each one is listed in
``ENVIRONMENT`` below so a caller can discover them without reading the code.
Nothing falls back to a site-specific default path: a missing image is an error,
because a factory that guesses an image is a factory that silently benchmarks the
wrong guest.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from .pool import DesktopPoolConfig, DesktopSessionPool, PortLease
from .qemu import BASE_CHECKPOINT, QemuRuntime
from .runtime import GuestPorts
from .session import DesktopSession

#: Environment variables read as fallbacks, and what each supplies.  An explicit
#: argument always wins; nothing here is consulted if one is passed.
ENVIRONMENT: dict[str, str] = {
    "DESKTOP_ENV_IMAGE": "guest qcow2 path (fallback for `image`)",
    "DESKTOP_ENV_QEMU_BIN": "qemu-system-x86_64 path or name",
    "DESKTOP_ENV_QEMU_IMG_BIN": "qemu-img path or name",
    "DESKTOP_ENV_VM_SMP": "vCPU count",
    "DESKTOP_ENV_VM_MEM": "guest memory, e.g. 8G",
    "DESKTOP_ENV_VM_LOG_DIR": "directory for QEMU stdout/serial logs",
    "DESKTOP_ENV_QMP_DIR": "directory for the QMP unix socket (must be short)",
    "DESKTOP_ENV_ACCEL": "force 'kvm' or 'tcg'; otherwise detected",
}


#: QEMU's ``-m`` size grammar, as far as this factory uses it: a count with an
#: optional binary suffix.  Checked HERE rather than left to QEMU, so a typo is
#: a config error instead of a VM that dies during startup wearing a boot
#: failure's error message.
_MEMORY_SIZE = re.compile(r"[0-9]+[KMGT]?", re.IGNORECASE)


class ConfigError(ValueError):
    """A runtime cannot be constructed from the configuration given."""


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _resolve_image(image: Path | str | None) -> Path:
    raw = image if image is not None else _env("DESKTOP_ENV_IMAGE")
    if not raw:
        raise ConfigError(
            "no guest image: pass image=..., or set DESKTOP_ENV_IMAGE. There is "
            "deliberately no default -- guessing an image would silently "
            "benchmark the wrong guest."
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ConfigError(f"guest image does not exist: {path}")
    return path.resolve()


def build_qemu_runtime(
    *,
    image: Path | str | None = None,
    qemu_binary: Path | str | None = None,
    qemu_img_binary: Path | str | None = None,
    smp: int | None = None,
    memory: str | None = None,
    log_dir: Path | str | None = None,
    qmp_dir: Path | str | None = None,
    overlay_dir: Path | str | None = None,
    accelerator: str | None = None,
    ports: GuestPorts | None = None,
    base_checkpoint: str = BASE_CHECKPOINT,
    runtime_id: str | None = None,
    boot_timeout_s: float = 300.0,
    restore_timeout_s: float = 120.0,
) -> QemuRuntime:
    """Construct one ``QemuRuntime`` from explicit config plus named fallbacks.

    ``accelerator`` left as ``None`` means detect: KVM when ``/dev/kvm`` is usable,
    otherwise TCG with a warning.  Passing ``"kvm"`` explicitly does NOT assert
    that KVM exists -- ``QemuRuntime.start`` will simply fail on a node without
    it, which is the honest outcome for a caller that demanded acceleration.
    """
    resolved_accelerator = accelerator or _env("DESKTOP_ENV_ACCEL")
    if resolved_accelerator is not None and resolved_accelerator not in {"kvm", "tcg"}:
        raise ConfigError(
            f"accelerator must be 'kvm' or 'tcg', got {resolved_accelerator!r}"
        )
    raw_smp = smp if smp is not None else _env("DESKTOP_ENV_VM_SMP")
    try:
        cpu_count = int(raw_smp) if raw_smp is not None else 4
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"smp must be an integer, got {raw_smp!r}") from exc
    if cpu_count < 1:
        raise ConfigError(f"smp must be at least 1, got {cpu_count}")
    resolved_memory = memory or _env("DESKTOP_ENV_VM_MEM") or "8G"
    if not _MEMORY_SIZE.fullmatch(resolved_memory):
        raise ConfigError(
            f"memory must be a QEMU -m size such as '8G' or '4096M', got "
            f"{resolved_memory!r}"
        )
    return QemuRuntime(
        image=_resolve_image(image),
        qemu_binary=(
            qemu_binary or _env("DESKTOP_ENV_QEMU_BIN") or "qemu-system-x86_64"
        ),
        qemu_img_binary=(
            qemu_img_binary or _env("DESKTOP_ENV_QEMU_IMG_BIN") or "qemu-img"
        ),
        smp=cpu_count,
        memory=resolved_memory,
        log_dir=Path(log_dir) if log_dir else _optional_path("DESKTOP_ENV_VM_LOG_DIR"),
        qmp_dir=Path(qmp_dir) if qmp_dir else _optional_path("DESKTOP_ENV_QMP_DIR"),
        overlay_dir=Path(overlay_dir) if overlay_dir else None,
        accelerator=resolved_accelerator,
        ports=ports,
        base_checkpoint=base_checkpoint,
        runtime_id=runtime_id,
        boot_timeout_s=boot_timeout_s,
        restore_timeout_s=restore_timeout_s,
    )


def _optional_path(variable: str) -> Path | None:
    raw = _env(variable)
    return Path(raw) if raw else None


def build_desktop_session(
    *,
    image: Path | str | None = None,
    scratch_root: Path | str | None = None,
    metadata_path: Path | str | None = None,
    session_id: str | None = None,
    require_single_task: bool = True,
    forbid_gpu_visibility: bool = False,
    transport_timeout_s: float = 30.0,
    ports: GuestPorts | None = None,
    **runtime_options: Any,
) -> DesktopSession:
    """A ``DesktopSession`` over a fresh ``QemuRuntime``.  Not started.

    Returned unstarted on purpose: ``start()`` acquires the per-task lock, makes
    the scratch directory, and boots, and a caller usually wants that inside its
    own ``with`` block or error handling rather than inside a constructor.
    """
    runtime = build_qemu_runtime(image=image, ports=ports, **runtime_options)
    return DesktopSession(
        runtime,
        scratch_root=Path(scratch_root) if scratch_root else None,
        metadata_path=Path(metadata_path) if metadata_path else None,
        session_id=session_id,
        require_single_task=require_single_task,
        forbid_gpu_visibility=forbid_gpu_visibility,
        transport_timeout_s=transport_timeout_s,
    )


def qemu_session_factory(
    *,
    startup_timeout_s: float,
    image: Path | str | None = None,
    require_single_task: bool = False,
    transport_timeout_s: float = 30.0,
    **runtime_options: Any,
) -> Callable[[PortLease], DesktopSession]:
    """A ``session_factory`` for ``DesktopSessionPool``, bound to one image.

    ``startup_timeout_s`` is the pool's budget for one session, and it is checked
    against the runtime's own phase timeouts HERE, once, rather than per session.

    Five couplings here are load-bearing and easy to get wrong separately:

    1. The lease's ports are pinned into the runtime.  The pool holds an
       ``flock`` on a port block for the lease's lifetime; if the runtime then
       allocated its own with ``bind(0)``, the lock would guard four ports nothing
       listens on while QEMU bound four unrelated ones, and two pooled sessions
       could still collide.  Passing ``lease.ports`` through is what makes the
       lease mean something.
    2. ``require_single_task`` defaults to False here.  The flag controls one
       thing only: whether a ``SLURM_NTASKS`` other than ``"1"`` is rejected.  It
       does NOT gate the session's one-VM-per-task ``flock``, which
       ``_prepare_isolation`` takes unconditionally.  ``False`` is the right
       default because a pool process legitimately runs under
       ``SLURM_NTASKS > 1``.
    3. The lease's ``workdir`` becomes the session's scratch root, so a retired
       session's scratch is released with its lease.  This -- not the flag above --
       is also what lets several pooled desktops coexist in one process: the task
       lock lives *inside* the session's scratch root, so two pooled sessions
       never contend for it.  Hand a pool an explicit shared ``scratch_root`` and
       the second session fails regardless of either setting.
    4. The metadata goes in the lease's ``logdir``, NOT its ``workdir``.  The
       workdir is scratch that ``PortLease.release`` removes with ``rmdir``, and
       whose refusal to disappear is the only signal that a session leaked
       something; writing a file we intend to keep into it made that signal
       unreadable, because the directory could then never be empty.
    5. The startup budget must exceed the runtime's own phase timeouts.  The pool
       publishes ``startup_timeout_s`` in its status file and a remote supervisor
       restarts anything ``starting`` for longer than that, so a budget below our
       own worst case means a slow but healthy first boot is classified stale and
       killed mid-snapshot -- which leaks the VM and then repeats.  Checked here
       because this is the one place that knows both numbers.
    """
    budget_s = build_qemu_runtime(image=image, **runtime_options).start_budget_s
    if startup_timeout_s < budget_s:
        raise ConfigError(
            f"startup_timeout_s={startup_timeout_s:.0f}s is below this runtime's "
            f"own worst-case start of {budget_s:.0f}s (boot "
            f"+ QMP connect + snapshot). A supervisor judging a starting desktop "
            f"by the smaller number restarts healthy VMs mid-boot; raise the pool "
            f"budget or lower the runtime's phase timeouts."
        )

    def factory(lease: PortLease) -> DesktopSession:
        session = build_desktop_session(
            image=image,
            scratch_root=lease.workdir,
            metadata_path=lease.logdir / "session.json",
            require_single_task=require_single_task,
            transport_timeout_s=transport_timeout_s,
            ports=GuestPorts(
                server=lease.ports.server,
                chromium=lease.ports.chromium,
                vnc=lease.ports.vnc,
                vlc=lease.ports.vlc,
            ),
            log_dir=lease.logdir,
            **runtime_options,
        )
        session.start()
        return session

    return factory


def build_desktop_pool(
    *,
    root_dir: Path | str,
    image: Path | str | None = None,
    config: DesktopPoolConfig | None = None,
    worker_name: str | None = None,
    **runtime_options: Any,
) -> DesktopSessionPool[DesktopSession]:
    """A pool of QEMU-backed sessions.  Not started.

    ``DesktopSessionPool.start()`` begins prewarming; until then nothing boots.
    The image and the runtime options are resolved and validated here rather than
    on the first session's startup thread, where a ``ConfigError`` would surface
    only as a failure counter.
    """
    pool_config = config or DesktopPoolConfig()
    return DesktopSessionPool(
        config=pool_config,
        root_dir=Path(root_dir),
        session_factory=qemu_session_factory(
            startup_timeout_s=pool_config.startup_timeout_s,
            image=image,
            **runtime_options,
        ),
        worker_name=worker_name,
    )


__all__ = [
    "ENVIRONMENT",
    "ConfigError",
    "build_desktop_pool",
    "build_desktop_session",
    "build_qemu_runtime",
    "qemu_session_factory",
]
