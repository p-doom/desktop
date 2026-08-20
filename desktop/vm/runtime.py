"""The runtime contract: what any desktop backing must be able to do.

The method surface -- ``start``/``stop``/``is_ready``/``suspend``/``resume``/
``ensure_base``/``fork``/``checkpoint``/``list_checkpoints``/
``delete_checkpoint`` -- was taken as a design checklist from trycua/cua's
``Runtime``, read as a reference.  No trycua code is imported or copied here.

``fork`` (copy-on-write children off one warm base) and snapshot-rewind are
different primitives that a rollout harness needs for different reasons:

  * ``checkpoint``/``resume`` serialize N rollouts through one VM: cheap per
    reset (measured 4.4-5.2 s against 13.6-16.6 s for reboot-revert), but the
    rollouts cannot overlap.
  * ``fork`` gives N *isolated children* off one warm base at once, at the cost
    of N times the memory.  For group-sampled rollouts (GRPO group_size, k-sample
    teacher collection) that is the difference between serial and parallel.

Both are declared here so a harness can ask which one a backing supports instead
of assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class GuestPorts:
    """Host ports forwarded to the guest's own services."""

    server: int
    chromium: int = 0
    vnc: int = 0
    vlc: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "server": self.server,
            "chromium": self.chromium,
            "vnc": self.vnc,
            "vlc": self.vlc,
        }


@dataclass(frozen=True)
class Checkpoint:
    """One restorable state of a runtime."""

    name: str
    created_monotonic_ns: int
    kind: str = "vm_state"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeState:
    """Everything a caller needs to reach a started runtime."""

    runtime_id: str
    ports: GuestPorts
    base_url: str
    accelerator: str
    log_path: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class RuntimeError_(RuntimeError):
    """A runtime could not be started, restored, forked, or stopped."""


@runtime_checkable
class Runtime(Protocol):
    """A startable, restorable desktop.

    Implementations in this package: ``desktop.vm.qemu.QemuRuntime``.  A
    caller that only needs a reachable guest should depend on this protocol and
    not on QEMU.
    """

    name: str

    #: The clean checkpoint ``DesktopSession`` resets to.  A required member, not
    #: something a caller probes with ``getattr``: a default here would be a second
    #: copy of the name that silently disagrees the moment the runtime's own
    #: changes, and every reset would then restore a checkpoint nobody captured.
    base_checkpoint: str

    def start(self) -> RuntimeState:
        """Boot to the point where ``is_ready`` can succeed."""
        ...

    def stop(self) -> None:
        """Tear down.  Must be idempotent and must not raise on a dead runtime."""
        ...

    def is_ready(self, *, timeout_s: float = 0.0) -> bool:
        """Whether the guest's own agent is answering.  ``timeout_s == 0`` polls once."""
        ...

    def suspend(self) -> None:
        """Freeze execution without discarding state."""
        ...

    def resume(self) -> None:
        """Unfreeze a suspended runtime."""
        ...

    def ensure_base(self) -> Checkpoint:
        """Guarantee a named clean checkpoint exists, creating it if needed."""
        ...

    def fork(self, *, name: str) -> "Runtime":
        """A new isolated runtime sharing this one's base image copy-on-write."""
        ...

    def checkpoint(self, name: str) -> Checkpoint:
        """Capture full state under ``name``."""
        ...

    def restore(self, name: str) -> Checkpoint:
        """Restore full state from ``name`` and wait for the guest to answer."""
        ...

    def has_checkpoint(self, name: str) -> bool:
        """Whether ``name`` can be restored without being captured first.

        A protocol member rather than something a caller probes with ``getattr``:
        ``DesktopSession.reset_to_checkpoint`` asks this to decide between
        restoring a warm post-setup snapshot and re-running the setup, and a
        runtime that quietly lacked the method would take the slow branch forever
        while looking like it was working.
        """
        ...

    def list_checkpoints(self) -> tuple[Checkpoint, ...]:
        ...

    def delete_checkpoint(self, name: str) -> None:
        """Drop a checkpoint so the backing store stays bounded."""
        ...


@runtime_checkable
class SupportsFork(Protocol):
    """Optional capability probe: can this runtime hand out CoW children?"""

    def fork(self, *, name: str) -> Runtime: ...


@runtime_checkable
class SupportsCheckpoints(Protocol):
    """Optional capability probe: can this runtime rewind in place?"""

    def checkpoint(self, name: str) -> Checkpoint: ...
    def restore(self, name: str) -> Checkpoint: ...
    def has_checkpoint(self, name: str) -> bool: ...
