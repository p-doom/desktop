"""Starting, resetting, pooling, and reaching a desktop.

Nothing here knows what an action grammar is either.  This subpackage's job is to
hand a caller a live guest that answers HTTP, and to prove that a reset actually
rewound it.
"""

from .factory import (
    ConfigError,
    build_desktop_pool,
    build_desktop_session,
    build_qemu_runtime,
    qemu_session_factory,
)
from .osworld_client import GuestAgentError, OSWorldClient
from .pool import (
    CheckedOutDesktopSession,
    DesktopPoolConfig,
    DesktopSessionPool,
    PortLease,
    WorkerPorts,
    allocate_worker_ports,
)
from .qemu import QemuError, QemuRuntime, QmpClient, free_port, kvm_available
from .readiness import (
    ScreenshotStatus,
    desktop_screenshot_ready,
    png_luma_samples,
    wait_for_desktop_ready,
    wait_for_screenshot_ready,
)
from .runtime import Checkpoint, GuestPorts, Runtime, RuntimeState, SupportsCheckpoints, SupportsFork
from .sandbox_protocol import (
    DEFAULT_TRANSFER_TIMEOUT_S,
    MAX_TRANSFER_BYTES,
    ApptainerSandboxProvider,
    ConnectableProvider,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxProvider,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
    SupportsSandboxEndpoint,
    TransferTooLargeError,
)
from .session import DesktopSession, GuestScript, ResetReceipt, SessionError

__all__ = [
    "DEFAULT_TRANSFER_TIMEOUT_S",
    "MAX_TRANSFER_BYTES",
    "ApptainerSandboxProvider",
    "Checkpoint",
    "CheckedOutDesktopSession",
    "ConfigError",
    "ConnectableProvider",
    "DesktopPoolConfig",
    "DesktopSession",
    "DesktopSessionPool",
    "GuestAgentError",
    "GuestPorts",
    "GuestScript",
    "OSWorldClient",
    "PortLease",
    "QemuError",
    "QemuRuntime",
    "QmpClient",
    "ResetReceipt",
    "Runtime",
    "RuntimeState",
    "SandboxEndpoint",
    "SandboxExecResult",
    "SandboxHandle",
    "SandboxProvider",
    "SandboxResources",
    "SandboxSpec",
    "SandboxStatus",
    "ScreenshotStatus",
    "SessionError",
    "SupportsCheckpoints",
    "SupportsFork",
    "SupportsSandboxEndpoint",
    "TransferTooLargeError",
    "WorkerPorts",
    "allocate_worker_ports",
    "build_desktop_pool",
    "build_desktop_session",
    "build_qemu_runtime",
    "desktop_screenshot_ready",
    "free_port",
    "kvm_available",
    "png_luma_samples",
    "qemu_session_factory",
    "wait_for_desktop_ready",
    "wait_for_screenshot_ready",
]
