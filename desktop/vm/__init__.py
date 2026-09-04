"""Starting, resetting, pooling, and reaching a desktop.

Nothing here knows what an action grammar is either.  This subpackage's job is to
hand a caller a live guest that answers HTTP, and to prove that a reset actually
rewound it.
"""

from .client import DesktopClient, GuestAgentError, GuestCommandResult
from .factory import (
    ConfigError,
    build_desktop_pool,
    build_desktop_session,
    build_qemu_runtime,
    qemu_session_factory,
)
from .image_build import (
    DebArtifact,
    DesktopImageBuildConfig,
    DesktopImageBuilder,
    GuestCommandError,
)
from .pool import (
    HUB_PORT_RANGE,
    CheckedOutDesktopSession,
    DesktopPoolConfig,
    DesktopSessionPool,
    PortLease,
    PortRangeLease,
    WorkerPorts,
    acquire_port_range,
    allocate_worker_ports,
)
from .qemu import QemuError, QemuRuntime, QmpClient, kvm_available
from .readiness import (
    ScreenshotStatus,
    desktop_screenshot_ready,
    screenshot_luma_samples,
    wait_for_desktop_ready,
    wait_for_screenshot_ready,
)
from .runtime import (
    Checkpoint,
    GuestPorts,
    Runtime,
    RuntimeState,
    SupportsCheckpoints,
    SupportsFork,
)
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
from .session import (
    DesktopResetMode,
    DesktopSession,
    GuestScript,
    ResetReceipt,
    SessionError,
)

__all__ = [
    "DEFAULT_TRANSFER_TIMEOUT_S",
    "HUB_PORT_RANGE",
    "MAX_TRANSFER_BYTES",
    "ApptainerSandboxProvider",
    "Checkpoint",
    "CheckedOutDesktopSession",
    "ConfigError",
    "ConnectableProvider",
    "DebArtifact",
    "DesktopImageBuildConfig",
    "DesktopImageBuilder",
    "DesktopClient",
    "DesktopPoolConfig",
    "DesktopResetMode",
    "DesktopSession",
    "DesktopSessionPool",
    "GuestAgentError",
    "GuestCommandError",
    "GuestCommandResult",
    "GuestPorts",
    "GuestScript",
    "PortLease",
    "PortRangeLease",
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
    "acquire_port_range",
    "allocate_worker_ports",
    "build_desktop_pool",
    "build_desktop_session",
    "build_qemu_runtime",
    "desktop_screenshot_ready",
    "kvm_available",
    "screenshot_luma_samples",
    "qemu_session_factory",
    "wait_for_desktop_ready",
    "wait_for_screenshot_ready",
]
