"""Apply operations to a guest and hand back a receipt.

NEW LOGIC.  The two predecessor executors (``NativeAbsoluteExecutor``,
``CompactRawExecutor``) each fused three jobs: parse a specific grammar, resolve
its coordinates, and drive the transport.  Only the third belongs here.  This
engine therefore starts from resolved ``Operation``s, and its optional
``apply_text`` reaches a grammar only through the ``Codec`` protocol -- so the
engine's own code contains no action names, no ``match`` on a grammar, and no
coordinate convention.

The cursor-readback verification is preserved from ``CompactRawExecutor``, where
it earned its place: the guest reports its own before/after cursor, and the host
reads the cursor independently on both sides.  A disagreement means the pointer
moved outside the action, which is the failure mode a delta-resolving grammar
cannot otherwise detect.  It is preserved *generically* here rather than only for
one grammar.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from ..geometry import DisplayGeometry, geometry_from_screen_size
from ..ir import Operation
from .guest_program import (
    PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    AtomicExecutionResult,
    ExecutionError,
)
from .transport import GuiTransport


@dataclass(frozen=True)
class StepReceipt:
    """What one applied action did, in enough detail to audit it later."""

    ok: bool
    operations: tuple[Operation, ...]
    requested_operations: tuple[Operation, ...]
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    host_cursor_before: tuple[int, int] | None
    host_cursor_after: tuple[int, int] | None
    cursor_readback_verified: bool
    error: str | None
    failure_kind: str | None
    atomic_state: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "operations": [item.as_dict() for item in self.operations],
            "requested_operations": [item.as_dict() for item in self.requested_operations],
            "cursor_before": list(self.cursor_before),
            "cursor_after": list(self.cursor_after),
            "host_cursor_before": (
                None if self.host_cursor_before is None else list(self.host_cursor_before)
            ),
            "host_cursor_after": (
                None if self.host_cursor_after is None else list(self.host_cursor_after)
            ),
            "cursor_readback_verified": self.cursor_readback_verified,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "atomic_state": dict(self.atomic_state),
        }


class Engine:
    """Applies resolved operations to one guest through one transport."""

    def __init__(
        self,
        transport: GuiTransport,
        *,
        click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
        verify_cursor_readback: bool = True,
    ) -> None:
        self.transport = transport
        self.click_backend = click_backend
        self.verify_cursor_readback = verify_cursor_readback
        self.receipts: list[StepReceipt] = []
        self._click_backend_supported = self._accepts_click_backend()

    # ------------------------------------------------------------- geometry
    def geometry(self) -> DisplayGeometry:
        """Ask the guest how big its screen is, as a ``DisplayGeometry``."""
        width, height = self.transport.screen_size()
        return geometry_from_screen_size(width, height)

    def cursor(self) -> tuple[int, int]:
        return self.transport.cursor_position()

    def resolution_context(self) -> tuple[DisplayGeometry, tuple[int, int]]:
        """The pair every ``Codec.compile`` needs, fetched together.

        Fetched together and returned together so a caller cannot accidentally
        resolve a delta against a cursor read after the geometry changed.
        """
        return self.geometry(), self.cursor()

    # -------------------------------------------------------------- applying
    def apply(self, operations: tuple[Operation, ...]) -> StepReceipt:
        """Send one action -- one guest process -- and build its receipt."""
        host_before: tuple[int, int] | None = None
        host_after: tuple[int, int] | None = None
        if self.verify_cursor_readback:
            host_before = self.transport.cursor_position()
        result = self._execute(operations)
        if self.verify_cursor_readback:
            host_after = self.transport.cursor_position()
        verified = self._verify(result, host_before, host_after)
        state = result.as_dict()
        state.update(
            {
                "host_cursor_before": None if host_before is None else list(host_before),
                "host_cursor_after": None if host_after is None else list(host_after),
                "cursor_readback_verified": verified,
                "executed_cursor_delta": [
                    result.cursor_after[0] - result.cursor_before[0],
                    result.cursor_after[1] - result.cursor_before[1],
                ],
            }
        )
        ok = result.ok and verified
        error = result.error
        failure_kind = result.failure_kind
        if result.ok and not verified:
            error = (
                "cursor readback mismatch: "
                f"host {host_before}->{host_after}, "
                f"guest {result.cursor_before}->{result.cursor_after}"
            )
            failure_kind = "verification"
            state.update({"ok": False, "error": error, "failure_kind": failure_kind})
        receipt = StepReceipt(
            ok=ok,
            operations=result.operations,
            requested_operations=operations,
            cursor_before=result.cursor_before,
            cursor_after=result.cursor_after,
            host_cursor_before=host_before,
            host_cursor_after=host_after,
            cursor_readback_verified=verified,
            error=error,
            failure_kind=failure_kind,
            atomic_state=state,
        )
        self.receipts.append(receipt)
        return receipt

    def apply_or_raise(self, operations: tuple[Operation, ...]) -> StepReceipt:
        """``apply``, but a failed action is an exception carrying the receipt."""
        receipt = self.apply(operations)
        if not receipt.ok:
            raise ExecutionError(
                f"guest action failed: {receipt.error}", evidence=receipt.as_dict()
            )
        return receipt

    def apply_text(self, text: str, codec: Any) -> StepReceipt:
        """Compile model output through a codec, then apply it.

        ``codec`` is anything satisfying ``desktop_env.codec_protocol.Codec``.
        The geometry and cursor are fetched here and passed *into* the codec, so
        the codec resolves against live state and this engine never learns which
        convention was resolved.
        """
        geometry, cursor = self.resolution_context()
        return self.apply(tuple(codec.compile(text, geometry, cursor)))

    # -------------------------------------------------------------- internals
    def _accepts_click_backend(self) -> bool:
        """Whether this transport's ``execute_atomic`` takes ``click_backend``.

        Decided by inspecting the signature once, at construction, rather than by
        calling and catching ``TypeError``.  A ``try/except TypeError`` around the
        call would also swallow a genuine ``TypeError`` raised from *inside* the
        transport -- turning a real bug into a silent retry with different
        arguments, which is the worst possible way to learn about it.
        """
        try:
            signature = inspect.signature(self.transport.execute_atomic)
        except (TypeError, ValueError):  # C-implemented or otherwise opaque
            return False
        parameters = signature.parameters
        if "click_backend" in parameters:
            return True
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _execute(self, operations: tuple[Operation, ...]) -> AtomicExecutionResult:
        if self._click_backend_supported:
            return self.transport.execute_atomic(  # type: ignore[call-arg]
                operations, click_backend=self.click_backend
            )
        # A transport that pins one click backend takes no such keyword. Any
        # TypeError from here on is a real bug and propagates.
        return self.transport.execute_atomic(operations)

    def _verify(
        self,
        result: AtomicExecutionResult,
        host_before: tuple[int, int] | None,
        host_after: tuple[int, int] | None,
    ) -> bool:
        if not self.verify_cursor_readback:
            return True
        if host_before is None or host_after is None:
            return True
        return host_before == result.cursor_before and host_after == result.cursor_after
