"""Apply operations to a guest and hand back a receipt.

The engine starts from resolved ``Operation``s and drives a transport, so its own
code contains no action names, no ``match`` on a grammar, and no coordinate
convention.

Every action is verified by cursor readback: the guest reports its own
before/after cursor, and the host reads the cursor independently on both sides.
A disagreement means the pointer moved outside the action, which is the failure
mode a delta-resolving grammar cannot otherwise detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..geometry import DisplayGeometry, geometry_from_screen_size
from ..ir import Operation
from .guest_program import AtomicExecutionResult, ExecutionError
from .transport import GuiTransport


@dataclass(frozen=True)
class StepReceipt:
    """What one applied action did, in enough detail to audit it later."""

    ok: bool
    operations: tuple[Operation, ...]
    requested_operations: tuple[Operation, ...]
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    host_cursor_before: tuple[int, int]
    host_cursor_after: tuple[int, int]
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
            "host_cursor_before": list(self.host_cursor_before),
            "host_cursor_after": list(self.host_cursor_after),
            "cursor_readback_verified": self.cursor_readback_verified,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "atomic_state": dict(self.atomic_state),
        }


class Engine:
    """Applies resolved operations to one guest through one transport."""

    def __init__(self, transport: GuiTransport) -> None:
        # There is no `verify_cursor_readback=False`.  With it off, `_verify`
        # returned True unconditionally, so the receipt said
        # `cursor_readback_verified: true` while `host_cursor_before/after` were
        # both None -- a receipt asserting a check that provably had not run, which
        # is worse than one that admits it did not.  Nothing outside a single test
        # ever passed False.
        self.transport = transport
        self.receipts: list[StepReceipt] = []

    def geometry(self) -> DisplayGeometry:
        """Ask the guest how big its screen is, as a ``DisplayGeometry``."""
        width, height = self.transport.screen_size()
        return geometry_from_screen_size(width, height)

    def cursor(self) -> tuple[int, int]:
        return self.transport.cursor_position()

    def apply(self, operations: tuple[Operation, ...]) -> StepReceipt:
        """Send one action -- one guest process -- and build its receipt."""
        host_before = self.transport.cursor_position()
        result = self._execute(operations)
        host_after = self.transport.cursor_position()
        verified = self._verify(result, host_before, host_after)
        state = result.as_dict()
        state.update(
            {
                "host_cursor_before": list(host_before),
                "host_cursor_after": list(host_after),
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

    def _execute(self, operations: tuple[Operation, ...]) -> AtomicExecutionResult:
        return self.transport.execute_atomic(operations)

    def _verify(
        self,
        result: AtomicExecutionResult,
        host_before: tuple[int, int],
        host_after: tuple[int, int],
    ) -> bool:
        return host_before == result.cursor_before and host_after == result.cursor_after
