"""Consumer-specific coupling lives here and nowhere else.

Every module under ``env_fleet.adapters`` may name a particular trainer,
orchestrator, or RL framework. Every module *outside* it must not: the core
speaks only of machines, registries, capacity, and Slurm. Adapters are not
imported by the core -- import direction is strictly adapter -> core -- so an
adapter can be deleted without touching a line of fleet logic.
"""

from __future__ import annotations

__all__: list[str] = []
