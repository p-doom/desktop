"""A stand-in ``apptainer`` binary, so the provider's real subprocess path runs.

The provider has never been executed.  What must be tested is OUR argv
construction, OUR base64 framing and OUR error handling -- not Apptainer's
container runtime.  So this stub speaks the three subcommands the provider uses
(``instance start``/``stop``/``list --json`` and ``exec``), tracks instances in a
state file, and runs an ``exec`` body directly on the host.

``emit_banner`` is the interesting knob: ``exec`` runs ``bash -lc``, a LOGIN
shell, so a container profile can print onto the same stdout a downloaded file
travels on.  The stub can reproduce that.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_SCRIPT = r"""#!/bin/bash
# A stand-in apptainer. Not a container runtime: it runs `exec` bodies on the host.
STATE="${FAKE_APPTAINER_STATE:?FAKE_APPTAINER_STATE is unset}"
touch "$STATE"
sub="$1"; shift
case "$sub" in
  exec)
    args=("$@")
    index=0
    target=""
    while [ $index -lt ${#args[@]} ]; do
      case "${args[$index]}" in
        instance://*) target="${args[$index]#instance://}"; break ;;
      esac
      index=$((index+1))
    done
    if [ -z "$target" ]; then echo "no instance:// target in argv" >&2; exit 64; fi
    if ! grep -qx "$target" "$STATE"; then echo "no such instance: $target" >&2; exit 65; fi
    printf '%s' "${FAKE_APPTAINER_BANNER:-}"
    exec "${args[@]:$((index+1))}"
    ;;
  instance)
    action="$1"; shift
    case "$action" in
      start)
        if [ -n "${FAKE_APPTAINER_START_FAILS:-}" ]; then
          echo "instance start refused by the stub" >&2; exit 1
        fi
        printf '%s\n' "${@: -1}" >> "$STATE"; exit 0 ;;
      stop)
        grep -vx "$1" "$STATE" > "$STATE.next" || true
        mv "$STATE.next" "$STATE"; exit 0 ;;
      list)
        printf '{"instances":['
        first=1
        while read -r name; do
          [ -z "$name" ] && continue
          [ $first -eq 0 ] && printf ','
          printf '{"instance":"%s","pid":1}' "$name"; first=0
        done < "$STATE"
        printf ']}\n'; exit 0 ;;
    esac
    echo "unsupported instance action: $action" >&2; exit 70 ;;
esac
echo "unsupported subcommand: $sub" >&2; exit 70
"""

_BROKEN_LIST_SCRIPT = r"""#!/bin/bash
# `instance list --json` answers with something that is not JSON.
if [ "$1" = "instance" ] && [ "$2" = "list" ]; then echo "not json at all"; exit 0; fi
exit 1
"""


def install(directory: Path, *, broken_list: bool = False) -> Path:
    """Write the stub into ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("apptainer-broken-list" if broken_list else "apptainer")
    path.write_text(_BROKEN_LIST_SCRIPT if broken_list else _SCRIPT)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return path


def state_file(directory: Path) -> Path:
    """Point the stub at a state file and return it."""
    path = directory / "instances.state"
    path.write_text("")
    os.environ["FAKE_APPTAINER_STATE"] = str(path)
    return path
