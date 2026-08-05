"""Shared fixtures and the two environment gates.

Two markers decide whether a test can run *here* rather than being deleted:

``needs_vm``
    Needs a real ``qemu-img`` / ``qemu-system-x86_64``.  Those binaries are not
    installed on every node -- on this cluster they exist only inside the
    ``osworld-guest-kvm`` container -- so the test reads them from
    ``DESKTOP_ENV_QEMU_BIN`` / ``DESKTOP_ENV_QEMU_IMG_BIN`` and skips when they
    do not resolve.  A one-line wrapper script that ``exec``s into the container
    is a legitimate value for either.

``needs_build``
    Needs a built ``.sif``.  Point ``DESKTOP_ENV_TEST_SIF`` at one.

Skipping is deliberate rather than absent: a suite that silently contains no VM
coverage looks identical to a suite whose VM coverage passes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def _resolve_binary(variable: str, default: str) -> str | None:
    """An executable for ``variable``, or ``None`` when it cannot be found."""
    raw = os.environ.get(variable) or default
    candidate = Path(raw).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())
    found = shutil.which(raw)
    return found


def qemu_img_binary() -> str | None:
    return _resolve_binary("DESKTOP_ENV_QEMU_IMG_BIN", "qemu-img")


def qemu_system_binary() -> str | None:
    return _resolve_binary("DESKTOP_ENV_QEMU_BIN", "qemu-system-x86_64")


def test_sif() -> Path | None:
    """The KVM-tier ``.sif``, via ``DESKTOP_ENV_TEST_SIF``."""
    raw = os.environ.get("DESKTOP_ENV_TEST_SIF")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def nonkvm_test_sif() -> Path | None:
    """The non-KVM-tier ``.sif``, via ``DESKTOP_ENV_TEST_NONKVM_SIF``.

    Separate from ``DESKTOP_ENV_TEST_SIF`` because the two tiers are different
    images with different contents, and a test written for one says nothing about
    the other -- which is the whole point of the two-tier split.
    """
    raw = os.environ.get("DESKTOP_ENV_TEST_NONKVM_SIF")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def pytest_configure(config: pytest.Config) -> None:
    for marker, description in (
        ("needs_vm", "requires a real qemu-img / qemu-system-x86_64 binary"),
        ("needs_build", "requires a built .sif (DESKTOP_ENV_TEST_SIF)"),
        (
            "needs_nonkvm_build",
            "requires a built non-KVM .sif (DESKTOP_ENV_TEST_NONKVM_SIF)",
        ),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    no_qemu = pytest.mark.skip(
        reason="no qemu binary: set DESKTOP_ENV_QEMU_BIN / DESKTOP_ENV_QEMU_IMG_BIN"
    )
    no_sif = pytest.mark.skip(reason="no built .sif: set DESKTOP_ENV_TEST_SIF")
    no_nonkvm_sif = pytest.mark.skip(
        reason="no built non-KVM .sif: set DESKTOP_ENV_TEST_NONKVM_SIF"
    )
    no_kvm = pytest.mark.skip(reason="/dev/kvm is not readable and writable")
    no_apptainer = pytest.mark.skip(reason="no apptainer binary")
    have_qemu = qemu_img_binary() is not None and qemu_system_binary() is not None
    have_sif = test_sif() is not None
    have_nonkvm_sif = nonkvm_test_sif() is not None
    have_kvm = os.access("/dev/kvm", os.R_OK | os.W_OK)
    have_apptainer = shutil.which("apptainer") is not None
    for item in items:
        if "needs_vm" in item.keywords and not have_qemu:
            item.add_marker(no_qemu)
        if "needs_build" in item.keywords and not have_sif:
            item.add_marker(no_sif)
        if "needs_nonkvm_build" in item.keywords and not have_nonkvm_sif:
            item.add_marker(no_nonkvm_sif)
        if "kvm" in item.keywords and not have_kvm:
            item.add_marker(no_kvm)
        if "apptainer" in item.keywords and not have_apptainer:
            item.add_marker(no_apptainer)


@pytest.fixture
def recording():
    """A fresh ``RecordingTransport``."""
    from desktop_env.execute.transport import RecordingTransport

    return RecordingTransport()
