"""The two Apptainer definitions.

Everything cheap is checked statically here, because the expensive checks need a
build.  The two ``needs_build`` tests at the end are the real ones, and the first
of them found the defect that mattered: the KVM tier named its qemu PACKAGE after
the BINARY (``qemu-system-x86_64``), which is not an Ubuntu package, so
``apt-get`` exited 100 and the build FATAL'd.  The tier that produces every
reportable number could not be built at all.

The two risks the files flag rather than solve -- ``--writable-tmpfs`` for a
writable browser profile, and ``instance start`` not behaving like a Docker
``ENTRYPOINT`` -- are asserted to still be flagged, so they cannot be silently
dropped, and the Chromium claim is checked against what is actually installed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

IMAGES_DIR = Path(__file__).resolve().parent.parent / "desktop" / "vm" / "images"
KVM_DEF = IMAGES_DIR / "osworld-guest-kvm.def"
NONKVM_DEF = IMAGES_DIR / "desktop-nonkvm.def"
IMAGES_README = IMAGES_DIR / "README.md"


def _section(text: str, name: str) -> str:
    """The body of one ``%section`` of a def-file."""
    match = re.search(rf"^%{name}\b(.*?)(?=^%|\Z)", text, re.MULTILINE | re.DOTALL)
    return "" if match is None else match.group(1)


@pytest.fixture(scope="module")
def kvm_text() -> str:
    return KVM_DEF.read_text()


@pytest.fixture(scope="module")
def nonkvm_text() -> str:
    return NONKVM_DEF.read_text()


def test_both_definitions_exist_and_are_packaged():
    assert KVM_DEF.is_file() and NONKVM_DEF.is_file() and IMAGES_README.is_file()
    pyproject = (IMAGES_DIR.parent.parent.parent / "pyproject.toml").read_text()
    assert '"desktop.vm.images" = ["*.def", "*.md", "*.patch"]' in pyproject


@pytest.mark.parametrize("definition", [KVM_DEF, NONKVM_DEF])
def test_a_definition_declares_a_pinned_base(definition):
    text = definition.read_text()
    assert text.startswith("Bootstrap: docker")
    assert re.search(r"^From: ubuntu:22\.04$", text, re.MULTILINE)


@pytest.mark.parametrize("definition", [KVM_DEF, NONKVM_DEF])
def test_a_definition_declares_its_parity_honestly(definition):
    labels = _section(definition.read_text(), "labels")
    assert "Parity" in labels


@pytest.mark.parametrize("definition", [KVM_DEF, NONKVM_DEF])
def test_a_definition_cleans_its_apt_lists(definition):
    """Otherwise the image carries tens of MB of package indices."""
    post = _section(definition.read_text(), "post")
    assert "apt-get clean" in post
    assert "rm -rf /var/lib/apt/lists" in post


@pytest.mark.parametrize("definition", [KVM_DEF, NONKVM_DEF])
def test_a_definition_installs_without_recommends(definition):
    post = _section(definition.read_text(), "post")
    for line in post.splitlines():
        if "apt-get" in line and "install" in line:
            assert "--no-install-recommends" in line, line


@pytest.mark.parametrize("definition", [KVM_DEF, NONKVM_DEF])
def test_a_definition_has_a_test_section(definition):
    assert _section(definition.read_text(), "test").strip()


@pytest.mark.parametrize("definition", [KVM_DEF, NONKVM_DEF])
def test_a_test_section_is_valid_shell(definition, tmp_path):
    """%test gates the build: a syntax error there fails an otherwise good image."""
    script = tmp_path / "test.sh"
    script.write_text(_section(definition.read_text(), "test"))
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_nonkvm_test_section_gives_pyautogui_the_display_it_requires(nonkvm_text):
    """FOUND BY BUILDING: this tier's own %test could never pass.

    ``import pyautogui`` is not display-free -- its X11 backend opens
    ``os.environ['DISPLAY']`` at import time -- so the bare
    ``python3 -c "import Xlib, pyautogui"`` in %test always failed, %test exited 1,
    and ``apptainer build`` ended in
    ``FATAL: ... failed to execute %test script: exit status 1``.  The tier had
    therefore never been built at all.

    The fix gives the import a display rather than weakening the check: Xvfb is
    already installed, so ``xvfb-run`` proves the real import path works.
    """
    test_section = _section(nonkvm_text, "test")
    assert "xvfb-run -a python3 -c" in test_section
    assert "import Xlib, pyautogui" in test_section
    # The bare, display-less form must not come back.
    assert '\n    python3 -c "import Xlib, pyautogui"' not in test_section
    # ... and the binary it now depends on is checked for.
    assert "xvfb-run" in test_section.split("for binary in")[1].split(";")[0]


def test_the_kvm_tier_names_the_qemu_PACKAGE_not_the_binary(kvm_text):
    """THE DEFECT THIS FILE'S BUILD FOUND.

    ``qemu-system-x86_64`` is the BINARY.  The Ubuntu 22.04 PACKAGE that ships it
    is ``qemu-system-x86``.  Naming the package after the binary makes ``apt-get``
    answer ``E: Unable to locate package qemu-system-x86_64`` and exit 100, which
    Apptainer turns into ``FATAL: ... while running %post section``.
    """
    post = _section(kvm_text, "post")
    assert re.search(r"^\s+qemu-system-x86\s*\\?$", post, re.MULTILINE), post
    assert "qemu-system-x86_64 \\" not in post
    assert "qemu-utils" in post


def test_the_kvm_tier_still_checks_for_the_qemu_BINARY_by_its_real_name(kvm_text):
    """The package is ``qemu-system-x86``; the binary it must provide is
    ``qemu-system-x86_64``, and that is what the runtime execs."""
    test_section = _section(kvm_text, "test")
    assert "command -v qemu-system-x86_64" in test_section
    assert "command -v qemu-img" in test_section
    import inspect

    from desktop.vm import factory
    from desktop.vm.factory import build_qemu_runtime  # noqa: F401

    assert '"qemu-system-x86_64"' in inspect.getsource(factory)


def test_the_kvm_tier_contains_no_desktop(kvm_text):
    """The guest desktop is a pinned qcow2 INPUT, so a container rebuild cannot
    silently change what a benchmark number means."""
    post = _section(kvm_text, "post")
    for desktop_package in ("xvfb", "libreoffice", "chromium", "firefox", "x11vnc", "mutter"):
        assert desktop_package not in post.lower(), desktop_package


def test_the_kvm_tier_pins_a_short_qmp_directory(kvm_text):
    """AF_UNIX caps the socket path, so it must not default to a deep scratch dir."""
    assert "export DESKTOP_ENV_QMP_DIR=/tmp" in _section(kvm_text, "environment")
    from desktop.vm.factory import ENVIRONMENT

    assert "DESKTOP_ENV_QMP_DIR" in ENVIRONMENT


def test_the_kvm_tier_documents_its_runtime_requirements(kvm_text):
    assert "/dev/kvm" in kvm_text
    assert "snapshot=on" in kvm_text
    assert "TMPDIR" in kvm_text
    assert "tcg" in kvm_text


def test_the_kvm_tier_needs_no_privileged_flags(kvm_text):
    assert "No --fakeroot and no privileged flags" in kvm_text


def test_the_nonkvm_tier_states_it_gives_no_parity(nonkvm_text):
    assert "IT GIVES NO OSWorld-Verified PARITY, EVER." in nonkvm_text
    assert "Parity   none" in _section(nonkvm_text, "labels")


def test_the_nonkvm_tier_installs_the_input_stack_the_executor_needs(nonkvm_text):
    """The executor gets Xlib; PyAutoGUI remains for OSWorld setup and graders."""
    post = _section(nonkvm_text, "post")
    assert "pyautogui==0.9.54" in post
    assert "python-xlib==0.33" in post


def test_the_nonkvm_tier_installs_the_xlib_binding_the_typing_path_imports(nonkvm_text):
    from desktop.vm.client import ACTION_EXECUTOR_PATH

    program = ACTION_EXECUTOR_PATH.read_text()
    assert "from Xlib import X" in program
    assert "Gtk" not in program and "pyautogui" not in program
    post = _section(nonkvm_text, "post")
    assert "python-xlib==0.33" in post


def test_the_nonkvm_tier_pins_its_novnc_checkout(nonkvm_text):
    post = _section(nonkvm_text, "post")
    assert "--branch v1.5.0" in post and "--branch v0.12.0" in post


def _start_desktop_script(text: str) -> str:
    """The body of the ``start-desktop`` heredoc inside ``%post``."""
    match = re.search(
        r"cat > /opt/desktop-env/bin/start-desktop <<'SCRIPT'\n(.*?)\nSCRIPT",
        text,
        re.DOTALL,
    )
    assert match is not None, "the start-desktop heredoc could not be located"
    return match.group(1)


def _commands_only(script: str) -> str:
    """The script with comment lines dropped, so ordering is about real commands."""
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


def test_the_nonkvm_startup_waits_for_xvfb_before_the_window_manager(nonkvm_text):
    """Starting mutter before Xvfb listens gives an instance that looks alive and
    has no window manager -- which presents as "clicks do nothing"."""
    script = _commands_only(_start_desktop_script(nonkvm_text))
    assert "until xdpyinfo" in script
    assert script.index("Xvfb ") < script.index("until xdpyinfo") < script.index("mutter")
    assert script.index("until xdpyinfo") < script.index("x11vnc")


def test_the_nonkvm_startup_bounds_its_wait_for_xvfb(nonkvm_text):
    """An unbounded ``until`` loop would hang the instance instead of failing."""
    script = _start_desktop_script(nonkvm_text)
    assert "deadline=" in script and "exit 1" in script


def test_the_nonkvm_startup_script_is_valid_shell(nonkvm_text, tmp_path):
    """The heredoc body is never syntax-checked by the build, so check it here."""
    script = tmp_path / "start-desktop"
    script.write_text(_start_desktop_script(nonkvm_text))
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_nonkvm_tier_creates_the_home_and_runtime_dirs_it_declares(nonkvm_text):
    """``%environment`` points HOME and XDG_RUNTIME_DIR at /tmp paths; pyautogui
    will not import without them."""
    environment = _section(nonkvm_text, "environment")
    assert "XDG_RUNTIME_DIR=/tmp/runtime" in environment
    assert "HOME=/tmp/home" in environment
    for section in ("startscript", "runscript"):
        assert "mkdir -p /tmp/home /tmp/runtime" in _section(nonkvm_text, section)


def test_a_bare_exec_into_the_nonkvm_tier_creates_those_dirs(nonkvm_text):
    """Only %startscript and %runscript used to create HOME and XDG_RUNTIME_DIR.

    ``apptainer exec`` runs neither -- and ``exec`` is exactly how the README says
    to drive this tier, since it has no in-guest HTTP agent -- so the documented
    invocation pointed HOME at a directory that did not exist and pyautogui failed
    to import.  Creating them in %post would not have helped either: /tmp is a
    fresh tmpfs per instance.
    """
    environment = _section(nonkvm_text, "environment")
    assert "mkdir -p /tmp/home /tmp/runtime" in environment
    assert "|| true" in environment, "a failed mkdir must not break the shell"
    # The path HOME points at is the one that gets created.
    assert "HOME=/tmp/home" in environment
    assert "XDG_RUNTIME_DIR=/tmp/runtime" in environment


def test_the_nonkvm_environment_section_is_valid_shell(nonkvm_text, tmp_path):
    """%environment is SOURCED, so a syntax error there breaks every invocation."""
    script = tmp_path / "environment.sh"
    script.write_text(_section(nonkvm_text, "environment"))
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_absent_browser_is_stated_as_a_fact_with_its_evidence(nonkvm_text):
    """The file used to flag a Chromium/Firefox profile risk while installing
    NEITHER browser, so the risk as written could not be triggered and a reader
    could reasonably assume a browser was present.

    On ubuntu:22.04 there is no apt-installable browser at all -- verified:
    ``chromium-browser`` has no installation candidate (a snap transitional stub),
    ``chromium`` does not exist, ``firefox`` is ``1:1snap1-0ubuntu2``.  Snaps need
    systemd and a writable /snap, neither of which an Apptainer instance has.  So
    the honest resolution is to say so, with the evidence, rather than paste a
    package name into the install list and assume the build proves it.
    """
    post = _section(nonkvm_text, "post").lower()
    assert "chromium" not in post and "firefox" not in post
    assert "THERE IS NO BROWSER IN THIS TIER" in nonkvm_text
    # The evidence, not just the conclusion (the phrase wraps in the def-file).
    assert "installation candidate" in nonkvm_text
    assert "a snap" in nonkvm_text and "transitional stub" in nonkvm_text
    assert "1:1snap1-0ubuntu2" in nonkvm_text
    assert "third-party PPA" in nonkvm_text
    assert "cannot check anything browser-shaped" in nonkvm_text


def test_the_writable_profile_mechanism_is_still_documented(nonkvm_text):
    """Kept for whoever adds a browser: a .sif is read-only, so a profile
    directory needs an overlay."""
    assert "--writable-tmpfs" in nonkvm_text
    assert "--overlay" in nonkvm_text
    from desktop.vm.sandbox_protocol import ApptainerSandboxProvider

    assert ApptainerSandboxProvider().writable_tmpfs is True


def test_the_readme_agrees_that_there_is_no_browser():
    readme = IMAGES_README.read_text()
    assert "no installation candidate" in readme
    assert "cannot check anything browser-shaped" in readme
    assert "**no browser**" in readme


def test_the_instance_start_versus_entrypoint_risk_is_still_flagged(nonkvm_text):
    assert "does not fork the whole startup" in nonkvm_text
    assert "%startscript" in nonkvm_text
    assert "non-daemonising" in nonkvm_text


def test_the_readme_agrees_with_the_definitions_about_parity():
    readme = IMAGES_README.read_text()
    assert "**none, ever**" in readme
    assert "**full**" in readme
    assert "osworld-guest-kvm.def" in readme and "desktop-nonkvm.def" in readme
    assert "Flagged, not faked" in readme


def test_the_readme_build_commands_name_the_real_files():
    readme = IMAGES_README.read_text()
    for definition in (KVM_DEF, NONKVM_DEF):
        assert definition.name in readme
        assert f"{definition.stem}.sif" in readme


@pytest.mark.needs_build
@pytest.mark.apptainer
def test_the_built_kvm_tier_provides_working_qemu_binaries():
    """Both binaries present AND able to enumerate accelerators.

    Run against a real build with:
        apptainer build osworld-guest-kvm.sif osworld-guest-kvm.def
        DESKTOP_ENV_TEST_SIF=osworld-guest-kvm.sif pytest -m needs_build
    """
    from tests.conftest import test_sif

    sif = test_sif()
    assert sif is not None
    for binary in ("qemu-system-x86_64", "qemu-img"):
        which = subprocess.run(
            ["apptainer", "exec", str(sif), "bash", "-lc", f"command -v {binary}"],
            capture_output=True,
            text=True,
        )
        assert which.returncode == 0, f"{binary} missing from the built image"
    accelerators = subprocess.run(
        ["apptainer", "exec", str(sif), "qemu-system-x86_64", "-accel", "help"],
        capture_output=True,
        text=True,
    )
    assert accelerators.returncode == 0
    assert "tcg" in accelerators.stdout
    assert "kvm" in accelerators.stdout


@pytest.mark.needs_build
@pytest.mark.apptainer
def test_the_built_kvm_tier_passes_its_own_test_section():
    from tests.conftest import test_sif

    sif = test_sif()
    assert sif is not None
    result = subprocess.run(["apptainer", "test", str(sif)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "qemu present" in result.stdout


@pytest.mark.needs_build
@pytest.mark.apptainer
def test_the_built_kvm_tier_pins_the_qmp_directory_at_runtime():
    from tests.conftest import test_sif

    sif = test_sif()
    assert sif is not None
    result = subprocess.run(
        ["apptainer", "exec", str(sif), "bash", "-lc", 'printf %s "$DESKTOP_ENV_QMP_DIR"'],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "/tmp"


# --------------------------------------------------------------------------- #
# needs_nonkvm_build: the non-KVM tier, against a real build
#
# Both assertions below correspond to defects found BY BUILDING this tier, which
# had never been built: its %test could not pass, and its documented invocation
# pointed HOME at a directory nothing created.
# --------------------------------------------------------------------------- #


@pytest.mark.needs_nonkvm_build
@pytest.mark.apptainer
def test_the_documented_bare_exec_invocation_works(tmp_path):
    """``apptainer exec`` runs neither %startscript nor %runscript, and ``exec`` is
    how the README says to drive this tier.  HOME and XDG_RUNTIME_DIR must exist
    and be writable, or pyautogui fails to import on the documented path."""
    from tests.conftest import nonkvm_test_sif

    sif = nonkvm_test_sif()
    assert sif is not None
    result = subprocess.run(
        [
            "apptainer",
            "exec",
            str(sif),
            "bash",
            "-lc",
            'test -d "$HOME" && test -w "$HOME" && test -d "$XDG_RUNTIME_DIR" '
            '&& printf %s "$HOME"',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp/home"


@pytest.mark.needs_nonkvm_build
@pytest.mark.apptainer
def test_pyautogui_imports_under_xvfb_but_not_bare():
    """The exact defect that made this tier unbuildable, both halves.

    ``import pyautogui`` opens ``os.environ['DISPLAY']`` at import time, so the
    bare form fails with no X server -- which is what the old %test ran, and why
    ``apptainer build`` ended in ``FATAL: ... %test script: exit status 1``.  Under
    ``xvfb-run`` the same import succeeds.
    """
    from tests.conftest import nonkvm_test_sif

    sif = nonkvm_test_sif()
    assert sif is not None
    bare = subprocess.run(
        ["apptainer", "exec", str(sif), "python3", "-c", "import Xlib, pyautogui"],
        capture_output=True,
        text=True,
    )
    assert bare.returncode != 0, "a display-less import unexpectedly succeeded"
    under_xvfb = subprocess.run(
        [
            "apptainer",
            "exec",
            str(sif),
            "xvfb-run",
            "-a",
            "python3",
            "-c",
            "import Xlib, pyautogui",
        ],
        capture_output=True,
        text=True,
    )
    assert under_xvfb.returncode == 0, under_xvfb.stderr


@pytest.mark.needs_nonkvm_build
@pytest.mark.apptainer
def test_the_nonkvm_tier_really_has_no_browser():
    """The claim the def-file now makes as fact, checked against the image."""
    from tests.conftest import nonkvm_test_sif

    sif = nonkvm_test_sif()
    assert sif is not None
    result = subprocess.run(
        [
            "apptainer",
            "exec",
            str(sif),
            "bash",
            "-lc",
            "command -v chromium chromium-browser firefox google-chrome 2>/dev/null | head -1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "", f"a browser IS present: {result.stdout!r}"


@pytest.mark.needs_nonkvm_build
@pytest.mark.apptainer
def test_the_nonkvm_tier_has_the_input_stack_the_executor_needs():
    """The one thing that makes this tier useful: the same primitive names."""
    from tests.conftest import nonkvm_test_sif

    sif = nonkvm_test_sif()
    assert sif is not None
    for binary in ("Xvfb", "xdotool", "x11vnc", "mutter", "tint2", "scrot", "xvfb-run"):
        result = subprocess.run(
            ["apptainer", "exec", str(sif), "bash", "-lc", f"command -v {binary}"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{binary} missing from the built image"
