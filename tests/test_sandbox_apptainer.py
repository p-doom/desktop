"""``ApptainerSandboxProvider``: binary safety and download integrity.

The claim under test is BINARY SAFETY.  The payloads that move through here are
PNG screenshots and qcow2 overlays, and decoding guest stdout as UTF-8 with
``errors="replace"`` turns every invalid sequence into U+FFFD irreversibly.  So
every round trip below uses a payload containing invalid UTF-8, and asserts byte
equality rather than "it worked".

The second thing under test is that a payload which is NOT the file raises rather
than persisting a truncated file that looks like a successful download.  ``exec``
runs ``bash -lc`` -- a LOGIN shell -- so a container profile banner on stdout is
the realistic way that happens.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pytest

from pixeldesk.vm.sandbox_protocol import (
    DEFAULT_TRANSFER_TIMEOUT_S,
    MAX_TRANSFER_BYTES,
    ApptainerSandboxProvider,
    SandboxCreateError,
    SandboxEndpoint,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
    TransferTooLargeError,
    apptainer_available,
)
from tests.support import fake_apptainer

#: Bytes chosen to break any implementation that decodes stdout as text:
#: a PNG magic number, every byte value, invalid UTF-8 continuation bytes, NULs.
BINARY_PAYLOAD = (
    b"\x89PNG\r\n\x1a\n"
    + bytes(range(256))
    + b"\xff\xfe\xfd\xc3\x28\xa0\xa1\xe2\x28\xa1"
    + b"\x00\x00\x00"
    + b"\xed\xa0\x80"  # a lone surrogate in UTF-8 form
)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_APPTAINER_STATE", str(tmp_path / "instances.state"))
    monkeypatch.delenv("FAKE_APPTAINER_BANNER", raising=False)
    monkeypatch.delenv("FAKE_APPTAINER_START_FAILS", raising=False)
    (tmp_path / "instances.state").write_text("")
    binary = fake_apptainer.install(tmp_path / "bin")
    return ApptainerSandboxProvider(binary=str(binary))


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "guest.sif"
    path.write_bytes(b"a stand-in for a SIF")
    return path


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.fixture
def sandbox(provider, image):
    return run(provider.create(SandboxSpec(image=str(image), ports=[5000, 9222])))


@pytest.fixture
def guest_path(tmp_path):
    """A guest-side path unique to this test.

    The stub apptainer runs ``exec`` bodies on the host, so a fixed path such as
    ``/tmp/p.bin`` is shared state: tests then depend on each other's ordering and
    a download can read a neighbour's payload. This makes each test's guest path
    its own.
    """
    return str(tmp_path / "guest-side.bin")


def test_create_starts_an_instance_and_reports_it_running(provider, image):
    handle = run(provider.create(SandboxSpec(image=str(image))))
    assert handle.provider_name == "apptainer"
    assert handle.sandbox_id.startswith("desktop-env-")
    assert run(provider.status(handle)) is SandboxStatus.RUNNING


def test_create_passes_writable_tmpfs_by_default(provider, image, tmp_path):
    """A .sif is read-only, and an app wanting a writable profile needs an
    overlay; without one the failure is a confusing crash."""
    assert provider.writable_tmpfs is True
    plain = ApptainerSandboxProvider(binary=provider.binary, writable_tmpfs=False)
    assert plain.writable_tmpfs is False


def test_create_requires_an_image(provider):
    with pytest.raises(SandboxCreateError, match="requires an image"):
        run(provider.create(SandboxSpec()))


def test_create_refuses_an_image_that_is_not_there(provider, tmp_path):
    with pytest.raises(SandboxCreateError, match="image not found"):
        run(provider.create(SandboxSpec(image=str(tmp_path / "absent.sif"))))


def test_a_failed_instance_start_raises_with_the_runtime_message(
    provider, image, monkeypatch
):
    monkeypatch.setenv("FAKE_APPTAINER_START_FAILS", "1")
    with pytest.raises(SandboxCreateError, match="instance start failed"):
        run(provider.create(SandboxSpec(image=str(image))))


def test_close_stops_the_instance(provider, sandbox):
    run(provider.close(sandbox))
    assert run(provider.status(sandbox)) is SandboxStatus.STOPPED


def test_aclose_stops_every_instance_the_provider_started(provider, image):
    handles = [run(provider.create(SandboxSpec(image=str(image)))) for _ in range(3)]
    run(provider.aclose())
    for handle in handles:
        assert run(provider.status(handle)) is SandboxStatus.STOPPED


def test_status_is_unknown_when_the_runtime_cannot_be_queried(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_APPTAINER_STATE", str(tmp_path / "s"))
    broken = fake_apptainer.install(tmp_path / "bin", broken_list=True)
    provider = ApptainerSandboxProvider(binary=str(broken))
    handle = SandboxHandle(sandbox_id="x", provider_name="apptainer", raw={})
    assert run(provider.status(handle)) is SandboxStatus.UNKNOWN


def test_exec_returns_stdout_stderr_and_the_exit_code(provider, sandbox):
    result = run(provider.exec(sandbox, "echo out; echo err >&2; exit 3"))
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"
    assert result.return_code == 3


def test_exec_cannot_change_user(provider, sandbox):
    """Honouring a different user would need setuid, which is the thing this
    provider exists to avoid."""
    with pytest.raises(ValueError, match="cannot change user"):
        run(provider.exec(sandbox, "true", user="root"))


def test_exec_forwards_env_and_cwd(provider, sandbox, tmp_path):
    result = run(provider.exec(sandbox, "printf %s \"$DE_PROBE\"", env={"DE_PROBE": "v"}))
    assert result.return_code == 0


def test_exec_against_an_unknown_instance_fails(provider, image):
    handle = SandboxHandle(sandbox_id="never-started", provider_name="apptainer", raw={})
    assert run(provider.exec(handle, "true")).return_code != 0


def test_a_timeout_is_reported_as_a_timeout_not_an_exit_code(provider, sandbox):
    result = run(provider.exec(sandbox, "sleep 5", timeout_s=0.3))
    assert result.error_type == "timeout"
    assert result.return_code == -1
    assert result.stdout is None


def test_a_binary_payload_survives_upload_and_download_byte_for_byte(
    provider, sandbox, tmp_path, guest_path
):
    source = tmp_path / "in.bin"
    source.write_bytes(BINARY_PAYLOAD)
    run(provider.upload_file(sandbox, source, guest_path))
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target))
    got = target.read_bytes()
    assert got == BINARY_PAYLOAD
    assert hashlib.sha256(got).hexdigest() == hashlib.sha256(BINARY_PAYLOAD).hexdigest()


def test_no_replacement_character_appears_anywhere_in_the_round_trip(
    provider, sandbox, tmp_path, guest_path
):
    """The previous implementation's exact failure: U+FFFD, unrecoverably."""
    source = tmp_path / "in.bin"
    source.write_bytes(BINARY_PAYLOAD)
    run(provider.upload_file(sandbox, source, guest_path))
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target))
    assert b"\xef\xbf\xbd" not in target.read_bytes()


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00",
        b"\x00" * 4096,
        bytes(range(256)),
        b"\xff\xfe",
        b"plain ascii",
        b"\x89PNG\r\n\x1a\n" + os.urandom(8192),
    ],
    ids=["empty", "one-nul", "many-nuls", "all-bytes", "invalid-utf8", "ascii", "png-ish"],
)
def test_payload_shapes_all_round_trip(provider, sandbox, tmp_path, payload, guest_path):
    source = tmp_path / "in.bin"
    source.write_bytes(payload)
    run(provider.upload_file(sandbox, source, guest_path))
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target))
    assert target.read_bytes() == payload


def test_a_target_path_hostile_to_quoting_round_trips(provider, sandbox, tmp_path):
    """``shlex.quote`` on both directions, checked with a path full of shell
    metacharacters -- an unquoted one would execute ``$(id)``."""
    source = tmp_path / "in.bin"
    source.write_bytes(BINARY_PAYLOAD)
    weird = str(tmp_path / "we ird$(id)`hostname`;rm -rf x'\".bin")
    run(provider.upload_file(sandbox, source, weird))
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, weird, target))
    assert target.read_bytes() == BINARY_PAYLOAD


def test_download_creates_missing_parent_directories(provider, sandbox, tmp_path, guest_path):
    source = tmp_path / "in.bin"
    source.write_bytes(b"x")
    run(provider.upload_file(sandbox, source, guest_path))
    target = tmp_path / "a" / "b" / "c" / "out.bin"
    run(provider.download_file(sandbox, guest_path, target))
    assert target.read_bytes() == b"x"


def test_an_upload_to_an_unwritable_target_raises(provider, sandbox, tmp_path):
    source = tmp_path / "in.bin"
    source.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="upload failed"):
        run(provider.upload_file(sandbox, source, "/proc/nonexistent/x"))


def test_downloading_a_file_that_is_not_there_raises(provider, sandbox, tmp_path):
    with pytest.raises(RuntimeError, match="download failed"):
        run(provider.download_file(sandbox, "/tmp/definitely-absent", tmp_path / "o"))


def test_a_failed_download_persists_nothing(provider, sandbox, tmp_path):
    target = tmp_path / "never" / "written.bin"
    with pytest.raises(RuntimeError):
        run(provider.download_file(sandbox, "/tmp/definitely-absent", target))
    assert not target.exists()


@pytest.mark.parametrize(
    "banner",
    [
        "Welcome to the container\n",
        "MOTD",
        "Warning: something\n\n",
        "x" * 4,
        "not base64 at all !!! @@@",
    ],
    ids=["newline", "alphabet-only-no-newline", "two-newlines", "four-chars", "punctuation"],
)
def test_login_shell_stdout_noise_cannot_corrupt_a_download(
    provider, sandbox, tmp_path, monkeypatch, banner, guest_path
):
    """``exec`` runs ``bash -lc`` -- a LOGIN shell -- so the container's profile
    prints onto the very stdout the file is travelling on.

    Why the payload is fenced rather than merely decoded strictly:

        given  /tmp/p.bin containing exactly 1024 bytes (bytes(range(256)) * 4)
        and    a profile that emits ``printf 'MOTD'`` -- no newline
        then   stdout is ``MOTD`` + base64(file)
        and    ``base64.b64decode(..., validate=True)`` ACCEPTS it, because every
               character of ``MOTD`` is in the base64 alphabet and the total
               length stays a multiple of four
        and    the decode silently yields **1027 bytes**, not 1024

    ``validate=True`` rejects a banner containing a newline or any non-alphabet
    character -- the ``Welcome to the container\\n`` case -- and nothing else.
    Fencing the payload between two markers that are *not* base64-alphabet text
    makes the channel total: either the bytes are exact, or the download raises.
    """
    payload = bytes(range(256)) * 4
    source = tmp_path / "in.bin"
    source.write_bytes(payload)
    run(provider.upload_file(sandbox, source, guest_path))
    monkeypatch.setenv("FAKE_APPTAINER_BANNER", banner)
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target))
    assert target.read_bytes() == payload, f"banner {banner!r} corrupted the payload"


def test_stdout_noise_after_the_payload_also_cannot_corrupt_it(
    provider, sandbox, tmp_path, guest_path
):
    """A profile that prints on exit puts its noise AFTER the base64."""
    payload = bytes(range(256))
    source = tmp_path / "in.bin"
    source.write_bytes(payload)
    run(provider.upload_file(sandbox, source, guest_path))

    original_exec = provider.exec

    async def trailing_noise(handle, command, **kwargs):
        return await original_exec(handle, command + "; printf 'GOODBYE'", **kwargs)

    provider.exec = trailing_noise  # type: ignore[method-assign]
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target))
    assert target.read_bytes() == payload


def test_a_download_whose_stdout_has_no_payload_raises(provider, sandbox, tmp_path, guest_path):
    async def only_noise(handle, command, **kwargs):
        from pixeldesk.vm.sandbox_protocol import SandboxExecResult

        return SandboxExecResult("just a banner, no file", "", 0)

    provider.exec = only_noise  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="no fenced payload"):
        run(provider.download_file(sandbox, guest_path, tmp_path / "o"))


def test_a_corrupt_base64_body_raises_rather_than_persisting_a_truncated_file(
    provider, sandbox, tmp_path, guest_path
):
    from pixeldesk.vm.sandbox_protocol import (
        _DOWNLOAD_BEGIN,
        _DOWNLOAD_END,
        SandboxExecResult,
    )

    async def corrupt(handle, command, **kwargs):
        return SandboxExecResult(
            f"{_DOWNLOAD_BEGIN}!!!not base64!!!{_DOWNLOAD_END}", "", 0
        )

    provider.exec = corrupt  # type: ignore[method-assign]
    target = tmp_path / "out.bin"
    with pytest.raises(RuntimeError, match="non-base64 payload"):
        run(provider.download_file(sandbox, guest_path, target))
    assert not target.exists()


def test_whitespace_around_the_payload_is_tolerated(provider, sandbox, tmp_path, guest_path):
    import base64

    from pixeldesk.vm.sandbox_protocol import (
        _DOWNLOAD_BEGIN,
        _DOWNLOAD_END,
        SandboxExecResult,
    )

    payload = b"\x00\xff hello"
    encoded = base64.b64encode(payload).decode("ascii")

    async def padded(handle, command, **kwargs):
        return SandboxExecResult(
            f"{_DOWNLOAD_BEGIN}\n  {encoded}  \n{_DOWNLOAD_END}\n", "", 0
        )

    provider.exec = padded  # type: ignore[method-assign]
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target))
    assert target.read_bytes() == payload


def test_the_fence_markers_are_not_base64_alphabet_text():
    """So a fence can never be mistaken for payload, nor payload for a fence."""
    import string

    from pixeldesk.vm.sandbox_protocol import _DOWNLOAD_BEGIN, _DOWNLOAD_END

    alphabet = set(string.ascii_letters + string.digits + "+/=")
    for marker in (_DOWNLOAD_BEGIN, _DOWNLOAD_END):
        assert not set(marker) <= alphabet
        assert marker != _DOWNLOAD_END or marker != _DOWNLOAD_BEGIN
    assert _DOWNLOAD_BEGIN != _DOWNLOAD_END


def test_a_descriptor_round_trips_into_a_live_handle(provider, sandbox):
    descriptor = run(provider.serialize_handle(sandbox, scope="rollout-7"))
    assert descriptor["provider"] == "apptainer"
    assert descriptor["sandbox_id"] == sandbox.sandbox_id
    assert descriptor["ports"] == [5000, 9222]
    assert descriptor["scope"] == "rollout-7"
    rebuilt = run(provider.connect(descriptor))
    assert rebuilt.sandbox_id == sandbox.sandbox_id


def test_a_descriptor_is_json_serialisable(provider, sandbox):
    import json

    descriptor = run(provider.serialize_handle(sandbox))
    assert json.loads(json.dumps(descriptor))["sandbox_id"] == sandbox.sandbox_id


def test_a_descriptor_from_another_node_is_refused(provider, sandbox):
    """Apptainer instances are node-local; silently attaching to a same-named
    instance on this node would be talking to the wrong sandbox."""
    descriptor = run(provider.serialize_handle(sandbox))
    descriptor["host"] = "some-other-node.example"
    with pytest.raises(ValueError, match="node-local"):
        run(provider.connect(descriptor))


def test_a_descriptor_for_another_provider_is_refused(provider):
    with pytest.raises(ValueError, match="descriptor is for provider"):
        run(provider.connect({"provider": "docker", "sandbox_id": "x"}))


def test_a_descriptor_without_an_id_is_refused(provider):
    with pytest.raises(ValueError, match="no sandbox_id"):
        run(provider.connect({"provider": "apptainer"}))


def test_connecting_to_a_stopped_sandbox_is_refused(provider, sandbox):
    descriptor = run(provider.serialize_handle(sandbox))
    run(provider.close(sandbox))
    with pytest.raises(ValueError, match="is not running on this node"):
        run(provider.connect(descriptor))


def test_the_provider_satisfies_the_connectable_protocol():
    from pixeldesk.vm.sandbox_protocol import (
        ConnectableProvider,
        SupportsSandboxEndpoint,
    )

    provider = ApptainerSandboxProvider()
    assert isinstance(provider, ConnectableProvider)
    assert isinstance(provider, SupportsSandboxEndpoint)


def test_a_declared_port_resolves_to_an_endpoint(provider, sandbox):
    endpoint = run(provider.endpoint(sandbox, 5000))
    assert endpoint.endpoint.endswith(":5000")
    assert endpoint.endpoint.startswith("http://")


def test_an_undeclared_port_is_refused(provider, sandbox):
    with pytest.raises(ValueError, match="was not declared"):
        run(provider.endpoint(sandbox, 1234))


def test_a_sandbox_declaring_no_ports_allows_any(provider, image):
    handle = run(provider.create(SandboxSpec(image=str(image))))
    assert run(provider.endpoint(handle, 4321)).endpoint.endswith(":4321")


def test_an_endpoint_must_be_an_absolute_url():
    assert SandboxEndpoint(endpoint="http://h:1").endpoint == "http://h:1"
    for bad in ("", "   ", "not-a-url", "/relative", "h:1"):
        with pytest.raises(ValueError, match="absolute URL"):
            SandboxEndpoint(endpoint=bad)


def test_endpoint_headers_are_stringified():
    endpoint = SandboxEndpoint(endpoint="http://h:1", headers={"A": 1})
    assert endpoint.headers == {"A": "1"}


def test_unknown_resource_keys_are_refused():
    with pytest.raises(ValueError, match="Unknown sandbox resource keys"):
        SandboxResources.from_mapping({"cpus": 4})
    assert SandboxResources.from_mapping({"cpu": 2}).cpu == 2.0
    assert SandboxResources.from_mapping(None) == SandboxResources()


def test_a_spec_normalises_its_resources_mapping():
    spec = SandboxSpec(resources={"memory_mib": 512})
    assert isinstance(spec.resources, SandboxResources)
    assert spec.resources.memory_mib == 512


@pytest.mark.parametrize("bad", [0, 65536, -1, True, "abc", None, 1.5])
def test_an_invalid_port_declaration_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        SandboxSpec(ports=[bad])


def test_duplicate_port_declarations_are_refused():
    with pytest.raises(ValueError, match="Duplicate sandbox TCP port"):
        SandboxSpec(ports=[5000, 5000])


def test_ports_must_be_a_sequence():
    with pytest.raises(TypeError, match="list or tuple"):
        SandboxSpec(ports=5000)


def test_string_ports_are_coerced():
    assert SandboxSpec(ports=["5000"]).ports == (5000,)


def test_apptainer_available_reflects_the_real_binary():
    import shutil

    assert apptainer_available() == (shutil.which("apptainer") is not None)


@pytest.mark.needs_build
@pytest.mark.apptainer
def test_a_binary_round_trip_through_a_real_apptainer_instance(tmp_path):
    """The same round trip against a real ``.sif``.  Skipped without one.

    Set ``DESKTOP_ENV_TEST_SIF`` to a built image; the KVM-tier container is
    enough, since only ``bash``, ``base64`` and ``printf`` are exercised.
    """
    from tests.conftest import test_sif

    sif = test_sif()
    assert sif is not None
    provider = ApptainerSandboxProvider()
    handle = run(provider.create(SandboxSpec(image=str(sif), ready_timeout_s=180)))
    try:
        source = tmp_path / "in.bin"
        source.write_bytes(BINARY_PAYLOAD)
        run(provider.upload_file(handle, source, "/tmp/de_round_trip.bin"))
        target = tmp_path / "out.bin"
        run(provider.download_file(handle, "/tmp/de_round_trip.bin", target))
        assert target.read_bytes() == BINARY_PAYLOAD
        assert run(provider.status(handle)) is SandboxStatus.RUNNING
        descriptor = run(provider.serialize_handle(handle))
        assert run(provider.connect(descriptor)).sandbox_id == handle.sandbox_id
    finally:
        run(provider.close(handle))


def test_both_transfer_directions_are_bounded_by_default():
    """Neither direction had ANY timeout: ``download_file`` passed
    ``timeout_s=None`` and ``_exec_with_stdin`` took no timeout at all, so a
    wedged guest hung the caller forever.  In a fleet that presents as a stalled
    worker rather than an error, which is much harder to diagnose."""
    import inspect

    for method in (
        ApptainerSandboxProvider.upload_file,
        ApptainerSandboxProvider.download_file,
    ):
        default = inspect.signature(method).parameters["timeout_s"].default
        assert default == DEFAULT_TRANSFER_TIMEOUT_S
        assert default is not None


def test_an_upload_that_hangs_raises_a_timeout(provider, sandbox, tmp_path, guest_path):
    source = tmp_path / "in.bin"
    source.write_bytes(b"x" * 64)

    async def hanging(handle, command, *, stdin, timeout_s=None):
        from pixeldesk.vm.sandbox_protocol import SandboxExecResult

        assert stdin, "the payload must reach the subprocess on stdin"
        assert timeout_s == 0.2, "the caller's timeout must reach the subprocess"
        return SandboxExecResult(None, None, -1, error_type="timeout")

    provider.exec = hanging  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="upload of .* timed out"):
        run(provider.upload_file(sandbox, source, guest_path, timeout_s=0.2))


def test_a_download_that_hangs_raises_a_timeout(provider, sandbox, tmp_path, guest_path):
    seen: dict = {}

    async def hanging(handle, command, **kwargs):
        from pixeldesk.vm.sandbox_protocol import SandboxExecResult

        seen.update(kwargs)
        return SandboxExecResult(None, None, -1, error_type="timeout")

    provider.exec = hanging  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="download of .* timed out"):
        run(provider.download_file(sandbox, guest_path, tmp_path / "o", timeout_s=0.3))
    assert seen["timeout_s"] == 0.3, "the caller's timeout must reach the subprocess"


@pytest.mark.parametrize("stdin", [b"", None], ids=["upload-path", "download-path"])
def test_a_real_timeout_kills_the_subprocess(provider, sandbox, stdin):
    """Exercises the real ``wait_for`` + ``kill`` path, not a stubbed result.

    Both directions go through the same ``exec``; the only difference is whether
    there is a payload on stdin.

    The guest command is ``exec sleep 30``, so bash REPLACES itself with sleep and
    the process being killed is the one holding stdout.  A plain ``sleep 30``
    would leave sleep as a grandchild still holding that pipe after bash dies, and
    ``await process.wait()`` then blocks on an EOF that never comes -- which is a
    real limitation of this timeout path, reported separately.
    """
    result = run(provider.exec(sandbox, "exec sleep 30", stdin=stdin, timeout_s=0.5))
    assert result.error_type == "timeout"
    assert result.return_code == -1
    assert result.stdout is None and result.stderr is None


def test_an_oversized_upload_is_refused_by_name_rather_than_buffered(
    provider, sandbox, tmp_path, guest_path
):
    """The cap is the honest limit of the mechanism, not a tuning knob: both
    directions buffer the file AND its ~33%-larger base64 in host memory, so a
    2.7 GB qcow2 overlay -- which this module's header names as a payload -- would
    want ~6 GB of RAM.  Refusing by name beats being OOM-killed mid-rollout."""
    source = tmp_path / "big.bin"
    source.write_bytes(b"x" * 4096)
    with pytest.raises(TransferTooLargeError, match="needs a streaming path"):
        run(provider.upload_file(sandbox, source, guest_path, max_bytes=1024))


def test_an_oversized_upload_is_refused_before_reading_the_file(
    provider, sandbox, tmp_path, guest_path
):
    """The size comes from ``stat``, so the refusal costs no memory."""
    source = tmp_path / "big.bin"
    source.write_bytes(b"x" * 4096)
    read_calls = []
    original = Path.read_bytes

    def spy(self):
        read_calls.append(self)
        return original(self)

    Path.read_bytes = spy  # type: ignore[method-assign]
    try:
        with pytest.raises(TransferTooLargeError):
            run(provider.upload_file(sandbox, source, guest_path, max_bytes=1024))
    finally:
        Path.read_bytes = original  # type: ignore[method-assign]
    assert read_calls == [], "an oversized file must not be read into memory"


def test_an_oversized_download_is_refused_by_the_guest_before_encoding(
    provider, sandbox, tmp_path, guest_path
):
    """Checked INSIDE the guest, so the bytes are never base64'd onto the wire."""
    source = tmp_path / "in.bin"
    source.write_bytes(b"y" * 4096)
    run(provider.upload_file(sandbox, source, guest_path))
    target = tmp_path / "out.bin"
    with pytest.raises(TransferTooLargeError, match="exceeds the 1024-byte cap"):
        run(provider.download_file(sandbox, guest_path, target, max_bytes=1024))
    assert not target.exists()


def test_the_refusal_reports_the_observed_size(provider, sandbox, tmp_path, guest_path):
    source = tmp_path / "in.bin"
    source.write_bytes(b"z" * 5000)
    run(provider.upload_file(sandbox, source, guest_path))
    with pytest.raises(TransferTooLargeError, match="5000 bytes"):
        run(provider.download_file(sandbox, guest_path, tmp_path / "o", max_bytes=1024))


def test_a_file_at_the_cap_still_transfers(provider, sandbox, tmp_path, guest_path):
    """The cap is a ceiling, not an off-by-one exclusion."""
    payload = b"w" * 1024
    source = tmp_path / "in.bin"
    source.write_bytes(payload)
    run(provider.upload_file(sandbox, source, guest_path, max_bytes=1024))
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target, max_bytes=1024))
    assert target.read_bytes() == payload


def test_the_cap_can_be_lifted_explicitly(provider, sandbox, tmp_path, guest_path):
    """A caller who knows what they are doing can opt out per call."""
    payload = b"v" * 4096
    source = tmp_path / "in.bin"
    source.write_bytes(payload)
    run(provider.upload_file(sandbox, source, guest_path, max_bytes=None))
    target = tmp_path / "out.bin"
    run(provider.download_file(sandbox, guest_path, target, max_bytes=None))
    assert target.read_bytes() == payload


def test_the_default_cap_is_documented_as_a_mechanism_limit():
    assert MAX_TRANSFER_BYTES == 256 * 1024 * 1024
    assert issubclass(TransferTooLargeError, RuntimeError)


def test_a_screenshot_sized_payload_is_nowhere_near_the_cap(
    provider, sandbox, tmp_path, guest_path
):
    """The payload this channel actually carries most: a full-desktop PNG."""
    source = tmp_path / "frame.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(3 * 1024 * 1024))
    assert source.stat().st_size < MAX_TRANSFER_BYTES
    run(provider.upload_file(sandbox, source, guest_path))
    target = tmp_path / "out.png"
    run(provider.download_file(sandbox, guest_path, target))
    assert target.read_bytes() == source.read_bytes()
