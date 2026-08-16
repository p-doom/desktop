"""The readiness heuristic, its Pillow sampler, and the ``luma_sampler`` seam.

The load-bearing claim is that ``thumbnail`` **averages** over a resampling filter
and that the two thresholds (0.05 non-dark, 5.0 stddev) were calibrated against
that averaged form.  A point-sampling variant would keep decoding fine and keep
reporting plausible-looking numbers while meaning something different -- so the
calibration guard below is the most important test in this file: the SAME frame
must come out ``not_ready`` averaged and ``ready`` point-sampled.

The four non-READY statuses exist because they mean different things on a timeout,
so each is exercised separately through the seam.
"""

from __future__ import annotations

import asyncio
import functools
import io
import random

import pytest
from PIL import Image

from desktop.vm import readiness as R
from desktop.vm.readiness import (
    DEFAULT_THUMBNAIL_SIZE,
    ScreenshotStatus,
    desktop_screenshot_ready,
    png_luma_samples,
    wait_for_screenshot_ready,
)



def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@functools.lru_cache(maxsize=None)
def _flat(value: int, size=(1920, 1080)) -> bytes:
    return _png(Image.new("L", size, value))


@functools.lru_cache(maxsize=None)
def _per_pixel_noise(size=(1920, 1080), seed=0) -> bytes:
    random.seed(seed)
    image = Image.new("L", size)
    image.putdata([random.randrange(256) for _ in range(size[0] * size[1])])
    return _png(image)


@functools.lru_cache(maxsize=None)
def _desktop_like(size=(1920, 1080)) -> bytes:
    """A dark desktop with a bright panel and a bright window: real structure."""
    image = Image.new("L", size, 18)
    for y in range(0, 60):
        for x in range(size[0]):
            image.putpixel((x, y), 210)
    for y in range(200, 700):
        for x in range(300, 1300):
            image.putpixel((x, y), 240)
    return _png(image)


@functools.lru_cache(maxsize=None)
def _point_sampler(data: bytes, size=DEFAULT_THUMBNAIL_SIZE) -> list[int]:
    """The hand-rolled variant the module warns about: samples, never averages."""
    image = Image.open(io.BytesIO(data)).convert("L")
    width, height = image.size
    step_x, step_y = width // size[0], height // size[1]
    return [
        image.getpixel((x * step_x, y * step_y))
        for y in range(size[1])
        for x in range(size[0])
    ]


def test_the_calibrated_thumbnail_size_is_unchanged():
    assert DEFAULT_THUMBNAIL_SIZE == (160, 90)
    assert R.DESKTOP_READY_MIN_NON_DARK_RATIO == 0.05
    assert R.DESKTOP_READY_MIN_LUMA_STDDEV == 5.0
    assert R.DESKTOP_READY_DARK_THRESHOLD == 12


def test_the_sampler_downsamples_to_the_calibrated_pixel_count():
    assert len(png_luma_samples(_flat(0))) == 160 * 90


def test_the_sampler_averages_rather_than_point_sampling():
    """One-pixel-wide stripes average to mid-grey; point sampling would keep
    returning only 0 and 255.  This is the property the thresholds assume."""
    size = (1920, 1080)
    image = Image.new("L", size)
    image.putdata([255 if index % 2 else 0 for index in range(size[0] * size[1])])
    samples = png_luma_samples(_png(image))
    assert set(samples) <= {126, 127, 128, 129}, sorted(set(samples))[:8]


def test_the_sampler_is_not_restricted_to_png_despite_its_name():
    buffer = io.BytesIO()
    Image.open(io.BytesIO(_desktop_like())).convert("RGB").save(buffer, format="JPEG")
    status, _ = desktop_screenshot_ready(buffer.getvalue())
    assert status is ScreenshotStatus.READY


def test_the_sampler_raises_on_undecodable_bytes():
    with pytest.raises(Exception):
        png_luma_samples(b"not-an-image")


def test_a_point_sampling_variant_would_silently_drift_off_calibration():
    """Same bytes, two samplers, two different verdicts.

    Per-pixel noise is exactly what averaging is supposed to destroy: 12x12
    input pixels collapse into one output pixel, so the stddev falls below the
    5.0 threshold.  A point sampler keeps the full per-pixel variance and calls
    the identical frame READY.  If this test ever reports the same status for
    both samplers, the averaging has been "tidied" away.
    """
    frame = _per_pixel_noise()
    averaged, averaged_detail = desktop_screenshot_ready(frame)
    sampled, sampled_detail = desktop_screenshot_ready(frame, luma_sampler=_point_sampler)
    assert averaged is ScreenshotStatus.NOT_READY, averaged_detail
    assert sampled is ScreenshotStatus.READY, sampled_detail


def test_the_averaged_stddev_of_per_pixel_noise_sits_below_the_threshold():
    samples = png_luma_samples(_per_pixel_noise())
    mean = sum(samples) / len(samples)
    stddev = (sum((value - mean) ** 2 for value in samples) / len(samples)) ** 0.5
    assert stddev < R.DESKTOP_READY_MIN_LUMA_STDDEV
    point = _point_sampler(_per_pixel_noise())
    point_mean = sum(point) / len(point)
    point_stddev = (sum((v - point_mean) ** 2 for v in point) / len(point)) ** 0.5
    assert point_stddev > 10 * R.DESKTOP_READY_MIN_LUMA_STDDEV


def test_a_black_framebuffer_is_not_ready():
    status, detail = desktop_screenshot_ready(_flat(0))
    assert status is ScreenshotStatus.NOT_READY
    assert "non_dark_ratio=0.000" in detail


@pytest.mark.parametrize("value", [0, 5, 11])
def test_a_uniformly_dark_framebuffer_is_not_ready(value):
    assert desktop_screenshot_ready(_flat(value))[0] is ScreenshotStatus.NOT_READY


@pytest.mark.parametrize("value", [128, 200, 255])
def test_a_flat_bright_framebuffer_is_not_ready(value):
    """Bright but flat -- the boot splash / solid grey case.  Brightness alone
    must not satisfy the rule."""
    status, detail = desktop_screenshot_ready(_flat(value))
    assert status is ScreenshotStatus.NOT_READY
    assert "luma_stddev=0.00" in detail


def test_a_structured_desktop_frame_is_ready():
    status, detail = desktop_screenshot_ready(_desktop_like())
    assert status is ScreenshotStatus.READY, detail


def test_both_statistics_are_required_not_either():
    """A frame that passes only the stddev test stays not-ready.

    A 40px bright taskbar on a black 1080p screen is 3.7% non-dark, under the
    0.05 floor, so structure alone does not make it ready.  Widening the bar to
    5.5% flips it.  This pins the AND, which is the rule as implemented.
    """
    dark_structured = Image.new("L", (1920, 1080), 0)
    for y in range(1040, 1080):
        for x in range(1920):
            dark_structured.putpixel((x, y), 220)
    status, detail = desktop_screenshot_ready(_png(dark_structured))
    assert status is ScreenshotStatus.NOT_READY, detail
    assert "luma_stddev" in detail
    wider = Image.new("L", (1920, 1080), 0)
    for y in range(1020, 1080):
        for x in range(1920):
            wider.putpixel((x, y), 220)
    assert desktop_screenshot_ready(_png(wider))[0] is ScreenshotStatus.READY


def test_missing_bytes_are_distinguished_from_a_dark_frame():
    for value in (None, "a string", 42, bytearray(b"x")):
        status, _ = desktop_screenshot_ready(value)
        assert status is ScreenshotStatus.MISSING, value


def test_structurally_invalid_bytes_are_invalid_not_not_ready():
    status, detail = desktop_screenshot_ready(b"not-an-image")
    assert status is ScreenshotStatus.INVALID
    assert "invalid screenshot" in detail
    assert desktop_screenshot_ready(b"")[0] is ScreenshotStatus.INVALID


def test_an_empty_sample_list_is_empty_not_a_zero_division():
    status, detail = desktop_screenshot_ready(b"anything", luma_sampler=lambda data: [])
    assert status is ScreenshotStatus.EMPTY
    assert detail == "empty screenshot"


def test_the_seam_needs_no_image_file_at_all():
    """The point of ``luma_sampler``: a readiness rule testable on synthetic data."""
    ready = desktop_screenshot_ready(b"x", luma_sampler=lambda data: [0, 255] * 50)
    not_ready = desktop_screenshot_ready(b"x", luma_sampler=lambda data: [200] * 100)
    assert ready[0] is ScreenshotStatus.READY
    assert not_ready[0] is ScreenshotStatus.NOT_READY


@pytest.fixture
def fast_clock(monkeypatch):
    """A fake clock, so a 900 s wait is testable in microseconds."""
    state = {"now": 0.0, "sleeps": []}
    monkeypatch.setattr(R.time, "monotonic", lambda: state["now"])

    def sleep(seconds: float) -> None:
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(R.time, "sleep", sleep)
    return state


def test_the_wait_returns_the_first_ready_frame(fast_clock):
    frames = [_flat(0), _flat(0), _desktop_like()]
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        return frames[min(len(calls) - 1, len(frames) - 1)]

    result = wait_for_screenshot_ready(fetch, initial_delay_s=0.0, timeout_s=100.0, poll_s=5.0)
    assert result == frames[-1]
    assert len(calls) == 3


def test_the_wait_honours_the_initial_delay_before_polling(fast_clock):
    calls = []

    def fetch() -> bytes:
        calls.append(fast_clock["now"])
        return _desktop_like()

    wait_for_screenshot_ready(fetch, initial_delay_s=130.0, timeout_s=900.0)
    assert calls == [130.0]


def test_a_slow_vm_times_out_by_returning_the_last_frame(fast_clock):
    """``NOT_READY`` at timeout means "the VM was slow", which is not an error."""
    result = wait_for_screenshot_ready(
        lambda: _flat(0), initial_delay_s=0.0, timeout_s=20.0, poll_s=5.0
    )
    assert result == _flat(0)


def test_a_broken_guest_agent_raises_at_timeout_rather_than_returning_junk(fast_clock):
    for payload, message in (
        (b"not-an-image", "invalid screenshot bytes"),
        (None, "without screenshot bytes"),
    ):
        with pytest.raises(TimeoutError, match=message):
            wait_for_screenshot_ready(
                lambda payload=payload: payload,
                initial_delay_s=0.0,
                timeout_s=20.0,
                poll_s=5.0,
            )


def test_an_empty_screenshot_raises_at_timeout(fast_clock):
    with pytest.raises(TimeoutError, match="empty screenshot"):
        wait_for_screenshot_ready(
            lambda: b"x",
            initial_delay_s=0.0,
            timeout_s=20.0,
            poll_s=5.0,
            luma_sampler=lambda data: [],
        )


def test_a_raising_fetch_is_reported_as_missing_and_keeps_polling(fast_clock):
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionRefusedError("agent not up")
        return _desktop_like()

    assert wait_for_screenshot_ready(
        fetch, initial_delay_s=0.0, timeout_s=100.0, poll_s=5.0
    ) == _desktop_like()
    assert len(calls) == 3


def test_a_structurally_broken_agent_ends_the_wait_early(fast_clock):
    """A garbage-answering agent must not cost the full timeout.

    ``INVALID`` means the agent returned something that is not a screenshot, and
    no amount of waiting turns that into one.  This used to poll for the whole
    900 s and then raise the very error it could have raised in seconds.
    """
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        return b"not-an-image"

    with pytest.raises(TimeoutError, match="waiting longer cannot help"):
        wait_for_screenshot_ready(fetch, initial_delay_s=0.0, timeout_s=900.0, poll_s=5.0)
    assert len(calls) == R.DESKTOP_READY_STRUCTURAL_TOLERANCE
    assert fast_clock["now"] < 60.0, "it should give up in seconds, not minutes"


def test_an_empty_screenshot_also_ends_the_wait_early(fast_clock):
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        return b"x"

    with pytest.raises(TimeoutError, match="with an empty screenshot"):
        wait_for_screenshot_ready(
            fetch,
            initial_delay_s=0.0,
            timeout_s=900.0,
            poll_s=5.0,
            luma_sampler=lambda data: [],
        )
    assert len(calls) == R.DESKTOP_READY_STRUCTURAL_TOLERANCE


def test_one_malformed_answer_is_ridden_out(fast_clock):
    """The tolerance is not 1: a single bad answer during agent startup is
    plausible, and must not fail an otherwise healthy boot."""
    frames = [b"not-an-image", _flat(0), b"not-an-image", _desktop_like()]
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        return frames[min(len(calls) - 1, len(frames) - 1)]

    assert wait_for_screenshot_ready(
        fetch, initial_delay_s=0.0, timeout_s=900.0, poll_s=5.0
    ) == _desktop_like()
    assert len(calls) == 4


def test_a_not_ready_frame_resets_the_structural_streak(fast_clock):
    """Only CONSECUTIVE structural answers count.

    Alternating invalid/dark frames must poll to the timeout -- and end as a plain
    slow-VM timeout, which returns the last frame rather than raising, because the
    final observation was ``NOT_READY``.
    """
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        return b"not-an-image" if len(calls) % 2 else _flat(0)

    result = wait_for_screenshot_ready(
        fetch, initial_delay_s=0.0, timeout_s=60.0, poll_s=5.0
    )
    assert result == _flat(0)
    assert len(calls) > R.DESKTOP_READY_STRUCTURAL_TOLERANCE
    assert fast_clock["now"] >= 60.0


def test_a_missing_screenshot_never_short_circuits(fast_clock):
    """"No bytes yet" is the NORMAL state of a booting VM, so it must keep
    polling to the timeout rather than giving up in 15 seconds."""
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        return None  # type: ignore[return-value]

    with pytest.raises(TimeoutError, match="timed out without screenshot bytes"):
        wait_for_screenshot_ready(fetch, initial_delay_s=0.0, timeout_s=50.0, poll_s=5.0)
    assert len(calls) == 10


def test_a_failing_fetch_never_short_circuits(fast_clock):
    """An agent that is not up yet raises; that is not a structural failure."""
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        raise ConnectionRefusedError("not up yet")

    with pytest.raises(TimeoutError, match="timed out without screenshot bytes"):
        wait_for_screenshot_ready(fetch, initial_delay_s=0.0, timeout_s=50.0, poll_s=5.0)
    assert len(calls) == 10


def test_the_poll_interval_never_overshoots_the_deadline(fast_clock):
    with pytest.raises(TimeoutError):
        wait_for_screenshot_ready(
            lambda: None, initial_delay_s=0.0, timeout_s=12.0, poll_s=5.0
        )
    assert sum(fast_clock["sleeps"]) <= 12.0


def test_a_zero_timeout_raises_before_fetching_anything(fast_clock):
    """The initial status is ``MISSING``, which is structural, so a zero timeout
    raises rather than returning empty bytes that look like a screenshot."""
    calls = []
    with pytest.raises(TimeoutError, match="without screenshot bytes"):
        wait_for_screenshot_ready(
            lambda: calls.append(1) or b"", initial_delay_s=0.0, timeout_s=0.0
        )
    assert calls == []


@pytest.mark.parametrize("poll_s", [0.0, -1.0])
def test_a_non_positive_poll_interval_is_refused_before_the_first_poll(poll_s):
    """Rejected at the boundary rather than reinterpreted. The loop used to have
    a second exit for ``sleep_s <= 0``, which silently turned ``poll_s=0`` into
    "poll once, then give up" -- a different function from the one that was
    asked for."""
    with pytest.raises(ValueError, match="poll_s must be positive"):
        wait_for_screenshot_ready(lambda: b"", poll_s=poll_s)


def test_the_deadline_is_enforced_in_exactly_one_place(fast_clock):
    """A fetch that overruns the deadline must be ended by the loop head, not by
    a duplicate exit further down."""
    def fetch() -> bytes:
        fast_clock["now"] += 60.0
        return _flat(0)

    result = wait_for_screenshot_ready(
        fetch, initial_delay_s=0.0, timeout_s=20.0, poll_s=5.0
    )
    assert result == _flat(0)
    assert fast_clock["sleeps"] == [0.0]


@pytest.fixture
def fast_async_clock(monkeypatch):
    """Fake ``time.monotonic`` and ``asyncio.sleep``, so a 900s wait is instant."""
    state = {"now": 0.0, "sleeps": []}
    monkeypatch.setattr(R.time, "monotonic", lambda: state["now"])

    async def sleep(seconds: float) -> None:
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(R.asyncio, "sleep", sleep)
    return state


class _Observer:
    """A ``DesktopObserver`` whose frames are scripted."""

    def __init__(self, frames: list[object]) -> None:
        self.frames = frames
        self.calls: list[float | None] = []

    def observe(self, *, request_timeout: float | None = None) -> dict:
        self.calls.append(request_timeout)
        index = min(len(self.calls) - 1, len(self.frames) - 1)
        return {"screenshot": self.frames[index]}


def test_the_async_wait_returns_an_already_ready_initial_observation(fast_async_clock):
    """The fast path: no initial delay is paid when the desktop is already up."""
    observer = _Observer([_flat(0)])
    obs = asyncio.run(
        R.wait_for_desktop_ready(observer, initial_obs={"screenshot": _desktop_like()})
    )
    assert obs["screenshot"] == _desktop_like()
    assert observer.calls == [], "a ready initial observation must not poll"
    assert fast_async_clock["sleeps"] == []


def test_a_not_ready_initial_observation_falls_through_to_polling(fast_async_clock):
    observer = _Observer([_desktop_like()])
    obs = asyncio.run(
        R.wait_for_desktop_ready(observer, initial_obs={"screenshot": _flat(0)})
    )
    assert obs["screenshot"] == _desktop_like()
    assert len(observer.calls) == 1
    assert fast_async_clock["sleeps"][0] == R.DESKTOP_READY_INITIAL_DELAY_S


def test_the_async_wait_polls_until_the_desktop_paints(fast_async_clock):
    observer = _Observer([_flat(0), _flat(0), _flat(128), _desktop_like()])
    obs = asyncio.run(R.wait_for_desktop_ready(observer))
    assert obs["screenshot"] == _desktop_like()
    assert len(observer.calls) == 4


def test_the_async_wait_bounds_each_observation_request(fast_async_clock):
    observer = _Observer([_desktop_like()])
    asyncio.run(R.wait_for_desktop_ready(observer))
    assert observer.calls[0] == min(
        R.DESKTOP_READY_GET_OBS_TIMEOUT_S,
        R.DESKTOP_READY_TIMEOUT_S - R.DESKTOP_READY_INITIAL_DELAY_S,
    )


def test_a_slow_vm_times_out_by_returning_the_last_observation(fast_async_clock):
    observer = _Observer([_flat(0)])
    obs = asyncio.run(R.wait_for_desktop_ready(observer))
    assert obs["screenshot"] == _flat(0)
    assert fast_async_clock["now"] >= R.DESKTOP_READY_TIMEOUT_S


def test_a_broken_guest_agent_raises_at_the_async_timeout(fast_async_clock):
    for payload, message in (
        (b"not-an-image", "invalid screenshot bytes"),
        (None, "without screenshot bytes"),
    ):
        with pytest.raises(TimeoutError, match=message):
            asyncio.run(R.wait_for_desktop_ready(_Observer([payload])))


def test_the_async_wait_never_sleeps_past_its_deadline(fast_async_clock):
    with pytest.raises(TimeoutError):
        asyncio.run(R.wait_for_desktop_ready(_Observer([None])))
    assert sum(fast_async_clock["sleeps"]) <= R.DESKTOP_READY_TIMEOUT_S


def test_the_async_and_sync_waits_agree_on_the_same_frames():
    """Two entry points, one heuristic: they must classify identically."""
    for frame in (_flat(0), _flat(255), _desktop_like(), b"not-an-image", None):
        assert (
            desktop_screenshot_ready(frame)[0]
            is desktop_screenshot_ready({"screenshot": frame}.get("screenshot"))[0]
        )


def test_the_async_wait_also_ends_early_on_a_structurally_broken_agent(
    fast_async_clock,
):
    """Both entry points must give up on a broken agent, not just the sync one."""
    observer = _Observer([b"not-an-image"])
    with pytest.raises(TimeoutError, match="waiting longer cannot help"):
        asyncio.run(R.wait_for_desktop_ready(observer))
    assert len(observer.calls) == R.DESKTOP_READY_STRUCTURAL_TOLERANCE
    assert fast_async_clock["now"] < R.DESKTOP_READY_INITIAL_DELAY_S + 60.0


def test_the_async_wait_rides_out_one_malformed_observation(fast_async_clock):
    observer = _Observer([b"not-an-image", _flat(0), _desktop_like()])
    obs = asyncio.run(R.wait_for_desktop_ready(observer))
    assert obs["screenshot"] == _desktop_like()
    assert len(observer.calls) == 3


def test_the_async_wait_never_short_circuits_a_missing_screenshot(fast_async_clock):
    observer = _Observer([None])
    with pytest.raises(TimeoutError, match="timed out without screenshot bytes"):
        asyncio.run(R.wait_for_desktop_ready(observer))
    assert len(observer.calls) > R.DESKTOP_READY_STRUCTURAL_TOLERANCE


def test_both_waits_agree_on_when_to_give_up_early():
    """The sync and async loops must share one policy, not two."""
    import inspect

    for function in (R.wait_for_screenshot_ready, R.wait_for_desktop_ready):
        source = inspect.getsource(function)
        assert "_structural_streak(" in source
        assert "_raise_if_structurally_broken(" in source
