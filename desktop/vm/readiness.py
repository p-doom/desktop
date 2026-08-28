"""Is the desktop actually up, or is the framebuffer still black?

A guest whose agent answers ``/screenshot`` with a 200 is not necessarily a guest
with a *desktop*.  During boot the framebuffer is uniformly black or uniformly
grey for tens of seconds, and a harness that starts driving then produces a
rollout whose first N frames carry no information -- the "black-frame carry"
failure, which silently invalidates a training corpus.  Two statistics catch it:
the fraction of non-dark pixels, and the luma standard deviation.  Dark-and-flat
means not ready; either one alone is not enough, because a solid grey screen is
bright but flat and a mostly-black desktop with a bright taskbar is dark but
structured.

Readiness uses Pillow; screenshots cross every other boundary as raw ``bytes``.

The sampler must AVERAGE, not point-sample: ``thumbnail`` averages over a
resampling filter, and the two ratio/stddev thresholds below are calibrated
against that form.

The ``luma_sampler`` seam is what makes the readiness rule testable against
synthetic frames with no VM and no image file in the loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from enum import Enum
from typing import Any, Callable, Protocol

from PIL import Image

logger = logging.getLogger(__name__)

DESKTOP_READY_INITIAL_DELAY_S = 130.0
DESKTOP_READY_TIMEOUT_S = 900.0
DESKTOP_READY_POLL_S = 5.0
DESKTOP_READY_GET_OBS_TIMEOUT_S = 60.0
DESKTOP_READY_MIN_NON_DARK_RATIO = 0.05
DESKTOP_READY_MIN_LUMA_STDDEV = 5.0
DESKTOP_READY_DARK_THRESHOLD = 12

#: How many CONSECUTIVE structurally-broken observations end the wait early.
#:
#: ``INVALID`` and ``EMPTY`` mean the guest agent answered with something that is
#: not a screenshot, and no amount of further waiting turns that into one -- so
#: polling on for the full 900 s timeout costs a quarter of an hour per broken
#: guest and reports the same failure at the end.  This is not 1, because a single
#: malformed answer during agent startup is plausible and cheap to ride out;
#: beyond a couple in a row it is the agent, not the timing.  ``MISSING`` is
#: deliberately NOT counted: "no bytes yet" is the normal state of a booting VM.
DESKTOP_READY_STRUCTURAL_TOLERANCE = 3


class DesktopObserver(Protocol):
    def observe(self, *, request_timeout: float | None = ...) -> dict[str, Any]: ...


class ScreenshotStatus(Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    MISSING = "missing"
    INVALID = "invalid"
    EMPTY = "empty"


#: Downsample target.  The ratio and stddev thresholds above are calibrated
#: against a 160x90 average-resampled thumbnail; changing this changes what
#: "ready" means.
DEFAULT_THUMBNAIL_SIZE = (160, 90)


def screenshot_luma_samples(
    data: bytes, *, thumbnail_size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE
) -> list[int]:
    """Decode a screenshot and return downsampled luma values in 0..255.

    ``convert("L")`` then ``thumbnail`` averages over a resampling filter, which
    is what the calibrated thresholds expect.  Raises on anything undecodable --
    ``desktop_screenshot_ready`` turns that into ``INVALID``, which is a
    *structural* status that stops the wait rather than retrying forever.

    Whatever Pillow opens, this measures.
    """
    image = Image.open(io.BytesIO(data)).convert("L")
    image.thumbnail(thumbnail_size)
    return list(image.getdata())


LumaSampler = Callable[[bytes], list[int]]


def desktop_screenshot_ready(
    screenshot: bytes | None,
    *,
    luma_sampler: LumaSampler = screenshot_luma_samples,
) -> tuple[ScreenshotStatus, str]:
    """Classify one screenshot as ready / not-ready / structurally broken.

    The four non-``READY`` statuses are kept distinct because they mean different
    things on a timeout: ``NOT_READY`` means the VM was slow, while ``MISSING`` /
    ``INVALID`` / ``EMPTY`` mean the guest agent is broken and waiting longer
    would never have helped.
    """
    if not isinstance(screenshot, bytes):
        return ScreenshotStatus.MISSING, "missing screenshot bytes"
    try:
        pixels = luma_sampler(screenshot)
    except Exception as exc:
        return ScreenshotStatus.INVALID, f"invalid screenshot: {exc!r}"
    if not pixels:
        return ScreenshotStatus.EMPTY, "empty screenshot"
    mean = sum(pixels) / len(pixels)
    non_dark_ratio = sum(
        value > DESKTOP_READY_DARK_THRESHOLD for value in pixels
    ) / len(pixels)
    variance = sum((value - mean) ** 2 for value in pixels) / len(pixels)
    stddev = variance**0.5
    status = (
        ScreenshotStatus.READY
        if (
            non_dark_ratio >= DESKTOP_READY_MIN_NON_DARK_RATIO
            and stddev >= DESKTOP_READY_MIN_LUMA_STDDEV
        )
        else ScreenshotStatus.NOT_READY
    )
    detail = (
        f"mean_luma={mean:.2f}, "
        f"non_dark_ratio={non_dark_ratio:.3f}, "
        f"luma_stddev={stddev:.2f}"
    )
    return status, detail


def wait_for_screenshot_ready(
    fetch: Callable[[], bytes],
    *,
    initial_delay_s: float = DESKTOP_READY_INITIAL_DELAY_S,
    timeout_s: float = DESKTOP_READY_TIMEOUT_S,
    poll_s: float = DESKTOP_READY_POLL_S,
    luma_sampler: LumaSampler = screenshot_luma_samples,
) -> bytes:
    """Synchronous readiness wait over any screenshot-returning callable.

    Most of this package is synchronous, and importing ``asyncio`` to poll a VM is
    not a trade worth making for a caller with no event loop; the async form below
    is for callers that already have one.
    """
    if poll_s <= 0:
        raise ValueError(f"poll_s must be positive, got {poll_s}")
    started = time.monotonic()
    if initial_delay_s > 0:
        _log_wait(f"waiting {initial_delay_s:.1f}s before polling")
        time.sleep(initial_delay_s)
    screenshot = b""
    status = ScreenshotStatus.MISSING
    detail = "missing screenshot bytes"
    structural_streak = 0
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            _log_wait(f"timed out after {elapsed:.1f}s ({detail})")
            _raise_for_timeout(status, detail, elapsed)
        try:
            screenshot = fetch()
        except Exception as exc:
            status, detail = ScreenshotStatus.MISSING, f"fetch failed: {exc!r}"
        else:
            status, detail = desktop_screenshot_ready(
                screenshot, luma_sampler=luma_sampler
            )
        elapsed = time.monotonic() - started
        if status is ScreenshotStatus.READY:
            _log_wait(f"ready after {elapsed:.1f}s ({detail})")
            return screenshot
        structural_streak = _structural_streak(status, structural_streak)
        _raise_if_structurally_broken(status, detail, elapsed, structural_streak)
        # Clamped so the last sleep lands ON the deadline; the loop head is the
        # one place that ends the wait.
        sleep_s = min(poll_s, max(0.0, timeout_s - elapsed))
        _log_wait(
            f"not ready after {elapsed:.1f}s; status={status.value} ({detail}); "
            f"retrying in {sleep_s:.1f}s"
        )
        time.sleep(sleep_s)


async def wait_for_desktop_ready(
    env: DesktopObserver,
    *,
    initial_obs: dict[str, Any] | None = None,
    luma_sampler: LumaSampler = screenshot_luma_samples,
) -> dict[str, Any]:
    """Async readiness wait over an observation-returning environment."""
    started_at = time.monotonic()
    if initial_obs is not None:
        status, detail = desktop_screenshot_ready(
            initial_obs.get("screenshot"), luma_sampler=luma_sampler
        )
        if status is ScreenshotStatus.READY:
            _log_wait(f"ready immediately ({detail})")
            return initial_obs

    _log_wait(f"waiting {DESKTOP_READY_INITIAL_DELAY_S:.1f}s before polling")
    await asyncio.sleep(DESKTOP_READY_INITIAL_DELAY_S)

    obs: dict[str, Any] = {}
    status = ScreenshotStatus.MISSING
    detail = "missing screenshot bytes"
    structural_streak = 0
    while True:
        elapsed_s = time.monotonic() - started_at
        remaining_s = DESKTOP_READY_TIMEOUT_S - elapsed_s
        if remaining_s <= 0:
            _log_wait(f"timed out after {elapsed_s:.1f}s ({detail})")
            _raise_for_timeout(status, detail, elapsed_s)

        obs = await asyncio.to_thread(
            env.observe,
            request_timeout=min(DESKTOP_READY_GET_OBS_TIMEOUT_S, remaining_s),
        )
        status, detail = desktop_screenshot_ready(
            obs.get("screenshot"), luma_sampler=luma_sampler
        )
        elapsed_s = time.monotonic() - started_at

        if status is ScreenshotStatus.READY:
            _log_wait(f"ready after {elapsed_s:.1f}s ({detail})")
            return obs

        structural_streak = _structural_streak(status, structural_streak)
        _raise_if_structurally_broken(status, detail, elapsed_s, structural_streak)

        sleep_s = min(DESKTOP_READY_POLL_S, DESKTOP_READY_TIMEOUT_S - elapsed_s)
        _log_wait(
            f"not ready after {elapsed_s:.1f}s; status={status.value} ({detail}); "
            f"retrying in {sleep_s:.1f}s"
        )
        await asyncio.sleep(sleep_s)


#: The statuses that mean the guest agent is broken rather than slow.
_STRUCTURAL_SUFFIXES = {
    ScreenshotStatus.MISSING: "without screenshot bytes",
    ScreenshotStatus.INVALID: "with invalid screenshot bytes",
    ScreenshotStatus.EMPTY: "with an empty screenshot",
}

#: ...of those, the ones a longer wait cannot fix.  ``MISSING`` is excluded: a
#: booting VM has no screenshot yet, which is why the wait exists at all.
_UNRECOVERABLE = (ScreenshotStatus.INVALID, ScreenshotStatus.EMPTY)


def _structural_streak(status: ScreenshotStatus, streak: int) -> int:
    """Count CONSECUTIVE unrecoverable observations; anything else resets it."""
    return streak + 1 if status in _UNRECOVERABLE else 0


def _raise_if_structurally_broken(
    status: ScreenshotStatus, detail: str, elapsed_s: float, streak: int
) -> None:
    """End the wait early when the guest agent is broken rather than slow.

    Otherwise a garbage-answering agent is polled for the whole 900 s timeout and
    then raises the same error it could have raised in seconds.
    """
    if status not in _UNRECOVERABLE or streak < DESKTOP_READY_STRUCTURAL_TOLERANCE:
        return
    _log_wait(
        f"giving up after {elapsed_s:.1f}s: {streak} consecutive "
        f"{status.value} observations ({detail})"
    )
    raise TimeoutError(
        f"desktop readiness gave up {_STRUCTURAL_SUFFIXES[status]} after "
        f"{elapsed_s:.1f}s and {streak} consecutive observations; waiting longer "
        f"cannot help ({detail})"
    )


def _raise_for_timeout(
    status: ScreenshotStatus, detail: str, elapsed_s: float
) -> None:
    """End a timed-out wait as an error, whatever the last status was.

    ``NOT_READY`` used to return the last frame here, on the reading that a slow VM
    is not an error.  But the frame it returned was BY CONSTRUCTION dark and flat --
    the exact black framebuffer this module exists to keep out of a rollout -- and it
    came back with the same type and no flag, so a caller could not tell it from a
    ready one.  That is the black-frame carry, reintroduced by the guard against it.
    """
    suffix = _STRUCTURAL_SUFFIXES.get(status, "on a desktop that never became ready")
    raise TimeoutError(
        f"desktop readiness timed out {suffix} after {elapsed_s:.1f}s ({detail})"
    )


def _log_wait(message: str) -> None:
    full_message = f"desktop readiness: {message}"
    logger.info(full_message)
    print(full_message, flush=True)
