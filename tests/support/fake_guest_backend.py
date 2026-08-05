"""A fake ``pyautogui`` + python-xlib backend, faithful where the program looks.

Faithful in the places ``guest_program`` actually inspects or hooks:

* ``platformModule`` exposes ``_display`` (with ``sync``/``flush`` and a
  ``screen().root.query_pointer()``), ``fake_input``, ``X``, ``_mouseDown``,
  ``_mouseUp``, ``_moveTo`` and ``BUTTON_NAME_MAPPING`` -- the exact attribute set
  ``_de_install_attempt_hooks`` and ``_de_click`` probe for.
* the pointer button MASK is maintained from injected ButtonPress/ButtonRelease,
  so the program's own mask verification is a real check and not a tautology.
* ``_mouseUp`` emits the release-side ``MotionNotify`` that PyAutoGUI emits, which
  is the single event the ``direct_xtest_no_release_motion`` backend removes.

NOT faithful, and deliberately: no timing realism and no X server.
"""

from __future__ import annotations

import sys
import types


class X11Consts:
    """The subset of ``Xlib.X`` the generated program names."""

    MotionNotify = 6
    ButtonPress = 4
    ButtonRelease = 5


class _Pointer:
    def __init__(self, x: int, y: int, mask: int) -> None:
        self.root_x, self.root_y, self.mask = x, y, mask


class _Root:
    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    def query_pointer(self) -> _Pointer:
        return _Pointer(self._backend.x, self._backend.y, self._backend.mask)


class _Screen:
    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    @property
    def root(self) -> _Root:
        return _Root(self._backend)


class _Display:
    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    def screen(self) -> _Screen:
        return _Screen(self._backend)

    def sync(self) -> None:
        self._backend.events.append(("sync",))

    def flush(self) -> None:
        self._backend.events.append(("flush",))


class Backend:
    """Stands in for ``pyautogui.platformModule`` (the X11 backend module)."""

    __name__ = "fake_x11_backend"
    BUTTON_NAME_MAPPING = {"left": 1, "middle": 2, "right": 3}
    MASKS = {"left": 1 << 8, "middle": 1 << 9, "right": 1 << 10}

    def __init__(
        self,
        size: tuple[int, int] = (1920, 1080),
        cursor: tuple[int, int] = (50, 50),
        initial_mask: int = 0,
    ) -> None:
        self.size = size
        self.x, self.y = cursor
        self.mask = initial_mask
        self.events: list[tuple] = []
        self._display = _Display(self)
        self.X = X11Consts

    def _button_name(self, detail: int) -> str:
        for name, number in self.BUTTON_NAME_MAPPING.items():
            if number == detail:
                return name
        raise KeyError(detail)

    def fake_input(self, display, event_type, detail=0, **kwargs) -> None:
        self.events.append(("fake_input", int(event_type), int(detail)))
        if int(event_type) == X11Consts.ButtonPress:
            self.mask |= self.MASKS[self._button_name(int(detail))]
        elif int(event_type) == X11Consts.ButtonRelease:
            self.mask &= ~self.MASKS[self._button_name(int(detail))]

    def _moveTo(self, x, y) -> None:
        self.x, self.y = int(x), int(y)
        self.events.append(("_moveTo", self.x, self.y))
        self.fake_input(self._display, X11Consts.MotionNotify, 0, x=self.x, y=self.y)

    def _mouseDown(self, x, y, button) -> None:
        self._moveTo(x, y)
        self.fake_input(self._display, X11Consts.ButtonPress, self.BUTTON_NAME_MAPPING[button])
        self._display.sync()

    def _mouseUp(self, x, y, button) -> None:
        # PyAutoGUI's release side moves first; that MotionNotify is the whole
        # experimental delta between the two click backends.
        self._moveTo(x, y)
        self.fake_input(
            self._display, X11Consts.ButtonRelease, self.BUTTON_NAME_MAPPING[button]
        )
        self._display.sync()


class FakePyAutoGui(types.ModuleType):
    """Stands in for the ``pyautogui`` module itself."""

    FAILSAFE = True
    PAUSE = 0.1

    def __init__(self, backend: Backend) -> None:
        super().__init__("pyautogui")
        self.platformModule = backend
        self._backend = backend
        self.calls: list[list] = []

    def position(self) -> tuple[int, int]:
        return (self._backend.x, self._backend.y)

    def size(self) -> tuple[int, int]:
        return self._backend.size

    def moveTo(self, x, y, duration=0.0) -> None:
        self.calls.append(["moveTo", int(x), int(y), duration])
        self._backend._moveTo(x, y)

    def click(self, clicks=1, interval=0.0, button="left") -> None:
        self.calls.append(["click", button, clicks, interval])
        for _ in range(clicks):
            self._backend._moveTo(self._backend.x, self._backend.y)  # click premove
            self._backend._mouseDown(self._backend.x, self._backend.y, button)
            self._backend._mouseUp(self._backend.x, self._backend.y, button)

    def mouseDown(self, button="left") -> None:
        self.calls.append(["mouseDown", button])
        self._backend._mouseDown(self._backend.x, self._backend.y, button)

    def mouseUp(self, button="left") -> None:
        self.calls.append(["mouseUp", button])
        self._backend._mouseUp(self._backend.x, self._backend.y, button)

    def keyDown(self, key) -> None:
        self.calls.append(["keyDown", key])

    def keyUp(self, key) -> None:
        self.calls.append(["keyUp", key])

    def write(self, text, interval=0) -> None:
        self.calls.append(["write", text, interval])

    def hotkey(self, *keys) -> None:
        self.calls.append(["hotkey", list(keys)])

    def scroll(self, clicks) -> None:
        self.calls.append(["scroll", int(clicks)])

    def hscroll(self, dx) -> None:
        self.calls.append(["hscroll", int(dx)])


def install(
    size: tuple[int, int] = (1920, 1080),
    cursor: tuple[int, int] = (50, 50),
    initial_mask: int = 0,
) -> tuple[FakePyAutoGui, Backend]:
    backend = Backend(size=size, cursor=cursor, initial_mask=initial_mask)
    module = FakePyAutoGui(backend)
    sys.modules["pyautogui"] = module
    return module, backend
