"""In-process Xlib/XTEST server used by compiled guest-program tests."""

from __future__ import annotations

import json
import sys
import types

from desktop.execute.keymap import KEYSYMS, guest_key


class X11Consts(types.ModuleType):
    KeyPress = 2
    KeyRelease = 3
    ButtonPress = 4
    ButtonRelease = 5
    MotionNotify = 6


class _Geometry:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


class _Pointer:
    def __init__(self, x: int, y: int, mask: int) -> None:
        self.root_x = x
        self.root_y = y
        self.mask = mask


class _Root:
    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def query_pointer(self) -> _Pointer:
        return _Pointer(self._backend.x, self._backend.y, self._backend.mask)

    def get_geometry(self) -> _Geometry:
        return _Geometry(*self._backend.size)


class _Screen:
    def __init__(self, backend: Backend) -> None:
        self.root = _Root(backend)


class _Info:
    min_keycode = 8
    max_keycode = 255


class _WireDisplay:
    info = _Info()


class _Extension:
    present = True


def _initial_keyboard_mapping() -> dict[int, tuple[int, ...]]:
    mapping = {code: (0, 0, 0, 0) for code in range(8, 256)}
    for offset, keysym in enumerate(sorted(set(KEYSYMS.values()))):
        mapping[8 + offset] = (keysym, 0, 0, 0)
    return mapping


class Backend:
    """The subset of python-xlib used by the generated program."""

    MASKS = {1: 1 << 8, 2: 1 << 9, 3: 1 << 10}

    def __init__(
        self,
        *,
        size: tuple[int, int],
        cursor: tuple[int, int],
        initial_mask: int,
        initial_keys: tuple[str, ...],
        fail_xtest_at: int | None,
    ) -> None:
        self.size = size
        self.x, self.y = cursor
        self.mask = initial_mask
        self.events: list[tuple] = []
        self.mapping = _initial_keyboard_mapping()
        self.initial_mapping = dict(self.mapping)
        self.display = _WireDisplay()
        self.keys: set[int] = set()
        self.fail_xtest_at = fail_xtest_at
        self.xtest_calls = 0
        for key in initial_keys:
            keysym = KEYSYMS[guest_key(key)]
            self.keys.add(self.keysym_to_keycodes(keysym)[0][0])

    def screen(self) -> _Screen:
        return _Screen(self)

    def query_extension(self, name: str) -> _Extension | None:
        return _Extension() if name == "XTEST" else None

    def sync(self) -> None:
        self.events.append(("sync",))

    def close(self) -> None:
        self.events.append(("close",))
        events = json.dumps([list(event) for event in self.events])
        restored = json.dumps(self.mapping == self.initial_mapping)
        sys.stderr.write("X_EVENTS=" + events + "\n")
        sys.stderr.write("KEYMAP_RESTORED=" + restored + "\n")
        sys.stderr.write("HELD_KEYCODES=" + json.dumps(sorted(self.keys)) + "\n")

    def query_keymap(self) -> list[int]:
        bitmap = [0] * 32
        for code in self.keys:
            bitmap[code >> 3] |= 1 << (code & 7)
        return bitmap

    def keysym_to_keycodes(self, keysym: int) -> list[tuple[int, int]]:
        return [
            (code, index)
            for code, symbols in sorted(self.mapping.items())
            for index, symbol in enumerate(symbols)
            if symbol == keysym
        ]

    def get_keyboard_mapping(self, first_keycode: int, count: int) -> list[tuple[int, ...]]:
        return [self.mapping[code] for code in range(first_keycode, first_keycode + count)]

    def change_keyboard_mapping(
        self, first_keycode: int, keysyms: list[tuple[int, ...]]
    ) -> None:
        for offset, symbols in enumerate(keysyms):
            self.mapping[first_keycode + offset] = tuple(int(value) for value in symbols)
        self.events.append(
            ("change_keyboard_mapping", first_keycode, tuple(tuple(row) for row in keysyms))
        )

    def get_modifier_mapping(self) -> list[list[int]]:
        def code(name: str) -> int:
            return self.keysym_to_keycodes(KEYSYMS[name])[0][0]

        return [
            [code("shiftleft"), code("shiftright")],
            [code("capslock")],
            [code("ctrlleft"), code("ctrlright")],
            [code("altleft"), code("altright")],
            [code("numlock")],
            [],
            [code("winleft"), code("winright")],
            [],
        ]

    def xtest_fake_input(
        self, event_type: int, detail: int = 0, *, x: int = 0, y: int = 0
    ) -> None:
        self.xtest_calls += 1
        if self.xtest_calls == self.fail_xtest_at:
            raise RuntimeError(f"injected XTEST failure at event {self.xtest_calls}")
        event_type = int(event_type)
        detail = int(detail)
        self.events.append(("xtest", event_type, detail, int(x), int(y)))
        if event_type == X11Consts.MotionNotify:
            self.x, self.y = int(x), int(y)
        elif event_type == X11Consts.ButtonPress and detail in self.MASKS:
            self.mask |= self.MASKS[detail]
        elif event_type == X11Consts.ButtonRelease and detail in self.MASKS:
            self.mask &= ~self.MASKS[detail]
        elif event_type == X11Consts.KeyPress:
            self.keys.add(detail)
        elif event_type == X11Consts.KeyRelease:
            self.keys.discard(detail)


def install(
    *,
    size: tuple[int, int] = (1920, 1080),
    cursor: tuple[int, int] = (50, 50),
    initial_mask: int = 0,
    initial_keys: tuple[str, ...] = (),
    fail_xtest_at: int | None = None,
) -> Backend:
    backend = Backend(
        size=size,
        cursor=cursor,
        initial_mask=initial_mask,
        initial_keys=initial_keys,
        fail_xtest_at=fail_xtest_at,
    )
    x_module = X11Consts("Xlib.X")
    xlib_module = types.ModuleType("Xlib")
    xlib_module.__path__ = []
    xlib_module.X = x_module
    display_module = types.ModuleType("Xlib.display")
    display_module.Display = lambda: backend
    ext_module = types.ModuleType("Xlib.ext")
    ext_module.__path__ = []
    xtest_module = types.ModuleType("Xlib.ext.xtest")

    def fake_input(display, event_type, detail=0, time=0, root=0, x=0, y=0):
        assert display is backend
        backend.xtest_fake_input(event_type, detail, x=x, y=y)

    xtest_module.fake_input = fake_input
    ext_module.xtest = xtest_module
    sys.modules["Xlib"] = xlib_module
    sys.modules["Xlib.X"] = x_module
    sys.modules["Xlib.display"] = display_module
    sys.modules["Xlib.ext"] = ext_module
    sys.modules["Xlib.ext.xtest"] = xtest_module
    return backend
