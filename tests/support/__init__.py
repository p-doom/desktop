"""Test doubles that let the guest program be EXECUTED without a VM.

The point of this package: ``guest_program.compile_atomic_guest_program`` emits
~200 lines of Python that only ever ran inside the pinned Ubuntu guest.  Reading
that string proves nothing about whether it runs.  ``fake_guest_backend`` and
``fake_gi`` are thin stand-ins for ``pyautogui`` / ``python-xlib`` / PyGObject --
enough for the generated program to execute, verify its own pointer mask, take
the clipboard round trip, and print its result marker -- so the whole compiled
program is exercised on a CPU node.

These modules are imported by a SUBPROCESS (see ``guest_runner``), never by the
host test process, so nothing here can shadow a real dependency.
"""
