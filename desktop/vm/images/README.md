# VM image tiers

| tier | definition | purpose |
| --- | --- | --- |
| KVM | `osworld-guest-kvm.def` | **full** parity from a pinned guest qcow2; required for benchmark and timing runs |
| non-KVM | `desktop-nonkvm.def` | **none, ever**; Xvfb-based executor and transport checks only |

## KVM

The container contains QEMU and its host-side tools, not a desktop. Bind the
guest qcow2 read-only and provide a writable `TMPDIR` for QEMU state. The
runtime requires readable and writable `/dev/kvm`; TCG is used only when the
caller explicitly requests it.

`desktop.vm.DesktopImageBuilder` provisions an upstream qcow2, verifies the
guest agent and grader dependencies, and publishes the image with a sibling
`.build.json` manifest. It refuses to overwrite either input or output.

## Non-KVM

The non-KVM image provides Xvfb, a window manager, and Python Xlib. Guest input
uses XTEST through `python-xlib`; PyAutoGUI is retained for OSWorld setup and
evaluator code, and `xdotool` is a differential test oracle. This image has no
OSWorld guest agent.

## Flagged, not faked

The non-KVM tier has **no browser**. Ubuntu reports "no installation candidate"
for Chromium, and the available Chromium and Firefox packages are snap
transitional stubs that cannot run in this image.

This tier cannot check anything browser-shaped.

## Build

```bash
apptainer build osworld-guest-kvm.sif osworld-guest-kvm.def
apptainer build desktop-nonkvm.sif desktop-nonkvm.def
```

The definitions require neither `--fakeroot` nor privileged build flags.
