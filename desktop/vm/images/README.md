# Apptainer definitions

Two tiers, and the difference between them decides whether a number is real.

| tier | def-file | needs `/dev/kvm` | benchmark parity | what it is for |
|---|---|---|---|---|
| KVM | `osworld-guest-kvm.def` | yes (else silent TCG fallback) | **full** — set by the pinned guest image, not by the container | every measurement that will be reported |
| non-KVM | `desktop-nonkvm.def` | no | **none, ever** | plumbing iteration: transport, executor, readiness, pool |

## Why the KVM container has no desktop in it

The guest desktop lives in a pinned qcow2 that this container *boots*, bound in
read-only at run time. Application versions, the in-VM agent, and the display
geometry are what determine what a benchmark number means, so they must be an
artifact a container rebuild cannot change. Baking a desktop into the image would
make every `apptainer build` a silent change to the benchmark.

## Why the non-KVM tier is not a shortcut

`desktop-nonkvm.def` is translated from `anthropics/claude-quickstarts`
`computer-use-demo` (MIT): Xvfb, x11vnc, xdotool, mutter, tint2, noVNC, non-root,
no privileged flags. It is an apps-on-a-desktop runtime — LibreOffice, gedit,
pcmanfm, galculator, xpdf, and **no browser** (see below). It has no OSWorld guest
agent, no OSWorld task setup, and different application versions. It is a tier
*beneath* the KVM tier and never a replacement for it. Anything measured here is a
plumbing check, not a result.

It is still worth having: it runs on any node, it starts in seconds rather than
minutes, and it has `pyautogui` + `python-xlib` installed, so a guest program
compiled by `desktop/execute/guest_program.py` executes there. That makes the
executor testable on a node with no virtualization at all.

## Flagged, not faked

* **There is no browser in the non-KVM tier, and adding one is not a package
  name.** On `ubuntu:22.04`, `chromium-browser` has *no installation candidate*
  (it is a snap transitional stub), `chromium` does not exist, and `firefox` is
  `1:1snap1-0ubuntu2` — also a snap stub. Snaps need systemd and a writable
  `/snap`, neither of which an Apptainer instance has. A browser therefore means a
  third-party PPA or an upstream tarball: a supply-chain decision. Until one is
  made, **this tier cannot check anything browser-shaped.** Whoever adds a browser
  will need a writable profile directory — a `.sif` is read-only, so
  `--writable-tmpfs` (which `ApptainerSandboxProvider` passes by default) or an
  explicit `--overlay`; a startup crash naming the profile directory is that.
* **`apptainer instance start` does not behave like a Docker `ENTRYPOINT`.**
  `%startscript` backgrounds the X stack; a non-daemonising service is the usual
  cause of an instance that is up with no display. Check `/tmp/x11vnc.log`,
  `/tmp/mutter.log`.
* **`%test` in both files is weak on purpose.** It runs at build time, with no
  instance and no `/dev/kvm`, so it can only assert that binaries and imports are
  present. Everything behavioural is checked at run time.

## Building

```
apptainer build osworld-guest-kvm.sif  osworld-guest-kvm.def
apptainer build desktop-nonkvm.sif     desktop-nonkvm.def
```

Neither build needs `--fakeroot` on a system with unprivileged user namespaces.
