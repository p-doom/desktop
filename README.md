# desktop

This package boots desktop VMs and executes mouse and keyboard operations on
them. It applies `Operation`s — already resolved to absolute screen pixels — to a real
desktop, and it starts, resets, pools, and proves the state of the VMs those
desktops run on.

## Two distributions

| directory | distribution | handles |
|---|---|---|
| `desktop/` | `desktop` | the computer is *wrong* — black framebuffer, cursor did not land, reset did not rewind. Synchronous; the caller owns the lifetime. |
| `desktop_fleet/` | `desktop-fleet` | the computer is *gone* — Slurm preemption, a dead node, a stale heartbeat, a leaked process group, queue-or-fail. |

`desktop_fleet` may import `desktop`; `desktop` must never import
`desktop_fleet`, and must never learn what a fleet is. That is what keeps it
usable against one VM with no scheduler, and it is why the Pillow-only
dependency floor below can hold.

Nothing here knows what an action grammar is. Whichever coordinate convention a
model emits is resolved inside your own `Codec.compile(text, geometry, cursor)`
before an `Operation` exists, so an absolute-coordinate model and a
relative-delta model both use this package unchanged.

One Python runtime dependency: Pillow, imported in `vm/readiness.py` to
downsample a screenshot and measure whether the framebuffer is still black.
Everything else is stdlib, HTTP is `urllib.request`, and screenshots cross every
boundary as raw bytes. Two external binaries are assumed: `qemu-system-x86_64`
and `apptainer`.

## Usage

```python
from desktop import DisplayGeometry, Operation
from desktop.execute import Engine, HttpGuiTransport
from desktop.vm import DesktopSession, QemuRuntime

runtime = QemuRuntime(image=Path("/images/guest.qcow2"))
with DesktopSession(runtime) as session:
    engine = Engine(session.start())

    # Either hand it resolved operations directly...
    engine.apply((Operation("move_to", (640, 400)), Operation("click", ("left",))))

    # ...or let your own codec resolve model output first. The geometry and the
    # cursor are passed into the codec as data.
    engine.apply_text(model_output, codec=your_codec)

    session.reset()   # ~4.5 s, and it proves the guest actually rewound
```

## The codec contract

`your_codec` satisfies `desktop.codec_protocol.Codec` and lives in *your*
repository. That is the whole contract:

```python
class Codec(Protocol):
    name: str
    stop_sequences: tuple[str, ...]        # an attribute, not a method
    def parse(self, text) -> object: ...
    def format(self, action) -> str: ...   # the inverse: your SFT-target generator
    def compile(self, text, geometry, cursor) -> tuple[Operation, ...]: ...
    def describe(self) -> str: ...         # the system prompt, from docstrings
```

There is no coordinate-space enum in this package. The convention is an open
record inside each codec, and the resolution context arrives as data through
`compile(...)`.

## Layout

| path | what |
|---|---|
| `ir.py` | `Operation(kind: str, args: tuple)` — open `kind` vocabulary, never an `Enum`. Includes an explicit `drag(x0,y0,x1,y1)` so a zero-extent drag survives resolution instead of collapsing into a no-op. |
| `geometry.py` | `DisplayGeometry`, `scale_normalized_coordinate`, `anthropic_scale_coordinates` — copied verbatim from Harbor, offered to codecs as resolution tools. |
| `codec_protocol.py` | The `Codec` protocol, plus a docstring-as-single-source action-set skeleton vendored from BrowserGym: one declaration derives the prompt, examples, validation, lowering, and tool-JSON. |
| `execute/guest_program.py` | Compiles one action into exactly one ordered guest process, with verified pointer state and guaranteed cleanup. |
| `execute/transport.py` | The `GuiTransport` protocol, a `urllib` HTTP implementation, and `RecordingTransport` — an in-process double, so the executor is testable without a VM. |
| `execute/keymap.py` | The key and pointer-button name tables, and the chord and transition helpers over them. |
| `execute/engine.py` | Applies operations, verifies the cursor readback, emits a receipt. |
| `vm/runtime.py` | The `Runtime` protocol: start, stop, checkpoint, restore, fork. |
| `vm/qemu.py` | QMP `savevm`/`loadvm` (13.6–16.6 s reboot-revert → 4.4–5.2 s restore), plus CoW `fork` and an `-accel tcg` fallback. |
| `vm/readiness.py` | Whether the desktop is up or the framebuffer is still black, from the non-dark-pixel ratio and the luma standard deviation. The `luma_sampler` seam makes the rule testable against synthetic frames with no VM. |
| `vm/factory.py` | The constructor side: `build_qemu_runtime`, `build_desktop_session`, `qemu_session_factory`, `build_desktop_pool`. Plain functions, explicit config, no name registry and no plugin lookup — you pass an image path, not a provider name. |
| `vm/session.py` | One isolated desktop with an attested reset: a guest nonce must be gone afterwards, or the reset did not happen. |
| `vm/pool.py` | Prewarmed sessions, leases with activity timeouts, bounded reuse, an inspectable status file. |
| `vm/osworld_client.py` | Four endpoints (`/screenshot`, `/screen_size`, `/execute`, `/accessibility`) plus `/cursor_position`. No dispatch, no keymaps. |
| `vm/sandbox_protocol.py` | NeMo-Gym's sandbox protocol including `ConnectableProvider`, plus our own Apptainer implementation. |
| `vm/images/` | Two tiers: KVM (full parity) and non-KVM (no OSWorld parity, ever — see that directory's README). |
| `vm/image_build.py` | The producer of the guest qcow2 every other module treats as a pinned input: boots the upstream image, installs the grader libraries and tools, verifies them from inside the guest, and publishes the image beside a manifest of what went in. |

## Provenance

Copied verbatim or near-verbatim, with attribution in each file: Harbor
(`geometry.py`), BrowserGym (`codec_protocol.py` skeleton), NeMo-Gym
(`sandbox_protocol.py` types), `anthropics/claude-quickstarts` `computer-use-demo`
(`images/desktop-nonkvm.def`). `trycua/cua` was read as a reference only — the CoW
fork and TCG fallback in `vm/qemu.py` are reimplementations of ideas, not code.
