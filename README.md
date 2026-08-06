# pixeldesk

**Nothing in this repository imports a grammar, a codec, or anything from the
model-training repositories it serves — and nothing here knows what an action
grammar is.** It receives `Operation`s that have already been resolved to
absolute screen pixels and applies them to a real desktop, and it starts,
resets, pools, and proves the state of the VMs those desktops run on. A team
training an absolute-coordinate model and a team training a relative-delta model
use it unchanged, because whichever coordinate convention a model emits is fully
consumed inside that team's own `Codec.compile(text, geometry, cursor)` before an
`Operation` exists.

There is **exactly one Python runtime dependency, Pillow** — everything else is
stdlib, HTTP is `urllib.request`, and screenshots cross every boundary here as raw
**bytes**. Two external binaries are assumed: `qemu-system-x86_64` and
`apptainer`.

### Why Pillow, and only Pillow

Pillow is used in one place — `vm/readiness.py`, to downsample a screenshot and
measure whether the framebuffer is still black. A hand-written `zlib` PNG decoder
was written for that and then **deleted**, because its failure mode is silent: a
bug in scanline reconstruction makes the statistics wrong, wrong statistics make
readiness report *not ready*, and "not ready" is indistinguishable from a slow VM.
It would have been diagnosed as infrastructure for as long as it took someone to
doubt the decoder. Owning a failure mode that mimics infrastructure slowness is
worse than owning a dependency.

Pillow clears the bar this package applies to any dependency: it does one thing,
its correctness criterion is external to us, and its import floor is small. Going
back to it also restored a calibration the hand-rolled version had quietly given
up — `thumbnail` averages over a resampling filter, and the two readiness
thresholds were tuned against the averaged form.

The `luma_sampler` seam is kept, so the readiness rule is still testable against
synthetic frames with no VM and no image decoding in the loop.

## The seam

```python
from pixeldesk import DisplayGeometry, Operation
from pixeldesk.execute import Engine, HttpGuiTransport
from pixeldesk.vm import DesktopSession, QemuRuntime

runtime = QemuRuntime(image=Path("/images/guest.qcow2"))
with DesktopSession(runtime) as session:
    engine = Engine(session.start())

    # Either hand it resolved operations directly...
    engine.apply((Operation("move_to", (640, 400)), Operation("click", ("left",))))

    # ...or let your own codec resolve model output first. The geometry and the
    # cursor are passed *into* the codec as data; the engine never learns which
    # convention was resolved.
    engine.apply_text(model_output, codec=your_codec)

    session.reset()   # ~4.5 s, and it proves the guest actually rewound
```

`your_codec` satisfies `pixeldesk.codec_protocol.Codec` and lives in *your*
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

There is deliberately **no coordinate-space enum anywhere in this package.**
A closed enumeration of conventions would put grammar knowledge in shared code
and would have to be edited by every team that invented a convention. The
convention is an open record inside each codec; the resolution context arrives as
data through `compile(...)`.

## Layout

| path | what |
|---|---|
| `ir.py` | `Operation(kind: str, args: tuple)` — open `kind` vocabulary, never an `Enum`. Includes an explicit `drag(x0,y0,x1,y1)` so a genuine zero-extent drag survives resolution instead of collapsing into a no-op. |
| `geometry.py` | `DisplayGeometry`, `scale_normalized_coordinate`, `anthropic_scale_coordinates` — copied verbatim from Harbor, offered to codecs as resolution tools. |
| `codec_protocol.py` | The `Codec` protocol, plus a docstring-as-single-source action-set skeleton vendored from BrowserGym: one declaration derives the prompt, examples, validation, lowering, and tool-JSON. |
| `execute/guest_program.py` | The novel part: compiles one action into exactly **one** ordered guest process, with verified pointer state and guaranteed cleanup. |
| `execute/transport.py` | The `GuiTransport` protocol, a `urllib` HTTP implementation, and `RecordingTransport` — which is why the executor is testable without a VM. |
| `execute/keymap.py` | One key/button table where there were three that disagreed. |
| `execute/engine.py` | Applies operations, verifies the cursor readback, emits a receipt. |
| `vm/qemu.py` | QMP `savevm`/`loadvm` (13.6–16.6 s reboot-revert → 4.4–5.2 s restore), plus CoW `fork` and an `-accel tcg` fallback. Not a monkeypatch into anyone's tree. |
| `vm/factory.py` | The constructor side: `build_qemu_runtime`, `build_desktop_session`, `qemu_session_factory`, `build_desktop_pool`. Plain functions, explicit config, **no name registry and no plugin lookup** — you pass an image path, not a provider name. |
| `vm/session.py` | One isolated desktop with an attested reset: a guest nonce must be gone afterwards, or the reset did not happen. |
| `vm/pool.py` | Prewarmed sessions, leases with activity timeouts, bounded reuse, an inspectable status file. |
| `vm/osworld_client.py` | Four endpoints (`/screenshot`, `/screen_size`, `/execute`, `/accessibility`) plus `/cursor_position`. No dispatch, no keymaps. |
| `vm/sandbox_protocol.py` | NeMo-Gym's sandbox protocol including `ConnectableProvider`, plus our own Apptainer implementation. |
| `vm/images/` | Two tiers: KVM (full parity) and non-KVM (**no parity, ever** — see that directory's README). |

## Provenance

Copied verbatim or near-verbatim, with attribution in each file: Harbor
(`geometry.py`), BrowserGym (`codec_protocol.py` skeleton), NeMo-Gym
(`sandbox_protocol.py` types), `anthropics/claude-quickstarts` `computer-use-demo`
(`images/desktop-nonkvm.def`). `trycua/cua` was read as a reference only — the CoW
fork and TCG fallback in `vm/qemu.py` are reimplementations of ideas, not code.
Everything else is consolidated from our own `osworld_parity/` tree and the RL
repo's desktop layer, with the per-file headers recording what was merged and what
was deliberately left behind.
