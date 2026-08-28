"""The unioned keymap, its resolution order, and ``guest_button``'s strictness.

Three predecessor tables were unioned, and they genuinely disagreed:

  1. ``rung1``      knew ``Key<X>``, not ``Num<N>``/``Digit<N>``
  2. eval client    knew all three, plus punctuation names and uppercase aliases
  3. RL actions     only lowercased

so a trajectory recorded through one did not round-trip through another.  The
tests below assert that all three vocabularies now resolve, that the order is
fixed (which is what stops two call sites disagreeing), and that ``guest_button``
RAISES rather than defaulting -- a silently defaulted button is how a right-click
test passes while testing a left click.
"""

from __future__ import annotations

import pytest

from desktop.execute.guest_program import BUTTON_MASKS
from desktop.execute.keymap import (
    BUTTON_ALIASES,
    BUTTON_NUMBERS,
    KEY_ALIASES,
    KEY_NAMES,
    KEYSYMS,
    POINTER_BUTTONS,
    PRESSABLE_KEYS,
    KeymapError,
    button_transition,
    guest_button,
    guest_key,
    key_chord,
    key_press,
    key_transition,
)


@pytest.mark.parametrize(
    ("name", "expected"), [("KeyA", "a"), ("KeyZ", "z"), ("KeyQ", "q"), ("Keya", "a")]
)
def test_vocabulary_one_key_x(name, expected):
    assert guest_key(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Num0", "0"), ("Num9", "9"), ("Digit0", "0"), ("Digit7", "7")],
)
def test_vocabulary_two_numeric_shapes(name, expected):
    """The shapes the first predecessor did NOT know."""
    assert guest_key(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Comma", ","),
        ("Period", "."),
        ("Slash", "/"),
        ("Backslash", "\\"),
        ("Semicolon", ";"),
        ("Quote", "'"),
        ("Minus", "-"),
        ("Equal", "="),
        ("Backquote", "`"),
        ("BracketLeft", "["),
        ("BracketRight", "]"),
    ],
)
def test_vocabulary_two_punctuation(name, expected):
    assert guest_key(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CTRL", "ctrlleft"),
        ("Ctrl", "ctrlleft"),
        ("ctrl", "ctrlleft"),
        ("CONTROL", "ctrlleft"),
        ("CMD", "winleft"),
        ("cmd", "winleft"),
        ("META", "winleft"),
        ("super", "winleft"),
        ("WINDOWS", "winleft"),
        ("PAGE_UP", "pageup"),
        ("PAGE_DOWN", "pagedown"),
        ("DEL", "delete"),
        ("OPTION", "altleft"),
    ],
)
def test_vocabulary_two_case_insensitive_aliases(name, expected):
    assert guest_key(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("SPACE", "space"),
        ("XF86AudioRaiseVolume", None),
        ("Numpad5", None),
    ],
)
def test_unrecognised_lowercase_passthroughs_are_rejected(name, expected):
    if expected is None:
        with pytest.raises(KeymapError, match="unsupported X11 key"):
            guest_key(name)
    else:
        assert guest_key(name) == expected


@pytest.mark.parametrize("character", ["a", "A", "1", "/", "%", "z"])
def test_a_single_character_maps_to_its_lowercase_self(character):
    assert guest_key(character) == character.lower()


def test_a_trajectory_recorded_in_any_vocabulary_round_trips():
    """One chord, spelled three ways, lands on the same guest key names."""
    rung1_style = ["ControlLeft", "KeyA"]
    eval_style = ["CTRL", "Digit1"]
    rl_style = ["Ctrl", "a"]
    assert [guest_key(k) for k in rung1_style] == ["ctrlleft", "a"]
    assert [guest_key(k) for k in eval_style] == ["ctrlleft", "1"]
    assert [guest_key(k) for k in rl_style] == ["ctrlleft", "a"]


def test_resolution_order_is_exact_then_alias_then_function_then_shape():
    # 1. exact KEY_NAMES beats everything: 'Alt' is an exact hit.
    assert guest_key("Alt") == KEY_NAMES["Alt"] == "altleft"
    # 2. alias, matched on .upper()
    assert guest_key("cmd") == KEY_ALIASES["CMD"] == "winleft"
    # 3. function keys, case-insensitively
    assert guest_key("f5") == guest_key("F5") == "f5"
    assert guest_key("F24") == "f24"
    # 4. Key<X> / Num<N> / Digit<N>
    assert guest_key("KeyB") == "b"
    # 5. a pressable bare lowercase character
    assert guest_key("b") == "b"
    with pytest.raises(KeymapError, match="unsupported X11 key"):
        guest_key("Unheard")


def test_the_exact_table_wins_over_the_alias_table_for_sided_names():
    """The reason the two tables are not merged: exact matching is what keeps a
    sided name sided."""
    assert guest_key("MetaLeft") == "winleft"
    assert guest_key("META") == "winleft"
    assert guest_key("AltRight") == guest_key("AltGr") == "altright"
    assert guest_key("ALT") == "altleft"


def test_every_case_overlap_between_the_two_tables_agrees():
    """Documents a fact the KEY_ALIASES comment used to get wrong: the twelve
    entries that overlap by casing all map to the SAME guest name, so no present
    mapping depends on the split -- only the resolution order does."""
    overlapping = {
        name: (value, KEY_ALIASES[name.upper()])
        for name, value in KEY_NAMES.items()
        if name.upper() in KEY_ALIASES
    }
    assert overlapping, "expected some overlap"
    disagreeing = {k: v for k, v in overlapping.items() if v[0] != v[1]}
    assert disagreeing == {}


def test_resolution_is_deterministic_and_idempotent():
    """``guest_key`` of an already-mapped name must be that name again, or a
    second pass through the map would corrupt a trajectory."""
    for value in list(KEY_NAMES.values()) + list(KEY_ALIASES.values()):
        assert guest_key(value) == value


def test_whitespace_is_stripped_before_resolution():
    assert guest_key("  Return  ") == "enter"
    assert guest_button("  left  ") == "left"


@pytest.mark.parametrize("name", ["Return", "Enter", "RETURN", "enter"])
def test_the_return_key_has_one_guest_spelling(name):
    """A grammar emitting ``Return`` and one emitting ``Enter`` must not become
    two different guest keys."""
    assert guest_key(name) == "enter"


@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_an_empty_key_name_raises(bad):
    with pytest.raises(KeymapError, match="empty"):
        guest_key(bad)


@pytest.mark.parametrize("bad", [None, 5, 1.0, [], object()])
def test_a_non_string_key_raises(bad):
    with pytest.raises(KeymapError, match="must be a string"):
        guest_key(bad)


def test_every_key_name_survives_a_change_of_case():
    """REGRESSION TEST for a silent breakage of 20 of the 38 ``KEY_NAMES``.

    ``KEY_NAMES`` was matched only exactly, and the fallback lowercases, so a
    differently-cased spelling degraded to an UNPRESSABLE name instead of raising:
    ``guest_key("comma")`` returned ``"comma"``, which pyautogui does not know, so
    the keystroke was dropped with no error, while ``guest_key("Comma")`` returned
    ``","``.  Every arrow, both sided control/meta modifiers, ``AltGr`` and all
    eleven punctuation names were affected -- the keys a computer-use model presses
    most.  The count that used to be broken is pinned at 0.
    """
    broken = {
        name: {
            variant: guest_key(variant)
            for variant in (name.lower(), name.upper(), name.swapcase())
            if guest_key(variant) != value
        }
        for name, value in KEY_NAMES.items()
    }
    broken = {name: variants for name, variants in broken.items() if variants}
    assert broken == {}, f"{len(broken)} of {len(KEY_NAMES)} names broke: {broken}"


@pytest.mark.parametrize(
    ("spellings", "expected"),
    [
        (("Comma", "comma", "COMMA"), ","),
        (("Period", "period", "PERIOD"), "."),
        (("Slash", "slash", "SLASH"), "/"),
        (("Backslash", "backslash", "BACKSLASH"), "\\"),
        (("Semicolon", "semicolon", "SEMICOLON"), ";"),
        (("Quote", "quote", "QUOTE"), "'"),
        (("Minus", "minus", "MINUS"), "-"),
        (("Equal", "equal", "EQUAL"), "="),
        (("Backquote", "backquote", "BACKQUOTE"), "`"),
        (("BracketLeft", "bracketleft", "BRACKETLEFT"), "["),
        (("BracketRight", "bracketright", "BRACKETRIGHT"), "]"),
        (("ArrowUp", "arrowup", "ARROWUP"), "up"),
        (("ArrowDown", "arrowdown", "ARROWDOWN"), "down"),
        (("ArrowLeft", "arrowleft", "ARROWLEFT"), "left"),
        (("ArrowRight", "arrowright", "ARROWRIGHT"), "right"),
        (("ControlLeft", "controlleft", "CONTROLLEFT"), "ctrlleft"),
        (("ControlRight", "controlright", "CONTROLRIGHT"), "ctrlright"),
        (("MetaLeft", "metaleft", "METALEFT"), "winleft"),
        (("MetaRight", "metaright", "METARIGHT"), "winright"),
        (("AltGr", "altgr", "ALTGR"), "altright"),
    ],
)
def test_the_previously_broken_names_all_resolve_in_any_case(spellings, expected):
    for spelling in spellings:
        assert guest_key(spelling) == expected, spelling


def test_loose_modifier_aliases_resolve_to_concrete_left_hand_keys():
    assert guest_key("MetaLeft") == "winleft"
    assert guest_key("MetaRight") == "winright"
    assert guest_key("META") == guest_key("meta") == "winleft"
    assert guest_key("AltLeft") == "altleft"
    assert guest_key("AltRight") == guest_key("AltGr") == "altright"
    assert guest_key("ALT") == guest_key("Alt") == "altleft"
    assert guest_key("ControlLeft") == "ctrlleft"
    assert guest_key("CTRL") == guest_key("CONTROL") == "ctrlleft"


def test_the_folded_table_has_no_case_insensitive_collisions():
    """Two ``KEY_NAMES`` entries differing only in case would make the fold
    order-dependent; there are none, and a new one must be a visible change."""
    from desktop.execute.keymap import _KEY_NAMES_FOLDED

    assert len(_KEY_NAMES_FOLDED) == len(KEY_NAMES)
    assert set(_KEY_NAMES_FOLDED.values()) == set(KEY_NAMES.values())


def test_the_alias_table_still_takes_precedence_over_the_fold():
    """Where both could answer, the alias table is authoritative -- it is the one
    that knows what a prose-ish spelling means."""
    for alias, expected in KEY_ALIASES.items():
        assert guest_key(alias) == expected, alias


def test_an_unknown_multicharacter_name_fails_loudly():
    for name in ("Retrun", "F25"):
        with pytest.raises(KeymapError, match="unsupported X11 key"):
            guest_key(name)


def test_every_pressable_key_has_a_fixed_keysym():
    assert PRESSABLE_KEYS == frozenset(KEYSYMS)
    assert all(isinstance(keysym, int) and keysym > 0 for keysym in KEYSYMS.values())


def test_an_unsupported_key_fails_before_a_guest_program_is_dispatched():
    from desktop import ir
    from desktop.execute.guest_program import compile_atomic_guest_program

    with pytest.raises(KeymapError, match="unsupported X11 key"):
        compile_atomic_guest_program(
            (ir.key_down("Retrun"),), initial_buttons=set(), initial_keys=set()
        )


@pytest.mark.parametrize("name", ["left", "middle", "right"])
def test_the_canonical_button_names_pass_through(name):
    assert guest_button(name) == name


@pytest.mark.parametrize(("number", "expected"), [(1, "left"), (2, "middle"), (3, "right")])
def test_x11_button_numbers_resolve(number, expected):
    assert guest_button(number) == expected


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("LMB", "left"),
        ("MMB", "middle"),
        ("RMB", "right"),
        ("ButtonLeft", "left"),
        ("buttonright", "right"),
        ("LEFT", "left"),
        ("Middle", "middle"),
    ],
)
def test_recorder_button_aliases_resolve(alias, expected):
    assert guest_button(alias) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "Left1",
        "",
        "   ",
        "leftclick",
        "BUTTON_LEFT",
        "button4",
        0,
        4,
        -1,
        99,
        1.0,
        None,
        [],
        object(),
    ],
)
def test_an_unknown_button_raises_rather_than_defaulting_to_left(bad):
    with pytest.raises(KeymapError):
        guest_button(bad)


@pytest.mark.parametrize("bad", [True, False])
def test_a_bool_is_not_silently_an_x11_button_number(bad):
    """``True == 1`` in Python, so a bool would otherwise become a left click."""
    with pytest.raises(KeymapError):
        guest_button(bad)


def test_the_button_tables_stay_in_lockstep_with_the_pointer_masks():
    """``BUTTON_MASKS`` in ``guest_program`` must agree with ``POINTER_BUTTONS``."""
    assert set(BUTTON_MASKS) == set(POINTER_BUTTONS)
    assert set(BUTTON_NUMBERS.values()) == set(POINTER_BUTTONS)
    assert set(BUTTON_ALIASES.values()) <= set(POINTER_BUTTONS)


def test_every_button_alias_resolves_to_a_pressable_button():
    for alias in BUTTON_ALIASES:
        assert guest_button(alias) in POINTER_BUTTONS


def test_a_chord_presses_in_order_and_releases_in_reverse():
    """Releasing ctrl before a in ctrl+a delivers a bare 'a' on the pinned guest."""
    operations = key_chord(["CTRL", "KeyA"])
    assert [(op.kind, op.args) for op in operations] == [
        ("key_down", ("ctrlleft",)),
        ("key_down", ("a",)),
        ("key_up", ("a",)),
        ("key_up", ("ctrlleft",)),
    ]


def test_a_three_key_chord_releases_in_full_reverse_order():
    operations = key_chord(["ctrl", "shift", "KeyT"])
    assert [op.args[0] for op in operations] == [
        "ctrlleft",
        "shiftleft",
        "t",
        "t",
        "shiftleft",
        "ctrlleft",
    ]


def test_a_chord_emits_operations_and_never_pyautogui_source():
    """All three predecessors returned source text; the union must not."""
    for operation in key_chord(["ctrl", "a"]):
        assert "pyautogui" not in str(operation.args[0])


def test_an_empty_chord_raises():
    with pytest.raises(KeymapError, match="empty key chord"):
        key_chord([])


def test_key_press_is_a_mapped_down_then_up():
    assert [(op.kind, op.args) for op in key_press("Return")] == [
        ("key_down", ("enter",)),
        ("key_up", ("enter",)),
    ]


def test_key_transition_maps_the_name():
    assert key_transition("ControlLeft", pressed=True).args == ("ctrlleft",)
    assert key_transition("ControlLeft", pressed=False).kind == "key_up"


def test_button_transition_maps_and_validates():
    assert button_transition(3, pressed=True).args == ("right",)
    assert button_transition("LMB", pressed=False).kind == "mouse_up"
    with pytest.raises(KeymapError):
        button_transition("nope", pressed=True)


#
# The press and the release of one key must balance even when a trajectory spells
# the two halves differently.  Held state used to be keyed on the RAW operation
# argument while the lowering pressed ``guest_key(name)``, so ``key_down("Return")``
# + ``key_up("Enter")`` -- one physical key -- was rejected host-side as "key not
# held", and ``key_down("Return")`` + ``key_down("Enter")`` was accepted as two
# separate held keys.  Reachable from any grammar that passes rdev names through
# verbatim, which several deliberately do.


@pytest.mark.parametrize(
    ("down", "up"),
    [
        ("Return", "Enter"),
        ("Return", "enter"),
        ("Enter", "RETURN"),
        ("KeyA", "a"),
        ("KeyA", "A"),
        ("ShiftLeft", "shiftleft"),
        ("CTRL", "ctrl"),
        ("Escape", "ESC"),
        ("Digit1", "Num1"),
        ("Comma", ","),
        ("F5", "f5"),
    ],
)
def test_a_press_and_release_spelled_differently_still_balance(down, up):
    from desktop import ir
    from desktop.execute.guest_program import expected_atomic_input_state

    assert guest_key(down) == guest_key(up), "fixture must be two spellings of one key"
    buttons, keys = expected_atomic_input_state(
        (ir.key_down(down), ir.key_up(up)), initial_buttons=set(), initial_keys=set()
    )
    assert keys == set(), f"{down}/{up} did not balance"
    assert buttons == set()


def test_pressing_one_key_under_two_spellings_is_a_double_press():
    from desktop import ir
    from desktop.execute.guest_program import ExecutionError, expected_atomic_input_state

    with pytest.raises(ExecutionError, match="key already held: enter"):
        expected_atomic_input_state(
            (ir.key_down("Return"), ir.key_down("Enter")),
            initial_buttons=set(),
            initial_keys=set(),
        )


def test_held_state_is_reported_in_the_mapped_vocabulary():
    """So the host's held set and the guest's ``_de_touched_keys`` are comparable."""
    from desktop import ir
    from desktop.execute.guest_program import expected_atomic_input_state

    _, keys = expected_atomic_input_state(
        (ir.key_down("ControlLeft"), ir.key_down("KeyA")),
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert keys == {"ctrlleft", "a"}


def test_incoming_initial_keys_are_normalised_too():
    """An audit carried over from a previous action may hold either spelling."""
    from desktop import ir
    from desktop.execute.guest_program import expected_atomic_input_state

    _, keys = expected_atomic_input_state(
        (ir.key_up("Enter"),), initial_buttons=set(), initial_keys={"Return"}
    )
    assert keys == set()


def test_normalisation_is_idempotent_across_action_boundaries():
    """Held state feeds back in as ``initial_keys``; a second pass must be a
    no-op, or a long trajectory would drift."""
    from desktop import ir
    from desktop.execute.guest_program import expected_atomic_input_state

    _, first = expected_atomic_input_state(
        (ir.key_down("Return"),), initial_buttons=set(), initial_keys=set()
    )
    _, second = expected_atomic_input_state((), initial_buttons=set(), initial_keys=first)
    assert first == second == {"enter"}


def test_a_cross_spelling_pair_executes_in_the_guest_as_one_key():
    """End to end: the compiled program presses and releases the same key once."""
    from desktop import ir
    from tests.support.guest_runner import run_guest_program

    run = run_guest_program((ir.key_down("Return"), ir.key_up("Enter")))
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    assert [event["event"] for event in run.payload["x_injection_evidence"]] == [
        "key_press",
        "key_release",
    ]


def test_the_guest_reads_back_a_held_key_and_its_release():
    from desktop import ir
    from tests.support.guest_runner import run_guest_program

    down = run_guest_program((ir.key_down("ControlLeft"),))
    assert down.payload["held_keys"] == ["ctrlleft"]
    up = run_guest_program((ir.key_up("CTRL"),), initial_keys={"ControlLeft"})
    assert up.payload["held_keys"] == []


def test_initial_held_key_mismatch_is_a_verification_failure():
    from desktop import ir
    from tests.support.guest_runner import run_guest_program

    run = run_guest_program(
        (ir.key_up("ControlLeft"),),
        initial_keys={"ControlLeft"},
        backend_initial_keys=set(),
    )
    assert run.payload["ok"] is False
    assert run.payload["failure_kind"] == "verification"
    assert "initial held keycodes" in run.payload["error"]


def test_the_recording_double_balances_the_same_pair(recording):
    """The double keys held state the same way, or it rejects what the guest runs."""
    from desktop import ir

    result = recording.execute_atomic((ir.key_down("KeyA"), ir.key_up("a")))
    assert result.ok is True, result.error
    assert recording.audit.held_keys == set()
    # The TRACE keeps the raw spelling, matching what the guest reports.
    assert [(op.kind, op.args) for op in result.operations] == [
        ("key_down", ("KeyA",)),
        ("key_up", ("a",)),
    ]


def test_the_recording_double_and_the_guest_agree_on_the_trace(recording):
    from desktop import ir
    from tests.support.guest_runner import run_guest_program

    operations = (
        ir.key_down("ControlLeft"),
        ir.key_down("KeyA"),
        ir.key_up("a"),
        ir.key_up("ctrlleft"),
    )
    guest = run_guest_program(operations)
    result = recording.execute_atomic(operations)
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert result.ok is True and guest.payload["ok"] is True


def test_an_unbalanced_release_is_still_rejected():
    """The normalisation must not weaken the check it is part of."""
    from desktop import ir
    from desktop.execute.guest_program import ExecutionError, expected_atomic_input_state

    with pytest.raises(ExecutionError, match="key not held: shiftleft"):
        expected_atomic_input_state(
            (ir.key_down("ctrl"), ir.key_up("SHIFT")),
            initial_buttons=set(),
            initial_keys=set(),
        )


def test_a_held_state_contradiction_is_its_own_class():
    """The taxonomy a rollout harness reads: this one is the CALLER's fault.

    `ExecutionError` also covers the guest request failing and the wiring being
    wrong, which are ours. A harness that cannot tell them apart scores a model's
    unmatched `up(...)` as its own infrastructure failure and nulls the episode.
    """
    from desktop import ir
    from desktop.execute.guest_program import (
        ExecutionError,
        HeldStateError,
        expected_atomic_input_state,
    )

    assert issubclass(HeldStateError, ExecutionError), "callers catching the base still work"
    for operations in (
        (ir.key_up("ShiftLeft"),),
        (ir.key_down("ctrl"), ir.key_down("ctrl")),
        (ir.mouse_up("left"),),
    ):
        with pytest.raises(HeldStateError):
            expected_atomic_input_state(operations, initial_buttons=set(), initial_keys=set())


def test_two_genuinely_different_sided_keys_are_still_two_keys():
    from desktop import ir
    from desktop.execute.guest_program import expected_atomic_input_state

    _, keys = expected_atomic_input_state(
        (ir.key_down("MetaLeft"), ir.key_down("MetaRight")),
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert keys == {"winleft", "winright"}


def test_a_malformed_key_name_raises_instead_of_being_tracked():
    """``guest_key`` rejects an empty name, so held state cannot contain one."""
    from desktop import ir
    from desktop.execute.guest_program import expected_atomic_input_state

    with pytest.raises(KeymapError):
        expected_atomic_input_state(
            (ir.key_down("  "),), initial_buttons=set(), initial_keys=set()
        )
