"""``parse_calls`` (the ``ast`` parser) and ``ActionSet.lower``'s cursor threading.

``parse_calls`` replaced pyparsing's PEG with ``ast.parse`` in eval mode.  The
security-shaped claim -- arguments are evaluated by ``literal_eval``, so no name
lookup, attribute access or call can happen inside one -- is tested directly.

The cursor threading is the load-bearing half: a relative grammar resolves each
call against the cursor left by the previous one, so a multi-call action only
lands where intended if EVERY pointer-moving kind advances the thread.
"""

from __future__ import annotations

import pytest

from typing import Any, Callable

from pixeldesk import ir
from pixeldesk.codec_protocol import (
    ActionSet,
    MultiActionError,
    ParsedCall,
    UnknownActionError,
    action_spec,
    parse_action_docstring,
    parse_calls,
)
from pixeldesk.geometry import DisplayGeometry, resolve_relative

GEOMETRY = DisplayGeometry(desktop_width=1920, desktop_height=1080)


def move_rel(dx: int, dy: int, *, geometry, cursor):
    """Move the pointer by a delta.

    Examples:
        move_rel(10, 0)
        move_rel(-5, 5)
    """
    return (ir.move_to(*resolve_relative(cursor, dx, dy, geometry)),)


def drag_rel(dx: int, dy: int, *, geometry, cursor):
    """Drag from the cursor by a delta."""
    end = resolve_relative(cursor, dx, dy, geometry)
    return (ir.drag(cursor[0], cursor[1], end[0], end[1]),)


def glide_rel(dx: int, dy: int, *, geometry, cursor):
    """Sweep the pointer by a delta."""
    end = resolve_relative(cursor, dx, dy, geometry)
    return (ir.glide_to(end[0], end[1], 0.3),)


def click_only(button: str = "left"):
    """Click without moving."""
    return (ir.click(button),)


def nothing():
    """Produce no operations at all."""
    return ()


def returns_none():
    """A handler that returns None rather than an empty tuple."""
    return None


ACTIONS = ActionSet([move_rel, drag_rel, glide_rel, click_only, nothing, returns_none])


def test_a_bare_call_with_positional_and_keyword_literals_parses():
    (call,) = parse_calls("move_rel(10, dy=-5)")
    assert call == ParsedCall("move_rel", (10,), {"dy": -5})


def test_one_call_per_line_parses_in_order():
    calls = parse_calls("move_rel(1, 2)\nclick_only('right')\nmove_rel(3, 4)")
    assert [call.name for call in calls] == ["move_rel", "click_only", "move_rel"]


def test_a_trailing_semicolon_is_tolerated():
    assert parse_calls("move_rel(1, 2);")[0].args == (1, 2)


def test_blank_lines_and_comments_are_skipped():
    assert len(parse_calls("\n# a comment\n\nmove_rel(1, 2)\n   \n")) == 1


def test_chatty_prose_is_skipped_when_not_strict():
    calls = parse_calls("Let me click the button.\nmove_rel(1, 2)\nThat should do it.")
    assert [call.name for call in calls] == ["move_rel"]


@pytest.mark.parametrize(
    "literal",
    ["1", "-3", "1.5", "'text'", '"text"', "True", "False", "None", "b'bytes'",
     "[1, 2]", "{'a': 1}", "(1, 2)", "{1, 2}"],
)
def test_every_python_literal_shape_is_accepted(literal):
    (call,) = parse_calls(f"f({literal})", strict=True)
    assert len(call.args) == 1


@pytest.mark.parametrize(
    "source",
    [
        "f(name)",
        "f(obj.attr)",
        "f(other())",
        "f(__import__('os'))",
        "f(1 + 1)",
        "f(1 if True else 2)",
        "f(*[1, 2])",
        "f(x for x in [1])",
    ],
)
def test_a_non_literal_argument_is_rejected_in_strict_mode(source):
    with pytest.raises(ValueError):
        parse_calls(source, strict=True)


def test_no_name_lookup_attribute_access_or_call_can_happen_in_an_argument():
    """The whole reason ``literal_eval`` was chosen over ``eval``."""
    assert parse_calls("f(__import__('os').system('echo pwned'))") == []
    assert parse_calls("f(open('/etc/passwd').read())") == []


def test_keyword_unpacking_is_rejected_rather_than_silently_discarded():
    """``f(**mapping)`` used to yield ``ParsedCall('f', (), {})``: a call that
    looks well-formed and has lost every argument it was given."""
    with pytest.raises(ValueError):
        parse_calls("f(**{'a': 1})", strict=True)
    assert parse_calls("f(**{'a': 1})") == []
    # ... while ordinary keywords are untouched.
    (call,) = parse_calls("f(1, b=2)", strict=True)
    assert call.args == (1,) and call.kwargs == {"b": 2}


def test_an_attribute_call_is_not_a_bare_function_call():
    with pytest.raises(ValueError, match="not a bare function call"):
        parse_calls("module.action(1)", strict=True)


def test_a_non_call_expression_is_rejected_in_strict_mode():
    with pytest.raises(ValueError, match="not a bare function call"):
        parse_calls("42", strict=True)


def test_an_empty_action_is_rejected_in_strict_mode():
    for source in ("", "   ", "# only a comment"):
        with pytest.raises(ValueError, match="empty action"):
            parse_calls(source, strict=True)


def test_two_calls_on_one_line_both_parse():
    """Used to yield ZERO actions in non-strict mode: ``a(); b()`` is not a valid
    ``eval``-mode expression, so the line was a SyntaxError and was skipped as if
    it were prose.  Two actions the model emitted became none, silently."""
    calls = parse_calls("click_only('left'); move_rel(1, 2)")
    assert [call.name for call in calls] == ["click_only", "move_rel"]
    assert calls[0].args == ("left",) and calls[1].args == (1, 2)


def test_many_calls_on_one_line_all_parse_in_order():
    calls = parse_calls("move_rel(1, 1); move_rel(2, 2); move_rel(3, 3)", strict=True)
    assert [call.args for call in calls] == [(1, 1), (2, 2), (3, 3)]


def test_a_semicolon_separated_line_is_never_silently_empty():
    """The invariant: a line containing a parsable call yields at least one."""
    for source in (
        "click_only('left')",
        "click_only('left');",
        "click_only('left'); move_rel(1, 2)",
        "click_only('left') ; move_rel(1, 2) ;",
    ):
        assert parse_calls(source), source


def test_calls_split_across_lines_and_semicolons_are_equivalent():
    inline = parse_calls("click_only('left'); move_rel(1, 2)")
    multiline = parse_calls("click_only('left')\nmove_rel(1, 2)")
    assert inline == multiline


def test_a_non_call_statement_on_a_shared_line_is_still_refused():
    """Only bare calls count; an assignment mixed in is not an action."""
    assert parse_calls("x = 1; move_rel(1, 2)") == [ParsedCall("move_rel", (1, 2), {})]
    with pytest.raises(ValueError, match="not a bare function call"):
        parse_calls("x = 1; move_rel(1, 2)", strict=True)


def test_a_non_literal_argument_on_a_shared_line_is_still_refused():
    assert parse_calls("move_rel(nope); click_only('left')") == [
        ParsedCall("click_only", ("left",), {})
    ]
    with pytest.raises(ValueError, match="non-literal argument"):
        parse_calls("move_rel(nope); click_only('left')", strict=True)


def test_exec_mode_does_not_widen_what_counts_as_an_argument():
    """The security-shaped claim must survive the mode change."""
    assert parse_calls("f(__import__('os').system('echo pwned')); g(1)") == [
        ParsedCall("g", (1,), {})
    ]
    for source in ("f(name)", "f(obj.attr)", "f(other())", "f(1 + 1)"):
        with pytest.raises(ValueError):
            parse_calls(source, strict=True)


def test_a_compound_statement_is_not_an_action_line():
    """``exec`` mode accepts ``if``/``for``/``def``; none of them is a call."""
    for source in ("if True: move_rel(1, 2)", "for i in []: move_rel(1, 2)"):
        assert parse_calls(source) == []
        with pytest.raises(ValueError, match="not a bare function call"):
            parse_calls(source, strict=True)


def test_prose_is_still_skipped_after_the_mode_change():
    calls = parse_calls("Let me click the button.\nmove_rel(1, 2)\nDone.")
    assert [call.name for call in calls] == ["move_rel"]


def test_parsed_call_renders_back_to_source():
    assert ParsedCall("f", (1, "a"), {"k": 2}).render() == "f(1, 'a', k=2)"


def test_successive_relative_moves_thread_the_cursor():
    operations = ACTIONS.lower(
        parse_calls("move_rel(10, 10)\nmove_rel(10, 10)\nmove_rel(10, 10)"),
        GEOMETRY,
        (0, 0),
    )
    assert [op.args for op in operations] == [(10, 10), (20, 20), (30, 30)]


def test_a_drag_threads_the_cursor_from_its_END_point():
    operations = ACTIONS.lower(
        parse_calls("drag_rel(100, 0)\nmove_rel(1, 1)"), GEOMETRY, (0, 0)
    )
    assert operations[0].args == (0, 0, 100, 0)
    assert operations[1].args == (101, 1)


def test_a_glide_threads_the_cursor_like_a_move():
    """``glide_to`` lands the pointer exactly where ``move_to`` would; it only
    takes longer.  Not threading it left the next relative call resolving
    against a stale cursor -- silently, and off by the whole sweep."""
    operations = ACTIONS.lower(
        parse_calls("glide_rel(100, 0)\nmove_rel(1, 1)"), GEOMETRY, (0, 0)
    )
    assert operations[0].args == (100, 0, 0.3)
    assert operations[1].args == (101, 1)


def test_glide_and_move_thread_identically():
    glided = ACTIONS.lower(parse_calls("glide_rel(7, 9)\nmove_rel(0, 0)"), GEOMETRY, (0, 0))
    moved = ACTIONS.lower(parse_calls("move_rel(7, 9)\nmove_rel(0, 0)"), GEOMETRY, (0, 0))
    assert glided[-1].args == moved[-1].args


def test_a_click_does_not_move_the_threaded_cursor():
    operations = ACTIONS.lower(
        parse_calls("click_only('left')\nmove_rel(5, 5)"), GEOMETRY, (100, 100)
    )
    assert operations[-1].args == (105, 105)


def test_the_thread_survives_a_handler_that_produces_nothing():
    operations = ACTIONS.lower(
        parse_calls("move_rel(10, 0)\nnothing()\nmove_rel(10, 0)"), GEOMETRY, (0, 0)
    )
    assert [op.args for op in operations] == [(10, 0), (20, 0)]


def test_a_handler_returning_none_is_treated_as_no_operations():
    assert ACTIONS.lower(parse_calls("returns_none()"), GEOMETRY, (0, 0)) == ()


def test_threading_clamps_at_the_screen_edge_and_stays_clamped():
    operations = ACTIONS.lower(
        parse_calls("move_rel(99999, 99999)\nmove_rel(10, 10)"), GEOMETRY, (0, 0)
    )
    assert operations[0].args == (1919, 1079)
    assert operations[1].args == (1919, 1079)


def test_the_starting_cursor_is_the_one_passed_in():
    operations = ACTIONS.lower(parse_calls("move_rel(1, 1)"), GEOMETRY, (500, 600))
    assert operations[0].args == (501, 601)


def test_only_handlers_that_ask_for_context_receive_it():
    seen = {}

    def wants_neither(value: int):
        "No context."
        seen["neither"] = True
        return ()

    def wants_cursor(value: int, *, cursor):
        "Cursor only."
        seen["cursor"] = cursor
        return ()

    def wants_geometry(value: int, *, geometry):
        "Geometry only."
        seen["geometry"] = geometry
        return ()

    action_set = ActionSet([wants_neither, wants_cursor, wants_geometry])
    action_set.lower(
        parse_calls("wants_neither(1)\nwants_cursor(1)\nwants_geometry(1)"),
        GEOMETRY,
        (3, 4),
    )
    assert seen["neither"] is True
    assert seen["cursor"] == (3, 4)
    assert seen["geometry"] is GEOMETRY


def test_an_unknown_action_name_is_rejected():
    with pytest.raises(UnknownActionError, match="invalid action type 'nope'"):
        ACTIONS.lower(parse_calls("nope()"), GEOMETRY, (0, 0))


def test_a_single_action_set_rejects_multiple_calls():
    single = ActionSet([move_rel], multiaction=False)
    with pytest.raises(MultiActionError, match="received 2 calls"):
        single.lower(parse_calls("move_rel(1, 1)\nmove_rel(2, 2)"), GEOMETRY, (0, 0))


def test_an_empty_call_list_is_rejected():
    with pytest.raises(ValueError, match="empty action"):
        ACTIONS.validate([])


def test_an_action_set_needs_at_least_one_action():
    with pytest.raises(ValueError, match="at least one action"):
        ActionSet([])


def test_subsets_are_the_one_grammar_many_dialects_mechanism():
    action_set = ActionSet(
        subsets={"mouse": [move_rel, click_only], "gesture": [drag_rel]},
        names=["mouse", "gesture"],
    )
    assert set(action_set.functions) == {"move_rel", "click_only", "drag_rel"}


def test_an_unknown_subset_name_is_rejected():
    with pytest.raises(ValueError, match="unknown action subset"):
        ActionSet(subsets={"a": [move_rel]}, names=["b"])


def test_duplicate_actions_are_de_duplicated_in_declaration_order():
    action_set = ActionSet([move_rel, click_only, move_rel])
    assert list(action_set.functions) == ["move_rel", "click_only"]


def test_the_handler_table_is_the_dispatch_surface():
    assert ACTIONS.handlers["move_rel"] is move_rel
    ACTIONS.handlers["move_rel"] = None  # a copy, so mutation cannot poison it
    assert ACTIONS.handlers["move_rel"] is move_rel


def test_the_docstring_is_the_spec():
    description, examples = parse_action_docstring(move_rel.__doc__)
    assert description == "Move the pointer by a delta."
    assert examples == ("move_rel(10, 0)", "move_rel(-5, 5)")


def test_a_docstring_without_examples_still_yields_a_description():
    assert parse_action_docstring("Just prose.") == ("Just prose.", ())
    assert parse_action_docstring(None) == ("", ())


def test_the_spec_is_derived_from_the_signature():
    spec = action_spec(move_rel)
    assert spec.name == "move_rel"
    assert spec.parameters["required"] == ["dx", "dy"]
    assert "geometry" not in spec.parameters["properties"]
    assert "cursor" not in spec.parameters["properties"]
    assert spec.signature == "move_rel(dx: int, dy: int, *, geometry, cursor)"


def test_describe_is_assembled_from_the_declarations():
    text = ACTIONS.describe()
    assert "6 different types of actions are available." in text
    assert "move_rel(dx: int, dy: int, *, geometry, cursor)" in text
    assert "Description: Move the pointer by a delta." in text
    assert "move_rel(10, 0)" in text
    assert "Multiple actions can be provided at once" in text


def test_the_rendered_prompt_never_shows_quoted_annotation_types():
    """``describe()`` IS the system prompt the model reads.

    This module uses ``from __future__ import annotations``, as a real codec
    module will.  Reading raw ``__annotations__`` made the prompt render
    ``move_rel(dx: 'int', dy: 'int', ...)`` -- with the quotes -- so the grammar's
    own self-description disagreed with the grammar, in exactly the formats whose
    point is native-format fidelity.
    """
    text = ACTIONS.describe()
    assert "dx: 'int'" not in text
    assert "'int'" not in text and '"int"' not in text
    assert "move_rel(dx: int, dy: int, *, geometry, cursor)" in text
    assert "drag_rel(dx: int, dy: int, *, geometry, cursor)" in text
    assert "click_only(button: str = 'left')" in text


def test_describe_says_single_action_when_multiaction_is_off():
    assert "Only a single action" in ActionSet([move_rel], multiaction=False).describe()


def test_tool_json_uses_the_right_schema_key_per_api():
    (openai,) = ActionSet([move_rel]).to_tool_description(api="openai")
    (anthropic,) = ActionSet([move_rel]).to_tool_description(api="anthropic")
    assert openai["type"] == "function" and "parameters" in openai
    assert "input_schema" in anthropic and "type" not in anthropic
    assert "Examples:\n- move_rel(10, 0)" in anthropic["description"]


def test_an_unrecognised_tool_api_is_refused():
    """It used to serve a third shape -- the OpenAI schema key without
    ``"type": "function"`` -- for any unrecognised name, including a typo."""
    with pytest.raises(ValueError, match="unsupported tool API"):
        ActionSet([move_rel]).to_tool_description(api="openai ")


def test_defaults_appear_in_the_derived_schema():
    (tool,) = ActionSet([click_only]).to_tool_description()
    assert tool["parameters"]["properties"]["button"]["default"] == "left"
    assert tool["parameters"]["required"] == []


def _codec_with_postponed_annotations(source: str) -> Callable[..., Any]:
    """Compile an action in a module that postpones its annotations.

    ``dont_inherit`` matters twice over: without it the compiled source inherits
    THIS module's flags, so a test meaning to exercise the postponed case could
    accidentally exercise the eager one, or vice versa.
    """
    namespace: dict = {}
    exec(compile(source, "<codec>", "exec", dont_inherit=True), namespace)
    return namespace["act"]


POSTPONED_ACTION = (
    "from __future__ import annotations\n"
    "def act(count: int, ratio: float, flag: bool = False, label: str = 'x'):\n"
    "    'Doc.'\n"
    "    return ()\n"
)
EAGER_ACTION = (
    "def act(count: int, ratio: float, flag: bool = False, label: str = 'x'):\n"
    "    'Doc.'\n"
    "    return ()\n"
)


def test_annotations_survive_postponed_evaluation():
    """The defect this fixed: string annotations typed everything as "string"."""
    action = _codec_with_postponed_annotations(POSTPONED_ACTION)
    (tool,) = ActionSet([action]).to_tool_description()
    properties = tool["parameters"]["properties"]
    assert properties["count"]["type"] == "number"
    assert properties["ratio"]["type"] == "number"
    assert properties["flag"]["type"] == "boolean"
    assert properties["label"]["type"] == "string"


def test_postponed_and_eager_annotations_describe_identically():
    """Whether a codec module postpones its annotations must not be observable in
    what the model is served -- neither in the tool JSON nor in the prompt."""
    postponed = ActionSet([_codec_with_postponed_annotations(POSTPONED_ACTION)])
    eager = ActionSet([_codec_with_postponed_annotations(EAGER_ACTION)])
    assert postponed.to_tool_description() == eager.to_tool_description()
    assert postponed.describe() == eager.describe()
    assert "'int'" not in postponed.describe()


def test_the_served_tool_types_match_the_declared_annotations():
    """Every parameter's served JSON type is the mapping of its real annotation.

    Checked against ``_TYPE_JSON`` rather than a hand-written expectation, so this
    keeps holding if the mapping is extended, and fails if the derivation stops
    consulting the annotation at all.
    """
    from pixeldesk.codec_protocol import _TYPE_JSON, _resolved_signature

    for source in (POSTPONED_ACTION, EAGER_ACTION):
        action = _codec_with_postponed_annotations(source)
        (tool,) = ActionSet([action]).to_tool_description()
        properties = tool["parameters"]["properties"]
        signature = _resolved_signature(action)
        assert properties, "no parameters were derived at all"
        for name, parameter in signature.parameters.items():
            annotation = parameter.annotation
            assert not isinstance(annotation, str), f"{name} annotation left unresolved"
            assert properties[name]["type"] == _TYPE_JSON[annotation], name


def test_an_unresolvable_annotation_falls_back_instead_of_failing():
    """A forward reference this module cannot resolve must still describe itself."""
    action = _codec_with_postponed_annotations(
        "from __future__ import annotations\n"
        "def act(target: SomethingNotImportableHere, count: int = 1):\n"
        "    'Doc.'\n"
        "    return ()\n"
    )
    (tool,) = ActionSet([action]).to_tool_description()
    assert tool["parameters"]["properties"]["target"]["type"] == "string"
    assert "act(" in ActionSet([action]).describe()


def test_annotation_types_are_derived_when_evaluated_eagerly():
    """The derivation DOES work -- but only for a codec that does not postpone
    annotations, which is the narrow case.  Defined via ``exec`` because this
    test module itself postpones them."""
    namespace: dict = {}
    # ``dont_inherit`` matters: without it the compiled source inherits THIS
    # module's postponed-annotations flag and the test would prove nothing.
    exec(
        compile(
            "def act(count: int, flag: bool = False, label: str = 'x'):\n"
            "    'Doc.'\n"
            "    return ()\n",
            "<eager-codec>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    (tool,) = ActionSet([namespace["act"]]).to_tool_description()
    properties = tool["parameters"]["properties"]
    assert properties["count"]["type"] == "number"
    assert properties["flag"]["type"] == "boolean"
    assert properties["label"]["type"] == "string"


def test_a_codec_satisfies_the_runtime_checkable_protocol_structurally():
    """``isinstance`` against the protocol must be True for a real codec shape.

    ``stop_sequences`` is a class-level TUPLE and there is no ``handlers`` member:
    a codec compiles to ``Operation``s, it does not dispatch.  A protocol member
    no implementation can satisfy makes the gate read "not a codec" for every
    codec, which is worse than having no gate.
    """
    from pixeldesk.codec_protocol import Codec

    class Minimal:
        name = "minimal"
        stop_sequences: tuple[str, ...] = ()

        def parse(self, text):
            return text

        def format(self, action):
            return str(action)

        def compile(self, text, geometry, cursor):
            return ()

        def describe(self):
            return ""

    assert isinstance(Minimal(), Codec)


def test_the_protocol_declares_no_dispatch_member():
    """``handlers`` belongs to ``ActionSet``, not to the codec contract."""
    from pixeldesk.codec_protocol import Codec

    assert "handlers" not in getattr(Codec, "__protocol_attrs__", set())
    assert "stop_sequences" in getattr(Codec, "__protocol_attrs__", set())
    assert not callable(getattr(Codec, "stop_sequences", None))


def test_the_module_contains_no_action_name_of_its_own():
    """A grammar in the protocol module would defeat the whole seam."""
    import inspect

    from pixeldesk import codec_protocol

    source = inspect.getsource(codec_protocol)
    body = source.split('"""', 2)[-1]  # drop the module docstring
    for forbidden in ("moveRel", "pyautogui", "left_click", "hotkey"):
        assert forbidden not in body, forbidden
