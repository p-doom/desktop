"""The codec seam: the *only* place a grammar is allowed to exist.

This module defines the contract, not any grammar.  A codec lives outside this
package -- in whatever repository owns the model being trained -- and reaches
this package through exactly one function:

    codec.compile(text, geometry, cursor) -> tuple[Operation, ...]

Everything grammar-shaped happens on the left of that arrow.  Absolute pixels
come out on the right.  The resolution context (screen geometry, current cursor)
arrives as *data* in the call, so a codec never has to be configured into a mode
and this package never has to know which mode it is in.

The action-set skeleton below is vendored (~250 LOC) from BrowserGym's
``browsergym/core/action/`` (ServiceNow, Apache-2.0): declare the grammar once,
as Python functions whose *docstrings are the spec*, and derive prompt text,
examples, validation, lowering, and tool-JSON from that single source.  It is
vendored rather than imported because BrowserGym's leaf is Playwright-bound and
its ``main`` has been dormant since 2026-03-17.  Two substitutions were required
to hold the zero-dependency floor:

  * ``pyparsing``'s PEG for function calls -> ``ast.parse`` in eval mode, which
    accepts strictly the same literal-argument call syntax and rejects more.
  * ``pyparsing``'s docstring parser -> a plain ``Examples:`` section split.

Three of BrowserGym's gaps are exactly what a caller of *this* package must fill
and are therefore left as required protocol members rather than provided:
there is no inverse renderer here (``format`` is yours), actions are typed
``Operation``s and not strings, and coordinates are whatever your codec resolves
them to.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from .geometry import DisplayGeometry
from .ir import Operation

# A handler turns one parsed call's arguments into resolved operations.  This is
# ``ActionSet``'s dispatch table below and nothing else: it is how a *declared
# action set* lowers ``name(args)`` into ``Operation``s without a ``match`` on an
# action name anywhere in shared code.
#
# IT IS NOT PART OF THE ``Codec`` PROTOCOL, deliberately.  A codec does not
# dispatch; it ``compile``s to ``Operation``s, and the Operation vocabulary is
# closed by physics (a pointer moves, a button goes down, a wheel turns) rather
# than open per grammar.  Lowering an ``Operation`` is therefore a literal
# ``if kind ==`` chain in ``execute/guest_program.py``, which is correct: the set
# it switches over is fixed and shared, not grammar-specific.  Requiring
# ``handlers`` on a ``Codec`` described a second dispatch engine that does not
# exist, and made ``isinstance(codec, Codec)`` false for every real codec.
Handler = Callable[..., tuple[Operation, ...]]


@runtime_checkable
class Codec(Protocol):
    """One grammar, in both directions, plus its own self-description.

    ``@runtime_checkable`` makes ``isinstance(codec, Codec)`` a legal gate, so
    every member here must be one a real codec actually exposes, in the shape it
    exposes it.  A protocol member that no implementation can satisfy is worse
    than no protocol: the gate reads as "this is not a codec" for every codec.
    """

    name: str

    #: Decoder stop strings this grammar needs to terminate cleanly.  An
    #: ATTRIBUTE, not a method: it is a fixed property of the grammar's surface
    #: syntax, decided when the grammar is written, and every codec spells it as
    #: a class-level tuple.  Empty is the common case -- a grammar whose action
    #: line may be preceded by reasoning has no token sequence that ends a turn.
    stop_sequences: tuple[str, ...]

    def parse(self, text: str) -> object:
        """Model output text -> a structured action object of the codec's choice."""
        ...

    def format(self, action: object) -> str:
        """A structured action -> model output text.  The inverse of ``parse``.

        This direction is what generates supervised training targets, and it is
        the member most action libraries omit.
        """
        ...

    def compile(
        self, text: str, geometry: DisplayGeometry, cursor: tuple[int, int]
    ) -> tuple[Operation, ...]:
        """Model output text -> operations in ABSOLUTE SCREEN PIXELS.

        Every coordinate convention -- absolute, relative-to-cursor, normalized,
        downscaled -- is fully resolved here, using ``geometry`` and ``cursor``
        as data.  Downstream code sees only pixels.
        """
        ...

    def describe(self) -> str:
        """The system prompt for this grammar, derived from the docstrings."""
        ...


# --------------------------------------------------------------------------- #
# Docstring-as-single-source skeleton (vendored from BrowserGym)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActionSpec:
    """One action, as derived from its Python function."""

    name: str
    signature: str
    description: str
    examples: tuple[str, ...]
    parameters: dict[str, Any]


def parse_action_docstring(doc: str | None) -> tuple[str, tuple[str, ...]]:
    """Split a docstring into prose and its ``Examples:`` block.

    The convention is BrowserGym's: free prose, then a literal ``Examples:``
    line, then one call per line.  Both halves are optional.
    """
    if not doc:
        return "", ()
    head, _, tail = doc.partition("Examples:")
    description = " ".join(head.split())
    examples = tuple(line.strip() for line in tail.splitlines() if line.strip())
    return description, examples


_TYPE_JSON = {
    str: "string",
    float: "number",
    int: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
    tuple: "array",
}


def _resolved_signature(func: Callable[..., Any]) -> inspect.Signature:
    """A signature whose annotations are OBJECTS, not strings.

    ``eval_str=True`` is load-bearing, not tidiness.  Under ``from __future__
    import annotations`` -- which every module in this package uses, and which a
    codec module will therefore almost certainly use too -- ``__annotations__``
    holds the *source text* of each annotation.  Without evaluating it:

      * ``_TYPE_JSON`` looks up the string ``"int"``, misses, and falls back, so
        EVERY parameter is typed ``"string"`` in the tool JSON a model is served;
      * the rendered signature reads ``move_rel(dx: 'int', dy: 'int')`` -- with the
        quotes -- and that string goes straight into ``describe()``, which IS the
        system prompt.

    So the grammar's self-description silently disagreed with the grammar.  The
    fallback keeps a codec whose annotations cannot be resolved in this scope
    (a forward reference to something not importable here) working exactly as
    before rather than failing to describe itself at all.
    """
    try:
        return inspect.signature(func, eval_str=True)
    except (NameError, TypeError, AttributeError):
        return inspect.signature(func)


def _json_parameters(func: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON-Schema object from a function's own signature."""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    for name, param in _resolved_signature(func).parameters.items():
        if name in {"self", "geometry", "cursor"}:
            continue
        json_type = "string"
        if param.annotation is not inspect.Parameter.empty:
            json_type = _TYPE_JSON.get(param.annotation, "string")
        entry: dict[str, Any] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            parameters["required"].append(name)
        else:
            entry["default"] = param.default
        parameters["properties"][name] = entry
    return parameters


def action_spec(func: Callable[..., Any]) -> ActionSpec:
    """Derive an ``ActionSpec`` from one action function -- the single source."""
    description, examples = parse_action_docstring(func.__doc__)
    return ActionSpec(
        name=func.__name__,
        signature=f"{func.__name__}{_resolved_signature(func)}",
        description=description,
        examples=examples,
        parameters=_json_parameters(func),
    )


class UnknownActionError(ValueError):
    """A call named an action outside the declared set."""


class MultiActionError(ValueError):
    """Several calls arrived where the set permits only one."""


@dataclass
class ParsedCall:
    """One parsed action call: a name plus literal positional/keyword args."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        parts = [repr(value) for value in self.args]
        parts += [f"{key}={value!r}" for key, value in self.kwargs.items()]
        return f"{self.name}({', '.join(parts)})"


def parse_calls(text: str, *, strict: bool = False) -> list[ParsedCall]:
    """Parse ``name(literal, key=literal)`` calls out of model output.

    Replaces BrowserGym's pyparsing PEG with ``ast`` so the package stays
    dependency-free.  Arguments must be Python literals: ``ast.literal_eval``
    does the evaluating, so no name lookup, attribute access, or call can occur
    inside an argument.

    ``strict`` requires that the whole text be nothing but calls; otherwise
    non-call lines are skipped, which is what a chatty model needs.

    A line may carry MORE THAN ONE call -- ``click('left'); move_to(1, 2)`` -- and
    all of them are returned.  Parsing was in ``eval`` mode, which accepts exactly
    one expression, so such a line was a ``SyntaxError``: strict mode said so, but
    non-strict mode could not tell it apart from prose and skipped it, turning two
    actions the model emitted into ZERO with nothing logged.  ``exec`` mode plus a
    walk over the statements keeps every other rejection identical -- an argument
    is still evaluated by ``literal_eval`` alone, and anything that is not a bare
    call is still refused -- while making a semicolon separator mean what it says.
    """
    calls: list[ParsedCall] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            statements = ast.parse(stripped, mode="exec").body
        except SyntaxError:
            if strict:
                raise ValueError(f"not a parsable action line: {line!r}") from None
            continue
        for statement in statements:
            node = statement.value if isinstance(statement, ast.Expr) else None
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                if strict:
                    raise ValueError(f"not a bare function call: {line!r}")
                continue
            try:
                # ``**mapping`` arrives as a keyword whose ``arg`` is None.
                # Skipping those SILENTLY DISCARDS the arguments and yields a call
                # that looks well-formed with no arguments at all, so it is
                # rejected like any other non-literal argument instead.
                if any(kw.arg is None for kw in node.keywords):
                    raise ValueError("** unpacking is not a literal argument")
                args = tuple(ast.literal_eval(arg) for arg in node.args)
                kwargs = {
                    str(kw.arg): ast.literal_eval(kw.value) for kw in node.keywords
                }
            except ValueError:
                if strict:
                    raise ValueError(f"non-literal argument in {line!r}") from None
                continue
            calls.append(ParsedCall(node.func.id, args, kwargs))
    if strict and not calls:
        raise ValueError("received an empty action")
    return calls


class ActionSet:
    """A named, describable, validating set of action functions.

    Construct it from action functions -- or from named subsets of them, which is
    BrowserGym's "one grammar, many dialects" idea: the same declarations, served
    to different models as different action spaces.
    """

    def __init__(
        self,
        actions: list[Callable[..., Any]] | None = None,
        *,
        subsets: dict[str, list[Callable[..., Any]]] | None = None,
        names: list[str] | None = None,
        multiaction: bool = True,
        strict: bool = False,
    ) -> None:
        chosen: list[Callable[..., Any]] = list(actions or [])
        if names:
            table = subsets or {}
            for subset in names:
                if subset not in table:
                    raise ValueError(f"unknown action subset: {subset}")
                chosen.extend(table[subset])
        if not chosen:
            raise ValueError("an action set needs at least one action")
        # dict.fromkeys de-duplicates while preserving declaration order.
        self.functions: dict[str, Callable[..., Any]] = {
            func.__name__: func for func in dict.fromkeys(chosen)
        }
        self.specs: dict[str, ActionSpec] = {
            name: action_spec(func) for name, func in self.functions.items()
        }
        self.multiaction = multiaction
        self.strict = strict

    @property
    def handlers(self) -> dict[str, Handler]:
        """The dispatch table a ``Codec`` exposes.  No ``match`` anywhere."""
        return dict(self.functions)

    def validate(self, calls: list[ParsedCall]) -> list[ParsedCall]:
        """Reject unknown names, and multi-actions when they are not allowed."""
        if not calls:
            raise ValueError("received an empty action")
        if len(calls) > 1 and not self.multiaction:
            raise MultiActionError(
                f"received {len(calls)} calls; this action set allows one"
            )
        for call in calls:
            if call.name not in self.functions:
                raise UnknownActionError(f"invalid action type {call.name!r}")
        return calls

    def lower(
        self,
        calls: list[ParsedCall],
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Run the validated calls through their handlers into operations.

        Handlers that accept ``geometry`` and/or ``cursor`` are given them; that
        is how a relative or normalized grammar gets its resolution context, as
        data, without any coordinate-space flag existing anywhere.
        """
        operations: list[Operation] = []
        for call in self.validate(calls):
            func = self.functions[call.name]
            accepted = inspect.signature(func).parameters
            extra: dict[str, Any] = {}
            if "geometry" in accepted:
                extra["geometry"] = geometry
            if "cursor" in accepted:
                extra["cursor"] = cursor
            produced = func(*call.args, **call.kwargs, **extra)
            operations.extend(produced or ())
            for operation in produced or ():
                # Every kind that MOVES the pointer must advance the threaded
                # cursor, or the next relative call in the same action resolves
                # against a stale position.  ``glide_to`` lands the cursor exactly
                # like ``move_to`` -- it only takes longer getting there.
                if operation.kind in {"move_to", "glide_to"}:
                    cursor = (int(operation.args[0]), int(operation.args[1]))
                elif operation.kind == "drag":
                    cursor = (int(operation.args[2]), int(operation.args[3]))
        return tuple(operations)

    def example_action(self, *, max_examples: int = 3) -> str:
        picked: list[str] = []
        for spec in self.specs.values():
            picked.extend(spec.examples)
        if not picked:
            return ""
        return "\n".join(picked[:max_examples]) if self.multiaction else picked[0]

    def describe(self, *, with_long_description: bool = True, with_examples: bool = True) -> str:
        """The system prompt, assembled from the declarations themselves."""
        lines = [f"{len(self.specs)} different types of actions are available.", ""]
        for spec in self.specs.values():
            lines.append(spec.signature)
            if with_long_description and spec.description:
                lines.append(f"    Description: {spec.description}")
            if with_examples and spec.examples:
                lines.append("    Examples:")
                lines.extend(f"        {example}" for example in spec.examples)
            lines.append("")
        lines.append(
            "Multiple actions can be provided at once, one per line, and will be "
            "executed sequentially without any feedback in between."
            if self.multiaction
            else "Only a single action can be provided at once."
        )
        example = self.example_action()
        if example:
            lines += ["Example:", example]
        return "\n".join(lines) + "\n"

    def to_tool_description(self, *, api: str = "openai", add_examples: bool = True) -> list[dict]:
        """Tool-JSON for the OpenAI or Anthropic wire format.

        An unrecognised ``api`` is an error, not the OpenAI shape.  It used to
        fall back to the ``parameters`` key while skipping ``"type": "function"``,
        which is neither format: a typo produced a THIRD wire shape, served to a
        model, with nothing anywhere saying the requested API was not understood.
        """
        schema_keys = {"openai": "parameters", "anthropic": "input_schema"}
        if api not in schema_keys:
            raise ValueError(
                f"unsupported tool API {api!r}; expected one of {sorted(schema_keys)}"
            )
        schema_key = schema_keys[api]
        tools: list[dict] = []
        for spec in self.specs.values():
            description = spec.description
            if add_examples and spec.examples:
                description += "\n\nExamples:\n" + "\n".join(
                    f"- {example}" for example in spec.examples
                )
            tool: dict[str, Any] = {
                "name": spec.name,
                "description": description,
                schema_key: spec.parameters,
            }
            if api == "openai":
                tool["type"] = "function"
            tools.append(tool)
        return tools
