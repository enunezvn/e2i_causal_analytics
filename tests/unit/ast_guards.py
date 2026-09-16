"""AST guards shared by the tests that keep caught-exception text out of authored errors (#2020).

:func:`caught_exception_interpolations` finds each ``raise`` of a target class, inside an
``except … as NAME`` handler, whose arguments render the exception that handler caught. It is a
helper module, not a test module, so more than one guard can import it.

What counts as rendering: any read of ``NAME`` in the raised call — an f-string field, ``str()``
/ ``repr()``, ``%``, ``.format()``, ``NAME.args`` — and any read of a name the handler derived
from it: an assignment (plain, annotated, augmented or walrus), a ``for`` target over it, a
``with … as`` target, a nested function whose body reads it, or a prebuilt instance of a target
class raised by name (``err = PlanningError(f"{e}"); raise err``). ``raise … from NAME`` does not
count: the cause is kept for the log, not rendered. Neither do ``type(NAME)``,
``type(NAME).__name__``, ``NAME.__class__.__name__`` or ``isinstance(NAME, …)``, which name or
test the class and carry no text.

Limits, deliberately:

* Only handlers that bind a name. Nameless forms that read the active exception —
  ``traceback.format_exc()``, ``sys.exc_info()`` — are out of scope.
* Each handler is analysed on its own. The walk of a handler body does not enter a nested
  ``except`` clause (that clause is visited separately with only its own name tainted), so a
  nested handler that renders the OUTER exception is not seen — whether the nested clause binds
  a name of its own or is nameless (``except ValueError: raise X(str(e))``). A nested clause's
  assignments still add taint for the statements after its ``try``.
* Taint flows through bindings only. A mutation through a method call or an attribute store
  (``parts.append(str(e))``, ``err.args = …``, ``d.update(m=str(e))``) does not taint the
  receiver.
* Rebinding is flow-sensitive but path-insensitive. An assignment of an untainted value clears
  the name for the statements after it; branches merge by union and loops run to a fixpoint, so
  a rebinding on one branch only never clears the name.
* Calls are not followed, except a function defined inside the handler: its body is checked
  with the handler's names, and calling it renders what its body reads.
"""

from __future__ import annotations

import ast
from typing import Callable, Dict, FrozenSet, Iterable, Iterator, List, NamedTuple, Optional, Tuple

_STMT_LIST_FIELDS = frozenset({"body", "orelse", "finalbody", "handlers", "cases"})


class CaughtInterpolation(NamedTuple):
    """One raise that renders a caught exception."""

    line: int
    function: Optional[str]  # innermost enclosing def of the handler; None at module level
    caught_types: Tuple[str, ...]  # the handler's ``except`` types; () for a bare except
    raised: str
    name: str  # the handler's ``as`` name


class _Taint(NamedTuple):
    names: FrozenSet[str]
    instances: FrozenSet[Tuple[str, str]]  # (name, target class) of prebuilt instances

    def __or__(self, other: "_Taint") -> "_Taint":
        return _Taint(self.names | other.names, self.instances | other.instances)

    def gen(self, names: Iterable[str]) -> "_Taint":
        return _Taint(self.names | frozenset(names), self.instances)

    def kill(self, names: Iterable[str]) -> "_Taint":
        dropped = frozenset(names)
        return _Taint(
            self.names - dropped,
            frozenset(pair for pair in self.instances if pair[0] not in dropped),
        )

    def instance_class(self, name: str) -> Optional[str]:
        return next((cls for bound, cls in sorted(self.instances) if bound == name), None)


def _handler_caught_types(handler: ast.ExceptHandler) -> Tuple[str, ...]:
    """The class names an ``except`` clause catches, in source order; ``()`` for a bare except."""
    if handler.type is None:
        return ()
    elements = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return tuple(_class_name(element) or ast.unparse(element) for element in elements)


def _class_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_class_introspection(node: ast.AST) -> bool:
    """``type(N)`` and ``N.__class__`` name the class and ``isinstance(N, …)`` tests it; none
    renders the exception's text."""
    if isinstance(node, ast.Attribute) and node.attr == "__class__":
        return True
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords):
        return False
    return (node.func.id == "type" and len(node.args) == 1) or (
        node.func.id == "isinstance" and len(node.args) == 2
    )


def _read_names(node: ast.AST) -> Iterator[str]:
    stack = [node]
    while stack:
        current = stack.pop()
        if _is_class_introspection(current) or isinstance(current, ast.ExceptHandler):
            continue
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
            yield current.id
        stack.extend(ast.iter_child_nodes(current))


def _target_names(target: ast.AST, *, rebinding_only: bool) -> List[str]:
    """Names an assignment target binds.

    With ``rebinding_only`` just the names the target REPLACES (``x`` / ``x, y`` / ``*x``), so an
    untainted ``d["k"] = 1`` does not clear ``d``; otherwise every name in the target, so a
    tainted ``d["k"] = str(e)`` taints ``d``.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _target_names(target.value, rebinding_only=rebinding_only)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt, rebinding_only=rebinding_only)]
    if rebinding_only:
        return []
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


def _header_nodes(stmt: ast.stmt) -> Iterator[ast.AST]:
    """The parts of a statement evaluated before (or instead of) its nested statement lists."""
    for field, value in ast.iter_fields(stmt):
        if field in _STMT_LIST_FIELDS:
            continue
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, ast.AST):
                yield item


class _HandlerScan:
    def __init__(self, targets: FrozenSet[str]) -> None:
        self.targets = targets
        self.found: Dict[int, str] = {}

    def tainted(self, node: ast.AST, state: _Taint) -> bool:
        return any(name in state.names for name in _read_names(node))

    def block(self, stmts: List[ast.stmt], state: _Taint) -> _Taint:
        for stmt in stmts:
            state = self.stmt(stmt, state)
        return state

    def stmt(self, stmt: ast.stmt, state: _Taint) -> _Taint:
        for header in _header_nodes(stmt):
            state = self._walrus(header, state)
        if isinstance(stmt, ast.Raise):
            self._check_raise(stmt, state)
            return state
        if isinstance(stmt, ast.Assign):
            return self._bind(stmt.targets, stmt.value, state)
        if isinstance(stmt, ast.AnnAssign):
            return state if stmt.value is None else self._bind([stmt.target], stmt.value, state)
        if isinstance(stmt, ast.AugAssign):
            if self.tainted(stmt.value, state):
                return state.gen(_target_names(stmt.target, rebinding_only=False))
            return state
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            loop_target, loop_iter = stmt.target, stmt.iter

            def bind_target(s: _Taint) -> _Taint:
                if self.tainted(loop_iter, s):
                    return s.gen(_target_names(loop_target, rebinding_only=False))
                return s

            return self._loop(stmt.body, stmt.orelse, state, bind_target)
        if isinstance(stmt, ast.While):
            return self._loop(stmt.body, stmt.orelse, state, lambda s: s)
        if isinstance(stmt, ast.If):
            return self.block(stmt.body, state) | self.block(stmt.orelse, state)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None and self.tainted(item.context_expr, state):
                    state = state.gen(_target_names(item.optional_vars, rebinding_only=False))
            return self.block(stmt.body, state)
        if isinstance(stmt, (ast.Try, ast.TryStar)):
            body = self.block(stmt.body, state)
            after = self.block(stmt.orelse, body)
            # A nested ``except`` is visited on its own; only its bindings are carried here.
            merged = state | body | after
            for handler in stmt.handlers:
                merged = merged | self._bindings_only(handler.body, merged)
            return self.block(stmt.finalbody, merged)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in ast.walk(stmt.args) if isinstance(a, ast.arg)]
            inner = state.kill(params)
            self.block(stmt.body, inner)
            if any(self.tainted(node, inner) for node in stmt.body):
                return state.gen([stmt.name])
            return state.kill([stmt.name])
        # Match, ClassDef and anything else with nested statements: bindings only add taint.
        nested = [
            child
            for field, value in ast.iter_fields(stmt)
            if field in _STMT_LIST_FIELDS
            for child in value
        ]
        return state | self._bindings_only(nested, state)

    def _walrus(self, node: ast.AST, state: _Taint) -> _Taint:
        walruses = [n for n in ast.walk(node) if isinstance(n, ast.NamedExpr)]
        grew = True
        while grew:
            grew = False
            for walrus in walruses:
                name = walrus.target.id
                if name not in state.names and self.tainted(walrus.value, state):
                    state, grew = state.gen([name]), True
        return state

    def _bind(self, targets: List[ast.expr], value: ast.expr, state: _Taint) -> _Taint:
        replaced = [n for t in targets for n in _target_names(t, rebinding_only=True)]
        tainted = self.tainted(value, state)
        aliased = state.instance_class(value.id) if isinstance(value, ast.Name) else None
        built = _class_name(value.func) if isinstance(value, ast.Call) else None
        state = state.kill(replaced)
        if tainted:
            state = state.gen(n for t in targets for n in _target_names(t, rebinding_only=False))
        instance = aliased or (built if tainted and built in self.targets else None)
        if instance is not None:
            state = _Taint(state.names, state.instances | {(name, instance) for name in replaced})
        return state

    def _check_raise(self, stmt: ast.Raise, state: _Taint) -> None:
        exc = stmt.exc
        if isinstance(exc, ast.Call):
            raised = _class_name(exc.func)
            if raised in self.targets and self.tainted(exc, state):
                assert raised is not None
                self.found[stmt.lineno] = raised
        elif isinstance(exc, ast.Name):
            prebuilt = state.instance_class(exc.id)
            if prebuilt is not None:
                self.found[stmt.lineno] = prebuilt

    def _loop(
        self,
        body: List[ast.stmt],
        orelse: List[ast.stmt],
        state: _Taint,
        head: Callable[[_Taint], _Taint],
    ) -> _Taint:
        current = head(state)
        while True:
            widened = head(current | self.block(body, current))
            if widened == current:
                break
            current = widened
        return current | self.block(orelse, current)

    def _bindings_only(self, stmts: List[ast.stmt], state: _Taint) -> _Taint:
        """Taint added by any binding in ``stmts``, iterated to a fixpoint; no raise is checked."""
        nodes = [n for stmt in stmts for n in ast.walk(stmt)]
        grew = True
        while grew:
            grew = False
            for node in nodes:
                names: List[str] = []
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    if node.value is not None and self.tainted(node.value, state):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        names = [n for t in targets for n in _target_names(t, rebinding_only=False)]
                elif isinstance(node, ast.NamedExpr) and self.tainted(node.value, state):
                    names = [node.target.id]
                elif isinstance(node, (ast.For, ast.AsyncFor)) and self.tainted(node.iter, state):
                    names = _target_names(node.target, rebinding_only=False)
                new = [n for n in names if n not in state.names]
                if new:
                    state, grew = state.gen(new), True
        return state


class _Visitor(ast.NodeVisitor):
    def __init__(self, targets: FrozenSet[str]) -> None:
        self.targets = targets
        self.functions: List[str] = []
        self.found: List[CaughtInterpolation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            scan = _HandlerScan(self.targets)
            scan.block(node.body, _Taint(frozenset({node.name}), frozenset()))
            function = self.functions[-1] if self.functions else None
            caught = _handler_caught_types(node)
            self.found.extend(
                CaughtInterpolation(line, function, caught, raised, node.name)
                for line, raised in sorted(scan.found.items())
            )
        self.generic_visit(node)


def caught_exception_interpolations(
    tree: ast.AST, targets: FrozenSet[str]
) -> List[CaughtInterpolation]:
    """Every raise of a ``targets`` class that renders the exception its handler caught.

    See the module docstring for what counts as rendering and for the analysis's limits.
    """
    visitor = _Visitor(frozenset(targets))
    visitor.visit(tree)
    return sorted(visitor.found)
