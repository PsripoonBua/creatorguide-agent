"""Arithmetic evaluator for the `calculator` tool. No eval(), no names, no attribute access.

Only numbers, + - * / // % **, parentheses, unary +/-, and a few pure functions are allowed.
"""
from __future__ import annotations

import ast
import math
import operator
import re

MAX_LEN = 200
MAX_EXPONENT = 100
MAX_ABS = 1e15

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "min": min, "max": max, "round": round, "ceil": math.ceil, "floor": math.floor}
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # "3,200" -> "3200"


class MathError(ValueError):
    pass


def evaluate(expression: str) -> float | int:
    if not isinstance(expression, str) or not expression.strip():
        raise MathError("expression must be a non-empty string")
    if len(expression) > MAX_LEN:
        raise MathError(f"expression longer than {MAX_LEN} characters")
    cleaned = _THOUSANDS.sub("", expression.replace("_", ""))
    try:
        tree = ast.parse(cleaned.strip(), mode="eval")
    except SyntaxError as exc:
        raise MathError(f"not a valid arithmetic expression: {exc.msg}") from None
    value = _eval(tree.body)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise MathError("result is not a finite number")
    return value


def _check(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MathError("only numbers are allowed")
    if abs(value) > MAX_ABS:
        raise MathError("number too large")
    return value


def _eval(node):
    if isinstance(node, ast.Constant):
        return _check(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check(_UNARY[type(node.op)](_eval(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise MathError("exponent too large")
        try:
            return _check(_BIN[type(node.op)](left, right))
        except ZeroDivisionError:
            raise MathError("division by zero") from None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and len(node.args) <= 8):
        return _check(_FUNCS[node.func.id](*[_eval(a) for a in node.args]))
    raise MathError("unsupported syntax (numbers and + - * / // % ** ( ) only)")


def format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.6g}" if isinstance(value, float) else str(value)
