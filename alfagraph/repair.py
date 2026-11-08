"""Small, test-guided AST mutation search; no function-name/answer lookup tables."""

from __future__ import annotations

import ast
import copy

from alfagraph.execution import is_test, run_tests
from alfagraph.tasks import Attempt


def variants(source: str):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    nodes = list(ast.walk(tree))
    changes = []
    for index, node in enumerate(nodes):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            changes.append((index, "op", ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()))
        if isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
            changes.append((index, "op", ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            inverse = {
                ast.Lt: ast.Gt,
                ast.Gt: ast.Lt,
                ast.LtE: ast.GtE,
                ast.GtE: ast.LtE,
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
            }
            if type(node.ops[0]) in inverse:
                changes.append((index, "ops", [inverse[type(node.ops[0])]()]))
            if (
                isinstance(node.left, ast.BinOp)
                and isinstance(node.left.op, ast.Mod)
                and isinstance(node.left.right, ast.Constant)
                and node.left.right.value == 2
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value in (0, 1)
            ):
                changes.append(
                    (index, "comparators", [ast.Constant(1 - node.comparators[0].value)])
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
            and len(node.args) == 1
            and not node.keywords
        ):
            changes.append((index, "func", ast.Name(id="reversed", ctx=ast.Load())))
    for index, attribute, value in changes:
        candidate = copy.deepcopy(tree)
        setattr(list(ast.walk(candidate))[index], attribute, value)
        ast.fix_missing_locations(candidate)
        yield ast.unparse(candidate) + "\n"


async def repair(attempt: Attempt, max_candidates: int = 24) -> bool:
    baseline = await run_tests(attempt)
    if baseline["ok"]:
        return True
    best_score = baseline["failures"] + baseline["errors"]
    files = sorted(
        p
        for p in attempt.work.rglob("*.py")
        if not is_test(p.relative_to(attempt.work)) and p.stat().st_size < 20_000
    )
    tried = 0
    seen = set()
    for _round in range(3):
        improved = False
        for path in files:
            best = path.read_text()
            for candidate in variants(best):
                fingerprint = (str(path), candidate)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                tried += 1
                if tried > max_candidates:
                    return False
                path.write_text(candidate)
                try:
                    result = await run_tests(attempt)
                finally:
                    path.write_text(best)
                score = result["failures"] + result["errors"]
                if result["ok"]:
                    path.write_text(candidate)
                    return True
                if score < best_score:
                    best, best_score = candidate, score
                    path.write_text(best)
                    improved = True
                    break
        if not improved:
            break
    return False
