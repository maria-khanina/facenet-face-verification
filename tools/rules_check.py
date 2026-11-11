import ast
from collections.abc import Callable
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
ALLOWED_NUMBERS = {0, 1, -1, 1.0, 2.0}
ABBREVIATIONS = {
    "impl", "spec", "specs", "cfg", "idx", "num", "val", "obj", "fn", "ctx",
    "tmp", "arg", "args", "kwargs", "rng", "msg", "res", "resp", "req", "ep",
    "arch", "pv", "fmt", "attr", "attrs", "param", "params", "info", "doc",
    "docs", "lib", "repo", "dir", "dirs", "cls", "func", "var", "vars", "len",
}
STDLIB = set(sys.stdlib_module_names)
MAX_STATEMENTS = 10
CONFIG_FILE_SUFFIX = "_settings.py"
SKIPPED_DIRECTORIES = {"typings", "__pycache__", ".venv", "venv"}
CONTAINER_NODES = (
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
)
CONTAINER_CALLS = {"list", "dict", "set", "defaultdict"}

Report = list[str]
Check = Callable[[pathlib.Path, ast.Module, Report], None]
ImportEntry = tuple[int, str, int]


def find_project_packages(root: pathlib.Path) -> set[str]:
    root = root.resolve()
    bases: list[pathlib.Path] = [root, root.parent, root / "src", root.parent / "src"]
    packages: set[str] = {
        marker.parent.name
        for base in bases
        for marker in base.glob("*/__init__.py")
    }
    if (root / "__init__.py").exists():
        packages.add(root.name)
    return packages


PROJECT_PACKAGES = find_project_packages(ROOT)


def count_statements(node: ast.AST) -> int:
    total = 0
    for child in ast.walk(node):
        if isinstance(child, ast.stmt) and child is not node:
            total += 1
    return total


def has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if not node.body or not isinstance(node.body[0], ast.Expr):
        return False
    value = node.body[0].value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def check_function_sizes(path: pathlib.Path, tree: ast.Module, report: Report) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            size = count_statements(node) - (1 if has_docstring(node) else 0)
            if size > MAX_STATEMENTS:
                report.append(f"{path}:{node.lineno} {node.name} has {size} statements")


def collect_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    return []


def check_name(location: str, name: str, report: Report) -> None:
    if name.startswith("__") and name.endswith("__"):
        return
    if len(name) == 1 and name != "_":
        report.append(f"{location} single-letter name {name}")
    bad = set(name.lower().split("_")) & ABBREVIATIONS
    if bad and name != "len":
        report.append(f"{location} abbreviation {sorted(bad)} in {name}")
    if name.lower().endswith(("spec", "specs")) and name not in {"spec", "specs"}:
        report.append(f"{location} abbreviation Spec in {name}")


def check_names(path: pathlib.Path, tree: ast.Module, report: Report) -> None:
    for node in ast.walk(tree):
        for name in collect_names(node):
            check_name(f"{path}:{getattr(node, 'lineno', 0)}", name, report)


def check_defaults(path: pathlib.Path, tree: ast.Module, report: Report) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            keyword_defaults: list[ast.expr] = [
                default for default in node.args.kw_defaults if default is not None
            ]
            if node.args.defaults or keyword_defaults:
                report.append(f"{path}:{node.lineno} default parameter in {node.name}")


def check_numbers(path: pathlib.Path, tree: ast.Module, report: Report) -> None:
    if path.name.endswith(CONFIG_FILE_SUFFIX):
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        if isinstance(node.value, (int, float)) and node.value not in ALLOWED_NUMBERS:
            report.append(f"{path}:{node.lineno} number {node.value}")


def module_group(name: str) -> int:
    root = name.split(".")[0]
    if root in STDLIB:
        return 0
    if root in PROJECT_PACKAGES:
        return 2
    return 1


def collect_imports(tree: ast.Module) -> list[ImportEntry]:
    entries: list[ImportEntry] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            name = node.names[0].name
            entries.append((module_group(name), name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            entries.append((module_group(name), name, node.lineno))
    return entries


def sort_key_of_imported_name(name: str) -> tuple[bool, bool, str]:
    return not name.isupper(), name[0].islower(), name


def check_imports(path: pathlib.Path, tree: ast.Module, report: Report) -> None:
    entries = collect_imports(tree)
    for first, second in zip(entries, entries[1:]):
        if (first[0], first[1].lower()) > (second[0], second[1].lower()):
            order = f"{first[1]} before {second[1]}"
            report.append(f"{path}:{second[2]} import order {order}")
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names: list[str] = [alias.name for alias in node.names]
            if names != sorted(names, key=sort_key_of_imported_name):
                report.append(f"{path}:{node.lineno} import names order {names}")


def is_container_value(value: ast.expr) -> bool:
    if isinstance(value, CONTAINER_NODES):
        return True
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value.func.id in CONTAINER_CALLS
    return False


def check_container_hints(path: pathlib.Path, tree: ast.Module, report: Report) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or node in tree.body:
            continue
        if is_container_value(node.value):
            message = f"container without type hint: {target.id}"
            report.append(f"{path}:{node.lineno} {message}")


CHECKS: list[Check] = [
    check_function_sizes,
    check_names,
    check_defaults,
    check_numbers,
    check_imports,
    check_container_hints,
]


def is_skipped(path: pathlib.Path) -> bool:
    if SKIPPED_DIRECTORIES & set(path.parts):
        return True
    return path.name == pathlib.Path(__file__).name


def main() -> None:
    report: Report = []
    for path in sorted(ROOT.rglob("*.py")):
        if is_skipped(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for check in CHECKS:
            check(path.relative_to(ROOT), tree, report)
    print("\n".join(report))
    print(len(report), "findings")


main()
