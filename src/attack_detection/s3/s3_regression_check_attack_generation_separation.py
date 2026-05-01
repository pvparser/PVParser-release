#!/usr/bin/env python3
"""
Lightweight regression checks for attack4 generator separation.

This script is intentionally static and fast:
- verify attack4 control generator no longer cross-imports attack2
- verify attack4 entry point exposes ``pcap_output_window_mode``
- verify attack4 entry point exposes ``training_raw_folder_names``
- verify attack4 entry point exposes ``test_base_raw_folder_names``
- verify attack4 supervisory traffic defaults to ``csv_time_range``
- verify attack4 historian metadata keeps the same ``template_module`` schema
- verify attack4 historian manipulated-PV metadata matches replacement semantics
- verify supervisory historian generator writes unified attack injection logs
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


NAMESPACE_ROOT = Path(__file__).resolve().parent
NAMESPACE = NAMESPACE_ROOT.name

CONTROL_ATTACK4 = NAMESPACE_ROOT / "control_attack" / "s3_attack4_generation.py"
TRAFFIC_ATTACK4 = NAMESPACE_ROOT / "supervisory_traffic_attack" / "s3_attack4_generation.py"
HISTORIAN_ATTACK4 = NAMESPACE_ROOT / "supervisory_historian_attack" / "s3_attack4_generation.py"


def _load_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _function_arg_names(tree: ast.AST, function_name: str) -> list[str]:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            arg_names = [arg.arg for arg in node.args.args]
            arg_names.extend(arg.arg for arg in node.args.kwonlyargs)
            if node.args.vararg is not None:
                arg_names.append(node.args.vararg.arg)
            if node.args.kwarg is not None:
                arg_names.append(node.args.kwarg.arg)
            return arg_names
    raise AssertionError(f"Function {function_name!r} not found.")


def _defined_function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _function_default_value(tree: ast.AST, function_name: str, arg_name: str) -> object:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            positional_args = list(node.args.posonlyargs) + list(node.args.args)
            defaults_by_name: dict[str, object] = {}
            if node.args.defaults:
                for arg, default in zip(positional_args[-len(node.args.defaults) :], node.args.defaults):
                    defaults_by_name[arg.arg] = ast.literal_eval(default)
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None:
                    defaults_by_name[arg.arg] = ast.literal_eval(default)
            if arg_name not in defaults_by_name:
                raise AssertionError(
                    f"Argument {arg_name!r} on function {function_name!r} does not have a default."
                )
            return defaults_by_name[arg_name]
    raise AssertionError(f"Function {function_name!r} not found.")


def _assert_contains_all(text: str, required_snippets: Iterable[str], *, label: str) -> None:
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        raise AssertionError(f"{label} is missing required snippets: {missing}")


def main() -> None:
    for path in (
        CONTROL_ATTACK4,
        TRAFFIC_ATTACK4,
        HISTORIAN_ATTACK4,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")

    control_attack4_tree = _load_ast(CONTROL_ATTACK4)
    traffic_attack4_tree = _load_ast(TRAFFIC_ATTACK4)

    control_attack4_imports = _import_modules(control_attack4_tree)

    attack4_traffic_module = f"attack_detection.{NAMESPACE}.supervisory_traffic_attack.s3_attack4_generation"
    stale_attack2_imports = sorted(
        module
        for module in control_attack4_imports
        if "control_attack" in module and "attack2" in module
    )
    if stale_attack2_imports:
        raise AssertionError(
            "s3 control generator still imports attack2 control helpers: "
            f"{stale_attack2_imports}"
        )
    if attack4_traffic_module not in control_attack4_imports:
        raise AssertionError("s3 control generator does not import its dedicated s3 supervisory traffic module.")

    attack4_args = _function_arg_names(control_attack4_tree, "inject_historian_attack4_into_control_pcaps")
    if "pcap_output_window_mode" not in attack4_args:
        raise AssertionError("attack4 control entry point is missing pcap_output_window_mode.")
    if "training_raw_folder_names" not in attack4_args:
        raise AssertionError("attack4 control entry point is missing training_raw_folder_names.")
    if "test_base_raw_folder_names" not in attack4_args:
        raise AssertionError("attack4 control entry point is missing test_base_raw_folder_names.")

    traffic_attack4_functions = _defined_function_names(traffic_attack4_tree)
    if "build_attack4_traffic" not in traffic_attack4_functions:
        raise AssertionError("attack4 supervisory traffic module is missing build_attack4_traffic.")
    if "build_supervisory_attack4_pv_specs" not in traffic_attack4_functions:
        raise AssertionError("attack4 supervisory traffic module is missing build_supervisory_attack4_pv_specs.")
    if "build_attack2_traffic" in traffic_attack4_functions:
        raise AssertionError("attack4 supervisory traffic module still defines build_attack2_traffic.")
    if "build_supervisory_attack2_pv_specs" in traffic_attack4_functions:
        raise AssertionError("attack4 supervisory traffic module still defines build_supervisory_attack2_pv_specs.")
    if (
        _function_default_value(
            traffic_attack4_tree,
            "build_attack4_traffic",
            "pcap_output_window_mode",
        )
        != "csv_time_range"
    ):
        raise AssertionError(
            "attack4 supervisory traffic default pcap_output_window_mode is not csv_time_range."
        )

    historian_attack4_text = HISTORIAN_ATTACK4.read_text(encoding="utf-8")
    _assert_contains_all(
        historian_attack4_text,
        [
            '"template_module": "s3_attack4_LIT301_template.py"',
            '"template_module": "s3_attack4_FIT201_template.py"',
            '"template_module": "s3_attack4_MV201_template.py"',
            '"template_module": "s3_attack4_P101_template.py"',
        ],
        label="s3 supervisory historian metadata",
    )
    _assert_contains_all(
        historian_attack4_text,
        [
            "Manipulated PV template replacement begins at the injection point.",
            '"entry_transition": "template_replacement_begins_at_injection_start"',
        ],
        label="s3 supervisory historian manipulated-PV semantics",
    )
    if "instant_switch_to_template_value" in historian_attack4_text:
        raise AssertionError(
            "attack4 historian metadata still claims manipulated PVs instantly switch values."
        )
    _assert_contains_all(
        historian_attack4_text,
        [
            "write_attack_injection_log",
            '"attack_injection_log"',
        ],
        label="s3 supervisory historian labeling",
    )
    traffic_attack4_text = TRAFFIC_ATTACK4.read_text(encoding="utf-8")
    control_attack4_text = CONTROL_ATTACK4.read_text(encoding="utf-8")
    _assert_contains_all(
        traffic_attack4_text,
        [
            "materialize_attack_injection_log_for_run",
            '"attack_injection_log"',
        ],
        label="s3 supervisory traffic label propagation",
    )
    _assert_contains_all(
        control_attack4_text,
        ['"attack_injection_log"'],
        label="s3 control label propagation",
    )

    print("attack generator separation checks passed")


if __name__ == "__main__":
    main()
