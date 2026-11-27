#!/usr/bin/env python3
"""无 pytest 沙箱下的简易测试发现/运行器，行为接近 `pytest -q`。"""
import importlib.util
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "pytest_shim"))


def load_module(path):
    name = "testmod_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    failures = []
    passed = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            module = load_module(path)
        except ModuleNotFoundError as exc:
            if exc.name == "flask":
                print(f"SKIP {path.name}（本机无 flask，容器内由 pytest 运行）")
                continue
            raise
        for attr in sorted(vars(module)):
            if not attr.startswith("test_"):
                continue
            func = getattr(module, attr)
            if not callable(func):
                continue
            cases = getattr(func, "__parametrized__", None) or [func]
            for case in cases:
                try:
                    case()
                except Exception:  # noqa: BLE001 - 测试运行器需要捕获全部异常
                    failures.append((path.name, getattr(case, "__name__", attr), traceback.format_exc()))
                else:
                    passed += 1
    for file_name, test_name, formatted in failures:
        print(f"FAILED {file_name}::{test_name}")
        print(formatted)
    print(f"{passed} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
