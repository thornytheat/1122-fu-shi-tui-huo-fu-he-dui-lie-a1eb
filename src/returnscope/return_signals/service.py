from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .algorithm import calculate, public_baseline
from .contracts import ExpectedPackage, InvalidRow, parse


@dataclass(frozen=True, slots=True)
class ExpectedPackageRejection:
    line: int
    field: str
    code: str


def validate(rows: Iterable[Mapping[str, Any]]) -> tuple[list[ExpectedPackage], list[ExpectedPackageRejection]]:
    accepted = []
    rejected = []
    for line, raw in enumerate(rows, 1):
        try:
            accepted.append(parse(raw))
        except InvalidRow as issue:
            rejected.append(ExpectedPackageRejection(line, issue.field, issue.code))
    return accepted, rejected


def evaluate(rows: Iterable[Mapping[str, Any]], **options) -> dict:
    """计算退货特征，逐行隔离字段错误，保留原因码和行号。

    同时输出按物理称重设备重建的误差基线证据（样本量、设备下限、
    是否达到自动评分门槛），供工作台核查与运行指纹锁定。
    """
    accepted, rejected = validate(rows)
    outcome, baseline = calculate(accepted, **options)
    return {
        'records': outcome,
        'baseline': public_baseline(baseline),
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        'errors': [asdict(error) for error in rejected],
    }
