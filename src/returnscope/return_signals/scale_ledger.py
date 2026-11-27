"""称重设备台账：登记每台秤的服役区间，保证旧设备样本不会进入新设备基线。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class InvalidScaleLedger(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _text(row: Mapping, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80:
        raise InvalidScaleLedger('SCALE_LEDGER_EXPECTED_NONEMPTY_TEXT')
    return value.strip()


def _integer(row: Mapping, key: str) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0 or value > 10_000_000:
        raise InvalidScaleLedger('SCALE_LEDGER_EXPECTED_NON_NEGATIVE_INTEGER')
    return value


def _floor(row: Mapping, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise InvalidScaleLedger('SCALE_LEDGER_INVALID_INSTRUMENT_FLOOR')
    return float(value)


@dataclass(frozen=True, slots=True)
class ScaleEpoch:
    shop: str
    scale: str
    installed_at: int
    retired_at: int | None
    instrument_floor: float

    def covers(self, moment: int) -> bool:
        return self.installed_at <= moment and (self.retired_at is None or moment < self.retired_at)


def parse_ledger(rows: Any) -> tuple[ScaleEpoch, ...]:
    if rows is None:
        return ()
    if not isinstance(rows, (list, tuple)):
        raise InvalidScaleLedger('SCALE_LEDGER_EXPECTED_LIST')
    epochs: list[ScaleEpoch] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise InvalidScaleLedger('SCALE_LEDGER_EXPECTED_OBJECT')
        shop = _text(row, 'shop')
        scale = _text(row, 'scale')
        installed_at = _integer(row, 'installed_at')
        retired_value = row.get('retired_at')
        if retired_value is None:
            retired_at = None
        else:
            if type(retired_value) is not int or not installed_at < retired_value <= 10_000_000:
                raise InvalidScaleLedger('SCALE_LEDGER_INVALID_RETIREMENT')
            retired_at = retired_value
        epochs.append(ScaleEpoch(shop, scale, installed_at, retired_at,
                                 _floor(row, 'instrument_floor')))
    return tuple(epochs)


class ScaleRegistry:
    """按门店与到件时刻定位当时实际服役的称重设备。"""

    def __init__(self, epochs: Any):
        parsed = parse_ledger(epochs)
        self._epochs: dict[str, list[ScaleEpoch]] = {}
        for epoch in parsed:
            self._epochs.setdefault(epoch.shop, []).append(epoch)
        for shop, entries in self._epochs.items():
            entries.sort(key=lambda item: item.installed_at)
            for earlier, later in zip(entries, entries[1:]):
                if later.installed_at < (earlier.retired_at if earlier.retired_at is not None else 10**9):
                    raise InvalidScaleLedger('SCALE_LEDGER_OVERLAPPING_EPOCHS')
                if earlier.scale == later.scale:
                    raise InvalidScaleLedger('SCALE_LEDGER_DUPLICATE_SCALE_NAME')

    def resolve(self, shop: str, moment: int) -> ScaleEpoch | None:
        candidates = [epoch for epoch in self._epochs.get(shop, ()) if epoch.covers(moment)]
        if len(candidates) > 1:  # 解析阶段已拒绝重叠，这里只是防御性兜底
            raise InvalidScaleLedger('SCALE_LEDGER_OVERLAPPING_EPOCHS')
        return candidates[0] if candidates else None

    def shops(self) -> tuple[str, ...]:
        return tuple(sorted(self._epochs))

    def epochs_for(self, shop: str) -> tuple[ScaleEpoch, ...]:
        return tuple(self._epochs.get(shop, ()))
