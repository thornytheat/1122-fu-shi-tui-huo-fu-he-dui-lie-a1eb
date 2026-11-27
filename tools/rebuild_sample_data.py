#!/usr/bin/env python3
"""重建样例数据：引入称重设备台账，按物理设备重新生成称重误差与训练向量。

设计要点（与业务诉求一一对应）：
- 浙-南、粤-西 在第 88 天更换称重设备；台账登记服役区间，旧秤 retired_at=88。
- 旧设备历史原样保留，但因其设备编号不同，永远不会进入新设备的基线分组。
- 浙-南新秤已有 4 个针织历史样本，仍不足 minimum_group=5 → 新秤包裹转人工，
  不借其他门店或旧秤样本。
- 粤-西新秤三个商品组各 1~2 个样本，同样不足 → 转人工。
- learned.weight_z 由 return_signals 的同一条特征代码重算，训练/在线口径对齐。
- 仅依赖标准库，确定性随机种子，可重复生成。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from returnscope.return_signals.algorithm import calculate as build_features  # noqa: E402

CUTOFF = 102
REPLACEMENT_DAY = 88
MINIMUM_GROUP = 5
RNG_SEED = 20261127

# 各门店设备台账：instrument_floor 是该物理秤的最小可读误差
SCALES = [
    {'shop': '京-东', 'scale': 'JD-S01', 'installed_at': 0, 'retired_at': None,
     'instrument_floor': 0.015},
    {'shop': '川-西', 'scale': 'CX-S01', 'installed_at': 0, 'retired_at': None,
     'instrument_floor': 0.015},
    {'shop': '浙-南', 'scale': 'ZN-S01', 'installed_at': 0, 'retired_at': REPLACEMENT_DAY,
     'instrument_floor': 0.015},
    {'shop': '浙-南', 'scale': 'ZN-S02', 'installed_at': REPLACEMENT_DAY, 'retired_at': None,
     'instrument_floor': 0.004},
    {'shop': '粤-西', 'scale': 'YX-S01', 'installed_at': 0, 'retired_at': REPLACEMENT_DAY,
     'instrument_floor': 0.015},
    {'shop': '粤-西', 'scale': 'YX-S02', 'installed_at': REPLACEMENT_DAY, 'retired_at': None,
     'instrument_floor': 0.004},
]
LEGACY_SCALE = {'京-东': 'JD-S01', '川-西': 'CX-S01', '浙-南': 'ZN-S01', '粤-西': 'YX-S01'}
CURRENT_SCALE = {
    '京-东': 'JD-S01', '川-西': 'CX-S01', '浙-南': 'ZN-S02', '粤-西': 'YX-S02',
}


def label_scale(shop: str, arrival: int) -> str:
    if shop in ('浙-南', '粤-西') and arrival >= REPLACEMENT_DAY:
        return CURRENT_SCALE[shop]
    return LEGACY_SCALE[shop]


def new_scale_weight(listed_weight: int, rng: random.Random) -> int:
    """新设备：高精度，正常包裹的相对偏差只在 0.1%~0.5% 之间波动。"""
    ratio = rng.uniform(0.001, 0.005)
    sign = rng.choice((-1, 1))
    return max(1, int(round(listed_weight + sign * listed_weight * ratio)))


def main() -> int:
    source_path = ROOT / 'sample_data' / 'western_region_return_counter.json'
    data = json.loads(source_path.read_text(encoding='utf-8'))
    rng = random.Random(RNG_SEED)

    # 1) 历史称重：换秤门店在更换日之后用新设备精度重新生成，并标注设备编号
    for row in data['history']:
        scale = label_scale(row['original_shop'], row['arrival'])
        row['scale'] = scale
        if scale.endswith('-S02'):
            row['weight'] = new_scale_weight(row['listed_weight'], rng)

    # 2) 当班申请：同样按到件时刻标注设备；新设备包裹用新秤精度称重
    for row in data['requests']:
        scale = label_scale(row['original_shop'], row['arrival'])
        row['scale'] = scale
        if scale.endswith('-S02'):
            row['weight'] = new_scale_weight(row['listed_weight'], rng)

    # 3) 验包流水：扫描设备取「扫描门店 + 扫描时刻」对应的秤。
    #    异店转送面单的受理门店不在设备台账内，不标注设备；
    #    换店去重归并后以原门店授权行的设备为准。
    ledger_shops = {entry['shop'] for entry in SCALES}
    for row in data['scans']:
        if isinstance(row['weight'], int) and row['shop'] in ledger_shops:
            row['scale'] = label_scale(row['shop'], row['arrival'])

    # 4) 台账
    data['scales'] = SCALES

    # 5) 用与在线完全相同的特征代码重建训练向量（仅 weight_z 变化，标签等保留）
    feature_rows, baseline = build_features(
        data['history'], history=data['history'], cutoff=CUTOFF,
        minimum_group=MINIMUM_GROUP, scale_ledger=SCALES,
    )
    by_key = {(row['original_shop'], row['rma']): row for row in feature_rows}
    rebuilt = 0
    dropped = 0
    for label_row in data['learned']:
        feature = by_key[(label_row['original_shop'], label_row['rma'])]
        if not feature['valid'] or not feature['baseline_supported']:
            dropped += 1  # 基线不支持的设备×商品组不得进入训练集
            continue
        weight_z, open_seal, repeat_log, extra_days = feature['features']
        label_row['weight_z'] = weight_z
        label_row['open_seal'] = open_seal
        label_row['repeat_log'] = repeat_log
        label_row['extra_days'] = extra_days
        rebuilt += 1
    data['learned'] = [
        row for row in data['learned']
        if by_key[(row['original_shop'], row['rma'])]['valid']
        and by_key[(row['original_shop'], row['rma'])]['baseline_supported']
    ]

    source_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'history={len(data["history"])} requests={len(data["requests"])} '
          f'learned_rebuilt={rebuilt} learned_dropped={dropped} '
          f'baseline_id={baseline["baseline_id"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
