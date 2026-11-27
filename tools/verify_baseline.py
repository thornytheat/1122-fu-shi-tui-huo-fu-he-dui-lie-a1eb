#!/usr/bin/env python3
"""可核实的称重基线验收检查（只读，不依赖第三方库）。

逐项证明本班诉求：
1. 基线按物理称重设备 × 商品组隔离，旧设备样本不进入新设备分组。
2. 新设备样本不足时转人工，不借用其他门店或旧设备样本。
3. 训练向量与在线评分共用同一基线版本（baseline_id 一致）。
4. 同一批数据重复运行，基线与复核派位完全不漂移（幂等）。
5. 显式设备编号与台账服役区间冲突时直接拒绝，防止错配设备。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from returnscope.pipeline import MODEL_CUTOFF, run  # noqa: E402
from returnscope.pipeline import rebuild_learned_vectors  # noqa: E402
from returnscope.return_signals.algorithm import calculate as build_features  # noqa: E402
from returnscope.return_signals.scale_ledger import ScaleRegistry  # noqa: E402


def load():
    return json.loads((ROOT / 'sample_data' / 'western_region_return_counter.json').read_text(encoding='utf-8'))


def check(name, condition, detail=''):
    mark = 'PASS' if condition else 'FAIL'
    print(f'[{mark}] {name}' + (f' — {detail}' if detail else ''))
    if not condition:
        raise SystemExit(1)


def main() -> int:
    data = load()
    ledger = data['scales']
    registry = ScaleRegistry(ledger)

    # 1) 设备隔离：每个基线分组只含单一物理设备，新旧秤互不相混
    features, baseline = build_features(
        data['history'], history=data['history'], cutoff=MODEL_CUTOFF,
        scale_ledger=ledger,
    )
    seen_scales = {profile['scale'] for profile in baseline['profiles']}
    check('台账登记了换秤门店的新旧两台设备',
          {'ZN-S01', 'ZN-S02', 'YX-S01', 'YX-S02'} <= seen_scales,
          f'设备集合={sorted(seen_scales)}')
    for scale in ('ZN-S02', 'YX-S02'):
        epochs = [e for e in registry.epochs_for('浙-南' if scale.startswith('ZN') else '粤-西')]
        retired = next((e.retired_at for e in epochs if e.scale == scale[:2] + '-S01'), None)
        samples = [r for r in data['history'] if r['scale'] == scale]
        check(f'{scale} 全部样本均在旧秤退役之后',
              all(r['arrival'] >= retired for r in samples),
              f'旧秤退役日={retired}，新秤样本到达日={sorted(r["arrival"] for r in samples)}')
    # 旧秤分组中不允许出现换秤日后的样本
    old_scale_rows = [r for r in data['history'] if r['scale'] in ('ZN-S01', 'YX-S01')]
    check('旧设备分组不含换秤日之后的任何样本',
          all(r['arrival'] < 88 for r in old_scale_rows))

    # 2) 样本不足 → 不支持自动评分；且不存在跨店/跨设备回填
    unsupported = [p for p in baseline['profiles'] if not p['supported']]
    check('新设备×商品组全部标记为样本不足',
          {p['scale'] for p in unsupported} == {'ZN-S02', 'YX-S02'},
          f'不足分组={[(p["scale"], p["category"], p["sample_count"]) for p in unsupported]}')
    new_scale_requests = [r for r in data['requests'] if r['scale'].endswith('-S02')]
    online, _ = build_features(new_scale_requests, history=data['history'],
                               cutoff=MODEL_CUTOFF, scale_ledger=ledger)
    check('新设备当班包裹无一进入自动评分（全部转人工）',
          all(not row['baseline_supported'] for row in online))
    check('样本不足时容忍度只取设备误差下限，未借任何外部样本',
          all(abs(row['weight_tolerance'] - 0.004) < 1e-12 for row in online))

    # 3) 训练/在线同口径：同一 baseline_id，且 learned.weight_z 由同一特征代码重建
    vectors, training_baseline = rebuild_learned_vectors(
        data['history'], data['learned'], cutoff=MODEL_CUTOFF, scale_ledger=ledger,
    )
    check('训练与在线基线指纹一致',
          training_baseline['baseline_id'] == baseline['baseline_id'],
          baseline['baseline_id'])
    check('基线不支持的新设备样本未进入训练集',
          all(not any(v['rma'] == r['rma'] and v['original_shop'] == r['original_shop']
                      for v in vectors)
              for r in data['history'] if r['scale'].endswith('-S02')))

    # 4) 幂等：两次完整运行的行结果、复核位与运行指纹完全一致
    first, second = run(), run()
    check('重复运行复核队列不漂移', first['rows'] == second['rows'])
    check('重复运行基线指纹不漂移',
          first['reconciliation']['weight_baseline']['baseline_id']
          == second['reconciliation']['weight_baseline']['baseline_id'],
          first['reconciliation']['weight_baseline']['baseline_id'])
    check('重复运行基线明细不漂移',
          first['reconciliation']['weight_baseline']['profiles']
          == second['reconciliation']['weight_baseline']['profiles'])
    check('重复运行各环节证据不漂移',
          [stage['records'] for stage in first['stages']]
          == [stage['records'] for stage in second['stages']])

    # 5) 设备编号与台账冲突必须报错，防止把新包裹挂到旧设备上
    conflict = dict(new_scale_requests[0])
    conflict['scale'] = 'ZN-S01'  # 新到件却声称使用已退役旧秤
    try:
        build_features([conflict], history=data['history'], cutoff=MODEL_CUTOFF,
                       scale_ledger=ledger)
    except ValueError as exc:
        check('设备编号与台账服役区间冲突时拒绝处理', str(exc).startswith('SCALE_EPOCH_MISMATCH'))
    else:
        check('设备编号与台账服役区间冲突时拒绝处理', False)

    print('\n全部验收检查通过。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
