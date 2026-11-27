"""设备称重基线端到端契约：训练/在线同口径、样本不足转人工、重算不漂移。"""
import json

from returnscope.pipeline import (
    MODEL_CUTOFF,
    load_counter_statement,
    rebuild_learned_vectors,
    run,
)
from returnscope.return_signals.algorithm import calculate as build_features


def test_training_and_serving_share_one_baseline_fingerprint():
    data = load_counter_statement()
    _, baseline = build_features(data['history'], history=data['history'],
                                 cutoff=MODEL_CUTOFF, scale_ledger=data['scales'])
    _, training_baseline = rebuild_learned_vectors(
        data['history'], data['learned'], cutoff=MODEL_CUTOFF,
        scale_ledger=data['scales'],
    )
    assert training_baseline['baseline_id'] == baseline['baseline_id']


def test_unsupported_new_device_parcels_are_manually_handled_not_borrowed():
    result = run()
    new_scale_rows = [row for row in result['rows']
                      if row['shop'] in ('浙-南', '粤-西')]
    # 换秤门店的当班包裹没有一个拿到自动概率，全部进入人工链路
    assert all(row['confidence'] is None for row in new_scale_rows)
    dispatched = [row for row in new_scale_rows if row['place'] is not None]
    waiting = [row for row in new_scale_rows if row['place'] is None]
    assert {row['reason'] for row in dispatched} == {'INSUFFICIENT_DEVICE_SAMPLES'}
    assert {row['reason'] for row in waiting} <= {
        'SHOP_CONCENTRATION', 'DESK_FULL', 'SCAN_REVISION_CONFLICT',
    }


def test_recomputing_the_same_batch_keeps_baseline_and_desk_slots_stable():
    first = run(5)
    second = run(5)
    assert first['rows'] == second['rows']
    assert (first['reconciliation']['weight_baseline']
            == second['reconciliation']['weight_baseline'])
    assert (first['reconciliation']['weight_baseline']['baseline_id']
            == second['reconciliation']['weight_baseline']['baseline_id'])


def test_capacity_changes_desk_but_never_the_weight_baseline():
    small = run(3)
    large = run(9)
    assert (small['reconciliation']['weight_baseline']
            == large['reconciliation']['weight_baseline'])
    assert small['metrics']['本轮复核'] == 3
    assert large['metrics']['本轮复核'] == 9


def test_baseline_evidence_is_visible_and_json_serializable():
    result = run()
    evidence = result['reconciliation']['weight_baseline']
    assert len(evidence['profiles']) >= 1
    json.dumps(evidence, ensure_ascii=False, allow_nan=False)
    scales = {profile['scale'] for profile in evidence['profiles']}
    assert {'ZN-S01', 'ZN-S02', 'YX-S01', 'YX-S02'} <= scales
    # 旧秤分组里不允许出现换秤日（88）之后的样本
    for profile in evidence['profiles']:
        if profile['scale'] in ('ZN-S01', 'YX-S01'):
            assert all(row['arrival'] < 88 for row in load_counter_statement()['history']
                       if row['scale'] == profile['scale'])


def test_new_device_history_is_excluded_from_training_until_supported():
    data = load_counter_statement()
    vectors, _ = rebuild_learned_vectors(
        data['history'], data['learned'], cutoff=MODEL_CUTOFF,
        scale_ledger=data['scales'],
    )
    new_scale_rmas = {(row['original_shop'], row['rma'])
                      for row in data['history'] if row['scale'].endswith('-S02')}
    vector_keys = {(row['original_shop'], row['rma']) for row in vectors}
    assert new_scale_rmas.isdisjoint(vector_keys)
