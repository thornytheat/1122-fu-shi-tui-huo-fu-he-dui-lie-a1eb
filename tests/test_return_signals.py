import json

import pytest

from returnscope.return_signals.contracts import parse
from returnscope.return_signals.service import evaluate
from returnscope.return_signals import algorithm
from returnscope.return_signals.scale_ledger import ScaleRegistry, InvalidScaleLedger

LEDGER = [
    {'shop': 'S9', 'scale': 'S9-OLD', 'installed_at': 0, 'retired_at': 80,
     'instrument_floor': 0.015},
    {'shop': 'S9', 'scale': 'S9-NEW', 'installed_at': 80, 'retired_at': None,
     'instrument_floor': 0.004},
    {'shop': 'S8', 'scale': 'S8-OLD', 'installed_at': 0, 'retired_at': None,
     'instrument_floor': 0.015},
]


def measured(**changes):
    return dict(rma='R1', original_shop='S9', category='coats', arrival=90, submitted=82,
                weight=400, listed_weight=500, prior_returns=1, seal=False, amount=8990,
                priority=False, scale='S9-NEW') | changes


def history_row(day, scale, **changes):
    return dict(rma=f'H{day}', original_shop='S9', category='coats', arrival=day,
                submitted=max(0, day - 5), weight=490, listed_weight=500, prior_returns=1,
                seal=True, amount=8990, priority=False, scale=scale) | changes


def test_future_weight_does_not_enter_device_statistics():  # TRAP-42
    history = [history_row(81 + day, 'S9-NEW', rma=f'H{day}') for day in range(5)]
    future = history_row(95, 'S9-NEW', rma='J', weight=900)
    result = evaluate([measured()], history=history, cutoff=90, scale_ledger=LEDGER)
    extended = evaluate([measured()], history=history + [future], cutoff=90,
                        scale_ledger=LEDGER)
    assert result == extended
    # 5 个样本偏差均为 |490-500|/500=0.02，高于新秤下限 0.004
    assert result['records'][0]['weight_tolerance'] == .02


def test_exact_weight_variance_uses_instrument_floor():  # TRAP-32
    history = [history_row(81 + day, 'S9-NEW', weight=500, rma=f'H{day}') for day in range(8)]
    evidence = evaluate([measured()], history=history, cutoff=200, scale_ledger=LEDGER)
    assert evidence['records'][0]['weight_tolerance'] == .004
    assert evidence['records'][0]['baseline_supported'] is True


def test_no_history_has_explicit_business_prior_tolerance():
    evidence = evaluate([measured()], history=[], cutoff=200, scale_ledger=LEDGER)['records'][0]
    assert evidence['weight_tolerance'] == .06
    assert evidence['baseline_supported'] is False
    assert evidence['baseline_status'] == 'NO_DEVICE_HISTORY'


def test_invalid_chronology_is_not_a_zero_day_return():
    evidence = evaluate([measured(submitted=91)], history=[], cutoff=200,
                        scale_ledger=LEDGER)['records'][0]
    assert evidence['reason'] == 'AUTHORIZATION_AFTER_RECEIPT'
    assert evidence['valid'] is False


def test_elapsed_days_at_return_deadline():
    rows = evaluate([measured(arrival=96, submitted=82), measured(arrival=97, submitted=82)],
                    history=[], cutoff=200, scale_ledger=LEDGER)['records']
    assert rows[0]['outside_policy'] is False
    assert rows[1]['outside_policy'] is True
    assert rows[1]['features'][3] == pytest.approx(1 / 14)


def test_invalid_listed_weight_cannot_be_divisor():
    result = evaluate([measured(listed_weight=0), measured()], history=[], cutoff=200,
                      scale_ledger=LEDGER)
    assert result['rejected_count'] == 1
    assert len(result['records']) == 1


def test_limit_prevents_one_broken_scale_from_domination():
    row = evaluate([measured(weight=85000)], history=[], cutoff=200,
                   scale_ledger=LEDGER)['records'][0]
    assert row['features'][0] == 20
    assert row['delta_ratio'] == 169


def test_repeat_refund_authorization_counts_once_in_history():
    history = [history_row(84, 'S9-NEW', rma='DUP', weight=480),
               history_row(85, 'S9-NEW', rma='DUP', weight=450)]
    assert len(algorithm.fitting_rows([parse(value) for value in history], 100)) == 1


def test_old_device_samples_never_enter_new_device_baseline():
    # 旧秤 S9-OLD 带约 16% 的大偏差，新秤 S9-NEW 是高精度小偏差样本
    old = [history_row(70 + offset, 'S9-OLD', weight=420, rma=f'OLD{offset}')
           for offset in range(8)]
    new = [history_row(81 + offset, 'S9-NEW', weight=498, rma=f'NEW{offset}')
           for offset in range(5)]
    report = evaluate([measured()], history=old + new, cutoff=200, scale_ledger=LEDGER)
    evidence = report['records'][0]
    # 新秤中位偏差 0.004，恰为设备下限；旧秤 16% 的偏差不得混入新秤分组
    assert evidence['weight_tolerance'] == pytest.approx(.004)
    profiles = {(p['scale'], p['category']): p for p in report['baseline']['profiles']}
    assert profiles[('S9-NEW', 'coats')]['sample_count'] == 5
    assert profiles[('S9-OLD', 'coats')]['sample_count'] == 8


def test_insufficient_new_device_samples_force_manual_without_borrowing():
    # 新秤只有 4 个样本（minimum_group=5）；同店旧秤与 S8 门店样本再充足也不得借用
    new = [history_row(81 + offset, 'S9-NEW', weight=498, rma=f'NEW{offset}')
           for offset in range(4)]
    old = [history_row(70 + offset, 'S9-OLD', weight=420, rma=f'OLD{offset}')
           for offset in range(8)]
    other_shop = [history_row(81 + offset, 'S8-OLD', original_shop='S8',
                               rma=f'S8{offset}') for offset in range(6)]
    evidence = evaluate([measured()], history=new + old + other_shop, cutoff=200,
                        scale_ledger=LEDGER)['records'][0]
    assert evidence['baseline_supported'] is False
    assert evidence['baseline_status'] == 'INSUFFICIENT_DEVICE_SAMPLES'
    assert evidence['reason'] == 'INSUFFICIENT_DEVICE_SAMPLES'
    assert evidence['weight_tolerance'] == pytest.approx(.004)
    assert evidence['baseline_sample_count'] == 4


def test_explicit_scale_conflicting_with_ledger_epoch_is_rejected():
    valid = measured(arrival=70, scale='S9-OLD')  # 第 70 天旧秤仍在服役
    assert evaluate([valid], history=[], cutoff=200, scale_ledger=LEDGER)['records'][0]['scale'] == 'S9-OLD'
    conflict = measured(arrival=90, scale='S9-OLD')  # 旧秤已退役，编号与台账冲突
    with pytest.raises(ValueError, match='SCALE_EPOCH_MISMATCH'):
        evaluate([conflict], history=[], cutoff=200, scale_ledger=LEDGER)


def test_ledger_rejects_overlapping_and_duplicate_epochs():
    overlapping = LEDGER + [{'shop': 'S7', 'scale': 'S7-A', 'installed_at': 0,
                             'retired_at': None, 'instrument_floor': 0.01},
                            {'shop': 'S7', 'scale': 'S7-B', 'installed_at': 5,
                             'retired_at': None, 'instrument_floor': 0.01}]
    with pytest.raises(InvalidScaleLedger):
        ScaleRegistry(overlapping)
    duplicate = [dict(LEDGER[0], scale='DUP'), dict(LEDGER[1], scale='DUP')]
    with pytest.raises(InvalidScaleLedger):
        ScaleRegistry(duplicate)


def test_baseline_fingerprint_changes_when_device_history_changes():
    history = [history_row(81 + offset, 'S9-NEW', weight=498, rma=f'H{offset}')
               for offset in range(5)]
    first = evaluate([measured()], history=history, cutoff=200, scale_ledger=LEDGER)
    shifted = [dict(row, weight=490) for row in history]
    second = evaluate([measured()], history=shifted, cutoff=200, scale_ledger=LEDGER)
    assert first['baseline']['baseline_id'] != second['baseline']['baseline_id']


def test_missing_scale_field_falls_back_to_ledger_epoch():
    row = measured()
    row.pop('scale')
    evidence = evaluate([row], history=[], cutoff=200, scale_ledger=LEDGER)['records'][0]
    assert evidence['scale'] == 'S9-NEW'


def test_baseline_evidence_is_json_ready_and_excludes_internal_index():
    report = evaluate([measured()], history=[], cutoff=200, scale_ledger=LEDGER)
    encoded = json.dumps(report['baseline'], ensure_ascii=False)
    assert 'profiles' in encoded and 'baseline_id' in encoded
    assert all(isinstance(profile, dict) for profile in report['baseline']['profiles'])
    assert 'tolerances' not in report['baseline']
