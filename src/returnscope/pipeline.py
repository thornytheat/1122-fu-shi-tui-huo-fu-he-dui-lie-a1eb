from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from . import desk_quota, inspection_score, parcel_scan, return_replay, return_signals, shop_identity
from .return_signals.algorithm import calculate as build_features


DEFAULT_CAPACITY = 9
CLOSE = 140
MODEL_CUTOFF = 102


def load_counter_statement():
    journal = Path(__file__).resolve().parents[2] / 'sample_data' / 'western_region_return_counter.json'
    return json.loads(journal.read_text(encoding='utf-8'))


def join_identity(accepted_scans, requests):
    declared = {row['rma']: row for row in requests}
    unmatched = []
    qualified = []
    for scan in accepted_scans:
        declaration = declared.get(scan['rma'])
        if declaration:
            qualified.append(declaration | scan)
        else:
            unmatched.append({'rma': scan['rma'], 'reason': 'NO_DECLARED_AUTHORIZATION'})
    return qualified, unmatched


def prepare_score_inputs(feature_rows):
    scored = []
    for record in feature_rows:
        if record['valid'] and record['baseline_supported']:
            weight, seal, repeat, lateness = record['features']
            scored.append(record | {'weight_z': weight, 'open_seal': seal, 'repeat_log': repeat,
                                    'extra_days': lateness, 'found_mismatch': False, 'labeled': 0})
    return scored


def rebuild_learned_vectors(history, learned, *, cutoff, scale_ledger):
    """用与在线评分同一条特征口径重算训练向量，保证训练/在线不漂移。

    历史特征只允许使用结算时点之前可见的称重样本；设备台账使旧秤样本
    无法进入新秤分组。基线不支持的设备×商品组不进入训练集。
    """
    from .return_signals.contracts import parse as parse_package

    labels = {(row['original_shop'], row['rma']): row for row in learned}
    packages = [parse_package(row) for row in history
                if (row['original_shop'], row['rma']) in labels]
    feature_rows, baseline = build_features(
        packages, history=history, cutoff=cutoff, scale_ledger=scale_ledger,
    )
    vectors = []
    for feature in feature_rows:
        label = labels[(feature['original_shop'], feature['rma'])]
        if not feature['valid'] or not feature['baseline_supported']:
            continue
        weight, seal, repeat, lateness = feature['features']
        vectors.append({
            'rma': feature['rma'],
            'original_shop': feature['original_shop'],
            'category': feature['category'],
            'arrival': feature['arrival'],
            'weight_z': weight,
            'open_seal': seal,
            'repeat_log': repeat,
            'extra_days': lateness,
            'found_mismatch': label['found_mismatch'],
            'labeled': label['labeled'],
        })
    return vectors, baseline


def compute_rows(declared, identities, scans, desk):
    decisions = {item['rma']: item for item in desk}
    reason = {item['rma']: item['reason'] for item in scans if not item['valid']}
    reason.update({item['rma']: item['reason'] for item in identities if not item['valid']})
    result = []
    for authorization in declared:
        rma = authorization['rma']
        action = decisions.get(rma)
        confidence = action['score'] if action and action['in_scope'] else None
        result.append({
            'rma': rma,
            'shop': authorization['original_shop'],
            'category': authorization['category'],
            'refund': round(authorization['amount'] / 100, 2),
            'confidence': round(confidence, 3) if confidence is not None else None,
            'place': action['desk_slot'] if action else None,
            'status': action['state'] if action else '资料隔离',
            'reason': action['reason'] if action else reason.get(rma, 'MISSING_INSPECTABLE_PARCEL'),
        })
    return sorted(result, key=lambda row: (row['place'] is None, row['place'] or 0, row['shop'], row['rma']))


def run(capacity: int | None = None) -> dict:
    source = load_counter_statement()
    scale_ledger = source.get('scales', [])
    scans = parcel_scan.evaluate(source['scans'], statement_close=CLOSE)
    valid_scans = [row for row in scans['records'] if row['valid']]
    authorizations, unmatched = join_identity(valid_scans, source['requests'])
    linked = shop_identity.evaluate(authorizations)
    candidates = {row['rma']: row for row in source['requests']}
    merged = [candidates[row['rma']] | row for row in linked['records'] if row['valid']]
    features = return_signals.evaluate(merged, history=source['history'], cutoff=MODEL_CUTOFF,
                                       scale_ledger=scale_ledger)
    serving_rows = prepare_score_inputs(features['records'])
    learned_vectors, training_baseline = rebuild_learned_vectors(
        source['history'], source['learned'], cutoff=MODEL_CUTOFF, scale_ledger=scale_ledger,
    )
    # 训练与在线必须共用同一版本的称重误差基线，否则评分口径漂移
    if training_baseline['baseline_id'] != features['baseline']['baseline_id']:
        raise RuntimeError('WEIGHT_BASELINE_DRIFT_BETWEEN_TRAIN_AND_SERVE')
    scores = inspection_score.evaluate(serving_rows, history=learned_vectors, cutoff=MODEL_CUTOFF,
                                       baseline_id=features['baseline']['baseline_id'])
    supported = {row['rma'] for row in serving_rows}
    feature_by_rma = {row['rma']: row for row in features['records']}
    requests = []
    for entry in scores['records']:
        source_row = feature_by_rma[entry['rma']]
        requests.append(source_row | entry | {'score': entry['score'] or 0})
    # 基线样本不足的正常包裹：宁可转人工，也不借其他门店或旧设备顶上
    for row in features['records']:
        if row['valid'] and row['rma'] not in supported:
            requests.append(row | {
                'reason': row['baseline_status'], 'score': 0, 'in_scope': False,
            })
    desk = desk_quota.evaluate(requests, capacity=capacity if capacity is not None else DEFAULT_CAPACITY)
    replay = return_replay.evaluate(source['settled'], as_of=CLOSE)
    rows = compute_rows(source['requests'], linked['records'], scans['records'], desk['records'])
    states = Counter(row['status'] for row in rows)
    summaries = [
        ('验包流水', f'收到 {len(valid_scans)} 笔可检面单，另记录撤回与异常', scans),
        ('换店去重', f"归并到 {len(linked['records'])} 个授权", linked),
        ('退货特征', '按门店称重设备重建误差基线，旧设备样本不进入新设备', features),
        ('验包评分', '训练向量与在线评分共用同一基线，样本不足转人工', scores),
        ('复核配额', f"已派件 {states['已派复核']} 单，门店集中度受限", desk),
        ('退货复盘', f"仅复核 {replay['records'][0]['settled']} 单已结单据", replay),
    ]
    return {
        'reconciliation': {
            'counter_health': parcel_scan.inspect_counter_health(scans,statement_close=CLOSE),
            'refund_register': shop_identity.reconcile_refund_register(source['requests'], linked['records'], desk['records'], scans['records']),
            'rolling_validation': inspection_score.validate_availability(source['learned'], with_labels_before=MODEL_CUTOFF),
            'staffing_frontier': desk_quota.compare_desk_plans(requests),
            'location_review': return_replay.reconcile_counter_locations(source['settled'],as_of=CLOSE),
            'weight_baseline': features['baseline'],
        },
        'metrics': {'退货申请': len(rows), '可放行': states['放行推荐'], '本轮复核': states['已派复核'],
                    '人工待处理': states['人工待处理'] + states['资料隔离']},
        'columns': {'rma': '申请编号', 'shop': '原门店', 'category': '商品组', 'refund': '退款金额',
                    'confidence': '模型概率', 'place': '复核位', 'status': '当前处理', 'reason': '复核依据'},
        'rows': rows,
        'series': [{'label': f"{round(item['threshold']*100)}%", 'value': item['missed_cents']/100}
                   for item in replay['records'][1:]],
        'stages': [{'title': title, 'summary': summary,
                    'records': record['records'] + (unmatched if title == '换店去重' else []),
                    'errors': record['errors'],
                    **({'baseline': record['baseline']} if 'baseline' in record else {})}
                   for title, summary, record in summaries],
    }
