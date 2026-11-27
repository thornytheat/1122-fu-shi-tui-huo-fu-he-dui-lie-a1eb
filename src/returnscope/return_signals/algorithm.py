from __future__ import annotations

from .contracts import ExpectedPackage
from .scale_ledger import ScaleRegistry

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from math import log1p
from statistics import median

DEFAULT_INSTRUMENT_FLOOR = .015
UNPROFILED_PRIOR = .06


def package_delta(measured, expected):
    return abs(measured - expected) / expected


def fitting_rows(rows, cutoff):
    eligible = sorted([row for row in rows if row.arrival < cutoff and row.submitted <= row.arrival],
                      key=lambda row: (row.arrival, row.original_shop, row.rma))
    seen = set()
    unique = []
    for row in eligible:
        identity = row.original_shop, row.rma
        if identity in seen:
            continue
        unique.append(row)
        seen.add(identity)
    return unique


def resolve_scale(row: ExpectedPackage, registry: ScaleRegistry):
    """优先采用记录自带的设备编号，再由台账按到件时刻定位；两者冲突直接报错。"""
    epoch = registry.resolve(row.original_shop, row.arrival)
    if row.scale is not None and epoch is not None and row.scale != epoch.scale:
        raise ValueError(
            f'SCALE_EPOCH_MISMATCH:{row.original_shop}/{row.rma}/{row.scale}!={epoch.scale}'
        )
    if row.scale is not None:
        return row.scale
    if epoch is not None:
        return epoch.scale
    # 无台账时按门店兜底为一台隐式设备，保证跨店样本永不合并
    return f'{row.original_shop}#LEGACY'


def instrument_floor_for(row: ExpectedPackage, scale: str, registry: ScaleRegistry) -> float:
    epoch = registry.resolve(row.original_shop, row.arrival)
    if epoch is not None and epoch.scale == scale:
        return epoch.instrument_floor
    return DEFAULT_INSTRUMENT_FLOOR


def fit_variance(history, cutoff, minimum_group, registry: ScaleRegistry):
    """按「门店设备 × 商品组」分别估计绝对称重偏差中位数。

    旧设备的服役区间在台账中已经终止，新设备的样本只来自其服役区间之后；
    分组键包含物理设备编号，因此同一门店换秤前后的样本天然隔离，
    也不会借用其他门店或其他设备的样本。
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    floors: dict[str, float] = {}
    for row in fitting_rows(history, cutoff):
        scale = resolve_scale(row, registry)
        floors[scale] = instrument_floor_for(row, scale, registry)
        grouped[(scale, row.category)].append(package_delta(row.weight, row.listed_weight))

    profiles = []
    tolerances: dict[tuple[str, str], dict[str, object]] = {}
    for (scale, category), values in sorted(grouped.items()):
        floor = floors.get(scale, DEFAULT_INSTRUMENT_FLOOR)
        sample_count = len(values)
        supported = sample_count >= minimum_group
        if supported:
            tolerance = max(floor, median(values))
            status = 'DEVICE_PROFILED'
        else:
            # 样本不足：不借其他门店/旧设备/全局样本，仅记录设备误差下限并转人工
            tolerance = floor
            status = 'INSUFFICIENT_DEVICE_SAMPLES'
        tolerances[(scale, category)] = {
            'tolerance': tolerance,
            'sample_count': sample_count,
            'instrument_floor': floor,
            'supported': supported,
            'status': status,
        }
        profiles.append({
            'scale': scale,
            'category': category,
            'sample_count': sample_count,
            'minimum_group': minimum_group,
            'tolerance': tolerance,
            'instrument_floor': floor,
            'supported': supported,
            'status': status,
        })
    ledger_material = tuple(
        (epoch.shop, epoch.scale, epoch.installed_at, epoch.retired_at, epoch.instrument_floor)
        for shop in registry.shops() for epoch in registry.epochs_for(shop)
    )
    baseline_id = sha256(repr((
        tuple(sorted((key, value['tolerance'], value['sample_count'], value['supported'])
                     for key, value in tolerances.items())),
        ledger_material, cutoff, minimum_group, 'device-category-median-v1',
    )).encode()).hexdigest()[:16]
    return {
        'profiles': profiles,
        'tolerances': tolerances,
        'baseline_id': baseline_id,
        'unprofiled_prior': UNPROFILED_PRIOR,
    }


def public_baseline(baseline):
    """剥离以元组为键的内部索引，输出可 JSON 序列化、可核查的基线证据。"""
    return {
        'profiles': baseline['profiles'],
        'baseline_id': baseline['baseline_id'],
        'unprofiled_prior': baseline['unprofiled_prior'],
    }


def feature_record(row, baseline, maximum_days, registry):
    delta = package_delta(row.weight, row.listed_weight)
    elapsed = row.arrival - row.submitted
    scale = resolve_scale(row, registry)
    profile = baseline['tolerances'].get((scale, row.category))
    if profile is None:
        # 该设备×商品组没有任何历史样本：无法自动估计，沿用业务先验并强制人工
        profile = {
            'tolerance': baseline['unprofiled_prior'], 'sample_count': 0,
            'instrument_floor': instrument_floor_for(row, scale, registry),
            'supported': False, 'status': 'NO_DEVICE_HISTORY',
        }
    reference = profile['tolerance']
    chronology_valid = elapsed >= 0
    baseline_supported = bool(profile['supported'])
    x = [min(20, delta / reference), 0 if row.seal else 1,
         log1p(row.prior_returns), max(0, elapsed - maximum_days) / maximum_days]
    if not chronology_valid:
        reason = 'AUTHORIZATION_AFTER_RECEIPT'
    elif not baseline_supported:
        reason = profile['status']
    else:
        reason = 'FEATURES_READY'
    return asdict(row) | {
        'scale': scale,
        'features': x,
        'delta_ratio': delta,
        'weight_tolerance': reference,
        'baseline_supported': baseline_supported,
        'baseline_status': profile['status'],
        'baseline_sample_count': profile['sample_count'],
        'baseline_id': baseline['baseline_id'],
        'elapsed_days': elapsed,
        'outside_policy': elapsed > maximum_days,
        'valid': chronology_valid,
        'reason': reason,
    }


def calculate(records, *, history, cutoff, minimum_group=5, maximum_days=14, scale_ledger=None):
    from .contracts import parse
    if maximum_days < 1 or minimum_group < 1:
        raise ValueError('INVALID_RETURN_FEATURE_POLICY')
    registry = ScaleRegistry(scale_ledger)
    parsed_records = [row if isinstance(row, ExpectedPackage) else parse(row) for row in records]
    parsed_history = [row if isinstance(row, ExpectedPackage) else parse(row) for row in history]
    baseline = fit_variance(parsed_history, cutoff, minimum_group, registry)
    return [feature_record(row, baseline, maximum_days, registry) for row in parsed_records], baseline
