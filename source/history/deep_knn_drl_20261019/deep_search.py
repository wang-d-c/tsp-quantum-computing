"""Shared U/D search: classical proposals, one selected exact test, diverse pool.

The neural selector acts before global LKH. It does not replace LKH internals.
The static sparse attention encoder is evaluated once within the paid budget;
the candidate decoder observes the current parent tour on every decision.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import numpy as np

from search_core import (TourPool, build_knn, enumerate_2opt_candidates,
                         evaluate_d_action, make_rng_streams,
                         sample_policy_action, tour_length)

DEFAULT_SETTINGS = dict(knn=16, d_candidate_count=192, d_exact_quota=1,
                        pool_size=3, local_max_steps=128, audit_decisions=128)


def context_for(pool, parent, step, step_cap=128):
    return dict(initial_mean_edge=pool.initial_length / len(parent['route']),
                parent_excess=(parent['cost'] - pool.best_length) / max(pool.best_length, 1),
                step_fraction=step / step_cap,
                best_fraction=pool.best_length / max(pool.initial_length, 1))


def _charge(ledger, name, began):
    ledger[name] = ledger.get(name, 0.) + time.perf_counter() - began


def optimize(route, distances, mode, deadline, emit, policy=None, seed=1,
             settings=None, coords=None):
    mode = str(mode).upper()
    if mode not in ('U', 'D'):
        raise ValueError('mode must be U or D')
    config = dict(DEFAULT_SETTINGS)
    config.update(settings or {})
    if config['d_exact_quota'] != 1:
        raise ValueError('exactly one sampled candidate is evaluated')
    if mode == 'D' and (policy is None or coords is None):
        raise ValueError('D requires a trained deep policy and coordinates')
    began = time.perf_counter()
    ledger = {}
    stats = dict(mode=mode, settings=config, steps=0, accepted_moves=0,
                 evaluations=0, proposed_candidates=0, gain=0, best_gain_sum=0,
                 parent_gain_sum=0, pool_insertions=0, max_pool_size=1,
                 parent_use_counts={}, diverse_parent_steps=0, decisions=[],
                 ledger=ledger, stop_reason='step_limit')
    proposal_rng, selection_rng = make_rng_streams(seed)
    started = time.perf_counter()
    pool = TourPool(route, distances, config['pool_size'])
    _charge(ledger, 'pool_seconds', started)
    knn = build_knn(distances, config['knn'], deadline, ledger)
    if knn is not None and mode == 'D' and time.perf_counter() < deadline:
        started = time.perf_counter()
        policy.prepare(coords, knn)
        _charge(ledger, 'encoder_seconds', started)
        stats['parameter_count'] = int(policy.parameter_count)

    def publish(stage):
        started_publish = time.perf_counter()
        stats.update(elapsed=time.perf_counter() - began,
                     best_length=pool.best_length, gain=pool.initial_length - pool.best_length)
        if emit is not None:
            emit(route.copy(), stage, copy.deepcopy(stats))
        _charge(ledger, 'publication_seconds', started_publish)

    for step in range(int(config['local_max_steps'])):
        if knn is None or time.perf_counter() >= deadline:
            stats['stop_reason'] = 'deadline'
            break
        started = time.perf_counter()
        parent = pool.parent(step)
        parent_id = str(parent['id'])
        stats['parent_use_counts'][parent_id] = stats['parent_use_counts'].get(parent_id, 0) + 1
        stats['diverse_parent_steps'] += int(parent['cost'] > pool.best_length)
        _charge(ledger, 'pool_seconds', started)
        candidates, old_features = enumerate_2opt_candidates(
            parent['route'], distances, proposal_rng,
            config['d_candidate_count'], knn, ledger)
        if not candidates:
            stats['stop_reason'] = 'empty_candidates'
            break
        context = context_for(pool, parent, step, config['local_max_steps'])
        started = time.perf_counter()
        if mode == 'D':
            logits = np.asarray(policy.logits(parent['route'], candidates, distances, context), dtype=np.float64)
            if logits.shape != (len(candidates),) or not np.isfinite(logits).all():
                raise ValueError('invalid neural logits')
            chosen = sample_policy_action(logits, selection_rng)
            p = np.exp(logits - logits.max())
            probability = float(p[chosen] / p.sum())
        else:
            logits = np.zeros(len(candidates))
            chosen = int(selection_rng.integers(len(candidates)))
            probability = 1. / len(candidates)
        _charge(ledger, 'inference_seconds', started)
        if time.perf_counter() >= deadline:
            stats['stop_reason'] = 'deadline'
            break
        action = candidates[chosen]
        result = evaluate_d_action(parent['route'], distances, action, 1, deadline, ledger)
        if result['deadline_hit']:
            stats['stop_reason'] = 'deadline'
            break
        stats['steps'] += 1
        stats['evaluations'] += result['evaluations']
        stats['proposed_candidates'] += len(candidates)
        started = time.perf_counter()
        before = pool.best_length
        transition = pool.update(parent, result)
        stats['accepted_moves'] += int(result['accepted'])
        stats['parent_gain_sum'] += transition['parent_gain']
        stats['best_gain_sum'] += transition['best_gain']
        stats['pool_insertions'] += int(transition['inserted'])
        stats['max_pool_size'] = max(stats['max_pool_size'], len(pool.entries))
        improved = pool.best_length < before
        if improved:
            route[:] = pool.best_route
        _charge(ledger, 'pool_seconds', started)
        started = time.perf_counter()
        if len(stats['decisions']) < config['audit_decisions']:
            stats['decisions'].append(dict(
                step=step, parent_id=parent['id'], parent_length=parent['cost'],
                action_count=len(candidates), selected_index=chosen,
                selected_action=dict(action), selected_features=old_features[chosen].tolist(),
                selected_logit=float(logits[chosen]), selected_probability=probability,
                logit_std=float(logits.std()), context=context,
                candidate_sha256=hashlib.sha256(json.dumps(candidates, sort_keys=True,
                    separators=(',', ':')).encode('ascii')).hexdigest(),
                gain=int(result['gain']), parent_gain=transition['parent_gain'],
                best_gain=transition['best_gain'], evaluations=result['evaluations'],
                move=result['move'], pool_size=len(pool.entries), best_length=pool.best_length))
        _charge(ledger, 'audit_seconds', started)
        if improved:
            publish('deep_candidate' if mode == 'D' else 'uniform_candidate')
    route[:] = pool.best_route
    stats.update(best_length=pool.best_length, gain=pool.initial_length - pool.best_length,
                 elapsed=time.perf_counter() - began,
                 final_pool=[dict(id=e['id'], cost=e['cost'], route=e['route'].tolist()) for e in pool.entries])
    stats['accounted_seconds'] = sum(ledger.values())
    stats['unaccounted_seconds'] = max(0., stats['elapsed'] - stats['accounted_seconds'])
    if stats['gain'] != stats['best_gain_sum']:
        raise AssertionError('incumbent gain does not telescope')
    return stats
