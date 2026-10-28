"""Cold-budgeted classical search environments shared by fixed and RL policies.

No action features contain a complete proposed-move cost or true gain. B and C
share action generation, candidate quotas, parent scheduling and pool retention;
only the action selector differs. All local windows are open fixed-endpoint
paths, including windows that wrap around the closed tour's array boundary.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np


ACTION_FEATURE_NAMES = (
    'operator_2opt', 'operator_or1', 'operator_or2', 'operator_or3',
    'window_fraction', 'old_mean_edge', 'old_max_edge', 'old_std_edge',
    'outside_old_edges', 'parent_excess', 'start_cos', 'start_sin',
)
EDGE_FEATURE_NAMES = (
    'old_edge_i', 'old_edge_j', 'arc_fraction', 'short_arc_fraction',
    'knn_ac', 'knn_bd', 'preceding_old_edge', 'following_old_edge',
    'kind_knn', 'kind_longedge', 'kind_explore', 'old_edge_imbalance',
)
DEFAULT_SETTINGS = {
    'local_max_steps': 128, 'action_count': 12, 'window_sizes': [24, 48, 64],
    'candidate_quota': 128, 'pool_size': 3,
    'd_candidate_count': 192, 'd_exact_quota': 1, 'knn': 16,
    'audit_decisions': 128,
}
_LOCAL_TABLES = {}
SELECTION_SEED_XOR = 0x5E1EC7


def make_rng_streams(seed):
    """Independent proposal and policy-selection streams shared across B/C/D.

    In particular C's categorical draws cannot perturb later proposal seeds,
    so forced-identical actions produce B/C-identical candidate trajectories.
    """
    return (np.random.default_rng(int(seed)),
            np.random.default_rng(int(seed) ^ SELECTION_SEED_XOR))


def _charge(ledger, name, started):
    if ledger is not None:
        ledger[name] = ledger.get(name, 0.0) + time.perf_counter() - started


def tour_length(route, distances):
    return int(np.asarray(distances[route, np.roll(route, -1)], dtype=np.int64).sum())


def _features_finite(features):
    return np.clip(np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)


def enumerate_actions(route, distances, rng=None, count=12,
                      window_sizes=(24, 48, 64), parent_excess=0.0,
                      ledger=None):
    """Return (action dictionaries, Nx12 features) without evaluating moves.

    A dictionary is a self-contained reproducible action: cyclic start, window
    size, one of 2opt/or1/or2/or3, and a private seed for its bounded candidate
    sample. Arrays are not mutated. Pass the same rng state for the same pool.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(route)
    if n < 4 or count < 1:
        _charge(ledger, 'action_construction_seconds', started)
        return [], np.empty((0, 12), dtype=np.float64)
    sizes = tuple(max(4, min(n, int(v))) for v in window_sizes)
    if not sizes:
        raise ValueError('window_sizes must not be empty')
    # First two windows deliberately use a common base and a 32-position shift.
    # Other windows explore a random location.
    base = (int(rng.integers(max(1, math.ceil(n / 64)))) * 64) % n
    actions = []
    window_count = math.ceil(count / 4)
    for w in range(window_count):
        size = sizes[w % len(sizes)]
        if w == 0:
            start = base
        elif w == 1:
            start = (base + 32) % n
        else:
            start = int(rng.integers(n))
        for operator in ('2opt', 'or1', 'or2', 'or3'):
            if len(actions) >= count:
                break
            actions.append({'start': start, 'size': size, 'operator': operator,
                            'seed': int(rng.integers(0, 2**31 - 1))})
    _charge(ledger, 'action_construction_seconds', started)
    started = time.perf_counter()
    old_edges = np.asarray(distances[route, np.roll(route, -1)], dtype=np.float64)
    normalizer = max(float(old_edges.mean()), 1.0)
    features = np.zeros((len(actions), 12), dtype=np.float64)
    for q, action in enumerate(actions):
        start, size = action['start'], action['size']
        indices = (np.arange(size, dtype=np.int64) + start) % n
        internal = old_edges[indices[:-1]]
        op = ('2opt', 'or1', 'or2', 'or3').index(action['operator'])
        features[q, op] = 1.0
        angle = 2.0 * math.pi * start / n
        features[q, 4:] = (
            size / max(sizes), float(internal.mean()) / normalizer,
            float(internal.max()) / normalizer, float(internal.std()) / normalizer,
            (old_edges[(start - 1) % n] + old_edges[indices[-1]]) / normalizer,
            parent_excess, math.cos(angle), math.sin(angle),
        )
    features = _features_finite(features)
    _charge(ledger, 'feature_seconds', started)
    return actions, features


def _candidate_table(size, operator):
    key = (size, operator)
    if key in _LOCAL_TABLES:
        return _LOCAL_TABLES[key]
    if operator == '2opt':
        a, b = np.triu_indices(size - 1, k=2)
        table = np.column_stack((a, b)).astype(np.int64)
    elif operator in ('or1', 'or2', 'or3'):
        length = int(operator[-1])
        table = np.array([(s, t) for s in range(1, size - length)
                          for t in range(size - 1)
                          if not s - 1 <= t <= s + length - 1], dtype=np.int64)
        table = table.reshape((-1, 2))
    else:
        raise ValueError('unknown operator')
    _LOCAL_TABLES[key] = table
    return table


def _unchanged(route, ledger, evaluations=0, deadline_hit=False):
    started = time.perf_counter()
    result = {'route': route.copy(), 'gain': 0, 'evaluations': int(evaluations),
              'accepted': False, 'move': None, 'deadline_hit': deadline_hit}
    _charge(ledger, 'application_seconds', started)
    return result


def evaluate_action(route, distances, action, budget=128, deadline=None,
                    ledger=None):
    """Evaluate at most budget sampled candidates; return an improved COPY.

    Exact gains are only computed here, after an action has been selected.
    One best positive-gain move is applied; there is no hidden local descent.
    deadline is absolute perf_counter. Candidate batches are indivisible.
    """
    if deadline is not None and time.perf_counter() >= deadline:
        return _unchanged(route, ledger, deadline_hit=True)
    started = time.perf_counter()
    n = len(route)
    size = min(n, int(action['size']))
    if size < 4 or budget <= 0:
        _charge(ledger, 'candidate_construction_seconds', started)
        return _unchanged(route, ledger)
    indices = (np.arange(size, dtype=np.int64) + int(action['start'])) % n
    path = route[indices]
    operator = action['operator']
    table = _candidate_table(size, operator)
    quota = min(len(table), int(budget))
    rng = np.random.default_rng(int(action['seed']))
    chosen = rng.choice(len(table), size=quota, replace=False)
    candidates = table[chosen]
    _charge(ledger, 'candidate_construction_seconds', started)
    if deadline is not None and time.perf_counter() >= deadline:
        return _unchanged(route, ledger, deadline_hit=True)
    if quota == 0:
        return _unchanged(route, ledger)
    started = time.perf_counter()
    x, y = candidates.T
    if operator == '2opt':
        a, b, c, d = path[x], path[x + 1], path[y], path[y + 1]
        gain = (distances[a, b].astype(np.int64) + distances[c, d]
                - distances[a, c] - distances[b, d])
    else:
        length = int(operator[-1])
        a, b = path[x - 1], path[x]
        c, d, u, v = path[x + length - 1], path[x + length], path[y], path[y + 1]
        gain = (distances[a, b].astype(np.int64) + distances[c, d] + distances[u, v]
                - distances[a, d] - distances[u, b] - distances[c, v])
    best = int(np.argmax(gain))
    best_gain = int(gain[best])
    _charge(ledger, 'exact_evaluation_seconds', started)
    if best_gain <= 0:
        return _unchanged(route, ledger, quota)
    started = time.perf_counter()
    x, y = map(int, candidates[best])
    newpath = path.copy()
    if operator == '2opt':
        newpath[x + 1:y + 1] = path[x + 1:y + 1][::-1]
    else:
        length = int(operator[-1])
        segment = path[x:x + length]
        if y < x - 1:
            newpath = np.concatenate((path[:y + 1], segment,
                                      path[y + 1:x], path[x + length:]))
        else:
            newpath = np.concatenate((path[:x], path[x + length:y + 1],
                                      segment, path[y + 1:]))
    newroute = route.copy()
    newroute[indices] = newpath
    result = {'route': newroute, 'gain': best_gain, 'evaluations': quota,
              'accepted': True, 'move': {'operator': operator, 'x': x, 'y': y,
                                         'start': int(action['start']), 'size': size},
              'deadline_hit': False}
    _charge(ledger, 'application_seconds', started)
    return result


def build_knn(distances, count=16, deadline=None, ledger=None):
    """Build a paid global nearest-neighbor table; None means deadline stop."""
    started = time.perf_counter()
    n = len(distances)
    k = min(max(int(count), 0), max(n - 1, 0))
    neighbors = np.empty((n, k), dtype=np.int64)
    take = min(n, k + 1)
    for city in range(n):
        if deadline is not None and time.perf_counter() >= deadline:
            _charge(ledger, 'knn_seconds', started)
            return None
        if k == 0:
            continue
        row = distances[city]
        candidates = np.argpartition(row, take - 1)[:take]
        candidates = candidates[candidates != city]
        order = np.lexsort((candidates, row[candidates]))
        neighbors[city] = candidates[order[:k]]
    _charge(ledger, 'knn_seconds', started)
    return neighbors


def enumerate_2opt_candidates(route, distances, rng=None, candidate_count=192,
                              knn=None, ledger=None):
    """Mix KNN, pairs of current long edges, and uniform exploration.

    Returns candidate dictionaries and Nx12 features. Features inspect removed
    and neighboring OLD edges plus KNN membership, never full added-edge costs.
    Training can build_knn once per map and pass it on successive states.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if knn is None:
        knn = build_knn(distances, ledger=ledger)
    started = time.perf_counter()
    n = len(route)
    if n < 4 or candidate_count <= 0:
        _charge(ledger, 'candidate_construction_seconds', started)
        return [], np.empty((0, 12), dtype=np.float64)
    old_edges = np.asarray(distances[route, np.roll(route, -1)], dtype=np.float64)
    long = np.argsort(old_edges)[-min(n, max(8, candidate_count // 4)):]
    position = np.empty(n, dtype=np.int64)
    position[route] = np.arange(n)
    candidates, seen = [], set()

    def add(i, j, kind):
        i, j = sorted((int(i), int(j)))
        if j - i < 2 or (i == 0 and j == n - 1) or (i, j) in seen:
            return
        seen.add((i, j))
        candidates.append({'i': i, 'j': j, 'kind': kind})

    third = max(1, int(candidate_count) // 3)
    for kind in ('knn', 'longedge', 'explore'):
        target = min(int(candidate_count), len(candidates) + third)
        for attempt in range(max(32, 8 * third)):
            if len(candidates) >= target:
                break
            if kind == 'knn' and knn.shape[1]:
                i = int(long[int(rng.integers(len(long)))]) if attempt % 2 else int(rng.integers(n))
                city = int(knn[route[i], int(rng.integers(knn.shape[1]))])
                j = int(position[city])
            elif kind == 'longedge':
                i, j = rng.choice(long, size=2, replace=False)
            else:
                i, j = rng.integers(n, size=2)
            add(i, j, kind)
    for _ in range(max(32, int(candidate_count) * 8)):
        if len(candidates) >= candidate_count:
            break
        i, j = rng.integers(n, size=2)
        add(i, j, 'explore')
    _charge(ledger, 'candidate_construction_seconds', started)
    started = time.perf_counter()
    normalizer = max(float(old_edges.mean()), 1.0)
    features = np.zeros((len(candidates), 12), dtype=np.float64)
    for q, candidate in enumerate(candidates):
        i, j = candidate['i'], candidate['j']
        a, b, c, d = route[i], route[(i + 1) % n], route[j], route[(j + 1) % n]
        features[q, :8] = (
            old_edges[i] / normalizer, old_edges[j] / normalizer, (j - i) / n,
            min(j - i, n - j + i) / n,
            float(np.any(knn[a] == c) or np.any(knn[c] == a)),
            float(np.any(knn[b] == d) or np.any(knn[d] == b)),
            old_edges[(i - 1) % n] / normalizer,
            old_edges[(j + 1) % n] / normalizer,
        )
        features[q, 8 + ('knn', 'longedge', 'explore').index(candidate['kind'])] = 1.0
        features[q, 11] = abs(old_edges[i] - old_edges[j]) / normalizer
    features = _features_finite(features)
    _charge(ledger, 'feature_seconds', started)
    return candidates, features


def evaluate_d_candidates(route, distances, candidates, order=None, budget=32,
                          deadline=None, ledger=None):
    """Evaluate only a bounded ranked prefix, then apply its best positive gain."""
    if deadline is not None and time.perf_counter() >= deadline:
        return _unchanged(route, ledger, deadline_hit=True)
    started = time.perf_counter()
    order = np.arange(len(candidates)) if order is None else np.asarray(order)
    chosen = order[:max(0, int(budget))]
    if len(chosen) == 0:
        _charge(ledger, 'candidate_selection_seconds', started)
        return _unchanged(route, ledger)
    pairs = np.array([(candidates[int(q)]['i'], candidates[int(q)]['j'])
                      for q in chosen], dtype=np.int64)
    i, j = pairs.T
    n = len(route)
    if np.any(i < 0) or np.any(j >= n) or np.any(j - i < 2) or np.any((i == 0) & (j == n - 1)):
        raise ValueError('invalid nonadjacent 2opt cuts')
    _charge(ledger, 'candidate_selection_seconds', started)
    if deadline is not None and time.perf_counter() >= deadline:
        return _unchanged(route, ledger, deadline_hit=True)
    started = time.perf_counter()
    a, b = route[i], route[i + 1]
    c, d = route[j], route[(j + 1) % n]
    gains = (distances[a, b].astype(np.int64) + distances[c, d]
             - distances[a, c] - distances[b, d])
    best = int(np.argmax(gains))
    gain = int(gains[best])
    _charge(ledger, 'exact_evaluation_seconds', started)
    if gain <= 0:
        return _unchanged(route, ledger, len(chosen))
    started = time.perf_counter()
    x, y = map(int, pairs[best])
    newroute = route.copy()
    newroute[x + 1:y + 1] = route[x + 1:y + 1][::-1]
    result = {'route': newroute, 'gain': gain, 'evaluations': len(chosen),
              'accepted': True, 'move': {'operator': '2opt', 'i': x, 'j': y,
                                         'candidate_index': int(chosen[best])},
              'deadline_hit': False}
    _charge(ledger, 'application_seconds', started)
    return result


def evaluate_d_action(route, distances, candidate, budget=1, deadline=None,
                      ledger=None):
    return evaluate_d_candidates(route, distances, [candidate], budget=min(1, budget),
                                 deadline=deadline, ledger=ledger)


def _policy_scores(policy, features):
    if hasattr(policy, 'score'):
        scores = np.asarray(policy.score(features), dtype=np.float64)
    elif isinstance(policy, dict):
        weights = np.asarray(policy['weights'], dtype=np.float64)
        transformed = features
        if 'feature_mean' in policy:
            transformed = transformed - np.asarray(policy['feature_mean'])
        if 'feature_scale' in policy:
            transformed = transformed / np.maximum(np.asarray(policy['feature_scale']), 1e-12)
        scores = transformed @ weights + float(policy.get('bias', 0.0))
    else:
        raise ValueError('C/D require a trained policy object or weights dictionary')
    if scores.shape != (len(features),) or not np.all(np.isfinite(scores)):
        raise ValueError('policy must return finite one-dimensional action logits')
    return scores


def _edge_signature(route):
    return frozenset((min(int(a), int(b)), max(int(a), int(b)))
                     for a, b in zip(route, np.roll(route, -1)))


def _insert_pool(pool, route, cost, next_id, limit):
    signature = _edge_signature(route)
    if any(entry['signature'] == signature for entry in pool):
        return pool, next_id, False
    child = {'route': route, 'cost': int(cost), 'id': next_id,
             'signature': signature}
    pool = pool + [child]
    best = min(pool, key=lambda entry: (entry['cost'], entry['id']))
    others = [entry for entry in pool if entry is not best]
    # Retain genuine alternate parents, preferring edge diversity to incumbent.
    others.sort(key=lambda entry: (-len(entry['signature'] - best['signature']),
                                   entry['cost'], entry['id']))
    retained = [best] + others[:max(0, limit - 1)]
    inserted = any(entry['id'] == next_id for entry in retained)
    return retained, next_id + 1, inserted


class TourPool:
    """Shared training/test state and deterministic parent/retention rules.

    Initialization contains just the paid NN route. Accepted children grow a
    pool of at most pool_size tours. The incumbent is always retained; other
    entries maximize undirected-edge difference from it. Parent selection is
    step % current pool size, so archived parents actually drive later moves.
    """
    def __init__(self, route, distances, pool_size=3):
        if pool_size < 1:
            raise ValueError('pool_size must be positive')
        self.limit = int(pool_size)
        self.initial_length = tour_length(route, distances)
        self.best_length = self.initial_length
        self.best_route = route.copy()
        self.next_id = 1
        self.entries = [{'route': route.copy(), 'cost': self.initial_length,
                         'id': 0, 'signature': _edge_signature(route)}]

    def parent(self, step):
        return self.entries[int(step) % len(self.entries)]

    def update(self, parent, result):
        parent_gain = int(result['gain'])
        if parent_gain < 0 or (bool(result['accepted']) != (parent_gain > 0)):
            raise ValueError('only strictly improving accepted moves are valid')
        best_gain, inserted = 0, False
        if result['accepted']:
            child_cost = int(parent['cost']) - parent_gain
            self.entries, self.next_id, inserted = _insert_pool(
                self.entries, result['route'], child_cost, self.next_id, self.limit)
            if child_cost < self.best_length:
                best_gain = self.best_length - child_cost
                self.best_length = child_cost
                self.best_route = result['route']
        return {'parent_gain': parent_gain, 'best_gain': best_gain,
                'inserted': inserted}


def sample_policy_action(scores, rng):
    """Stable softmax categorical sampling shared with REINFORCE evaluation."""
    scores = np.asarray(scores, dtype=np.float64)
    probability = np.exp(scores - float(scores.max()))
    probability /= probability.sum()
    return int(rng.choice(len(scores), p=probability))


def optimize(route, distances, mode, deadline, emit, policy=None, seed=1,
             settings=None, coords=None):
    """Mutate route to best checkpoint and return a serializable cost ledger.

    B/C use the same rotating parent schedule, diverse pool, window candidates
    and 128-evaluation quota; B cycles operators and uses a fixed old-edge
    priority while C samples from trained softmax logits. D samples one shared
    classical edge candidate from learned softmax logits and evaluates it.
    Every clock is wall time; ledger categories are exclusive. Model loading,
    dense-matrix preparation and common NN are charged by the calling worker.
    """
    started = time.perf_counter()
    mode = str(mode).upper()
    if mode not in ('B', 'C', 'D'):
        raise ValueError('mode must be B, C or D')
    config = dict(DEFAULT_SETTINGS)
    if settings:
        config.update(settings)
    if config['pool_size'] < 1:
        raise ValueError('pool_size must be positive')
    ledger = {}
    stats = {'mode': mode, 'settings': config, 'steps': 0, 'accepted_moves': 0,
             'gain': 0, 'evaluations': 0, 'proposed_candidates': 0,
             'pool_insertions': 0, 'max_pool_size': 1, 'parent_use_counts': {},
             'diverse_parent_steps': 0, 'parent_gain_sum': 0,
             'best_gain_sum': 0, 'decisions': [], 'ledger': ledger,
             'stop_reason': 'step_limit', 'feature_names': list(
                 EDGE_FEATURE_NAMES if mode == 'D' else ACTION_FEATURE_NAMES)}
    proposal_rng, selection_rng = make_rng_streams(seed)
    setup_started = time.perf_counter()
    tour_pool = TourPool(route, distances, config['pool_size'])
    initial_cost = tour_pool.initial_length
    _charge(ledger, 'pool_seconds', setup_started)
    best_cost = initial_cost
    knn = None
    if mode == 'D':
        knn = build_knn(distances, config['knn'], deadline, ledger)
        if knn is None:
            stats['stop_reason'] = 'deadline'

    def publish(stage):
        publish_started = time.perf_counter()
        stats['elapsed'] = time.perf_counter() - started
        stats['best_length'] = best_cost
        stats['gain'] = initial_cost - best_cost
        if emit is not None:
            emit(route.copy(), stage, copy.deepcopy(stats))
        _charge(ledger, 'publication_seconds', publish_started)

    for step in range(int(config['local_max_steps'])):
        if time.perf_counter() >= deadline or (mode == 'D' and knn is None):
            stats['stop_reason'] = 'deadline'
            break
        parent_started = time.perf_counter()
        parent = tour_pool.parent(step)
        parent_id = str(parent['id'])
        stats['parent_use_counts'][parent_id] = stats['parent_use_counts'].get(parent_id, 0) + 1
        stats['diverse_parent_steps'] += int(parent['cost'] > best_cost)
        _charge(ledger, 'pool_seconds', parent_started)
        if mode in ('B', 'C'):
            actions, features = enumerate_actions(
                parent['route'], distances, proposal_rng, config['action_count'],
                config['window_sizes'], (parent['cost'] - best_cost) / max(best_cost, 1), ledger)
            if not actions:
                stats['stop_reason'] = 'empty_actions'
                break
            choose_started = time.perf_counter()
            if mode == 'B':
                # Same features and pool as C; no true candidate gain is known.
                scores = features[:, 6] + 0.5 * features[:, 5] + 0.25 * features[:, 7]
                scores = scores + 100.0 * features[:, step % 4]
            else:
                scores = _policy_scores(policy, features)
            selected = (int(np.argmax(scores)) if mode == 'B'
                        else sample_policy_action(scores, selection_rng))
            _charge(ledger, 'inference_seconds', choose_started)
            if time.perf_counter() >= deadline:
                stats['stop_reason'] = 'deadline'
                break
            action = actions[selected]
            result = evaluate_action(parent['route'], distances, action,
                                     config['candidate_quota'], deadline, ledger)
            stats['proposed_candidates'] += len(actions)
        else:
            actions, features = enumerate_2opt_candidates(
                parent['route'], distances, proposal_rng, config['d_candidate_count'], knn, ledger)
            if not actions:
                stats['stop_reason'] = 'empty_candidates'
                break
            choose_started = time.perf_counter()
            scores = _policy_scores(policy, features)
            selected = sample_policy_action(scores, selection_rng)
            _charge(ledger, 'inference_seconds', choose_started)
            if time.perf_counter() >= deadline:
                stats['stop_reason'] = 'deadline'
                break
            action = actions[selected]
            if config['d_exact_quota'] != 1:
                raise ValueError('D must evaluate one sampled candidate to match training')
            result = evaluate_d_action(parent['route'], distances, actions[selected],
                                      1, deadline, ledger)
            stats['proposed_candidates'] += len(actions)
        stats['steps'] += 1
        stats['evaluations'] += result['evaluations']
        audit_started = time.perf_counter()
        if len(stats['decisions']) < config['audit_decisions']:
            stats['decisions'].append({
                'step': step, 'parent_id': parent['id'], 'parent_length': parent['cost'],
                'action_count': len(actions), 'selected_index': selected,
                'selected_action': dict(action), 'selected_features': features[selected].tolist(),
                'selected_logit': float(scores[selected]), 'gain': result['gain'],
                'parent_gain': result['gain'], 'best_gain': 0,
                'evaluations': result['evaluations'], 'move': result['move'],
                'pool_size': len(tour_pool.entries),
            })
        _charge(ledger, 'audit_seconds', audit_started)
        if result['accepted']:
            stats['accepted_moves'] += 1
            pool_started = time.perf_counter()
            transition = tour_pool.update(parent, result)
            stats['parent_gain_sum'] += transition['parent_gain']
            stats['best_gain_sum'] += transition['best_gain']
            if stats['decisions'] and stats['decisions'][-1]['step'] == step:
                stats['decisions'][-1]['best_gain'] = transition['best_gain']
            stats['pool_insertions'] += int(transition['inserted'])
            stats['max_pool_size'] = max(stats['max_pool_size'], len(tour_pool.entries))
            improved = transition['best_gain'] > 0
            if improved:
                best_cost = tour_pool.best_length
                route[:] = tour_pool.best_route
            _charge(ledger, 'pool_seconds', pool_started)
            if improved and (stats['accepted_moves'] % 8 == 0):
                publish('local_improved')
        if result['deadline_hit']:
            stats['stop_reason'] = 'deadline'
            break
    publish('local_finished')
    stats['elapsed'] = time.perf_counter() - started
    stats['best_length'] = best_cost
    stats['gain'] = initial_cost - best_cost
    stats['accounted_seconds'] = sum(ledger.values())
    stats['unattributed_seconds'] = max(0.0, stats['elapsed'] - stats['accounted_seconds'])
    stats['pool_final'] = [{'id': p['id'], 'length': p['cost']} for p in tour_pool.entries]
    return stats
