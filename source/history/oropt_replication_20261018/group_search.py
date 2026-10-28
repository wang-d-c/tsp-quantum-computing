"""Time-budgeted grouped improvements for a symmetric integer TSP.

The route is a permutation without a repeated closing city.  All windows are
OPEN paths with their first and last city fixed; the surrounding tour edges
therefore keep their costs.  optimize_grouped mutates route in place and emits
copied, valid full-tour checkpoints.  Numba cache is deliberately disabled:
compilation happens inside, and is charged to, the supplied absolute deadline.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from numba import njit


@njit(cache=False)
def _or_opt_step(path, distances):
    """Best strictly improving, orientation-preserving relocation of 1..3 cities."""
    m = len(path)
    best_gain = np.int64(0)
    best_s = -1
    best_l = 0
    best_t = -1
    for length in range(1, 4):
        for s in range(1, m - length):
            e = s + length - 1
            a, b, c, d = path[s - 1], path[s], path[e], path[e + 1]
            for t in range(m - 1):
                if s - 1 <= t <= e:
                    continue
                u, v = path[t], path[t + 1]
                old = (np.int64(distances[a, b]) + distances[c, d]
                       + distances[u, v])
                new = (np.int64(distances[a, d]) + distances[u, b]
                       + distances[c, v])
                gain = old - new
                if gain > best_gain:
                    best_gain, best_s, best_l, best_t = gain, s, length, t
    if best_gain > 0:
        previous = path.copy()
        s, e, t = best_s, best_s + best_l - 1, best_t
        q = 0
        if t < s - 1:
            for p in range(t + 1):
                path[q] = previous[p]
                q += 1
            for p in range(s, e + 1):
                path[q] = previous[p]
                q += 1
            for p in range(t + 1, s):
                path[q] = previous[p]
                q += 1
            for p in range(e + 1, m):
                path[q] = previous[p]
                q += 1
        else:
            for p in range(s):
                path[q] = previous[p]
                q += 1
            for p in range(e + 1, t + 1):
                path[q] = previous[p]
                q += 1
            for p in range(s, e + 1):
                path[q] = previous[p]
                q += 1
            for p in range(t + 1, m):
                path[q] = previous[p]
                q += 1
    return best_gain, best_l, best_s, best_t


@njit(cache=False)
def _candidate_mask(path, neighbors):
    """A symmetric global-KNN relation, indexed by current window positions."""
    m = len(path)
    mask = np.zeros((m, m), dtype=np.bool_)
    for a in range(m):
        for b in range(a + 1, m):
            linked = False
            for p in range(neighbors.shape[1]):
                if neighbors[path[a], p] == path[b] or neighbors[path[b], p] == path[a]:
                    linked = True
                    break
            mask[a, b] = linked
            mask[b, a] = linked
    return mask


@njit(cache=False)
def _three_opt_step(path, distances, mask):
    """Best of the four pure 3-opt reconnections, with a KNN new-edge filter.

    For cuts (a,b), (c,d), (e,f), let B=b..c and C=d..e.  The four
    patterns are rev(B) rev(C), C B, rev(C) B, and C rev(B).
    The other three nonidentity reconnections are 2-opt and are omitted.
    Cut edges must be pairwise nonadjacent on the open window path.
    At least one ADDED edge must be in the symmetric KNN candidate relation.
    """
    m = len(path)
    best_gain = np.int64(0)
    best_pattern, best_i, best_j, best_k = 0, -1, -1, -1
    for i in range(m - 5):
        a, b = path[i], path[i + 1]
        for j in range(i + 2, m - 3):
            c, d = path[j], path[j + 1]
            for k in range(j + 2, m - 1):
                e, f = path[k], path[k + 1]
                old = (np.int64(distances[a, b]) + distances[c, d]
                       + distances[e, f])
                if mask[i, j] or mask[i + 1, k] or mask[j + 1, k + 1]:
                    new = (np.int64(distances[a, c]) + distances[b, e]
                           + distances[d, f])
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        best_pattern, best_i, best_j, best_k = 1, i, j, k
                if mask[i, j + 1] or mask[k, i + 1] or mask[j, k + 1]:
                    new = (np.int64(distances[a, d]) + distances[e, b]
                           + distances[c, f])
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        best_pattern, best_i, best_j, best_k = 2, i, j, k
                if mask[i, k] or mask[j + 1, i + 1] or mask[j, k + 1]:
                    new = (np.int64(distances[a, e]) + distances[d, b]
                           + distances[c, f])
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        best_pattern, best_i, best_j, best_k = 3, i, j, k
                if mask[i, j + 1] or mask[k, j] or mask[i + 1, k + 1]:
                    new = (np.int64(distances[a, d]) + distances[e, c]
                           + distances[b, f])
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        best_pattern, best_i, best_j, best_k = 4, i, j, k
    if best_gain > 0:
        previous = path.copy()
        i, j, k = best_i, best_j, best_k
        q = i + 1
        if best_pattern == 1:
            for p in range(j, i, -1):
                path[q] = previous[p]
                q += 1
            for p in range(k, j, -1):
                path[q] = previous[p]
                q += 1
        elif best_pattern == 2:
            for p in range(j + 1, k + 1):
                path[q] = previous[p]
                q += 1
            for p in range(i + 1, j + 1):
                path[q] = previous[p]
                q += 1
        elif best_pattern == 3:
            for p in range(k, j, -1):
                path[q] = previous[p]
                q += 1
            for p in range(i + 1, j + 1):
                path[q] = previous[p]
                q += 1
        else:
            for p in range(j + 1, k + 1):
                path[q] = previous[p]
                q += 1
            for p in range(j, i, -1):
                path[q] = previous[p]
                q += 1
    return best_gain, best_pattern, best_i, best_j, best_k


def _window_indices(n, group_size, shift=0):
    """A cyclic partition; its final short window never duplicates a city."""
    for consumed in range(0, n, group_size):
        size = min(group_size, n - consumed)
        yield (np.arange(size, dtype=np.int64) + shift + consumed) % n


def _build_neighbors(distances, candidate_count, deadline):
    n = len(distances)
    k = min(max(0, candidate_count), n - 1)
    result = np.empty((n, k), dtype=np.int64)
    if k == 0:
        return result
    take = min(n, k + 1)
    for city in range(n):
        if time.perf_counter() >= deadline:
            return None
        row = distances[city]
        candidates = np.argpartition(row, take - 1)[:take]
        candidates = candidates[candidates != city]
        order = np.lexsort((candidates, row[candidates]))
        result[city] = candidates[order[:k]]
    return result


def optimize_grouped(route, distances, mode, deadline, emit,
                     group_size=64, candidate_count=12):
    """Mutate route in place and return budget/accounting statistics.

    B: at most two passes of original-window Or-opt, 32 moves/window/pass.
    C: each of at most two rounds runs original Or-opt, original pure 3-opt,
       half-window-shifted Or-opt, shifted pure 3-opt.  3-opt has an eight-move
       cap per window/round.  Only improving moves are accepted.  The fixed
       caps ensure one difficult window does not monopolize a stage.

    deadline is an ABSOLUTE time.perf_counter() value.  The caller must start
    its total clock before imports/common construction; all time here counts.
    Checks occur between kernels and during KNN construction.  A first-call
    Numba compilation cannot be interrupted within this function and may
    overshoot; first_kernel_call_seconds reports compilation plus execution.
    emit(route_copy, stage_string, deep_copied_stats) is called at stage ends,
    including an interrupted stage.  Callback time is inside the budget.
    """
    started = time.perf_counter()
    mode = str(mode).upper()
    if mode not in ("B", "C"):
        raise ValueError("mode must be B or C")
    if route.ndim != 1 or distances.shape != (len(route), len(route)):
        raise ValueError("route and distance matrix shapes differ")
    if group_size < 4:
        raise ValueError("group_size must be at least four")
    stats = {
        "mode": mode, "group_size": int(group_size),
        "candidate_count": int(candidate_count), "rounds_started": 0,
        "stages_reached": [], "stage_details": [], "window_passes": 0,
        "or_moves": 0, "three_moves": 0, "gain": 0,
        "jit_compilation_included": True,
        "first_kernel_call_seconds": {}, "knn_seconds": 0.0,
        "stop_reason": "converged", "elapsed": 0.0,
    }
    n = len(route)
    neighbors = None

    def checkpoint(stage):
        stats["elapsed"] = time.perf_counter() - started
        if emit is not None:
            emit(route.copy(), stage, copy.deepcopy(stats))

    def charged_call(label, kernel, *args):
        if label not in stats["first_kernel_call_seconds"]:
            call_started = time.perf_counter()
            result = kernel(*args)
            stats["first_kernel_call_seconds"][label] = time.perf_counter() - call_started
            return result
        return kernel(*args)

    def stage_run(stage, shift, operation):
        nonlocal neighbors
        detail = {"stage": stage, "windows": 0, "moves": 0, "gain": 0,
                  "completed": False}
        stats["stages_reached"].append(stage)
        stats["stage_details"].append(detail)
        stage_started = time.perf_counter()
        if operation == "three" and neighbors is None:
            knn_started = time.perf_counter()
            neighbors = _build_neighbors(distances, candidate_count, deadline)
            stats["knn_seconds"] += time.perf_counter() - knn_started
            if neighbors is None:
                stats["stop_reason"] = "deadline"
                detail["elapsed"] = time.perf_counter() - stage_started
                checkpoint(stage)
                return False
        for indices in _window_indices(n, group_size, shift):
            if time.perf_counter() >= deadline:
                stats["stop_reason"] = "deadline"
                break
            if len(indices) < (4 if operation == "or" else 6):
                continue
            path = np.asarray(route[indices], dtype=np.int64).copy()
            detail["windows"] += 1
            stats["window_passes"] += 1
            move_limit = 32 if operation == "or" else 8
            for _ in range(move_limit):
                if time.perf_counter() >= deadline:
                    stats["stop_reason"] = "deadline"
                    break
                if operation == "or":
                    result = charged_call("or_opt", _or_opt_step, path, distances)
                else:
                    mask = charged_call("candidate_mask", _candidate_mask, path, neighbors)
                    if time.perf_counter() >= deadline:
                        stats["stop_reason"] = "deadline"
                        break
                    result = charged_call("three_opt", _three_opt_step,
                                          path, distances, mask)
                gain = int(result[0])
                if gain <= 0:
                    break
                route[indices] = path
                detail["moves"] += 1
                detail["gain"] += gain
                stats["gain"] += gain
                stats["or_moves" if operation == "or" else "three_moves"] += 1
            if stats["stop_reason"] == "deadline":
                break
        else:
            detail["completed"] = True
        detail["elapsed"] = time.perf_counter() - stage_started
        checkpoint(stage)
        return detail["completed"]

    for round_index in range(1, 3):
        if time.perf_counter() >= deadline:
            stats["stop_reason"] = "deadline"
            break
        stats["rounds_started"] = round_index
        before = stats["gain"]
        if not stage_run(f"original_or_r{round_index}", 0, "or"):
            break
        if mode == "C":
            if not stage_run(f"original_three_r{round_index}", 0, "three"):
                break
            if not stage_run(f"shifted_or_r{round_index}", group_size // 2, "or"):
                break
            if not stage_run(f"shifted_three_r{round_index}", group_size // 2, "three"):
                break
        if stats["gain"] == before:
            break
    else:
        stats["stop_reason"] = "round_limit"
    if time.perf_counter() >= deadline:
        stats["stop_reason"] = "deadline"
    stats["elapsed"] = time.perf_counter() - started
    return stats
