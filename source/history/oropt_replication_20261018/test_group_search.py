"""Independent cost/permutation checks; run `python work/test_group_search.py`."""

import time
import unittest

import numpy as np

from group_search import (_candidate_mask, _or_opt_step, _three_opt_step,
                          _window_indices, optimize_grouped)


def matrix(seed, n):
    xy = np.random.default_rng(seed).uniform(0, 1000, size=(n, 2))
    return np.floor(np.linalg.norm(xy[:, None] - xy[None, :], axis=2) + 0.5).astype(np.int32)


def path_cost(path, distances):
    return sum(int(distances[a, b]) for a, b in zip(path[:-1], path[1:]))


def tour_cost(path, distances):
    return path_cost(path, distances) + int(distances[path[-1], path[0]])


def reconnect(path, i, j, k, pattern):
    b, c = list(path[i + 1:j + 1]), list(path[j + 1:k + 1])
    middle = {1: b[::-1] + c[::-1], 2: c + b,
              3: c[::-1] + b, 4: c + b[::-1]}[pattern]
    return np.array(list(path[:i + 1]) + middle + list(path[k + 1:]), dtype=np.int64)


def added_position_edges(i, j, k, pattern):
    return {
        1: [(i, j), (i + 1, k), (j + 1, k + 1)],
        2: [(i, j + 1), (k, i + 1), (j, k + 1)],
        3: [(i, k), (j + 1, i + 1), (j, k + 1)],
        4: [(i, j + 1), (k, j), (i + 1, k + 1)],
    }[pattern]


class GroupSearchTests(unittest.TestCase):
    def test_or_opt_against_full_cost_enumeration(self):
        for seed in range(15):
            n = 9
            distances = matrix(seed, n)
            old = np.random.default_rng(seed + 50).permutation(n).astype(np.int64)
            best = 0
            original = path_cost(old, distances)
            for length in (1, 2, 3):
                for s in range(1, n - length):
                    e = s + length - 1
                    for t in range(n - 1):
                        if s - 1 <= t <= e:
                            continue
                        segment = list(old[s:e + 1])
                        remaining = list(old[:s]) + list(old[e + 1:])
                        insert_at = t + 1 if t < s else t + 1 - length
                        new = remaining[:insert_at] + segment + remaining[insert_at:]
                        best = max(best, original - path_cost(new, distances))
            new = old.copy()
            gain, *_ = _or_opt_step(new, distances)
            self.assertEqual(int(gain), best)
            self.assertEqual(original - path_cost(new, distances), int(gain))
            self.assertEqual(tour_cost(old, distances) - tour_cost(new, distances), int(gain))
            np.testing.assert_array_equal(np.sort(old), np.sort(new))
            self.assertEqual((old[0], old[-1]), (new[0], new[-1]))

    def test_three_opt_four_patterns_and_candidate_filter(self):
        seen_patterns = set()
        for seed in range(100):
            n = 9
            distances = matrix(seed + 200, n)
            old = np.random.default_rng(seed).permutation(n).astype(np.int64)
            rng = np.random.default_rng(seed + 300)
            mask = rng.random((n, n)) < 0.17
            mask |= mask.T
            original = path_cost(old, distances)
            best = 0
            for i in range(n - 5):
                for j in range(i + 2, n - 3):
                    for k in range(j + 2, n - 1):
                        for pattern in range(1, 5):
                            if not any(mask[a, b] for a, b in added_position_edges(i, j, k, pattern)):
                                continue
                            new = reconnect(old, i, j, k, pattern)
                            best = max(best, original - path_cost(new, distances))
            new = old.copy()
            gain, pattern, i, j, k = _three_opt_step(new, distances, mask)
            self.assertEqual(best, int(gain))
            self.assertEqual(original - path_cost(new, distances), int(gain))
            self.assertEqual(tour_cost(old, distances) - tour_cost(new, distances), int(gain))
            np.testing.assert_array_equal(np.sort(old), np.sort(new))
            self.assertEqual((old[0], old[-1]), (new[0], new[-1]))
            if gain > 0:
                seen_patterns.add(pattern)
                np.testing.assert_array_equal(new, reconnect(old, i, j, k, pattern))
                self.assertTrue(any(mask[a, b] for a, b in added_position_edges(i, j, k, pattern)))
        self.assertEqual(seen_patterns, {1, 2, 3, 4})

    def test_candidate_mask_uses_both_knn_directions(self):
        neighbors = np.array([[2], [0], [1], [2]], dtype=np.int64)
        path = np.array([3, 0, 2, 1], dtype=np.int64)
        actual = _candidate_mask(path, neighbors)
        for i, a in enumerate(path):
            for j, b in enumerate(path):
                expected = i != j and (b in neighbors[a] or a in neighbors[b])
                self.assertEqual(bool(actual[i, j]), expected)

    def test_wrapped_window_gain_is_full_tour_gain(self):
        n = 23
        distances = matrix(900, n)
        route = np.random.default_rng(88).permutation(n).astype(np.int64)
        windows = list(_window_indices(n, 8, 4))
        np.testing.assert_array_equal(np.sort(np.concatenate(windows)), np.arange(n))
        self.assertTrue(any(np.any(np.diff(w) < 0) for w in windows))
        accepted = 0
        for indices in windows:
            old_cost = tour_cost(route, distances)
            old_window = route[indices].copy()
            new_window = old_window.copy()
            gain, *_ = _or_opt_step(new_window, distances)
            route[indices] = new_window
            self.assertEqual(old_cost - tour_cost(route, distances), int(gain))
            self.assertEqual((old_window[0], old_window[-1]), (new_window[0], new_window[-1]))
            accepted += int(gain > 0)
        self.assertGreater(accepted, 0)

    def test_modes_checkpoints_and_accounted_gain(self):
        for mode in ("B", "C"):
            n = 113
            distances = matrix(52, n)
            route = np.random.default_rng(53).permutation(n).astype(np.int64)
            initial = route.copy()
            original = tour_cost(route, distances)
            last_cost = original
            stages = []

            def emit(checkpoint, stage, stats):
                nonlocal last_cost
                np.testing.assert_array_equal(np.sort(checkpoint), np.arange(n))
                current = tour_cost(checkpoint, distances)
                self.assertLessEqual(current, last_cost)
                self.assertEqual(original - current, stats["gain"])
                stages.append(stage)
                last_cost = current

            stats = optimize_grouped(route, distances, mode,
                                     time.perf_counter() + 30, emit,
                                     group_size=16, candidate_count=6)
            self.assertEqual(original - tour_cost(route, distances), stats["gain"])
            self.assertGreater(stats["or_moves"], 0)
            self.assertGreater(len(stages), 0)
            if mode == "B":
                for window in _window_indices(n, 16):
                    np.testing.assert_array_equal(route[window[[0, -1]]], initial[window[[0, -1]]])
            else:
                self.assertIn("shifted_three_r1", stages)
                self.assertGreater(stats["three_moves"], 0)

    def test_expired_deadline_leaves_route_unchanged(self):
        route = np.arange(12, dtype=np.int64)
        original = route.copy()
        stats = optimize_grouped(route, matrix(1, 12), "C",
                                 time.perf_counter() - 1, None, group_size=8)
        np.testing.assert_array_equal(route, original)
        self.assertEqual(stats["stop_reason"], "deadline")
        self.assertEqual(stats["stages_reached"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
