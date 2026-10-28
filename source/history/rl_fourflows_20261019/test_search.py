"""Independent correctness checks for the shared training/test search engine."""
import time
import unittest

import numpy as np

import search


def distances(seed, n):
    xy = np.random.default_rng(seed).uniform(0, 1000, (n, 2))
    return np.floor(np.linalg.norm(xy[:, None] - xy[None, :], axis=2) + .5).astype(np.int32)


def independent_cost(route, dist):
    return sum(int(dist[route[i], route[(i + 1) % len(route)]]) for i in range(len(route)))


def independent_window_moves(route, action):
    n = len(route)
    indices = (np.arange(action['size']) + action['start']) % n
    path = list(route[indices])
    m = len(path)
    if action['operator'] == '2opt':
        for i in range(m - 3):
            for j in range(i + 2, m - 1):
                new = path[:i + 1] + path[i + 1:j + 1][::-1] + path[j + 1:]
                result = route.copy()
                result[indices] = new
                yield result
    else:
        length = int(action['operator'][-1])
        for s in range(1, m - length):
            segment = path[s:s + length]
            remaining = path[:s] + path[s + length:]
            for t in range(m - 1):
                if s - 1 <= t <= s + length - 1:
                    continue
                at = t + 1 if t < s else t + 1 - length
                new = remaining[:at] + segment + remaining[at:]
                result = route.copy()
                result[indices] = new
                yield result


class SearchTests(unittest.TestCase):
    def test_all_local_operators_against_full_tour_enumeration(self):
        for seed in range(8):
            d = distances(seed, 13)
            route = np.random.default_rng(seed + 12).permutation(13)
            original = independent_cost(route, d)
            for start in (1, 10):
                for operator in ('2opt', 'or1', 'or2', 'or3'):
                    action = {'start': start, 'size': 9, 'operator': operator, 'seed': 77}
                    best = max([0] + [original - independent_cost(r, d)
                                     for r in independent_window_moves(route, action)])
                    result = search.evaluate_action(route, d, action, budget=100000)
                    new = result['route']
                    self.assertEqual(result['gain'], best)
                    self.assertEqual(original - independent_cost(new, d), best)
                    np.testing.assert_array_equal(np.sort(new), np.arange(13))
                    fixed = [start % 13, (start + 8) % 13]
                    np.testing.assert_array_equal(new[fixed], route[fixed])
                    outside = [i for i in range(13) if i not in ((np.arange(9) + start) % 13)]
                    np.testing.assert_array_equal(new[outside], route[outside])

    def test_candidate_quota_reproducibility_and_deadline(self):
        d = distances(10, 65)
        route = np.random.default_rng(9).permutation(65)
        actions, _ = search.enumerate_actions(route, d, np.random.default_rng(2))
        for action in actions:
            first = search.evaluate_action(route, d, action, 17)
            second = search.evaluate_action(route, d, action, 17)
            self.assertLessEqual(first['evaluations'], 17)
            self.assertEqual(first['gain'], second['gain'])
            np.testing.assert_array_equal(first['route'], second['route'])
        expired = search.evaluate_action(route, d, actions[0], deadline=time.perf_counter() - 1)
        self.assertEqual(expired['evaluations'], 0)
        self.assertTrue(expired['deadline_hit'])
        np.testing.assert_array_equal(expired['route'], route)

    def test_feature_scale_and_no_unselected_gain_queries(self):
        d = distances(22, 80)
        route = np.random.default_rng(4).permutation(80)
        actions, x = search.enumerate_actions(route, d, np.random.default_rng(7))
        actions2, y = search.enumerate_actions(route, d * 7, np.random.default_rng(7))
        self.assertEqual(actions, actions2)
        np.testing.assert_allclose(x, y, atol=1e-12)
        # Modifying every distance that is not an OLD route edge cannot change
        # window features: no action candidate's full true gain is evaluated.
        modified = d.copy()
        old_edges = {tuple(sorted((int(a), int(b)))) for a, b in zip(route, np.roll(route, -1))}
        for a in range(80):
            for b in range(a + 1, 80):
                if (a, b) not in old_edges:
                    modified[a, b] = modified[b, a] = int(d[a, b]) + 30000
        _, z = search.enumerate_actions(route, modified, np.random.default_rng(7))
        np.testing.assert_array_equal(x, z)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertEqual(x.shape, (12, 12))

    def test_d_candidates_and_reversals_have_exact_global_gain(self):
        for seed in range(10):
            n = 31
            d = distances(seed + 90, n)
            route = np.random.default_rng(seed).permutation(n)
            knn = search.build_knn(d, 16)
            candidates, features = search.enumerate_2opt_candidates(
                route, d, np.random.default_rng(seed + 200), 96, knn)
            self.assertEqual(features.shape, (len(candidates), 12))
            self.assertEqual(len({(c['i'], c['j']) for c in candidates}), len(candidates))
            original = independent_cost(route, d)
            expected = 0
            for candidate in candidates:
                i, j = candidate['i'], candidate['j']
                new = route.copy()
                new[i + 1:j + 1] = new[i + 1:j + 1][::-1]
                gain = max(0, original - independent_cost(new, d))
                result = search.evaluate_d_action(route, d, candidate)
                self.assertEqual(gain, result['gain'])
                self.assertEqual(original - independent_cost(result['route'], d), result['gain'])
                self.assertEqual(result['route'][0], route[0])
                np.testing.assert_array_equal(np.sort(result['route']), np.arange(n))
                expected = max(expected, gain)
            all_result = search.evaluate_d_candidates(route, d, candidates, budget=len(candidates))
            self.assertEqual(all_result['gain'], expected)
            # The reverse orientation is the same undirected tour and has the
            # same improvement available when both cut positions are mapped.
            first = candidates[0]
            i, j = first['i'], first['j']
            reversed_route = route[::-1].copy()
            ri, rj = sorted(((n - 2 - i) % n, (n - 2 - j) % n))
            mirrored = search.evaluate_d_action(reversed_route, d, {'i': ri, 'j': rj})
            normal = search.evaluate_d_action(route, d, first)
            self.assertEqual(mirrored['gain'], normal['gain'])

    def test_d_features_do_not_contain_added_edge_costs(self):
        d = distances(3, 80)
        route = np.random.default_rng(2).permutation(80)
        knn = search.build_knn(d, 16)
        candidates, features = search.enumerate_2opt_candidates(route, d, np.random.default_rng(1), 96, knn)
        # With KNN supplied/fixed, only existing route-edge values are queried.
        modified = d.copy()
        old = {tuple(sorted((int(a), int(b)))) for a, b in zip(route, np.roll(route, -1))}
        for a in range(80):
            for b in range(a + 1, 80):
                if (a, b) not in old:
                    modified[a, b] = modified[b, a] = 123456
        again, other = search.enumerate_2opt_candidates(route, modified, np.random.default_rng(1), 96, knn)
        self.assertEqual(candidates, again)
        np.testing.assert_array_equal(features, other)
        _, scaled = search.enumerate_2opt_candidates(route, d * 7, np.random.default_rng(1), 96, knn)
        np.testing.assert_allclose(features, scaled, atol=1e-12)

    def test_pool_transition_shared_best_reward_and_alternate_parents(self):
        d = distances(101, 96)
        original = np.random.default_rng(99).permutation(96)
        pool = search.TourPool(original, d, 3)
        rng = np.random.default_rng(12)
        rewards = 0
        visited = set()
        for step in range(40):
            parent = pool.parent(step)
            visited.add(parent['id'])
            actions, _ = search.enumerate_actions(parent['route'], d, rng)
            result = search.evaluate_action(parent['route'], d, actions[step % len(actions)])
            before = pool.best_length
            change = pool.update(parent, result)
            self.assertEqual(change['best_gain'], before - pool.best_length)
            self.assertEqual(change['parent_gain'], result['gain'])
            rewards += change['best_gain']
            self.assertEqual(pool.best_length, independent_cost(pool.best_route, d))
            self.assertLessEqual(len(pool.entries), 3)
            for entry in pool.entries:
                self.assertEqual(entry['cost'], independent_cost(entry['route'], d))
                np.testing.assert_array_equal(np.sort(entry['route']), np.arange(96))
        self.assertGreater(len(visited), 1)
        self.assertEqual(rewards, independent_cost(original, d) - pool.best_length)

    def test_optimize_budget_ledger_and_policies(self):
        d = distances(33, 96)
        original = np.random.default_rng(20).permutation(96)
        for mode in ('B', 'C', 'D'):
            route = original.copy()
            lengths = []

            def emit(new, stage, stats):
                lengths.append(independent_cost(new, d))
                np.testing.assert_array_equal(np.sort(new), np.arange(96))

            stats = search.optimize(route, d, mode, time.perf_counter() + 10,
                                    emit, {'weights': [0.0] * 12}, seed=7,
                                    settings={'local_max_steps': 40})
            actual = independent_cost(original, d) - independent_cost(route, d)
            self.assertEqual(stats['gain'], actual)
            self.assertEqual(stats['best_gain_sum'], actual)
            self.assertGreaterEqual(stats['parent_gain_sum'], actual)
            self.assertEqual(lengths, sorted(lengths, reverse=True))
            self.assertLessEqual(stats['accounted_seconds'], stats['elapsed'] + 1e-6)
            self.assertGreater(stats['ledger']['candidate_construction_seconds'], 0)
            self.assertGreater(stats['ledger']['feature_seconds'], 0)
            self.assertGreater(stats['ledger']['inference_seconds'], 0)
            self.assertGreater(stats['ledger']['pool_seconds'], 0)
            self.assertGreater(len(stats['parent_use_counts']), 1)
            self.assertTrue(all(x['evaluations'] <= (1 if mode == 'D' else 128)
                                for x in stats['decisions']))
        route = original.copy()
        expired = search.optimize(route, d, 'C', time.perf_counter() - 1,
                                  None, {'weights': [0.0] * 12})
        self.assertEqual(expired['steps'], 0)
        np.testing.assert_array_equal(route, original)

    def test_b_c_identical_actions_do_not_diverge_proposal_streams(self):
        class ForceFixedRule:
            def __init__(self):
                self.step = 0

            def score(self, features):
                value = features[:, 6] + .5 * features[:, 5] + .25 * features[:, 7]
                value += 100.0 * features[:, self.step % 4]
                self.step += 1
                output = np.full(len(value), -1000.0)
                output[int(np.argmax(value))] = 0.0
                return output

        d = distances(87, 128)
        original = np.random.default_rng(18).permutation(128)
        b, c = original.copy(), original.copy()
        options = {'local_max_steps': 36}
        bs = search.optimize(b, d, 'B', time.perf_counter() + 10, None,
                             seed=43, settings=options)
        cs = search.optimize(c, d, 'C', time.perf_counter() + 10, None,
                             ForceFixedRule(), seed=43, settings=options)
        np.testing.assert_array_equal(b, c)
        self.assertEqual(bs['best_gain_sum'], cs['best_gain_sum'])
        self.assertEqual([v['selected_action'] for v in bs['decisions']],
                         [v['selected_action'] for v in cs['decisions']])
        self.assertEqual([v['parent_id'] for v in bs['decisions']],
                         [v['parent_id'] for v in cs['decisions']])

    def test_insert_pool_reports_actual_retention(self):
        d = distances(19, 9)
        route = np.arange(9)
        pool = search.TourPool(route, d, pool_size=1)
        other = np.roll(route[::-1], 2).copy()
        # Same undirected tour is duplicate and cannot count as an insertion.
        entries, next_id, inserted = search._insert_pool(
            pool.entries, other, pool.best_length, 1, 1)
        self.assertFalse(inserted)
        self.assertEqual(len(entries), 1)
        changed = route.copy()
        changed[1:4] = changed[1:4][::-1]
        # Worse unique child is immediately evicted by a one-slot pool.
        entries, next_id, inserted = search._insert_pool(
            pool.entries, changed, pool.best_length + 100, 1, 1)
        self.assertFalse(inserted)
        self.assertEqual(len(entries), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
