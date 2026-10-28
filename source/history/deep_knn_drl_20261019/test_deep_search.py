import time
import unittest
import numpy as np

import deep_search as search


class ReplayPolicy:
    parameter_count = 1

    def __init__(self, choices):
        self.choices = iter(choices)

    def prepare(self, coords, knn):
        pass

    def logits(self, route, candidates, distances, context):
        logits = np.full(len(candidates), -1e6)
        logits[next(self.choices)] = 1e6
        return logits


class SearchChecks(unittest.TestCase):
    def fixture(self):
        xy = np.random.default_rng(102).integers(0, 10000, (97, 2))
        dist = np.floor(np.linalg.norm(xy[:, None] - xy[None, :], axis=2) + .5).astype(np.int32)
        route = np.random.default_rng(103).permutation(len(xy))
        return xy, dist, route

    def test_forced_same_actions_reproduce_uniform_trajectory(self):
        xy, dist, initial = self.fixture()
        uniform_route = initial.copy()
        settings = dict(local_max_steps=40, d_candidate_count=48)
        uniform = search.optimize(uniform_route, dist, 'U', time.perf_counter()+60,
                                  None, seed=9, settings=settings, coords=xy)
        choices = [d['selected_index'] for d in uniform['decisions']]
        deep_route = initial.copy()
        deep = search.optimize(deep_route, dist, 'D', time.perf_counter()+60,
                               None, ReplayPolicy(choices), 9, settings, xy)
        self.assertEqual(uniform['steps'], 40)
        self.assertEqual(uniform['final_pool'], deep['final_pool'])
        for u, d in zip(uniform['decisions'], deep['decisions']):
            for key in ('candidate_sha256', 'parent_id', 'selected_action',
                        'gain', 'best_gain', 'best_length'):
                self.assertEqual(u[key], d[key])
        np.testing.assert_array_equal(uniform_route, deep_route)

    def test_exact_gain_reconstructs_every_valid_2opt_cut(self):
        _, dist, route = self.fixture()
        old = search.tour_length(route, dist)
        for i in range(len(route)-2):
            for j in range(i+2, len(route)):
                if i == 0 and j == len(route)-1:
                    continue
                proposed = route.copy()
                proposed[i+1:j+1] = proposed[i+1:j+1][::-1]
                expected = max(0, old-search.tour_length(proposed, dist))
                result = search.evaluate_d_action(route, dist, dict(i=i,j=j,kind='knn'))
                self.assertEqual(result['gain'], expected)
                self.assertEqual(old-search.tour_length(result['route'], dist), expected)

    def test_expired_budget_does_no_move(self):
        xy, dist, route = self.fixture()
        before = route.copy()
        stats = search.optimize(route, dist, 'U', time.perf_counter()-1, None, coords=xy)
        self.assertEqual(stats['steps'], 0)
        self.assertEqual(stats['gain'], 0)
        np.testing.assert_array_equal(route, before)


if __name__ == '__main__':
    unittest.main()
