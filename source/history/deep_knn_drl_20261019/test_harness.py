"""Regression checks for stock-LKH isolation and balanced paired scheduling."""
from collections import Counter
import itertools
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import run
import worker


class HarnessTests(unittest.TestCase):
    def test_ordinary_worker_never_constructs_nearest_neighbor(self):
        cache = Path(__file__).resolve().parent / 'test_cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='deep_worker_', dir=cache) as directory:
            origin = time.perf_counter()
            task = dict(directory=directory, cold_started=origin, deadline=origin + 10.,
                        coords=[[0, 0], [10, 0], [10, 10], [0, 10]],
                        method='A', seed=101, initial_route=[0, 1, 2, 3], budget=10.,
                        settings={'preprocessing_fraction_target': .25})
            path = Path(directory) / 'task.json'
            path.write_text(json.dumps(task), encoding='utf-8')
            handshake = io.StringIO(json.dumps({'go': True, 'deadline': task['deadline']}) + '\n')
            with patch.object(worker.sys, 'argv', ['worker.py', str(path)]), \
                 patch.object(worker.sys, 'stdin', handshake), \
                 patch.object(worker, 'nearest_neighbor', side_effect=AssertionError('NN is forbidden in A')), \
                 patch.object(worker, 'lkh_search', return_value={'fixture_only': True}) as search:
                worker.main()
            search.assert_called_once()
            result = json.loads((Path(directory) / 'worker_result.json').read_text())
            self.assertEqual(result['nearest_neighbor_seconds'], 0.)
            self.assertTrue(result['ordinary_lkh_normal_initialization'])
            self.assertNotIn('preprocessing_search', result)
            self.assertTrue((Path(directory) / 'done.json').exists())

    def test_stock_lkh_does_not_receive_an_external_initial_tour(self):
        coords = [[0, 0], [10, 0], [10, 10], [0, 10]]
        cache = Path(__file__).resolve().parent / 'test_cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='deep_harness_', dir=cache) as directory:
            for method in ('A', 'U', 'D'):
                path = Path(directory) / method
                path.mkdir()
                origin = time.perf_counter()
                publisher = worker.Publisher(path, coords, origin + 10., origin)
                publisher([0, 1, 2, 3], 'test')
                task = dict(directory=str(path), cold_started=origin, coords=coords,
                            method=method, seed=101, deadline=origin - 1.,
                            lkh_binary='not_launched.exe')
                result = worker.lkh_search(task, publisher, [])
                self.assertTrue(result['skipped'])
                par = (path / 'lkh/run.par').read_text(encoding='ascii')
                if method == 'A':
                    self.assertNotIn('INITIAL_TOUR', par)
                    self.assertFalse((path / 'lkh/initial.tour').exists())
                else:
                    self.assertIn('INITIAL_TOUR_FILE', par)
                    self.assertIn('INITIAL_TOUR_FRACTION = 1', par)
                    self.assertEqual(worker.parse_tour(
                        (path / 'lkh/initial.tour').read_text(), 4), [0, 1, 2, 3])
                self.assertNotIn('INPUT_TOUR_FILE', par)
                self.assertNotIn('OPTIMUM =', par.replace('STOP_AT_OPTIMUM = NO', ''))

    def test_six_orders_balance_positions_and_directed_adjacencies(self):
        self.assertEqual(set(run.LATIN_ORDERS), set(itertools.permutations(run.METHODS)))
        orders = list(run.LATIN_ORDERS) * 10
        positions = Counter((method, pos) for order in orders
                            for pos, method in enumerate(order))
        adjacencies = Counter(pair for order in orders for pair in zip(order, order[1:]))
        self.assertEqual(set(positions.values()), {20})
        self.assertEqual(set(adjacencies.values()), {20})
        self.assertEqual(len(adjacencies), 6)


if __name__ == '__main__':
    unittest.main()
