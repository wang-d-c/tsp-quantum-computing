"""Regression checks for common paid fallback and internal-policy isolation."""
from collections import Counter
import itertools
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import run
import worker


class HarnessTests(unittest.TestCase):
    def test_native_statistics_survive_incomplete_final_append(self):
        cache=Path(__file__).resolve().parent/'test_cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='native_stats_',dir=cache) as temporary:
            path=Path(temporary)/'native.json'
            path.write_text('{"best5opt_calls":0}\n{"best5opt_calls":3}\n{"best5opt_calls":',encoding='utf-8')
            self.assertEqual(worker.last_complete_statistics(path),{'best5opt_calls':3})

    def test_native_policy_env_is_explicit_and_ordinary_has_no_intervention(self):
        coords=[[0,0],[10,0],[10,10],[0,10]]
        cache=Path(__file__).resolve().parent/'test_cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='internal_env_',dir=cache) as temporary:
            for method in ('A','U','D'):
                directory=Path(temporary)/method
                directory.mkdir()
                origin=time.perf_counter()
                publisher=worker.Publisher(directory,coords,origin+10,origin)
                publisher([0,1,2,3],'fixture')
                task=dict(directory=str(directory),cold_started=origin,deadline=origin+10,
                          method=method,coords=coords,seed=101,lkh_binary='fixture.exe',
                          settings={'policy_by_test_seed':{'101':{'D':'selected_native.txt'}}})
                process=Mock(returncode=0)
                process.poll.return_value=0
                process.wait.return_value=0
                with patch.dict(worker.os.environ,{'INTERNAL_RL_MODEL':'poison','INTERNAL_RL_DUMP':'poison'}), \
                     patch.object(worker.subprocess,'Popen',return_value=process) as launch:
                    result=worker.lkh_search(task,publisher,[])
                env=launch.call_args.kwargs['env']
                keys={key for key in env if key.upper().startswith('INTERNAL_RL_')}
                if method=='A':self.assertFalse(keys)
                else:
                    self.assertEqual(env['INTERNAL_RL_SEED'],'101')
                    self.assertTrue(env['INTERNAL_RL_STATS'].endswith('internal_stats.json'))
                    self.assertTrue(env['INTERNAL_RL_SELECTED'].endswith('selected_candidates.csv'))
                    self.assertTrue(env['INTERNAL_RL_SCORES'].endswith('scored_candidates.csv'))
                    self.assertNotIn('INTERNAL_RL_DUMP',env)
                    if method=='U':
                        self.assertEqual(env['INTERNAL_RL_MODE'],'uniform')
                        self.assertNotIn('INTERNAL_RL_MODEL',env)
                    else:
                        self.assertEqual(env['INTERNAL_RL_MODEL'],'selected_native.txt')
                        self.assertNotIn('INTERNAL_RL_MODE',env)
                self.assertFalse(result['initial_tour_enabled'])
                self.assertFalse(result['external_route_pool'])

    def test_all_workers_pay_same_nn_and_never_run_external_optimization(self):
        cache = Path(__file__).resolve().parent / 'test_cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='deep_worker_', dir=cache) as directory:
            origin = time.perf_counter()
            hashes=[]
            for method in ('A','U','D'):
                target=Path(directory)/method
                target.mkdir()
                task = dict(directory=str(target), cold_started=origin, deadline=origin + 10.,
                            coords=[[0, 0], [10, 0], [10, 10], [0, 10]],
                            method=method, seed=101, initial_route=[0, 1, 2, 3], budget=10., settings={})
                path = target / 'task.json'
                path.write_text(json.dumps(task), encoding='utf-8')
                handshake = io.StringIO(json.dumps({'go': True, 'deadline': task['deadline']}) + '\n')
                with patch.object(worker.sys, 'argv', ['worker.py', str(path)]), \
                     patch.object(worker.sys, 'stdin', handshake), \
                     patch.object(worker, 'nearest_neighbor', return_value=[0,1,2,3]) as nn, \
                     patch.object(worker, 'lkh_search', return_value={'fixture_only': True}) as search:
                    worker.main()
                nn.assert_called_once_with(task['coords'],101)
                search.assert_called_once()
                result = json.loads((target / 'worker_result.json').read_text())
                self.assertGreater(result['nearest_neighbor_seconds'], 0.)
                self.assertTrue(result['ordinary_lkh_normal_initialization'])
                self.assertNotIn('preprocessing_search', result)
                self.assertFalse(result['external_route_pool'])
                hashes.append(result['nearest_neighbor_route_sha256'])
                self.assertTrue((target / 'done.json').exists())
            self.assertEqual(len(set(hashes)),1)

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
                self.assertNotIn('INITIAL_TOUR', par)
                self.assertFalse((path / 'lkh/initial.tour').exists())
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
