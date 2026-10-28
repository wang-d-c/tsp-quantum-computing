"""Tests for the deep native candidate model and actual action likelihood."""
from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TEST_TEMP = Path(__file__).resolve().parent / 'testtmp'
TEST_TEMP.mkdir(exist_ok=True)

from edge_model import (CandidateGraph, build_torch_model, deterministic_order,
                        initialize_policy, load_native_actor, load_policy,
                        native_reference_logits, read_candidate_graph,
                        read_native_stats, sample_plackett_luce, write_actions)


def torch_import():
    try:
        import torch
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'deep_rl_benchmark' / 'deps'))
        import torch
    return torch


class EdgeModelTests(unittest.TestCase):
    def test_append_only_stats_ignore_unfinished_tail(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            path = Path(folder) / 'stats.json'
            path.write_text('{"best5opt_calls":0}\n{"best5opt_calls":4}\n{"best5opt_calls":', encoding='ascii')
            self.assertEqual(read_native_stats(path)['best5opt_calls'], 4)

    def test_three_way_forward_and_export(self):
        torch = torch_import()
        torch.set_num_threads(1)
        policy = initialize_policy(73)
        self.assertEqual(policy.actor_parameter_count, 2945)
        self.assertEqual(policy.parameter_count, 3810)
        x = np.random.default_rng(19).normal(size=(75, 12)).astype(np.float32)
        critic = np.random.default_rng(20).normal(size=25).astype(np.float32)
        model = build_torch_model(policy)
        logits, value = model(torch.from_numpy(x), torch.from_numpy(critic))
        np.testing.assert_allclose(logits.detach().numpy(), policy.logits(x), atol=2e-7, rtol=2e-5)
        self.assertAlmostEqual(float(value.detach()), policy.value(critic), places=6)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            native, archival = Path(folder) / 'native.txt', Path(folder) / 'model.json'
            policy.export_native(native)
            policy.save(archival)
            np.testing.assert_array_equal(load_policy(archival).logits(x), policy.logits(x))
            np.testing.assert_allclose(native_reference_logits(x, load_native_actor(native)),
                                       policy.logits(x), atol=2e-7, rtol=2e-5)

    def test_pl_probability_and_gradient_match_enumerated_formula(self):
        torch = torch_import()
        logits = torch.tensor([.2, -.4, .7, .1, -.2], dtype=torch.float64, requires_grad=True)
        index = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
        valid = torch.tensor([[True, True, True], [True, True, False]])
        generator = torch.Generator().manual_seed(713)
        order, logp, entropy = sample_plackett_luce(logits, index, valid, generator)
        explicit = 0
        for row, count in zip(order, [3, 2]):
            for j in range(count):
                explicit = explicit + logits[row[j]] - torch.logsumexp(logits[row[j:count]], dim=0)
        explicit = explicit / 5
        self.assertAlmostEqual(float(logp.detach()), float(explicit.detach()), places=12)
        first = torch.autograd.grad(logp, logits, retain_graph=True)[0]
        second = torch.autograd.grad(explicit, logits, retain_graph=True)[0]
        torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.isfinite(torch.autograd.grad(entropy, logits)[0]).all())
        # Conditional on the sampled action, compare autograd to finite difference.
        def fixed_logp(values):
            total = 0.
            for row, count in zip(order.numpy(), [3, 2]):
                for j in range(count):
                    rest = values[row[j:count]]
                    maximum = rest.max()
                    total += values[row[j]] - maximum - np.log(np.exp(rest - maximum).sum())
            return total / 5
        base = logits.detach().numpy()
        numeric = []
        for i in range(len(base)):
            plus, minus = base.copy(), base.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numeric.append((fixed_logp(plus) - fixed_logp(minus)) / 2e-6)
        np.testing.assert_allclose(first.numpy(), numeric, atol=2e-10, rtol=2e-8)

    def test_sampling_is_plackett_luce_first_choice(self):
        torch = torch_import()
        logits = torch.tensor([0., np.log(2.), np.log(3.)], dtype=torch.float32)
        n = 12000
        index = torch.arange(3).expand(n, 3)
        valid = torch.ones((n, 3), dtype=torch.bool)
        order, logp, entropy = sample_plackett_luce(logits, index, valid,
                                                 torch.Generator().manual_seed(99))
        empirical = np.bincount(order[:, 0].numpy(), minlength=3) / n
        np.testing.assert_allclose(empirical, [1 / 6, 2 / 6, 3 / 6], atol=.014, rtol=0)
        self.assertTrue(torch.isfinite(logp) & torch.isfinite(entropy))

    def test_csv_actions_are_exact_permutations(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            path = Path(folder) / 'features.csv'
            with path.open('w', encoding='ascii', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['node', 'target', 'protected'] + ['f' + str(i) for i in range(12)])
                for node in range(1, 5):
                    for rank, target in enumerate(v for v in range(1, 5) if v != node):
                        writer.writerow([node, target, int(rank < 2)] + [rank / 3] * 12)
            graph = read_candidate_graph(path)
            order = deterministic_order(graph, np.arange(12))
            target = Path(folder) / 'actions.csv'
            write_actions(target, graph, order)
            with target.open(encoding='ascii') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual([int(v['target']) for v in rows[:3]], graph.targets[order[0]].tolist())
            invalid = order.copy()
            invalid[0, 1] = invalid[0, 0]
            with self.assertRaises(ValueError):
                write_actions(target, graph, invalid)


if __name__ == '__main__':
    unittest.main()
