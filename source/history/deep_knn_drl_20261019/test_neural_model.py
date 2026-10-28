"""Mathematical, serialization and training/export checks for deep policy."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import neural_model as model


def fixture(n=24, seed=317):
    rng = np.random.default_rng(seed)
    coords = np.rint(rng.uniform(0, 10000, size=(n, 2))).astype(np.int64)
    distances = np.floor(np.linalg.norm(coords[:, None] - coords[None, :], axis=-1) + .5).astype(np.int32)
    # Strict tie breaking mirrors stable distance/city-id nearest neighbors.
    knn = np.asarray([np.lexsort((np.arange(n), distances[i]))[1:9] for i in range(n)])
    route = rng.permutation(n)
    candidates = [{"i": i, "j": j, "kind": ["knn", "longedge", "explore"][q % 3]}
                  for q, (i, j) in enumerate([(0, 3), (1, 9), (2, 21), (5, 13), (9, 17)])]
    return coords, distances, knn, route, candidates


class ExportTests(unittest.TestCase):
    def test_forward_candidate_permutation_and_translation(self):
        coords, distances, knn, route, candidates = fixture()
        policy = model.initialize_policy(7)
        policy.prepare(coords, knn)
        logits = policy.logits(route, candidates, distances)
        self.assertEqual(policy.parameter_count, 32450)
        self.assertEqual(logits.shape, (5,))
        order = [3, 1, 4, 0, 2]
        np.testing.assert_allclose(policy.logits(route, [candidates[q] for q in order], distances),
                                   logits[order], atol=2e-7, rtol=2e-5)
        policy.prepare(coords + np.array([1000, -3000]), knn)
        np.testing.assert_allclose(policy.logits(route, candidates, distances), logits,
                                   atol=2e-6, rtol=2e-5)

    def test_model_roundtrip_and_tamper_rejection(self):
        original = model.initialize_policy(3)
        temp_root = Path(__file__).with_name("test_cache")
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path = Path(tmp) / "model.json"
            original.save(path)
            loaded = model.load_policy(path)
            for name, value in original.parameters.items():
                np.testing.assert_array_equal(value, loaded.parameters[name])
            record = json.loads(path.read_text())
            record["parameters"]["actor2_bias"][0] += 1
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                model.load_policy(path)

    def test_action_inputs_do_not_read_future_edge_costs(self):
        coords, distances, knn, route, candidates = fixture()
        reference = model.dynamic_inputs(coords, route, candidates, knn, distances)
        changed = distances.copy()
        old_edges = {tuple(sorted((int(a), int(b)))) for a, b in zip(route, np.roll(route, -1))}
        for c in candidates:
            i, j = c["i"], c["j"]
            for a, b in [(route[i], route[j]), (route[(i + 1) % len(route)], route[(j + 1) % len(route)])]:
                if tuple(sorted((int(a), int(b)))) not in old_edges:
                    changed[a, b] += 98765
                    changed[b, a] += 98765
        actual = model.dynamic_inputs(coords, route, candidates, knn, changed)
        for one, two in zip(reference, actual):
            np.testing.assert_array_equal(one, two)

    def test_torch_export_and_attention_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch training dependency not on this interpreter path")
        torch.set_num_threads(1)
        coords, distances, knn, route, candidates = fixture()
        architecture = dict(model.DEFAULT_ARCHITECTURE, embedding_dim=64, encoder_ff_dim=128,
                            actor_hidden=[128, 64], critic_hidden=64)
        policy = model.initialize_policy(801, architecture)
        self.assertEqual(policy.parameter_count, 124290)
        training = model.build_torch_model(policy)
        x, neighbors = model.graph_inputs(coords, knn)
        endpoints, dynamic, state = model.dynamic_inputs(coords, route, candidates, knn, distances)
        embedding = training.encode(torch.from_numpy(x), torch.from_numpy(neighbors))
        logits, value = training(embedding, torch.from_numpy(endpoints),
                                 torch.from_numpy(dynamic), torch.from_numpy(state))
        policy.prepare(coords, knn)
        np_logits, np_value = policy.forward_inputs(endpoints, dynamic, state)
        np.testing.assert_allclose(embedding.detach().numpy(), policy.embeddings, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(logits.detach().numpy(), np_logits, atol=2e-6, rtol=2e-5)
        self.assertAlmostEqual(float(value.detach()), np_value, places=5)
        actor_only = torch.log_softmax(logits, dim=0)[2]
        actor_only.backward(retain_graph=True)
        self.assertGreater(float(training.params["block0_q_weight"].grad.norm()), 1e-9)
        training.zero_grad()
        objective = torch.log_softmax(logits, dim=0)[2] * .71 + .5 * (value - .31) ** 2
        objective.backward()
        actor_grad = training.params["actor2_weight"].grad.detach().clone()
        encoder_grad = training.params["block0_q_weight"].grad.detach().clone()
        self.assertGreater(float(actor_grad.norm()), 1e-7)
        self.assertGreater(float(encoder_grad.norm()), 1e-8)
        # A float64 autograd finite difference catches detached encoder mistakes.
        training = training.double()
        x_t = torch.from_numpy(x).double()
        dyn_t = torch.from_numpy(dynamic).double()
        state_t = torch.from_numpy(state).double()
        def objective_fn():
            em = training.encode(x_t, torch.from_numpy(neighbors))
            lo, va = training(em, torch.from_numpy(endpoints), dyn_t, state_t)
            return torch.log_softmax(lo, dim=0)[2] * .71 + .5 * (va - .31) ** 2
        training.zero_grad()
        objective_fn().backward()
        checks = [("actor2_weight", (0, 3)), ("block0_q_weight", (2, 4)),
                  ("input_weight", (1, 0)), ("critic1_weight", (0, 2))]
        epsilon = 1e-5
        for name, index in checks:
            analytic = float(training.params[name].grad[index])
            with torch.no_grad():
                old = float(training.params[name][index])
                training.params[name][index] = old + epsilon
                high = float(objective_fn())
                training.params[name][index] = old - epsilon
                low = float(objective_fn())
                training.params[name][index] = old
            self.assertAlmostEqual(analytic, (high - low) / (2 * epsilon), places=6)


if __name__ == "__main__":
    unittest.main()
