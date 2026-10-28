"""Deep candidate-edge actor used by native LKH, with an offline value critic.

The actor is deliberately small enough to execute inside the C search engine.
It contains two independently trainable nonlinear hidden layers.  Its actions
are candidate permutations, not tours or external 2-opt moves.  The native
engine keeps its two ALPHA backbone edges and uses the sampled/learned order
to select three further edges and order all five search candidates.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    'dx_over_extent', 'dy_over_extent', 'euclidean_over_extent',
    'log_alpha_normalized', 'alpha_rank_over_5', 'knn_rank_over_16',
    'is_alpha', 'is_knn', 'source_knn_mean_over_extent',
    'target_knn_mean_over_extent', 'source_knn_min_over_extent',
    'target_knn_min_over_extent',
]
ACTOR_SHAPES = {
    'actor0_weight': (64, 12), 'actor0_bias': (64,),
    'actor1_weight': (32, 64), 'actor1_bias': (32,),
    'actor2_weight': (1, 32), 'actor2_bias': (1,),
}
CRITIC_SHAPES = {
    'critic0_weight': (32, 25), 'critic0_bias': (32,),
    'critic1_weight': (1, 32), 'critic1_bias': (1,),
}
PARAMETER_SHAPES = ACTOR_SHAPES | CRITIC_SHAPES


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode('ascii')


def content_digest(value):
    value = dict(value)
    value.pop('sha256', None)
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_native_stats(path):
    """Read the latest complete append-only native JSONL snapshot."""
    last = None
    for line in Path(path).read_text(encoding='ascii').splitlines(keepends=True):
        if not line.endswith('\n'):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last = value
    if last is None:
        raise ValueError('No complete native statistics snapshot')
    return last


class EdgePolicy:
    def __init__(self, parameters, metadata=None):
        self.parameters = {k: np.asarray(v, dtype=np.float32)
                           for k, v in parameters.items()}
        self.metadata = {} if metadata is None else dict(metadata)
        if set(self.parameters) != set(PARAMETER_SHAPES):
            raise ValueError('Unexpected model parameter names')
        for name, expected in PARAMETER_SHAPES.items():
            if self.parameters[name].shape != expected or not np.isfinite(self.parameters[name]).all():
                raise ValueError('Invalid model parameter: ' + name)

    @property
    def actor_parameter_count(self):
        return sum(self.parameters[k].size for k in ACTOR_SHAPES)

    @property
    def parameter_count(self):
        return sum(v.size for v in self.parameters.values())

    def _linear(self, x, name):
        return x @ self.parameters[name + '_weight'].T + self.parameters[name + '_bias']

    def logits(self, features):
        x = np.asarray(features, dtype=np.float32)
        x = np.maximum(self._linear(x, 'actor0'), 0)
        x = np.maximum(self._linear(x, 'actor1'), 0)
        return np.clip(self._linear(x, 'actor2').reshape(-1), -10, 10)

    def value(self, graph_features):
        x = np.maximum(self._linear(np.asarray(graph_features, dtype=np.float32), 'critic0'), 0)
        return float(self._linear(x, 'critic1').reshape(()))

    def to_dict(self):
        result = {
            'schema': 1, 'model_type': 'internal_lkh_deep_edge_actor_critic',
            'actor_architecture': [12, 64, 32, 1],
            'actor_activations': ['relu', 'relu', 'clip[-10,10]'],
            'critic_architecture': [25, 32, 1],
            'actor_parameter_count': self.actor_parameter_count,
            'parameter_count': self.parameter_count,
            'feature_names': FEATURE_NAMES, 'dtype': 'float32',
            'parameters': {k: v.tolist() for k, v in self.parameters.items()},
            'metadata': self.metadata,
        }
        result['sha256'] = content_digest(result)
        return result

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding='ascii')

    def export_native(self, path):
        # Native C reads in precisely this order; the critic is offline only.
        with Path(path).open('w', encoding='ascii', newline='\n') as stream:
            stream.write('IRL_MLP_V1 12 64 32 1\n')
            for name in ACTOR_SHAPES:
                stream.write(' '.join(format(float(v), '.17g')
                                      for v in self.parameters[name].reshape(-1)) + '\n')


def initialize_policy(seed=1):
    rng = np.random.default_rng(seed)
    result = {}
    for name, shape in PARAMETER_SHAPES.items():
        if name.endswith('_bias'):
            value = np.zeros(shape, dtype=np.float32)
        else:
            value = rng.normal(0, np.sqrt(2 / (shape[0] + shape[1])), shape).astype(np.float32)
            if name == 'actor2_weight':
                value *= .05
        result[name] = value
    return EdgePolicy(result)


def load_policy(path):
    value = json.loads(Path(path).read_text(encoding='ascii'))
    if value.get('sha256') != content_digest(value):
        raise ValueError('Model content digest mismatch')
    if value.get('model_type') != 'internal_lkh_deep_edge_actor_critic':
        raise ValueError('Wrong policy kind')
    policy = EdgePolicy(value['parameters'], value.get('metadata'))
    if value.get('parameter_count') != policy.parameter_count:
        raise ValueError('Parameter count mismatch')
    return policy


def load_native_actor(path):
    words = Path(path).read_text(encoding='ascii').split()
    if words[:5] != ['IRL_MLP_V1', '12', '64', '32', '1']:
        raise ValueError('Invalid native weight header')
    flat = np.asarray([float(v) for v in words[5:]], dtype=np.float64)
    if len(flat) != sum(int(np.prod(v)) for v in ACTOR_SHAPES.values()) or not np.isfinite(flat).all():
        raise ValueError('Invalid native weight values')
    result, start = {}, 0
    for name, shape in ACTOR_SHAPES.items():
        count = int(np.prod(shape))
        result[name] = flat[start:start + count].reshape(shape)
        start += count
    return result


def native_reference_logits(features, parameters):
    """Scalar/double-equivalent native forward, independent of Torch."""
    result = np.asarray(features, dtype=np.float64)
    for prefix in ('actor0', 'actor1', 'actor2'):
        result = result @ parameters[prefix + '_weight'].T + parameters[prefix + '_bias']
        result = np.maximum(result, 0) if prefix != 'actor2' else np.clip(result, -10, 10)
    return result.reshape(-1)


@dataclass
class CandidateGraph:
    nodes: np.ndarray
    targets: np.ndarray
    protected: np.ndarray
    features: np.ndarray
    groups: list

    @property
    def n(self):
        return len(self.groups)

    def critic_features(self):
        return np.concatenate((self.features.mean(axis=0), self.features.std(axis=0),
                               [np.log(max(self.n, 2)) / 10])).astype(np.float32)

    def padded_indices(self):
        width = max(len(v) for v in self.groups)
        index = np.zeros((self.n, width), dtype=np.int64)
        mask = np.zeros((self.n, width), dtype=bool)
        for i, group in enumerate(self.groups):
            index[i, :len(group)] = group
            mask[i, :len(group)] = True
        return index, mask


def read_candidate_graph(path):
    with Path(path).open(encoding='ascii', newline='') as stream:
        reader = csv.DictReader(stream)
        expected = ['node', 'target', 'protected'] + ['f' + str(i) for i in range(12)]
        if reader.fieldnames != expected:
            raise ValueError('Native feature dump schema mismatch')
        rows = list(reader)
    nodes = np.asarray([int(r['node']) for r in rows], dtype=np.int64)
    targets = np.asarray([int(r['target']) for r in rows], dtype=np.int64)
    protected = np.asarray([int(r['protected']) for r in rows], dtype=np.int64)
    features = np.asarray([[float(r['f' + str(i)]) for i in range(12)] for r in rows], dtype=np.float32)
    if not len(rows) or not np.isfinite(features).all():
        raise ValueError('Empty/nonfinite feature dump')
    n = int(nodes.max())
    if not np.array_equal(np.unique(nodes), np.arange(1, n + 1)):
        raise ValueError('Native node IDs must be consecutive and one based')
    if np.any(targets < 1) or np.any(targets > n) or np.any(targets == nodes):
        raise ValueError('Invalid candidate endpoint')
    if not np.isin(protected, (0, 1)).all():
        raise ValueError('Invalid protected flag')
    groups = [np.flatnonzero(nodes == node) for node in range(1, n + 1)]
    for group in groups:
        if len(np.unique(targets[group])) != len(group):
            raise ValueError('Duplicate candidate endpoint')
        if int(protected[group].sum()) != min(2, len(group)):
            raise ValueError('Native ALPHA backbone count mismatch')
    return CandidateGraph(nodes, targets, protected.astype(bool), features, groups)


def write_actions(path, graph, order):
    """Write every union candidate exactly once, in policy order per node."""
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 2 or order.shape[0] != graph.n:
        raise ValueError('Action permutation shape mismatch')
    with Path(path).open('w', encoding='ascii', newline='\n') as stream:
        stream.write('node,target\n')
        for group, row in zip(graph.groups, order):
            chosen = row[:len(group)]
            if sorted(chosen.tolist()) != sorted(group.tolist()):
                raise ValueError('Action is not an exact candidate permutation')
            for edge in chosen:
                stream.write(f'{int(graph.nodes[edge])},{int(graph.targets[edge])}\n')


def deterministic_order(graph, logits):
    index, _ = graph.padded_indices()
    result = index.copy()
    for i, group in enumerate(graph.groups):
        # Native ties use the smaller target ID, independent of union insertion order.
        result[i, :len(group)] = group[np.lexsort((graph.targets[group], -np.asarray(logits)[group]))]
    return result


def build_torch_model(policy):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class TorchEdgeActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v.copy()))
                                            for k, v in policy.parameters.items()})

        def linear(self, x, prefix):
            return F.linear(x, self.params[prefix + '_weight'], self.params[prefix + '_bias'])

        def forward(self, features, critic_features):
            hidden = F.relu(self.linear(features, 'actor0'))
            hidden = F.relu(self.linear(hidden, 'actor1'))
            logits = torch.clamp(self.linear(hidden, 'actor2').reshape(-1), -10, 10)
            hidden_value = F.relu(self.linear(critic_features, 'critic0'))
            value = self.linear(hidden_value, 'critic1').reshape(())
            return logits, value

        def export(self, metadata=None):
            return EdgePolicy({k: v.detach().cpu().numpy().copy() for k, v in self.params.items()}, metadata)

    return TorchEdgeActorCritic()


def sample_plackett_luce(logits, index, valid, generator):
    """Batched exact PL permutation, with its differentiable log probability.

    The complete joint log probability is divided by the fixed number of edge
    decisions for numerical scaling.  Within an episode this is a positive
    action-independent constant, preserving the policy-gradient direction.
    """
    import torch
    scores = logits[index]
    uniforms = torch.rand(scores.shape, generator=generator, dtype=scores.dtype,
                          device=scores.device).clamp_(1e-7, 1 - 1e-7)
    priorities = scores - torch.log(-torch.log(uniforms))
    priorities = priorities.masked_fill(~valid, -torch.inf)
    permutation = torch.argsort(priorities, dim=1, descending=True, stable=True)
    ordered_edges = torch.gather(index, 1, permutation)
    ordered_scores = torch.gather(scores, 1, permutation)
    ordered_valid = torch.gather(valid, 1, permutation)
    # A finite mask avoids undefined (-inf)-(-inf) derivatives in padded tails.
    # exp(-1e9) is exactly zero at the supported float32 training precision.
    ordered_scores = ordered_scores.masked_fill(~ordered_valid, -1e9)
    denominators = torch.logcumsumexp(ordered_scores.flip(1), dim=1).flip(1)
    terms = torch.where(ordered_valid, ordered_scores - denominators,
                        torch.zeros_like(ordered_scores))
    mean_log_probability = terms.sum() / valid.sum()
    first_logprobs = torch.log_softmax(scores.masked_fill(~valid, -1e9), dim=1)
    entropy = -(first_logprobs.exp() * first_logprobs).sum(dim=1).mean()
    return ordered_edges, mean_log_probability, entropy
