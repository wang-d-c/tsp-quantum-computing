"""Sparse attention improvement policy, exported to dependency-light NumPy.

The trainable network has two multi-head graph-attention encoder blocks,
nonlinear candidate-endpoint actor and a learned state-value critic.  Training
uses PyTorch autograd; this module imports it only inside build_torch_model.
The public inference path needs NumPy alone and uses the same learned weights.
No action input contains an evaluated 2-opt gain or future tour length.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_ARCHITECTURE = {
    "embedding_dim": 32, "heads": 4, "encoder_layers": 2,
    "encoder_ff_dim": 64, "actor_hidden": [64, 32],
    "critic_hidden": 32, "node_feature_count": 6,
    "candidate_feature_count": 22, "state_feature_count": 7,
    "layer_norm_epsilon": 1e-5,
}
CANDIDATE_FEATURE_NAMES = [
    "ab_dx", "ab_dy", "ac_dx", "ac_dy", "bd_dx", "bd_dy", "cd_dx", "cd_dy",
    "old_ab", "old_cd", "old_preceding_a", "old_following_d", "arc_fraction",
    "short_arc_fraction", "knn_ac", "knn_bd", "kind_knn", "kind_longedge",
    "kind_explore", "parent_excess", "step_fraction", "log_n_over_10",
]


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def model_digest(value):
    clean = dict(value)
    clean.pop("sha256", None)
    return hashlib.sha256(canonical_bytes(clean)).hexdigest()


def _softmax(x, axis=-1):
    values = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


def graph_inputs(coords, knn):
    """Static geometric state; each attention row includes self and kNN nodes."""
    coords = np.asarray(coords, dtype=np.float32)
    knn = np.asarray(knn, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2 or knn.shape[0] != len(coords):
        raise ValueError("Invalid coordinates or kNN table")
    extent = max(float(np.ptp(coords, axis=0).max()), 1.0)
    center = coords.mean(axis=0, keepdims=True)
    xy = (coords - center) / extent
    if knn.shape[1]:
        relative = (coords[knn] - coords[:, None, :]) / extent
        lengths = np.sqrt((relative * relative).sum(axis=-1))
        summary = np.column_stack((lengths.mean(axis=1), lengths.min(axis=1),
                                   lengths.max(axis=1), np.ones(len(coords))))
    else:
        summary = np.column_stack((np.zeros((len(coords), 3)), np.ones(len(coords))))
    x = np.column_stack((xy, summary)).astype(np.float32)
    neighbors = np.column_stack((np.arange(len(coords), dtype=np.int64), knn))
    return x, neighbors


def dynamic_inputs(coords, route, candidates, knn, distances, context=None):
    """Construct dynamic candidate and critic features without exact gains.

    Relative coordinate vectors expose geometry as in an improvement attention
    model.  Exact added-edge lengths and exact selected-move evaluation are the
    environment's responsibility, after action selection.
    """
    context = {} if context is None else context
    coords = np.asarray(coords, dtype=np.float32)
    route = np.asarray(route, dtype=np.int64)
    n = len(route)
    if not candidates:
        raise ValueError("Empty action set")
    pairs = np.asarray([(int(c["i"]), int(c["j"])) for c in candidates], dtype=np.int64)
    i, j = pairs.T
    if (np.any(i < 0) or np.any(j >= n) or np.any(j - i < 2)
            or np.any((i == 0) & (j == n - 1))):
        raise ValueError("Invalid nonadjacent two-opt cuts")
    endpoints = np.column_stack((route[i], route[(i + 1) % n],
                                 route[j], route[(j + 1) % n]))
    a, b, c, d = endpoints.T
    old = np.asarray(distances[route, np.roll(route, -1)], dtype=np.float32)
    normalizer = max(float(context.get("initial_mean_edge", old.mean())), 1.0)
    relative = np.column_stack((coords[b] - coords[a], coords[c] - coords[a],
                                 coords[d] - coords[b], coords[d] - coords[c])) / normalizer
    membership_ac = np.any(knn[a] == c[:, None], axis=1) | np.any(knn[c] == a[:, None], axis=1)
    membership_bd = np.any(knn[b] == d[:, None], axis=1) | np.any(knn[d] == b[:, None], axis=1)
    kinds = np.asarray([("knn", "longedge", "explore").index(x["kind"])
                        for x in candidates], dtype=np.int64)
    onehot = np.eye(3, dtype=np.float32)[kinds]
    parent_excess = float(context.get("parent_excess", 0.0))
    step_fraction = float(context.get("step_fraction", 0.0))
    dynamic = np.column_stack((
        relative, old[i] / normalizer, old[j] / normalizer,
        old[(i - 1) % n] / normalizer, old[(j + 1) % n] / normalizer,
        (j - i) / n, np.minimum(j - i, n - j + i) / n,
        membership_ac, membership_bd, onehot,
        np.full(len(i), parent_excess), np.full(len(i), step_fraction),
        np.full(len(i), math.log(max(n, 2)) / 10.0),
    )).astype(np.float32)
    dynamic = np.clip(dynamic, -10.0, 10.0)
    state = np.asarray((old.mean() / normalizer, old.std() / normalizer,
                        old.max() / normalizer, parent_excess, step_fraction,
                        float(context.get("best_fraction", 1.0)),
                        math.log(max(n, 2)) / 10.0), dtype=np.float32)
    state = np.clip(state, -10.0, 10.0)
    if dynamic.shape[1] != 22 or not np.all(np.isfinite(dynamic)):
        raise AssertionError("Invalid dynamic neural state")
    return endpoints, dynamic, state


class DeepPolicy:
    def __init__(self, parameters, architecture=None, metadata=None):
        self.architecture = dict(DEFAULT_ARCHITECTURE if architecture is None else architecture)
        self.parameters = {k: np.asarray(v, dtype=np.float32) for k, v in parameters.items()}
        self.metadata = {} if metadata is None else dict(metadata)
        self.embeddings = None
        self.coords = None
        self.knn = None
        self._validate()

    def _validate(self):
        expected = parameter_shapes(self.architecture)
        if set(self.parameters) != set(expected):
            raise ValueError("Neural parameter names differ from architecture")
        for name, shape in expected.items():
            if self.parameters[name].shape != shape or not np.all(np.isfinite(self.parameters[name])):
                raise ValueError(f"Invalid neural parameter {name}")

    @property
    def parameter_count(self):
        return sum(v.size for v in self.parameters.values())

    def _linear(self, x, prefix):
        p = self.parameters
        return x @ p[prefix + "_weight"].T + p[prefix + "_bias"]

    def _norm(self, x, prefix):
        p = self.parameters
        centered = x - x.mean(axis=-1, keepdims=True)
        scaled = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True)
                                    + self.architecture["layer_norm_epsilon"])
        return scaled * p[prefix + "_weight"] + p[prefix + "_bias"]

    def encode(self, node_features, neighbors):
        a = self.architecture
        n, h, d = len(node_features), a["heads"], a["embedding_dim"]
        head = d // h
        x = np.tanh(self._linear(np.asarray(node_features, dtype=np.float32), "input"))
        for layer in range(a["encoder_layers"]):
            prefix = f"block{layer}"
            q = self._linear(x, prefix + "_q").reshape(n, h, head)
            k = self._linear(x, prefix + "_k").reshape(n, h, head)
            v = self._linear(x, prefix + "_v").reshape(n, h, head)
            logits = np.einsum("nhd,nkhd->nkh", q, k[neighbors]) / math.sqrt(head)
            attention = _softmax(logits, axis=1)
            message = np.einsum("nkh,nkhd->nhd", attention, v[neighbors]).reshape(n, d)
            x = self._norm(x + self._linear(message, prefix + "_out"), prefix + "_norm1")
            ff = self._linear(np.maximum(self._linear(x, prefix + "_ff1"), 0.0), prefix + "_ff2")
            x = self._norm(x + ff, prefix + "_norm2")
        return x

    def prepare(self, coords, knn):
        self.coords = np.asarray(coords, dtype=np.float32)
        self.knn = np.asarray(knn, dtype=np.int64)
        features, neighbors = graph_inputs(self.coords, self.knn)
        self.embeddings = self.encode(features, neighbors)
        self.graph_mean = self.embeddings.mean(axis=0)
        return self.embeddings

    def forward_inputs(self, endpoints, dynamic, state_features):
        if self.embeddings is None:
            raise RuntimeError("Call prepare(coords,knn) before inference")
        action_x = np.column_stack((self.embeddings[np.asarray(endpoints)].reshape(len(endpoints), -1),
                                   np.broadcast_to(self.graph_mean, (len(endpoints), len(self.graph_mean))),
                                   dynamic)).astype(np.float32)
        hidden = np.maximum(self._linear(action_x, "actor0"), 0.0)
        hidden = np.maximum(self._linear(hidden, "actor1"), 0.0)
        # The fixed tanh bound is independent of candidate ordering.
        logits = 10.0 * np.tanh(self._linear(hidden, "actor2").reshape(-1))
        value_x = np.concatenate((self.graph_mean, state_features)).astype(np.float32)
        value = float(self._linear(np.maximum(self._linear(value_x, "critic0"), 0.0), "critic1")[0])
        return logits, value

    def logits(self, route, candidates, distances, context=None):
        inputs = dynamic_inputs(self.coords, route, candidates, self.knn, distances, context)
        logits, _ = self.forward_inputs(*inputs)
        if logits.shape != (len(candidates),) or not np.all(np.isfinite(logits)):
            raise ValueError("Nonfinite candidate logits")
        return logits

    def value(self, route, candidates, distances, context=None):
        return self.forward_inputs(*dynamic_inputs(self.coords, route, candidates, self.knn, distances, context))[1]

    def to_dict(self):
        value = {"schema_version": 2, "model_type": "sparse_knn_attention_actor_critic",
                 "architecture": self.architecture, "parameter_count": self.parameter_count,
                 "dtype": "float32", "parameters": {k: v.tolist() for k, v in self.parameters.items()},
                 "candidate_feature_names": CANDIDATE_FEATURE_NAMES, "metadata": self.metadata}
        value["sha256"] = model_digest(value)
        return value

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2,
                                        allow_nan=False), encoding="utf-8")


def parameter_shapes(architecture=None):
    a = DEFAULT_ARCHITECTURE if architecture is None else architecture
    d, ff = a["embedding_dim"], a["encoder_ff_dim"]
    shapes = {}
    def linear(prefix, out, inp):
        shapes[prefix + "_weight"] = (out, inp)
        shapes[prefix + "_bias"] = (out,)
    def norm(prefix):
        shapes[prefix + "_weight"] = (d,)
        shapes[prefix + "_bias"] = (d,)
    linear("input", d, a["node_feature_count"])
    for layer in range(a["encoder_layers"]):
        prefix = f"block{layer}"
        for kind in ("q", "k", "v", "out"):
            linear(prefix + "_" + kind, d, d)
        linear(prefix + "_ff1", ff, d)
        linear(prefix + "_ff2", d, ff)
        norm(prefix + "_norm1")
        norm(prefix + "_norm2")
    linear("actor0", a["actor_hidden"][0], d * 5 + a["candidate_feature_count"])
    linear("actor1", a["actor_hidden"][1], a["actor_hidden"][0])
    linear("actor2", 1, a["actor_hidden"][1])
    linear("critic0", a["critic_hidden"], d + a["state_feature_count"])
    linear("critic1", 1, a["critic_hidden"])
    return shapes


def initialize_policy(seed=1, architecture=None):
    a = dict(DEFAULT_ARCHITECTURE if architecture is None else architecture)
    if a["embedding_dim"] % a["heads"]:
        raise ValueError("Embedding dimension must divide evenly into heads")
    rng = np.random.default_rng(seed)
    parameters = {}
    for name, shape in parameter_shapes(a).items():
        if name.endswith("bias"):
            value = np.zeros(shape, dtype=np.float32)
        elif "norm" in name:
            value = np.ones(shape, dtype=np.float32)
        else:
            # Xavier initialization; actor output initially close to uniform.
            scale = math.sqrt(2.0 / (shape[0] + shape[1]))
            value = rng.normal(0, scale, size=shape).astype(np.float32)
            if name == "actor2_weight":
                value *= 0.01
        parameters[name] = value
    return DeepPolicy(parameters, a)


def load_policy(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("sha256") != model_digest(value):
        raise ValueError("Neural model content hash mismatch")
    if value.get("model_type") != "sparse_knn_attention_actor_critic":
        raise ValueError("Unexpected neural model type")
    policy = DeepPolicy(value["parameters"], value["architecture"], value.get("metadata"))
    if policy.parameter_count != value["parameter_count"]:
        raise ValueError("Neural parameter count mismatch")
    return policy


def build_torch_model(policy):
    """Autograd training model with an explicitly equivalent NumPy forward."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class TorchAttentionPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = dict(policy.architecture)
            self.params = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v.copy()))
                                            for k, v in policy.parameters.items()})

        def linear(self, x, prefix):
            return F.linear(x, self.params[prefix + "_weight"], self.params[prefix + "_bias"])

        def norm(self, x, prefix):
            return F.layer_norm(x, (self.architecture["embedding_dim"],),
                                self.params[prefix + "_weight"], self.params[prefix + "_bias"],
                                self.architecture["layer_norm_epsilon"])

        def encode(self, features, neighbors):
            a = self.architecture
            n, h, d = len(features), a["heads"], a["embedding_dim"]
            head = d // h
            x = torch.tanh(self.linear(features, "input"))
            for layer in range(a["encoder_layers"]):
                prefix = f"block{layer}"
                q = self.linear(x, prefix + "_q").reshape(n, h, head)
                k = self.linear(x, prefix + "_k").reshape(n, h, head)
                v = self.linear(x, prefix + "_v").reshape(n, h, head)
                logits = torch.einsum("nhd,nkhd->nkh", q, k[neighbors]) / math.sqrt(head)
                attention = torch.softmax(logits, dim=1)
                message = torch.einsum("nkh,nkhd->nhd", attention, v[neighbors]).reshape(n, d)
                x = self.norm(x + self.linear(message, prefix + "_out"), prefix + "_norm1")
                ff = self.linear(F.relu(self.linear(x, prefix + "_ff1")), prefix + "_ff2")
                x = self.norm(x + ff, prefix + "_norm2")
            return x

        def forward(self, embeddings, endpoints, dynamic, state_features):
            graph_mean = embeddings.mean(dim=0)
            action_x = torch.cat((embeddings[endpoints].reshape(len(endpoints), -1),
                                  graph_mean.expand(len(endpoints), -1), dynamic), dim=-1)
            hidden = F.relu(self.linear(action_x, "actor0"))
            hidden = F.relu(self.linear(hidden, "actor1"))
            logits = 10.0 * torch.tanh(self.linear(hidden, "actor2").reshape(-1))
            value_x = torch.cat((graph_mean, state_features))
            value = self.linear(F.relu(self.linear(value_x, "critic0")), "critic1").reshape(())
            return logits, value

        def export(self, metadata=None):
            return DeepPolicy({k: v.detach().cpu().numpy().copy() for k, v in self.params.items()},
                              self.architecture, metadata)

    return TorchAttentionPolicy()
