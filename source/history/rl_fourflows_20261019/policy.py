"""Auditable NumPy linear-softmax policies; no training occurs at inference.

This is a lightweight reinforcement-learning policy, not a deep network.
Training uses sequential sampled actions and actual environment rewards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FEATURE_COUNT = 12


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def model_digest(value):
    clean = dict(value)
    clean.pop("sha256", None)
    return hashlib.sha256(canonical_bytes(clean)).hexdigest()


class LinearPolicy:
    def __init__(self, weights=None, feature_mean=None, feature_scale=None,
                 metadata=None):
        self.weights = np.asarray(np.zeros(FEATURE_COUNT) if weights is None
                                  else weights, dtype=np.float64)
        self.feature_mean = np.asarray(np.zeros(len(self.weights)) if feature_mean
                                       is None else feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(np.ones(len(self.weights)) if feature_scale
                                        is None else feature_scale, dtype=np.float64)
        if not (self.weights.ndim == 1 and self.feature_mean.shape == self.weights.shape
                and self.feature_scale.shape == self.weights.shape
                and np.all(self.feature_scale > 0)
                and np.all(np.isfinite(self.weights))):
            raise ValueError("Invalid model parameters")
        self.metadata = {} if metadata is None else metadata

    def transform(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.weights):
            raise ValueError("Expected action-by-feature matrix")
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite policy features")
        return (x - self.feature_mean) / self.feature_scale

    def score(self, features):
        return self.transform(features) @ self.weights

    def probabilities(self, features):
        logits = self.score(features)
        if not len(logits):
            raise ValueError("Empty action set")
        values = np.exp(logits - np.max(logits))
        return values / values.sum()

    def sample(self, features, rng):
        probabilities = self.probabilities(features)
        return int(rng.choice(len(probabilities), p=probabilities)), probabilities

    def to_dict(self):
        value = {"schema_version": 1, "model_type": "linear_softmax_reinforce",
                 "feature_count": len(self.weights),
                 "weights": self.weights.tolist(),
                 "feature_mean": self.feature_mean.tolist(),
                 "feature_scale": self.feature_scale.tolist(),
                 "metadata": self.metadata}
        value["sha256"] = model_digest(value)
        return value

    @classmethod
    def from_dict(cls, value):
        if value.get("sha256") != model_digest(value):
            raise ValueError("Model content hash mismatch")
        return cls(value["weights"], value.get("feature_mean"),
                   value.get("feature_scale"), value.get("metadata"))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                        indent=2), encoding="utf-8")


def load_policy(path):
    return LinearPolicy.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def reinforce_gradient(features, chosen, probabilities, advantage,
                       entropy_coefficient=0.0):
    """Exact gradient of advantage*log pi(a|s) + beta*entropy(pi)."""
    x = np.asarray(features, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    grad = advantage * (x[int(chosen)] - p @ x)
    if entropy_coefficient:
        logp = np.log(np.maximum(p, 1e-300))
        entropy = -np.dot(p, logp)
        grad -= entropy_coefficient * ((p * (logp + entropy)) @ x)
    return grad


def finite_difference_check(seed=90001):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(9, FEATURE_COUNT))
    policy = LinearPolicy(rng.normal(scale=0.2, size=FEATURE_COUNT))
    chosen, advantage, beta = 3, 0.73, 0.013
    analytic = reinforce_gradient(x, chosen, policy.probabilities(x), advantage, beta)

    def objective():
        p = policy.probabilities(x)
        return advantage * np.log(p[chosen]) - beta * np.dot(p, np.log(p))

    numeric = np.zeros(FEATURE_COUNT)
    epsilon = 1e-6
    for j in range(FEATURE_COUNT):
        old = policy.weights[j]
        policy.weights[j] = old + epsilon
        hi = objective()
        policy.weights[j] = old - epsilon
        lo = objective()
        policy.weights[j] = old
        numeric[j] = (hi - lo) / (2 * epsilon)
    error = float(np.max(np.abs(analytic - numeric)))
    if error > 1e-7:
        raise AssertionError(f"Policy gradient failed finite differences: {error}")
    return {"seed": seed, "epsilon": epsilon, "max_absolute_error": error,
            "passed": True}


if __name__ == "__main__":
    print(json.dumps(finite_difference_check(), indent=2))
