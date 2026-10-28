"""Train five independently seeded C/D REINFORCE policies on disjoint maps.

All choices here are made without reading any formal benchmark test map.
Validation selects checkpoints; no gradient is computed on validation maps.
Run from any directory: python train.py --output-dir models
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from policy import LinearPolicy, finite_difference_check, reinforce_gradient


TRAINING_SEEDS = [1101, 1211, 1307, 1401, 1503]
TEST_SEEDS = [101, 211, 307, 401, 503]
SIZES = [128, 256, 512, 1000]
TRAIN_MAP_BASE = 710000
VALIDATION_MAP_BASE = 810000
ACTION_SEED_XOR = 0x5E1EC7


def coordinate_hash(coords):
    return hashlib.sha256(json.dumps(coords, separators=(",", ":"),
                                    ensure_ascii=True).encode("ascii")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def make_map(split, index):
    seed = (TRAIN_MAP_BASE if split == "train" else VALIDATION_MAP_BASE) + index
    n = SIZES[index % len(SIZES)]
    kind = ["uniform", "clustered"][(index // len(SIZES)) % 2]
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        xy = rng.uniform(0, 100000, size=(n, 2))
    else:
        centers = rng.uniform(15000, 85000, size=(8, 2))
        labels = rng.integers(0, len(centers), size=n)
        xy = centers[labels] + rng.normal(0, 4500, size=(n, 2))
        xy = np.clip(xy, 0, 100000)
    coords = np.rint(xy).astype(np.int64).tolist()
    return {"id": f"{split}_{index:04d}", "split": split, "seed": seed,
            "n": n, "distribution": kind, "coordinate_sha256": coordinate_hash(coords),
            "coords": coords}


def make_manifest(output_dir, split, count):
    mapdir = output_dir / "maps" / split
    mapdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(count):
        record = make_map(split, i)
        filename = f"maps/{split}/{record['id']}.json"
        dump_json(output_dir / filename, record)
        manifest.append({k: v for k, v in record.items() if k != "coords"}
                        | {"path": filename})
    dump_json(output_dir / f"{split}_manifest.json", manifest)
    return manifest


def load_environment(output_dir, entry, start_seed):
    record = json.loads((output_dir / entry["path"]).read_text(encoding="utf-8"))
    coords = np.asarray(record["coords"], dtype=np.float64)
    difference = coords[:, None, :] - coords[None, :, :]
    distances = np.floor(np.sqrt(np.sum(difference * difference, axis=2)) + 0.5).astype(np.int32)
    start = int(np.random.default_rng(start_seed).integers(len(coords)))
    route = np.empty(len(coords), dtype=np.int64)
    remaining = np.ones(len(coords), dtype=bool)
    route[0] = start
    remaining[start] = False
    for pos in range(1, len(coords)):
        values = np.where(remaining, distances[route[pos - 1]], np.iinfo(np.int32).max)
        city = int(np.argmin(values))
        route[pos] = city
        remaining[city] = False
    return route, distances


def route_length(route, distances):
    return int(np.sum(distances[route, np.roll(route, -1)], dtype=np.int64))


def discounted_returns(rewards, gamma):
    values = np.zeros(len(rewards), dtype=np.float64)
    total = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        total = float(rewards[t]) + gamma * total
        values[t] = total
    return values


def action_key(action):
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    return json.loads(json.dumps(action, default=default))


def rollout(output_dir, entry, kind, policy, steps, pool_seed, action_seed,
            start_seed, candidate_count, budget, collect_gradient=False,
            uniform=False):
    import search

    route, distances = load_environment(output_dir, entry, start_seed)
    pool = search.TourPool(route, distances, pool_size=3)
    knn = search.build_knn(distances, count=16) if kind == "D" else None
    initial_length = pool.best_length
    reward_scale = initial_length / len(route)
    pool_rng = np.random.default_rng(pool_seed)
    action_rng = np.random.default_rng(action_seed)
    trajectory = []
    observed_gain = 0
    for step in range(steps):
        parent = pool.parent(step)
        parent_route = parent["route"]
        parent_excess = (parent["cost"] - pool.best_length) / max(pool.best_length, 1)
        if kind == "C":
            actions, features = search.enumerate_actions(
                parent_route, distances, rng=pool_rng, count=12,
                window_sizes=(24, 48, 64), parent_excess=parent_excess)
        else:
            actions, features = search.enumerate_2opt_candidates(
                parent_route, distances, rng=pool_rng, candidate_count=candidate_count, knn=knn)
        if not len(actions):
            break
        if uniform:
            probabilities = np.full(len(actions), 1.0 / len(actions))
            chosen = int(action_rng.integers(len(actions)))
        else:
            chosen, probabilities = policy.sample(features, action_rng)
        before = pool.best_length
        parent_before = route_length(parent_route, distances)
        if kind == "C":
            result = search.evaluate_action(parent_route, distances, actions[chosen], budget=budget)
        else:
            result = search.evaluate_d_action(parent_route, distances, actions[chosen], budget=1)
        child_route = np.asarray(result["route"], dtype=np.int64)
        child_length = route_length(child_route, distances)
        parent_gain = parent_before - child_length
        if parent_gain != int(result["gain"]) or parent_gain < 0:
            raise AssertionError(f"Exact parent gain mismatch: {parent_gain} versus {result['gain']}")
        if len(np.unique(child_route)) != len(child_route):
            raise AssertionError("Environment returned nonpermutation")
        transition = pool.update(parent, result)
        after = pool.best_length
        gain = before - after
        if gain != int(transition["best_gain"]) or gain < 0:
            raise AssertionError("Pool global-best reward mismatch")
        if after != route_length(pool.best_route, distances):
            raise AssertionError("Pool best length differs from actual best tour")
        observed_gain += gain
        record = {"step": step, "action_index": chosen,
                  "parent_id": int(parent["id"]), "parent_excess": parent_excess,
                  "action_count": len(actions), "action": action_key(actions[chosen]),
                  "chosen_features": np.asarray(features[chosen]).tolist(),
                  "policy_probability": float(probabilities[chosen]),
                  "entropy": float(-np.dot(probabilities, np.log(np.maximum(probabilities, 1e-300)))),
                  "exact_length_before": before, "exact_length_after": after,
                  "parent_exact_length_before": parent_before,
                  "child_exact_length": child_length, "parent_gain": parent_gain,
                  "exact_gain": gain, "reward": gain / reward_scale,
                  "evaluations": int(result.get("evaluations", 0))}
        if collect_gradient:
            record["_features"] = policy.transform(features)
            record["_probabilities"] = probabilities
        trajectory.append(record)
    final_length = route_length(pool.best_route, distances)
    if observed_gain != initial_length - final_length:
        raise AssertionError("Cumulative reward does not telescope to true tour gain")
    return {"map_id": entry["id"], "coordinate_sha256": entry["coordinate_sha256"],
            "kind": kind, "n": entry["n"], "distribution": entry["distribution"],
            "pool_seed": pool_seed, "action_seed": action_seed, "start_seed": start_seed,
            "initial_length": initial_length, "final_length": final_length,
            "total_exact_gain": observed_gain,
            "improvement_fraction": observed_gain / initial_length,
            "reward_scale_initial_mean_edge": reward_scale,
            "sum_reward": sum(x["reward"] for x in trajectory),
            "reward_identity_passed": True, "steps": trajectory}


def public_episode(episode):
    return {k: ([{key: value for key, value in step.items() if not key.startswith("_")}
                 for step in v] if k == "steps" else v)
            for k, v in episode.items()}


def validation(output_dir, entries, kind, policy, config, checkpoint, logpath, uniform=False):
    started = time.perf_counter()
    fractions = []
    with logpath.open("a", encoding="utf-8") as handle:
        for i, entry in enumerate(entries):
            proposal_seed = 91000000 + i * 97 + (kind == "D")
            episode = rollout(output_dir, entry, kind, policy, config["steps"],
                              proposal_seed, proposal_seed ^ ACTION_SEED_XOR,
                              93000000 + i, config["candidate_count"],
                              config["candidate_budget"], uniform=uniform)
            fractions.append(episode["improvement_fraction"])
            value = public_episode(episode)
            value.update(checkpoint=checkpoint, uniform_baseline=uniform)
            handle.write(json.dumps(value, ensure_ascii=True) + "\n")
    return {"checkpoint": checkpoint, "uniform_baseline": uniform,
            "mean_improvement_fraction": float(np.mean(fractions)),
            "mean_remaining_fraction": 1.0 - float(np.mean(fractions)),
            "map_count": len(entries), "seconds": time.perf_counter() - started}


def train_one(output_dir, train_entries, val_entries, kind, training_seed, config):
    import search

    root = output_dir / f"seed_{training_seed}"
    root.mkdir(parents=True, exist_ok=True)
    actual_seed = training_seed + (1000000 if kind == "D" else 0)
    rng = np.random.default_rng(actual_seed)
    policy = LinearPolicy(rng.normal(0, 0.01, size=12))
    baseline = np.zeros(config["steps"], dtype=np.float64)
    baseline_counts = np.zeros(config["steps"], dtype=np.int64)
    adam_m = np.zeros(12)
    adam_v = np.zeros(12)
    checkpoint_scores = []
    best_score = -np.inf
    best_value = None
    validation_seconds = 0.0
    training_seconds = 0.0
    logpath = root / f"training_{kind}.jsonl"
    vallogpath = root / f"validation_{kind}.jsonl"
    order = rng.permutation(len(train_entries))
    for episode_index, map_index in enumerate(order, 1):
        started = time.perf_counter()
        pool_seed = int(rng.integers(1, 2**62))
        action_seed = pool_seed ^ ACTION_SEED_XOR
        start_seed = int(rng.integers(1, 2**62))
        episode = rollout(output_dir, train_entries[int(map_index)], kind, policy,
                          config["steps"], pool_seed, action_seed, start_seed,
                          config["candidate_count"], config["candidate_budget"],
                          collect_gradient=True)
        returns = discounted_returns([s["reward"] for s in episode["steps"]], config["gamma"])
        gradient = np.zeros(12)
        for t, (step, ret) in enumerate(zip(episode["steps"], returns)):
            advantage = float(ret - baseline[t])
            discount_weight = config["gamma"] ** t
            gradient += discount_weight * reinforce_gradient(
                step["_features"], step["action_index"],
                step["_probabilities"], advantage, config["entropy_coefficient"])
            step["discounted_return"] = float(ret)
            step["baseline_before_update"] = float(baseline[t])
            step["advantage"] = advantage
            step["gradient_discount_weight"] = discount_weight
            if baseline_counts[t] == 0:
                baseline[t] = ret
            else:
                baseline[t] = 0.95 * baseline[t] + 0.05 * ret
            baseline_counts[t] += 1
        gradient /= max(1, len(returns))
        norm = float(np.linalg.norm(gradient))
        if norm > config["gradient_clip"]:
            gradient *= config["gradient_clip"] / norm
        adam_m = 0.9 * adam_m + 0.1 * gradient
        adam_v = 0.999 * adam_v + 0.001 * gradient * gradient
        corrected_m = adam_m / (1 - 0.9**episode_index)
        corrected_v = adam_v / (1 - 0.999**episode_index)
        policy.weights += config["learning_rate"] * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        # A fixed parameter constraint stabilizes the softmax; never test-tuned.
        policy.weights[:] = np.clip(policy.weights, -5.0, 5.0)
        duration = time.perf_counter() - started
        training_seconds += duration
        public = public_episode(episode)
        public.update(episode=episode_index, training_seed=training_seed,
                      gradient_norm_before_clip=norm, weights_after_update=policy.weights.tolist(),
                      episode_seconds=duration)
        with logpath.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(public, ensure_ascii=True) + "\n")
        if episode_index in config["checkpoints"] or episode_index == len(order):
            result = validation(output_dir, val_entries, kind, policy, config,
                                episode_index, vallogpath)
            validation_seconds += result["seconds"]
            checkpoint_scores.append(result)
            if result["mean_improvement_fraction"] > best_score:
                best_score = result["mean_improvement_fraction"]
                policy.metadata = {"kind": kind, "training_seed": training_seed,
                                   "actual_rng_seed": actual_seed,
                                   "checkpoint_episode": episode_index,
                                   "validation_mean_improvement_fraction": best_score,
                                   "training_config": config,
                                   "selection_rule": "maximize mean validation fractional tour gain; earliest checkpoint wins ties",
                                   "algorithm": "episodic REINFORCE with time-indexed previous-episode EMA baseline and Adam",
                                   "reward": "exact pool-global-best full-tour integer length decrease / initial mean edge length",
                                   "deep_neural_network": False}
                policy.metadata["feature_names"] = list(
                    search.ACTION_FEATURE_NAMES if kind == "C" else search.EDGE_FEATURE_NAMES)
                best_value = policy.to_dict()
            print(json.dumps({"training_seed": training_seed, "kind": kind,
                              "episode": episode_index, "validation": result["mean_improvement_fraction"],
                              "training_seconds": training_seconds,
                              "validation_seconds": validation_seconds}), flush=True)
    dump_json(root / f"model_{kind}.json", best_value)
    result = {"training_seed": training_seed, "kind": kind,
              "model_path": f"seed_{training_seed}/model_{kind}.json",
              "model_file_sha256": file_hash(root / f"model_{kind}.json"),
              "selected_checkpoint": best_value["metadata"]["checkpoint_episode"],
              "checkpoint_scores": checkpoint_scores,
              "training_seconds": training_seconds, "validation_seconds": validation_seconds,
              "episodes": len(order), "reward_identities_checked": len(order) + len(val_entries) * len(checkpoint_scores)}
    dump_json(root / f"training_report_{kind}.json", result)
    return result


def main():
    program_started = time.perf_counter()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("models"))
    parser.add_argument("--train-maps", type=int, default=160)
    parser.add_argument("--validation-maps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--candidate-budget", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=192)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=TRAINING_SEEDS)
    parser.add_argument("--checkpoint-count", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training_report.json").exists():
        raise SystemExit("Refusing to overwrite completed training; choose a new output directory")
    search_source_hash = file_hash(Path(__file__).with_name("search.py"))
    policy_source_hash = file_hash(Path(__file__).with_name("policy.py"))
    gradient_check = finite_difference_check()
    train_entries = make_manifest(output_dir, "train", args.train_maps)
    val_entries = make_manifest(output_dir, "validation", args.validation_maps)
    if {m["coordinate_sha256"] for m in train_entries} & {m["coordinate_sha256"] for m in val_entries}:
        raise AssertionError("Train/validation coordinate overlap")
    checkpoints = sorted(set(max(1, int(round(args.train_maps * (j + 1) / args.checkpoint_count)))
                             for j in range(args.checkpoint_count)))
    config = {"steps": args.steps, "candidate_budget": args.candidate_budget,
              "candidate_count": args.candidate_count, "gamma": 0.97,
              "learning_rate": 0.025, "entropy_coefficient": 0.01,
              "gradient_clip": 5.0, "checkpoints": checkpoints,
              "action_sampling": "categorical softmax; stochastic at train and validation",
              "rng_streams": "proposal_rng=seed; independent selection_rng=seed XOR 0x5E1EC7, identical to test",
              "initial_route": "nearest neighbor from seeded random starting city",
              "C_window_sizes": [24, 48, 64], "C_action_count": 12,
              "pool_size": 3, "pool_schedule": "step modulo current pool size",
              "pool_retention": "incumbent plus routes with most distinct undirected edges from incumbent",
              "C_parent_excess": "(parent_length-pool_best_length)/pool_best_length",
              "D_knn": 16, "D_exact_evaluations_per_action": 1,
              "reward_scale": "initial mean full-tour edge; fixed throughout episode"}
    dump_json(output_dir / "frozen_training_config.json", config)
    preparation_seconds = time.perf_counter() - program_started
    started = time.perf_counter()
    uniform_validation = {}
    for kind in ["C", "D"]:
        uniform_validation[kind] = validation(
            output_dir, val_entries, kind, LinearPolicy(), config,
            "uniform", output_dir / f"uniform_validation_{kind}.jsonl", uniform=True)
    results = []
    for seed in args.training_seeds:
        for kind in ["C", "D"]:
            results.append(train_one(output_dir, train_entries, val_entries, kind, seed, config))
    registry = {"schema": 1, "by_test_seed": {}}
    for seed in args.training_seeds:
        test_seed = TEST_SEEDS[TRAINING_SEEDS.index(seed)] if seed in TRAINING_SEEDS else seed
        registry["by_test_seed"][str(test_seed)] = {
            "training_seed": seed, "C": f"seed_{seed}/model_C.json", "D": f"seed_{seed}/model_D.json"}
    dump_json(output_dir / "model_registry.json", registry)
    if file_hash(Path(__file__).with_name("search.py")) != search_source_hash:
        raise RuntimeError("Search source changed during training; models cannot be frozen")
    if file_hash(Path(__file__).with_name("policy.py")) != policy_source_hash:
        raise RuntimeError("Policy source changed during training; models cannot be frozen")
    completed_utc = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    report = {"schema_version": 1, "pilot": args.pilot,
              "training_completed_before_formal_test": True,
              "test_data_read": False,
              "started_utc": started_utc, "completed_utc": completed_utc,
              "training_seeds": args.training_seeds,
              "test_seed_mapping": registry["by_test_seed"],
              "train_map_count": len(train_entries), "validation_map_count": len(val_entries),
              "train_seed_domain": [TRAIN_MAP_BASE, TRAIN_MAP_BASE + args.train_maps - 1],
              "validation_seed_domain": [VALIDATION_MAP_BASE, VALIDATION_MAP_BASE + args.validation_maps - 1],
              "coordinate_hash_spec": "SHA256 compact ASCII JSON coords (original city order)",
              "map_sizes": SIZES, "distributions": ["uniform", "clustered"],
              "validation_only_checkpoint_selection": True,
              "validation_gradients_computed": False,
              "gradient_finite_difference_check": gradient_check,
              "uniform_validation_baseline": uniform_validation,
              "config": config, "models": results,
              "total_seconds": time.perf_counter() - started,
              "total_seconds_scope": "training and validation phases; excludes preparation",
              "end_to_end_seconds": time.perf_counter() - program_started,
              "preparation_seconds": preparation_seconds,
              "training_seconds": sum(r["training_seconds"] for r in results),
              "validation_seconds": sum(r["validation_seconds"] for r in results),
              "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
              "train_source_sha256": file_hash(__file__),
              "policy_source_sha256": policy_source_hash,
              "search_source_sha256": search_source_hash,
              "prior_aborted_run": {
                  "path": "work/rl_benchmark/models",
                  "reason": "training/test random-stream alignment repair before any formal test",
                  "selected_models_from_aborted_run": False,
                  "test_result_driven": False},
              "limitations": ["Linear policy is not a deep neural model",
                              "Training sizes are 128..1000; larger test sizes measure generalization",
                              "Only checkpoint is selected by validation; no formal test tuning",
                              "Linear softmax has no nonlinear state-action interactions"]}
    dump_json(output_dir / "training_report.json", report)
    print(json.dumps({"completed": True, "output_dir": str(output_dir),
                      "total_seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
