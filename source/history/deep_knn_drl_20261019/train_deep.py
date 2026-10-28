"""Train independent sparse-attention actor-critic policies on fresh maps.

Only generated train/validation maps are opened here.  Validation chooses a
checkpoint; benchmark test files are never read and receive no gradient.
The exported NumPy model is checked against the live autograd model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

# Conda NumPy and PyTorch bundle separate Intel OpenMP runtimes on Windows.
# Sequential MKL avoids initializing a second runtime; it is not the unsafe
# KMP_DUPLICATE_LIB_OK bypass.  This affects offline training only.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from neural_model import (DEFAULT_ARCHITECTURE, build_torch_model, dynamic_inputs,
                          graph_inputs, initialize_policy)


TRAINING_SEEDS = [2101, 2211, 2307, 2401, 2503]
TEST_SEEDS = [101, 211, 307, 401, 503]
SIZES = [128, 256, 512, 1000]
TRAIN_MAP_BASE, VAL_MAP_BASE = 1710000, 1810000
ACTION_SEED_XOR = 0x5E1EC7
_ENV_CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate_sha(coords):
    return hashlib.sha256(json.dumps(coords, ensure_ascii=True,
                                    separators=(",", ":")).encode("ascii")).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2,
                                   allow_nan=False), encoding="utf-8")


def make_map(split, index):
    seed = (TRAIN_MAP_BASE if split == "train" else VAL_MAP_BASE) + index
    n = SIZES[index % len(SIZES)]
    kind = ["uniform", "clustered"][(index // len(SIZES)) % 2]
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        xy = rng.uniform(0, 100000, (n, 2))
    else:
        centers = rng.uniform(15000, 85000, (8, 2))
        labels = rng.integers(0, 8, n)
        xy = np.clip(centers[labels] + rng.normal(0, 4500, (n, 2)), 0, 100000)
    coords = np.rint(xy).astype(np.int64).tolist()
    return dict(id=f"{split}_{index:04d}", split=split, seed=seed, n=n,
                distribution=kind, coordinate_sha256=coordinate_sha(coords), coords=coords)


def make_manifest(output, split, count):
    folder = output / "maps" / split
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        value = make_map(split, i)
        relative = f"maps/{split}/{value['id']}.json"
        dump(output / relative, value)
        records.append({k: v for k, v in value.items() if k != "coords"} | {"path": relative})
    dump(output / f"{split}_manifest.json", records)
    return records


def environment(output, entry, start_seed, settings):
    import deep_search as search
    key = (str(output), entry["id"], settings["knn"])
    if key not in _ENV_CACHE:
        record = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
        coords_exact = np.asarray(record["coords"], dtype=np.float64)
        delta = coords_exact[:, None] - coords_exact[None, :]
        distances = np.floor(np.sqrt((delta * delta).sum(axis=-1)) + .5).astype(np.int32)
        coords = coords_exact.astype(np.float32)
        knn = search.build_knn(distances, count=settings["knn"])
        _ENV_CACHE[key] = (coords, distances, knn)
    coords, distances, knn = _ENV_CACHE[key]
    route = np.empty(len(coords), dtype=np.int64)
    remaining = np.ones(len(coords), dtype=bool)
    route[0] = random.Random(start_seed).randrange(len(coords))
    remaining[route[0]] = False
    for i in range(1, len(coords)):
        route[i] = np.argmin(np.where(remaining, distances[route[i - 1]], np.iinfo(np.int32).max))
        remaining[route[i]] = False
    return coords, distances, knn, route


def exact_length(route, distances):
    return int(np.sum(distances[route, np.roll(route, -1)], dtype=np.int64))


def pool_context(pool, parent, step, config):
    return {"initial_mean_edge": pool.initial_length / len(parent["route"]),
            "parent_excess": (parent["cost"] - pool.best_length) / max(pool.best_length, 1),
            "step_fraction": step / config["test_step_cap"],
            "best_fraction": pool.best_length / max(pool.initial_length, 1)}


def _apply(pool, parent, distances, candidate, config):
    import deep_search as search
    result = search.evaluate_d_action(parent["route"], distances, candidate, budget=1)
    before = pool.best_length
    parent_before = exact_length(parent["route"], distances)
    child_length = exact_length(result["route"], distances)
    if child_length != parent_before - int(result["gain"]):
        raise AssertionError("Action exact gain mismatch")
    if sorted(np.asarray(result["route"]).tolist()) != list(range(len(result["route"]))):
        raise AssertionError("Action produced invalid tour")
    transition = pool.update(parent, result)
    reward_gain = before - pool.best_length
    if reward_gain != transition["best_gain"] or pool.best_length != exact_length(pool.best_route, distances):
        raise AssertionError("Incumbent reward identity failed")
    return result, transition, reward_gain, parent_before, child_length


def rollout(output, entry, model, config, proposal_seed, start_seed, optimizer=None,
            uniform=False):
    import torch
    import deep_search as search
    coords, distances, knn, route = environment(output, entry, start_seed, config)
    pool = search.TourPool(route, distances, config["pool_size"])
    proposal_rng = np.random.default_rng(proposal_seed)
    selection_rng = np.random.default_rng(proposal_seed ^ ACTION_SEED_XOR)
    features, neighbors = graph_inputs(coords, knn)
    x_t, neighbor_t = torch.from_numpy(features), torch.from_numpy(neighbors)
    training = optimizer is not None
    reward_scale = pool.initial_length / len(route)
    records, updates = [], []
    step = 0
    observed_gain = 0
    while step < config["steps"]:
        # Re-encode after every gradient update; parameters remain constant
        # within one n-step trajectory and backpropagate through the encoder.
        with torch.set_grad_enabled(training):
            embeddings = model.encode(x_t, neighbor_t)
            segment = []
            last_candidates = None
            for _ in range(min(config["n_step"], config["steps"] - step)):
                parent = pool.parent(step)
                context = pool_context(pool, parent, step, config)
                candidates, _ = search.enumerate_2opt_candidates(
                    parent["route"], distances, proposal_rng, config["candidate_count"], knn)
                if not candidates:
                    break
                endpoints, dynamic, state = dynamic_inputs(
                    coords, parent["route"], candidates, knn, distances, context)
                logits, value = model(embeddings, torch.from_numpy(endpoints),
                                      torch.from_numpy(dynamic), torch.from_numpy(state))
                probabilities = torch.softmax(logits, dim=0)
                p = probabilities.detach().cpu().numpy().astype(np.float64)
                p /= p.sum()
                if uniform:
                    chosen = int(selection_rng.integers(len(candidates)))
                else:
                    chosen = int(selection_rng.choice(len(candidates), p=p))
                result, transition, gain, parent_before, child_length = _apply(
                    pool, parent, distances, candidates[chosen], config)
                reward = gain / reward_scale
                observed_gain += gain
                logp = torch.log_softmax(logits, dim=0)[chosen]
                entropy = -(probabilities * torch.log_softmax(logits, dim=0)).sum()
                record = {"step": step, "parent_id": parent["id"],
                          "parent_length": parent_before, "child_length": child_length,
                          "action_index": chosen, "action_count": len(candidates),
                          "action": candidates[chosen], "parent_gain": int(result["gain"]),
                          "best_gain": int(gain), "reward": float(reward),
                          "best_length": pool.best_length, "pool_size": len(pool.entries),
                          "chosen_probability": float(p[chosen]),
                          "value": float(value.detach()), "entropy": float(entropy.detach()),
                          "context": context}
                records.append(record)
                segment.append((logp, entropy, value, reward, record))
                last_candidates = candidates
                step += 1
            if not segment:
                break
            if training:
                # Bootstrap at intermediate four-step truncations. The fixed
                # finite-horizon episode ends with zero further reward.
                if step < config["steps"]:
                    parent = pool.parent(step)
                    bootstrap_context = pool_context(pool, parent, step, config)
                    with torch.no_grad():
                        inputs = dynamic_inputs(coords, parent["route"], last_candidates,
                                                knn, distances, bootstrap_context)
                        _, bootstrap = model(embeddings.detach(), *(torch.from_numpy(q) for q in inputs))
                        ret = bootstrap.detach()
                    bootstrap_record = {"terminal": False, "next_step": step,
                                        "parent_id": parent["id"], "parent_length": parent["cost"],
                                        "context": bootstrap_context,
                                        "state_features": inputs[2].tolist(),
                                        "value": float(ret)}
                else:
                    ret = torch.tensor(0.0)
                    bootstrap_record = {"terminal": True, "next_step": step,
                                        "parent_id": None, "parent_length": None,
                                        "context": None, "state_features": None, "value": 0.0}
                terms = []
                for logp, entropy, value, reward, record in reversed(segment):
                    ret = reward + config["gamma"] * ret
                    advantage = ret.detach() - value
                    terms.append(-logp * advantage.detach()
                                 - config["entropy_coefficient"] * entropy
                                 + config["critic_weight"] * advantage.square())
                    record["n_step_return"] = float(ret.detach())
                    record["advantage"] = float(advantage.detach())
                loss = torch.stack(terms).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite actor-critic gradient")
                encoder_norm = float(sum(float(p.grad.square().sum()) for k, p in model.params.items()
                                         if p.grad is not None and (k.startswith("block") or k.startswith("input"))) ** .5)
                optimizer.step()
                updates.append({"first_step": segment[0][4]["step"], "after_step": step,
                                "bootstrap": bootstrap_record,
                                "bootstrap_value": bootstrap_record["value"],
                                "bootstrap_after_step": step,
                                "finite_horizon_terminal": bootstrap_record["terminal"],
                                "loss": float(loss.detach()),
                                "gradient_norm_before_clip": float(norm),
                                "encoder_gradient_norm_after_clip": encoder_norm})
    final_length = exact_length(pool.best_route, distances)
    if pool.initial_length - final_length != observed_gain:
        raise AssertionError("Episode rewards do not telescope")
    return {"map_id": entry["id"], "coordinate_sha256": entry["coordinate_sha256"],
            "n": entry["n"], "distribution": entry["distribution"],
            "proposal_seed": int(proposal_seed), "selection_seed": int(proposal_seed ^ ACTION_SEED_XOR),
            "start_seed": int(start_seed), "initial_length": pool.initial_length,
            "final_length": final_length, "total_exact_gain": observed_gain,
            "improvement_fraction": observed_gain / pool.initial_length,
            "reward_scale_initial_mean_edge": reward_scale,
            "reward_identity_passed": True, "uniform_baseline": uniform,
            "steps": records, "updates": updates}


def validate(output, entries, model, config, checkpoint, logpath, uniform=False):
    started = time.perf_counter()
    improvements = []
    with logpath.open("a", encoding="utf-8") as handle:
        for i, entry in enumerate(entries):
            episode = rollout(output, entry, model, config, 19100000 + i * 97,
                              19300000 + i, uniform=uniform)
            episode["checkpoint_episode"] = checkpoint
            handle.write(json.dumps(episode, ensure_ascii=True, allow_nan=False) + "\n")
            improvements.append(episode["improvement_fraction"])
    return {"checkpoint_episode": checkpoint, "uniform_baseline": uniform,
            "mean_improvement_fraction": float(np.mean(improvements)),
            "map_count": len(entries), "seconds": time.perf_counter() - started}


def export_parity(output, entry, model, config):
    import torch
    import deep_search as search
    coords, distances, knn, route = environment(output, entry, 19700000, config)
    candidates, _ = search.enumerate_2opt_candidates(route, distances, np.random.default_rng(1),
                                                   config["candidate_count"], knn)
    x, neighbors = graph_inputs(coords, knn)
    inputs = dynamic_inputs(coords, route, candidates, knn, distances)
    exported = model.export()
    exported.prepare(coords, knn)
    expected_logits, expected_value = exported.forward_inputs(*inputs)
    with torch.no_grad():
        embeddings = model.encode(torch.from_numpy(x), torch.from_numpy(neighbors))
        logits, value = model(embeddings, *(torch.from_numpy(q) for q in inputs))
    error_logits = float(np.max(np.abs(logits.numpy() - expected_logits)))
    error_embeddings = float(np.max(np.abs(embeddings.numpy() - exported.embeddings)))
    error_value = abs(float(value) - expected_value)
    if max(error_logits, error_embeddings, error_value) > 1e-4:
        raise AssertionError("Torch/NumPy export parity failed")
    return {"passed": True, "max_logit_absolute_error": error_logits,
            "max_embedding_absolute_error": error_embeddings, "value_absolute_error": error_value}


def train_one(output, entries, validation_entries, training_seed, config):
    import torch
    folder = output / f"seed_{training_seed}"
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(training_seed)
    torch.manual_seed(training_seed)
    model = build_torch_model(initialize_policy(training_seed, config["architecture"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    initial = model.export()
    initial.save(folder / "initial_model_D.json")
    best_score, selected = -np.inf, None
    scores = []
    train_seconds = validation_seconds = 0.0
    episode_index = 0
    for epoch in range(config["epochs"]):
        for index in rng.permutation(len(entries)):
            started = time.perf_counter()
            proposal_seed = int(rng.integers(1, 2**62))
            start_seed = int(rng.integers(1, 2**62))
            episode = rollout(output, entries[int(index)], model, config, proposal_seed,
                              start_seed, optimizer=optimizer)
            episode_index += 1
            duration = time.perf_counter() - started
            train_seconds += duration
            episode.update(episode=episode_index, epoch=epoch + 1, training_seed=training_seed,
                           episode_seconds=duration)
            with (folder / "training_D.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode, ensure_ascii=True, allow_nan=False) + "\n")
            if episode_index % 16 == 0 and episode_index not in config["checkpoints"]:
                print(json.dumps({"training_seed": training_seed, "episode": episode_index,
                                  "training_seconds": train_seconds,
                                  "validation_seconds": validation_seconds,
                                  "phase": "training_progress"}), flush=True)
            if episode_index in config["checkpoints"]:
                score = validate(output, validation_entries, model, config, episode_index,
                                 folder / "validation_D.jsonl")
                validation_seconds += score["seconds"]
                scores.append(score)
                parity = export_parity(output, validation_entries[0], model, config)
                if score["mean_improvement_fraction"] > best_score:
                    best_score = score["mean_improvement_fraction"]
                    selected = model.export({"kind": "D", "training_seed": training_seed,
                                             "checkpoint_episode": episode_index,
                                             "validation_mean_improvement_fraction": best_score,
                                             "training_config": config,
                                             "selection_rule": "maximize mean validation incumbent tour gain; earliest checkpoint wins ties",
                                             "deep_neural_network": True,
                                             "algorithm": "n-step on-policy advantage actor-critic with learned critic, entropy and Adam",
                                             "reward": "pool global incumbent integer length decrease / initial mean edge length",
                                             "export_parity": parity})
                print(json.dumps({"training_seed": training_seed, "episode": episode_index,
                                  "validation": score["mean_improvement_fraction"],
                                  "training_seconds": train_seconds,
                                  "validation_seconds": validation_seconds}), flush=True)
    if selected is None:
        raise AssertionError("No validation-selected checkpoint")
    selected.save(folder / "model_D.json")
    model_parameter_change = float(sum(np.square(selected.parameters[k] - initial.parameters[k]).sum()
                                       for k in selected.parameters) ** .5)
    encoder_parameter_change = float(sum(np.square(selected.parameters[k] - initial.parameters[k]).sum()
                                        for k in selected.parameters if k.startswith("block") or k.startswith("input")) ** .5)
    actor_parameter_change = float(sum(np.square(selected.parameters[k] - initial.parameters[k]).sum()
                                      for k in selected.parameters if k.startswith("actor")) ** .5)
    critic_parameter_change = float(sum(np.square(selected.parameters[k] - initial.parameters[k]).sum()
                                       for k in selected.parameters if k.startswith("critic")) ** .5)
    if min(encoder_parameter_change, actor_parameter_change, critic_parameter_change) == 0:
        raise AssertionError("Encoder, actor and critic must all be trained")
    report = {"kind": "D", "training_seed": training_seed,
              "model_path": f"seed_{training_seed}/model_D.json",
              "model_file_sha256": sha(folder / "model_D.json"),
              "selected_checkpoint": selected.metadata["checkpoint_episode"],
              "checkpoint_scores": scores, "episodes": episode_index,
              "parameter_count": selected.parameter_count,
              "selected_parameter_change_l2": model_parameter_change,
              "selected_encoder_parameter_change_l2": encoder_parameter_change,
              "selected_actor_parameter_change_l2": actor_parameter_change,
              "selected_critic_parameter_change_l2": critic_parameter_change,
              "training_seconds": train_seconds, "validation_seconds": validation_seconds}
    dump(folder / "training_report_D.json", report)
    return report


def main():
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--train-maps", type=int, default=128)
    parser.add_argument("--validation-maps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-step", type=int, default=4)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=TRAINING_SEEDS)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("Training output must be empty; refusing overwrite")
    sources = {name: sha(Path(__file__).with_name(name))
               for name in ("train_deep.py", "neural_model.py", "deep_search.py", "search_core.py")}
    dump(output / "source_seal_start.json", {"created_utc": started_utc, "source_sha256": sources})
    train = make_manifest(output, "train", args.train_maps)
    validation = make_manifest(output, "validation", args.validation_maps)
    if {x["coordinate_sha256"] for x in train} & {x["coordinate_sha256"] for x in validation}:
        raise AssertionError("Train and validation overlap")
    dimension = args.embedding_dim
    architecture = dict(DEFAULT_ARCHITECTURE, embedding_dim=dimension,
                        encoder_ff_dim=dimension * 2,
                        actor_hidden=[dimension * 2, dimension], critic_hidden=dimension)
    config = {"architecture": architecture, "steps": args.steps, "n_step": args.n_step,
              "epochs": args.epochs, "gamma": .99, "learning_rate": args.learning_rate,
              "entropy_coefficient": .01, "critic_weight": .5, "gradient_clip": 1.0,
              "pool_size": 3, "candidate_count": 192, "knn": 16, "exact_quota": 1,
              "test_step_cap": 128, "checkpoints": [len(train) * (i + 1) for i in range(args.epochs)],
              "optimizer": "Adam", "training_device": "CPU", "torch_num_threads": 1,
              "sampling": "categorical softmax during training, validation and testing",
              "reward": "global incumbent improvement / initial mean edge length",
              "horizon": "finite episode: intermediate four-step segments bootstrap learned value; final episode boundary bootstrap is zero",
              "acceptance": "strictly improving two-opt only; selected worsening moves keep parent unchanged (declared deviation from paper)",
              "initial_route": "nearest neighbor from seeded random starting city",
              "encoder": "two sparse kNN+self attention blocks; static embeddings cached between trajectory updates",
              "rng_streams": "proposal RNG=seed; separate selection RNG=seed XOR 0x5E1EC7"}
    dump(output / "frozen_training_config.json", config)
    preparation_seconds = time.perf_counter() - started
    uniform_model = build_torch_model(initialize_policy(1, architecture))
    uniform = validate(output, validation, uniform_model, config, "uniform",
                       output / "uniform_validation_D.jsonl", uniform=True)
    reports = [train_one(output, train, validation, seed, config) for seed in args.training_seeds]
    registry = {"schema": 2, "by_test_seed": {}}
    for seed in args.training_seeds:
        test_seed = TEST_SEEDS[TRAINING_SEEDS.index(seed)] if seed in TRAINING_SEEDS else seed
        registry["by_test_seed"][str(test_seed)] = {"training_seed": seed,
                                                     "D": f"seed_{seed}/model_D.json"}
    dump(output / "model_registry.json", registry)
    for name, expected in sources.items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise RuntimeError("Training source changed during run: " + name)
    report = {"schema_version": 2, "pilot": args.pilot, "deep_neural_network": True,
              "training_completed_before_formal_test": True, "test_data_read": False,
              "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "training_seeds": args.training_seeds, "test_seed_mapping": registry["by_test_seed"],
              "train_map_count": len(train), "validation_map_count": len(validation),
              "train_seed_domain": [TRAIN_MAP_BASE, TRAIN_MAP_BASE + len(train) - 1],
              "validation_seed_domain": [VAL_MAP_BASE, VAL_MAP_BASE + len(validation) - 1],
              "coordinate_hash_spec": "SHA256 compact ASCII JSON original city-order coordinates",
              "map_sizes": SIZES, "distributions": ["uniform", "clustered"],
              "validation_only_checkpoint_selection": True, "validation_gradients_computed": False,
              "config": config, "models": reports, "uniform_validation_baseline": {"D": uniform},
              "end_to_end_seconds": time.perf_counter() - started,
              "preparation_seconds": preparation_seconds,
              "training_seconds": sum(r["training_seconds"] for r in reports),
              "validation_seconds": sum(r["validation_seconds"] for r in reports),
              "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
              "platform": platform.platform(), "source_sha256": sources,
              "mkl_threading_layer": os.environ.get("MKL_THREADING_LAYER"),
              "online_dependency": "NumPy only; torch needed for offline training only",
              "limitations": ["Paper-inspired sparse adaptation, not exact paper reproduction",
                              "Much smaller offline training scale than the paper",
                              "Train sizes128..1000; larger test sizes evaluate transfer",
                              "Shared actor-critic static geometry encoder differs from paper separate encoders",
                              "Static attention omits tour positions; dynamic decoder uses current route edges and arcs"]}
    dump(output / "training_report.json", report)
    print(json.dumps({"completed": True, "output": str(output),
                      "seconds": report["end_to_end_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
