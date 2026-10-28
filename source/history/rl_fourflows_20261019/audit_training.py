"""Independent ledger/model audit; never reads formal benchmark test inputs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from policy import load_policy


def audit(root):
    root = Path(root)
    report = json.loads((root / "training_report.json").read_text(encoding="utf-8"))
    for component in ["train", "policy", "search"]:
        source = Path(__file__).with_name(f"{component}.py")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == report[f"{component}_source_sha256"]
    checks = {"training_episodes": 0, "validation_episodes": 0, "steps": 0,
              "exact_global_reward_steps": 0, "parent_only_gain_steps": 0,
              "nonzero_gradient_episodes": 0}
    manifests = {}
    for split in ["train", "validation"]:
        entries = json.loads((root / f"{split}_manifest.json").read_text(encoding="utf-8"))
        manifests[split] = entries
        for entry in entries:
            record = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
            hashed = hashlib.sha256(json.dumps(record["coords"], ensure_ascii=True,
                                               separators=(",", ":")).encode("ascii")).hexdigest()
            assert hashed == entry["coordinate_sha256"]
    assert not ({m["coordinate_sha256"] for m in manifests["train"]}
                & {m["coordinate_sha256"] for m in manifests["validation"]})
    summaries = []
    weights = {"C": [], "D": []}
    for model in report["models"]:
        kind = model["kind"]
        seed = model["training_seed"]
        policy = load_policy(root / model["model_path"])
        assert hashlib.sha256((root / model["model_path"]).read_bytes()).hexdigest() == model["model_file_sha256"]
        selected = policy.metadata["checkpoint_episode"]
        best = max(model["checkpoint_scores"], key=lambda value: value["mean_improvement_fraction"])
        assert best["checkpoint"] == selected
        initial = np.random.default_rng(seed + (1000000 if kind == "D" else 0)).normal(0, 0.01, 12)
        weight_change = float(np.linalg.norm(policy.weights - initial))
        assert weight_change > 0
        weights[kind].append(policy.weights)
        uniform = report["uniform_validation_baseline"][kind]["mean_improvement_fraction"]
        summaries.append({"training_seed": seed, "kind": kind,
                          "selected_checkpoint": selected,
                          "selected_validation_improvement_percent": best["mean_improvement_fraction"] * 100,
                          "uniform_validation_improvement_percent": uniform * 100,
                          "difference_from_uniform_percentage_points": (best["mean_improvement_fraction"] - uniform) * 100,
                          "weight_change_L2_from_initialization": weight_change,
                          "model_file_sha256": model["model_file_sha256"]})
        for stage in ["training", "validation"]:
            log = root / f"seed_{seed}" / f"{stage}_{kind}.jsonl"
            for line in log.read_text(encoding="utf-8").splitlines():
                episode = json.loads(line)
                checks[f"{stage}_episodes"] += 1
                before = episode["initial_length"]
                total = 0
                for step in episode["steps"]:
                    assert step["exact_length_before"] == before
                    assert before - step["exact_length_after"] == step["exact_gain"]
                    assert (step["parent_exact_length_before"] - step["child_exact_length"]
                            == step["parent_gain"])
                    assert step["exact_gain"] >= 0
                    assert abs(step["reward"] * episode["reward_scale_initial_mean_edge"]
                               - step["exact_gain"]) < 1e-7
                    if step["exact_gain"] > 0:
                        checks["exact_global_reward_steps"] += 1
                    if step["parent_gain"] > 0 and step["exact_gain"] == 0:
                        checks["parent_only_gain_steps"] += 1
                    before = step["exact_length_after"]
                    total += step["exact_gain"]
                    checks["steps"] += 1
                assert before == episode["final_length"]
                assert total == episode["total_exact_gain"]
                assert total == episode["initial_length"] - episode["final_length"]
                if stage == "training" and episode["gradient_norm_before_clip"] > 0:
                    checks["nonzero_gradient_episodes"] += 1
                if stage == "training":
                    ret = 0.0
                    for step in reversed(episode["steps"]):
                        ret = step["reward"] + report["config"]["gamma"] * ret
                        assert abs(ret - step["discounted_return"]) < 1e-9
                        assert abs(step["advantage"] - (ret - step["baseline_before_update"])) < 1e-9
    differences = {}
    for kind, values in weights.items():
        pairs = [float(np.linalg.norm(a - b)) for a, b in itertools.combinations(values, 2)]
        assert min(pairs) > 0
        differences[kind] = {"model_count": len(values), "all_weights_distinct": True,
                             "minimum_pairwise_L2": min(pairs), "maximum_pairwise_L2": max(pairs)}
    value = {"passed": True, "no_formal_test_data_read": True,
             "training_started_utc": report["started_utc"],
             "training_completed_utc": report["completed_utc"],
             "end_to_end_seconds": report["end_to_end_seconds"],
             "checks": checks, "model_differences": differences,
             "validation_summary": summaries}
    (root / "training_audit.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.model_dir), indent=2))
