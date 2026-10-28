"""Audit frozen TSP study results and compare A/B without changing raw files.

Usage: python analyze.py STUDY_DIR
Only parent-certified accepted.jsonl events may establish scored improvements.
Raw LKH tours are deliberately never read by this offline analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.dont_write_bytecode = True
METHODS = ("A", "B")
LABELS = {
    "A": "贪心 → LKH",
    "B": "贪心 → 分组 Or-opt → LKH",
}
STRATA = ("synthetic_uniform", "synthetic_clustered", "tsplib")
BOOTSTRAP_SEED = 202610182399
BOOTSTRAP_REPLICATES = 20000
ENGINEERING_TOLERANCE_PERCENT = .01
SOURCE_NAMES = {"run.py", "worker.py", "supervisor.py", "resource_v3.py",
                "group_search.py", "test_group_search.py", "prepare_inputs.py"}


def stratum(metadata):
    """Keep the three preregistered strata separate during map resampling."""
    value = metadata.get("stratum", metadata.get("family"))
    return {"uniform": "synthetic_uniform", "clustered": "synthetic_clustered",
            "tsplib_public": "tsplib"}.get(value, value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def route_hash(route):
    payload = json.dumps(route, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def euc_2d_length(coords, route):
    """Exact nearest-integer Euclidean cost for integer coordinates, no solver import."""
    n = len(coords)
    if not isinstance(route, list) or any(type(v) is not int for v in route):
        raise ValueError("route must contain integer city IDs without coercion")
    if len(route) != n or set(route) != set(range(n)):
        raise ValueError("route is not a Hamiltonian permutation")
    total = 0
    for index, a in enumerate(route):
        b = route[(index + 1) % n]
        dx = coords[a][0] - coords[b][0]
        dy = coords[a][1] - coords[b][1]
        squared = dx * dx + dy * dy
        lower = math.isqrt(squared)
        # squared is integer; sqrt(squared) >= lower + .5 iff remainder > lower.
        total += lower + int(squared - lower * lower > lower)
    return total


def nearest_neighbor(coords, seed):
    """Independent paid-in-study initializer reconstruction; offline work is unscored."""
    import numpy as np

    points = np.asarray(coords, dtype=np.int64)
    used = np.zeros(len(coords), dtype=bool)
    route = []
    city = random.Random(seed).randrange(len(coords))
    for position in range(len(coords)):
        route.append(city)
        used[city] = True
        if position + 1 < len(coords):
            delta = points - points[city]
            costs = np.floor(np.sqrt(np.sum(delta * delta, axis=1)) + .5)
            costs[used] = np.inf
            city = int(np.argmin(costs))
    return route


def mean(values):
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return statistics.fmean(clean) if clean else None


def summary(values):
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return dict(count=len(clean), mean=mean(clean), minimum=min(clean) if clean else None,
                maximum=max(clean) if clean else None,
                sample_stdev=statistics.stdev(clean) if len(clean) > 1 else None)


class Auditor:
    def __init__(self, study):
        self.study = study
        self.errors = []
        self.warnings = []
        self.read_hashes = {}
        self.checked_route_counts = Counter()

    def error(self, context, message):
        self.errors.append(dict(context=str(context), message=str(message)))

    def require(self, condition, context, message):
        if not condition:
            self.error(context, message)
        return bool(condition)

    def read(self, path, *, optional=False, lines=False):
        path = Path(path)
        if not path.is_file():
            if not optional:
                self.error(path, "required file missing")
            return None
        try:
            self.read_hashes[str(path.relative_to(self.study))] = sha(path)
            text = path.read_text(encoding="utf-8-sig")
            if lines:
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            return json.loads(text)
        except (OSError, ValueError) as exc:
            self.error(path, f"cannot read JSON: {exc}")
            return None

    def route(self, coords, record, context, kind):
        try:
            value = euc_2d_length(coords, record.get("route"))
            self.checked_route_counts[kind] += 1
            self.require(type(record.get("length")) is int and record["length"] == value,
                         context, f"reported length {record.get('length')} differs from {value}")
            return record["route"], value
        except (ValueError, TypeError, KeyError) as exc:
            self.error(context, f"invalid tour: {exc}")
            return None, None


def _same_scored_record(audit, actual, expected, context):
    audit.require(actual.get("route") == expected["route"], context,
                  "route differs from reconstruction of parent-certified accepted events")
    audit.require(actual.get("length") == expected["length"], context,
                  "length differs from reconstruction of parent-certified accepted events")
    audit.require(actual.get("source") == expected["source"], context,
                  "source differs from reconstructed incumbent")


def audit_row(audit, row, metadata, coords, protocol, nn):
    job = row.get("job")
    context = f"job {job} {row.get('map_id')}/{row.get('seed')}/{row.get('method')}"
    directory = audit.study / f"j{job:03d}"
    task = audit.read(directory / "task.json") or {}
    result = audit.read(directory / "result.json") or {}
    accepted = audit.read(directory / "accepted.jsonl", lines=True) or []
    worker = audit.read(directory / "worker_result.json", optional=True)
    phase = audit.read(directory / "lkh" / "phase.json", optional=True)
    audit.require({k: v for k, v in row.items() if k not in ("job", "map_id", "family", "stratum")} == result,
                  context, "row differs from original per-job result.json")
    audit.require(row.get("map_id") == metadata.get("id"), context, "map identity mismatch")
    audit.require(row.get("n") == len(coords), context, "dimension mismatch")
    for key in ("method", "seed", "cold_started", "deadline"):
        audit.require(task.get(key) == row.get(key), context, f"task/result {key} mismatch")
    audit.require(task.get("coords") == coords, context, "task coordinates differ from frozen input")
    audit.require(task.get("settings") == protocol.get("settings"), context, "task settings differ from protocol")
    audit.require(row.get("settings") == protocol.get("settings"), context, "result settings differ from protocol")
    audit.require(row.get("family") == metadata.get("family"), context, "map family mismatch")
    audit.require(row.get("stratum") == stratum(metadata), context, "map stratum mismatch")
    audit.require(row.get("worker_result") == worker, context, "embedded worker ledger differs from file")
    audit.require(row.get("missing_worker_cost_ledger") == (worker is None), context,
                  "missing worker ledger flag inconsistent")
    if worker is None:
        audit.warnings.append(dict(context=context, message="worker ledger absent; no timing costs invented"))
        worker = {}

    origin, deadline = row.get("cold_started"), row.get("deadline")
    budget = protocol.get("total_budget_seconds")
    valid_clock = (type(origin) in (int, float) and type(deadline) in (int, float)
                   and math.isfinite(origin) and math.isfinite(deadline))
    audit.require(valid_clock, context, "missing or nonfinite clock")
    if not valid_clock:
        return None
    audit.require(math.isclose(deadline - origin, budget, abs_tol=1e-7, rel_tol=0), context,
                  "deadline minus origin differs from total budget")
    audit.require(row.get("budget_seconds") == budget == task.get("budget"), context,
                  "budget differs from frozen protocol")
    initial = list(range(len(coords)))
    initial_length = euc_2d_length(coords, initial)
    initial_at = origin + row.get("initial_validation_cold_seconds", math.inf)
    audit.require(origin <= initial_at < deadline, context, "initial validation outside cold budget")
    initial_record = dict(route=initial, length=initial_length, source="initial_identity",
                          validated_at=initial_at)
    audit.require(task.get("initial_route") == initial, context, "unexpected free initial route")
    audit.route(coords, row, context + " final", "final")
    previous = dict(initial_record)
    previous_at = initial_at
    certified_events = []
    for index, event in enumerate(accepted, 1):
        event_context = f"{context} accepted[{index}]"
        route, cost = audit.route(coords, event, event_context, "accepted")
        stamp = event.get("validated_at")
        valid_stamp = type(stamp) in (int, float) and math.isfinite(stamp)
        audit.require(valid_stamp, event_context, "missing/nonfinite certification timestamp")
        if not valid_stamp or route is None:
            continue
        audit.require(origin <= stamp < deadline, event_context, "certification not strictly before deadline")
        audit.require(stamp >= previous_at, event_context, "certification times decrease")
        audit.require(event.get("sequence") == index, event_context, "accepted sequence is not contiguous")
        audit.require(math.isclose(event.get("cold_seconds", math.inf), stamp - origin,
                                   abs_tol=1e-7, rel_tol=0), event_context, "cold timestamp mismatch")
        audit.require(cost < previous["length"], event_context, "accepted length does not strictly improve")
        if origin <= stamp < deadline and cost < previous["length"]:
            certified_events.append(event)
            previous = dict(event)
            previous_at = stamp
    audit.require(row.get("accepted_checkpoints") == len(accepted), context,
                  "accepted count differs from event ledger")
    audit.require(row.get("only_initial") == (len(accepted) == 0), context,
                  "only_initial inconsistent with accepted event ledger")
    _same_scored_record(audit, row, previous, context + " final reconstruction")
    marks = protocol.get("milestones_seconds", [])
    actual_milestones = row.get("milestones", {})
    audit.require(len(actual_milestones) == len(marks), context, "milestone count differs from protocol")
    milestone_values = {}
    for mark in marks:
        matches = [entry for entry in actual_milestones.values() if entry.get("seconds") == mark]
        if not audit.require(len(matches) == 1, context, f"missing/duplicated milestone {mark}"):
            continue
        actual = matches[0]
        expected = dict(initial_record)
        for event in certified_events:
            if event["validated_at"] < origin + mark and event["length"] < expected["length"]:
                expected = event
        audit.route(coords, actual, f"{context} milestone {mark}", "milestone")
        _same_scored_record(audit, actual, expected, f"{context} milestone {mark}")
        audit.require(actual.get("validated_at") == expected["validated_at"], context,
                      f"milestone {mark} certification differs from accepted ledger")
        audit.require(actual.get("validated_at", math.inf) < origin + mark, context,
                      f"milestone {mark} certification not strictly before mark")
        if "validation_cold_seconds" in actual:
            audit.require(math.isclose(actual["validation_cold_seconds"],
                                       expected["validated_at"] - origin,
                                       abs_tol=1e-7, rel_tol=0), context,
                          f"milestone {mark} cold certification timestamp mismatch")
        audit.require(actual.get("only_initial") == (expected["source"] == "initial_identity"),
                      context, f"milestone {mark} only_initial incorrect")
        milestone_values[str(mark)] = actual.get("length")
        if mark == budget:
            audit.require(actual.get("route") == row.get("route") and actual.get("length") == row.get("length"),
                          context, "final milestone and final route differ")

    expected_nn_route, expected_nn_length = nn
    expected_nn_start = random.Random(row["seed"]).randrange(len(coords))
    nn_events = [event for event in accepted if event.get("source") == "nearest_neighbor"]
    for event in nn_events:
        audit.require(event.get("route") == expected_nn_route, context, "NN checkpoint differs from reconstructed NN")
    if worker:
        audit.require(worker.get("nearest_neighbor_length") == expected_nn_length, context,
                      "worker NN length differs from reconstructed common initializer")
        audit.require(worker.get("nearest_neighbor_start_city") == expected_nn_start, context,
                      "worker NN start differs from paired seed")
        audit.require(worker.get("nearest_neighbor_route_sha256") == route_hash(expected_nn_route), context,
                      "worker NN route hash differs from independent initializer reconstruction")
        audit.require(worker.get("pre_lkh_length", math.inf) <= expected_nn_length, context,
                      "preprocessing made the NN route longer")
        audit.require(worker.get("method") == row.get("method"), context, "worker method differs")
        audit.require(worker.get("settings") == protocol.get("settings"), context, "worker settings differ")
        if phase is not None:
            audit.require(worker.get("lkh") == phase, context, "LKH phase file differs from worker ledger")

    for counter in ("invalid_checkpoints", "late_checkpoints", "observed_checkpoints"):
        audit.require(type(row.get(counter)) is int and row[counter] >= 0, context, f"invalid {counter} counter")
    audit.require(row.get("status") in ("completed", "budget_cutoff", "failed"), context, "unknown run status")
    audit.require(row.get("containment_cleanup_passed") is True, context, "process-tree cleanup not certified")
    accounting = row.get("job_final_accounting") or {}
    audit.require(accounting.get("active_processes") == 0, context, "active child processes after cleanup")
    lkh = worker.get("lkh") or phase or {}
    start = lkh.get("started_cold_seconds")
    guard = lkh.get("stop_guard_seconds", .20)
    available = max(0., budget - start - guard) if isinstance(start, (int, float)) else None
    if isinstance(start, (int, float)):
        audit.require(0 <= start <= budget, context, "LKH start outside cold budget")
    return dict(job=job, map_id=row["map_id"], family=row.get("family"),
                stratum=stratum(metadata), seed=row["seed"],
                method=row["method"], n=len(coords), length=row["length"],
                final_route_sha256=route_hash(row["route"]), milestones=milestone_values,
                status=row.get("status"), error=row.get("error"), source=row.get("source"),
                invalid_checkpoints=row.get("invalid_checkpoints"), late_checkpoints=row.get("late_checkpoints"),
                accepted_checkpoints=len(accepted), only_initial=row.get("only_initial"),
                cleanup_passed=row.get("containment_cleanup_passed"),
                cleanup_overrun_seconds=row.get("cleanup_overrun_seconds"),
                total_elapsed_including_cleanup_seconds=row.get("elapsed_including_cleanup_seconds"),
                nn_start_city=expected_nn_start, nn_length=expected_nn_length,
                nn_route_sha256=route_hash(expected_nn_route),
                nn_worker_length=worker.get("nearest_neighbor_length"),
                nn_worker_start_city=worker.get("nearest_neighbor_start_city"),
                nn_worker_route_sha256=worker.get("nearest_neighbor_route_sha256"),
                nn_parent_checkpoint_seen=bool(nn_events),
                nearest_neighbor_seconds=worker.get("nearest_neighbor_seconds"),
                group_preprocessing_seconds=worker.get("group_preprocessing_seconds"),
                pre_lkh_cold_seconds=worker.get("pre_lkh_cold_seconds"),
                preprocessing_target_overrun_seconds=worker.get("preprocessing_target_overrun_seconds"),
                pre_lkh_length=worker.get("pre_lkh_length"),
                lkh_preparation_seconds=lkh.get("preparation_seconds"),
                lkh_start_cold_seconds=start, lkh_available_seconds=available,
                lkh_elapsed_seconds=lkh.get("elapsed_seconds"),
                grouped_search=worker.get("grouped_search"),
                worker_ledger_present=bool(worker),
                parent_plus_job_cpu_seconds=row.get("parent_plus_job_cpu_seconds"),
                sampled_peak_rss_bytes=row.get("paired_peak_rss_bytes"))


def percentile(ordered, probability):
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def interval_interpretation(lower, upper):
    """Percent units; a CI spanning zero is insufficient evidence, not no effect."""
    tolerance = ENGINEERING_TOLERANCE_PERCENT
    if lower is None or upper is None:
        return dict(code="unavailable", text="完整分层配对数据不足，无法解释区间。",
                    empirical_interval_within_engineering_tolerance=False)
    if upper < -tolerance:
        code, text = "improvement_exceeds_threshold", "平均改善有证据，且整个区间低于 -0.01%，超过预设工程门槛。"
    elif upper < 0:
        code, text = "mean_improvement", "平均改善有证据，但尚未证明改善超过 0.01% 工程门槛。"
    elif lower > tolerance:
        code, text = "degradation_exceeds_threshold", "平均变差有证据，且整个区间高于 +0.01%，超过预设工程门槛。"
    elif lower > 0:
        code, text = "mean_degradation", "平均变差有证据，但尚未证明变差超过 0.01% 工程门槛。"
    else:
        code, text = "insufficient_evidence", "区间包含零，平均改善方向证据不足；这不等于证明无差异。"
    within = lower >= -tolerance and upper <= tolerance
    return dict(code=code, text=text,
                empirical_interval_within_engineering_tolerance=within,
                equivalence_description=("经验区间全部位于 ±0.01% 内，可描述为工程尺度上近似等效；未执行正式 TOST 等效性检验。"
                                         if within else None))


def stratified_map_bootstrap(map_rows, value_key="mean_relative_difference_percent_vs_A"):
    """Resample whole map means within each registered stratum, never seed rows."""
    groups = {name: [r[value_key] for r in map_rows
                     if r.get("stratum") == name and r.get(value_key) is not None]
              for name in STRATA}
    counts = {name: len(values) for name, values in groups.items()}
    valid = all(values for values in groups.values())
    details = dict(method="stratified percentile bootstrap of equal-weight map means",
                   sampling_unit="map; each map value is the mean of paired seed-relative differences",
                   strata=list(STRATA), maps_per_stratum=counts,
                   seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                   confidence_level=.95, seeds_resampled_as_independent_maps=False,
                   strata_weights="observed map counts / total map count; formal design is 4/12 each",
                   percentile_interpolation="linear at (replicates - 1) * probability")
    if not valid:
        return dict(**details, available=False, mean=None, lower=None, upper=None,
                    interpretation=interval_interpretation(None, None))
    generator = random.Random(BOOTSTRAP_SEED)
    count = sum(counts.values())
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.
        for values in groups.values():
            total += sum(generator.choices(values, k=len(values)))
        draws.append(total / count)
    draws.sort()
    lower, upper = percentile(draws, .025), percentile(draws, .975)
    return dict(**details, available=True, mean=mean([r[value_key] for r in map_rows]),
                lower=lower, upper=upper,
                interpretation=interval_interpretation(lower, upper))


def aggregate(records):
    paired = []
    indexed = defaultdict(dict)
    for record in records:
        indexed[(record["map_id"], record["seed"])][record["method"]] = record
    for (map_id, seed), methods in sorted(indexed.items()):
        if not all(method in methods for method in METHODS):
            continue
        baseline, current = methods["A"], methods["B"]
        delta = current["length"] - baseline["length"]
        milestone_differences = {}
        for mark, length in current["milestones"].items():
            if mark in baseline["milestones"]:
                initial = baseline["milestones"][mark]
                milestone_differences[mark] = dict(delta_length=length-initial,
                    relative_difference_percent=100.*(length-initial)/initial)
        paired.append(dict(map_id=map_id, stratum=current["stratum"], seed=seed, method="B",
            baseline_A_length=baseline["length"], length=current["length"], delta_length=delta,
            relative_difference_percent=100.*delta/baseline["length"],
            result="win" if delta < 0 else "tie" if delta == 0 else "loss",
            milestones=milestone_differences,
            includes_failed_run=any(methods[m]["status"] == "failed" for m in METHODS)))
    by_map = []
    for map_id in sorted({r["map_id"] for r in records}):
        selected_map = [r for r in records if r["map_id"] == map_id]
        pairs = [p for p in paired if p["map_id"] == map_id]
        differences = [p["relative_difference_percent"] for p in pairs]
        map_mean = mean(differences)
        outcome = "win" if map_mean is not None and map_mean < 0 else "loss" if map_mean is not None and map_mean > 0 else "tie" if map_mean == 0 else "missing"
        item = dict(map_id=map_id, stratum=selected_map[0]["stratum"], n=selected_map[0]["n"],
                    paired_seeds=len(pairs), seed_ids=[p["seed"] for p in pairs],
                    mean_relative_difference_percent_vs_A=map_mean,
                    median_relative_difference_percent_vs_A=statistics.median(differences) if differences else None,
                    map_result=outcome,
                    seed_outcomes=dict(Counter(p["result"] for p in pairs)), methods={})
        item["milestones"] = {}
        for mark in sorted({m for p in pairs for m in p["milestones"]}, key=float):
            values = [p["milestones"][mark]["relative_difference_percent"] for p in pairs if mark in p["milestones"]]
            item["milestones"][mark] = dict(paired_seeds=len(values),
                mean_relative_difference_percent_vs_A=mean(values),
                median_relative_difference_percent_vs_A=statistics.median(values) if values else None)
        for method in METHODS:
            selected = [r for r in selected_map if r["method"] == method]
            if not selected:
                continue
            item["methods"][method] = dict(runs=len(selected), seeds=[r["seed"] for r in selected],
                n=selected[0]["n"], family=selected[0]["family"], stratum=selected[0]["stratum"],
                final_length=summary([r["length"] for r in selected]),
                mean_relative_difference_percent_vs_A=map_mean if method == "B" else 0.,
                mean_pre_lkh_cold_seconds=mean([r["pre_lkh_cold_seconds"] for r in selected]),
                mean_group_preprocessing_seconds=mean([r["group_preprocessing_seconds"] for r in selected]),
                mean_lkh_available_seconds=mean([r["lkh_available_seconds"] for r in selected]))
        by_map.append(item)
    overall = {}
    for method in METHODS:
        selected = [r for r in records if r["method"] == method]
        outcomes = Counter(p["result"] for p in paired) if method == "B" else Counter()
        differences = [p["relative_difference_percent"] for p in paired] if method == "B" else []
        map_means = [r["mean_relative_difference_percent_vs_A"] for r in by_map
                     if r["mean_relative_difference_percent_vs_A"] is not None]
        map_outcomes = Counter(r["map_result"] for r in by_map) if method == "B" else Counter()
        overall[method] = dict(runs=len(selected), paired_runs=len(paired) if method == "B" else 0,
            mean_relative_difference_percent_vs_A=mean(differences) if method == "B" else 0.,
            equal_map_weight_mean_relative_difference_percent_vs_A=mean(map_means) if method == "B" else 0.,
            median_paired_relative_difference_percent=statistics.median(differences) if differences else None,
            median_map_mean_relative_difference_percent=statistics.median(map_means) if method == "B" and map_means else None,
            wins=outcomes["win"], ties=outcomes["tie"], losses=outcomes["loss"],
            map_wins=map_outcomes["win"], map_ties=map_outcomes["tie"], map_losses=map_outcomes["loss"],
            mean_pre_lkh_cold_seconds=mean([r["pre_lkh_cold_seconds"] for r in selected]),
            mean_nearest_neighbor_seconds=mean([r["nearest_neighbor_seconds"] for r in selected]),
            mean_group_preprocessing_seconds=mean([r["group_preprocessing_seconds"] for r in selected]),
            mean_preprocessing_target_overrun_seconds=mean([r["preprocessing_target_overrun_seconds"] for r in selected]),
            mean_lkh_available_seconds=mean([r["lkh_available_seconds"] for r in selected]),
            status_counts=dict(Counter(r["status"] for r in selected)),
            invalid_checkpoints=sum(r["invalid_checkpoints"] or 0 for r in selected),
            late_checkpoints=sum(r["late_checkpoints"] or 0 for r in selected),
            only_initial_runs=sum(bool(r["only_initial"]) for r in selected),
            missing_worker_ledgers=sum(not r["worker_ledger_present"] for r in selected),
            cleanup_failures=sum(r["cleanup_passed"] is not True for r in selected),
            cleanup_overrun_seconds=summary([r["cleanup_overrun_seconds"] for r in selected]))
    primary = stratified_map_bootstrap(by_map)
    secondary = {}
    for mark in sorted({m for r in by_map for m in r["milestones"]}, key=float):
        if float(mark) != 10.:
            continue
        mark_maps = [dict(map_id=r["map_id"], stratum=r["stratum"],
                         mean_relative_difference_percent_vs_A=r["milestones"][mark]["mean_relative_difference_percent_vs_A"])
                     for r in by_map if mark in r["milestones"]]
        secondary[mark] = dict(purpose="secondary diagnostic only; never early stopping or parameter selection",
                              bootstrap=stratified_map_bootstrap(mark_maps), by_map=mark_maps)
    by_stratum = {}
    for name in STRATA:
        selected = [r for r in by_map if r["stratum"] == name]
        values = [r["mean_relative_difference_percent_vs_A"] for r in selected
                  if r["mean_relative_difference_percent_vs_A"] is not None]
        by_stratum[name] = dict(maps=len(selected), equal_map_weight_mean_relative_difference_percent=mean(values),
            median_map_mean_relative_difference_percent=statistics.median(values) if values else None,
            map_outcomes=dict(Counter(r["map_result"] for r in selected)))
    return dict(by_map=by_map, paired=paired, overall=overall, by_stratum=by_stratum,
                primary=dict(endpoint="certified final route strictly before total-budget deadline",
                             estimator="mean across maps of each map's mean paired-seed relative difference",
                             engineering_tolerance_percent=ENGINEERING_TOLERANCE_PERCENT,
                             bootstrap=primary), secondary_milestones=secondary,
                relative_difference_definition="100 * (B_length - A_length) / A_length; negative is shorter",
                units="original instance coordinates, integer-rounded EUC_2D, closed tours",
                all_failed_and_initial_only_runs_retained=True)


def fmt(value, digits=6):
    return "缺失" if value is None else f"{value:.{digits}f}"


def report_text(study, protocol, audit, results):
    design = results["design"]
    primary = results["primary"]["bootstrap"]
    overall = results["overall"]["B"]
    lines = ["轻量分组 Or-opt：新实例、配对种子、相同总预算复验", "",
             f"原始目录：{study}",
             f"每次冷启动总预算：{protocol.get('total_budget_seconds')} 秒；方法 A、B。",
             "A：贪心 → LKH；B：贪心 → 分组 Or-opt → LKH。",
             f"实例 {design['maps']} 张，配对种子 {len(protocol.get('seeds', []))} 个，完整配对 {len(results['paired'])} 对，独立运行 {len(results['records'])} 次。",
             "距离使用各实例原坐标的 EUC_2D 整数单位，闭环包含最后一城返回首城的边。",
             "相对差 = 100 × (B 长度 − A 长度) / A 长度；负值更短，不是距最优解的差距。", "",
             f"审计：{'通过' if not audit.errors else '存在错误，下面比较不可视为有效结论'}；错误 {len(audit.errors)} 项，提示 {len(audit.warnings)} 项。",
             f"独立重算路线数：{dict(audit.checked_route_counts)}。",
             "最终成绩和 10 秒里程碑均从严格截止前完成父进程独立认证的 accepted.jsonl 重建；没有读取原始 best.tour 给分。", "",
             "主要终点：30 秒总预算内最终长度（实际预算以冻结 protocol 为准）。",
             "先算每图各配对种子的相对差均值，再对地图等权平均；种子不是独立地图。",
             f"地图等权相对差均值：{fmt(primary['mean'])}%；95% 分层地图 bootstrap 区间 [{fmt(primary['lower'])}%, {fmt(primary['upper'])}%]。",
             f"地图均值中位数：{fmt(overall['median_map_mean_relative_difference_percent'])}%；全部配对差中位数：{fmt(overall['median_paired_relative_difference_percent'])}%。",
             f"图级胜/平/负：{overall['map_wins']}/{overall['map_ties']}/{overall['map_losses']}（按每图配对相对差均值的符号）。",
             f"种子配对胜/平/负：{overall['wins']}/{overall['ties']}/{overall['losses']}。",
             f"预设工程容差 ±{ENGINEERING_TOLERANCE_PERCENT:.2f}%；{primary['interpretation']['text']}"]
    if primary['interpretation']['equivalence_description']:
        lines.append(primary['interpretation']['equivalence_description'])
    lines.extend([f"Bootstrap：{BOOTSTRAP_REPLICATES} 次，固定随机种子 {BOOTSTRAP_SEED}；在三个层内分别重采样整张地图，并保留该图的全部种子均值。",
                  f"各层图数：{primary['maps_per_stratum']}；正式设计为均匀、聚簇、公开 TSPLIB 各四张。",
                  "该区间刻画此地图样本下的地图间变异；不构成对任意 TSP 分布的保证。", "",
                  "各图 A/B 最终长度均值、配对相对差均值及图级结果："])
    for item in results["by_map"]:
        parts = [f"{m}={fmt(item['methods'].get(m, {}).get('final_length', {}).get('mean'), 3)}" for m in METHODS]
        lines.append(f"{item['map_id']} [{item['stratum']}] n={item['n']}：" + "；".join(parts)
                     + f"；B−A={fmt(item['mean_relative_difference_percent_vs_A'])}%；中位数={fmt(item['median_relative_difference_percent_vs_A'])}%；{item['map_result']}。")
    lines.extend(["", "各层描述统计（每张地图等权）："])
    for name, item in results["by_stratum"].items():
        lines.append(f"{name}：{item['maps']} 图，均值 {fmt(item['equal_map_weight_mean_relative_difference_percent'])}%；图级结果 {item['map_outcomes']}。")
    lines.extend(["", "全部同图同种子配对差（不删除失败或仅保底运行）："])
    for item in results["paired"]:
        lines.append(f"{item['map_id']} seed={item['seed']}：A={item['baseline_A_length']}，B={item['length']}，"
                     f"长度差={item['delta_length']}，相对差={fmt(item['relative_difference_percent'])}%，{item['result']}。")
    lines.extend(["", "10 秒次要终点（仅诊断，不用于早停、改参数或代替主要终点）："])
    if not results["secondary_milestones"]:
        lines.append("冻结记录中没有完整 10 秒里程碑。")
    for mark, item in results["secondary_milestones"].items():
        value = item["bootstrap"]
        lines.append(f"{mark} 秒：地图等权相对差 {fmt(value['mean'])}%；95% 区间 [{fmt(value['lower'])}%, {fmt(value['upper'])}%]。")
    lines.extend(["", "每次运行平均耗时（秒，全部计入总预算）："])
    for method in METHODS:
        item = results["overall"][method]
        lines.append(f"{method}：NN {fmt(item['mean_nearest_neighbor_seconds'], 3)}；组内前处理 {fmt(item['mean_group_preprocessing_seconds'], 3)}；"
                     f"冷启动至 LKH 准备前 {fmt(item['mean_pre_lkh_cold_seconds'], 3)}；LKH 可用预算 {fmt(item['mean_lkh_available_seconds'], 3)}；"
                     f"超过合作式前处理目标 {fmt(item['mean_preprocessing_target_overrun_seconds'], 3)}。")
    lines.extend(["LKH 可用预算 = 总预算 − LKH 开始时的冷启动累计时间 − 0.20 秒统一停止余量。",
                  "进程、导入、NN、矩阵、分组、Numba JIT 和局部优化均付费；每次独立空编译缓存。",
                  "B 使用冻结的 25% 合作式前处理目标；矩阵、导入和 JIT 可能超目标，实际时间全部扣除。",
                  "输入复制与离线审计在预算外；操作系统文件缓存未清空；截止后的清理不能产生计分解。", "",
                  "运行状态与异常："])
    for method in METHODS:
        item = results["overall"][method]
        lines.append(f"{method}：状态 {item['status_counts']}；无效 checkpoint {item['invalid_checkpoints']}；晚于截止认证 {item['late_checkpoints']}；"
                     f"仅保底 {item['only_initial_runs']}；进程清理失败 {item['cleanup_failures']}；缺失 worker 账本 {item['missing_worker_ledgers']}。")
    lines.extend(["", "同图同种子独立重建 NN 并核对起点、长度、SHA256。七个冻结源文件和输入均检查 hash。",
                  "audit_handoff.py 另行独立检查 LKH 输入路线、EUC_2D 问题、共同参数、预算及组内 gain 账本。",
                  "分析脚本 hash 单独记录；运行中不依赖分析脚本，没有执行量子计算。",
                  "结果仅比较本配置、本预算和注册实例；若方向证据不足，不应解释为算法无效或等效证明。",
                  "原始文件未修改；results.json 包含全部种子差、图级统计、区间和路线 hash；audit.json 包含错误与读取文件 hash。"])
    if audit.errors:
        lines.extend(["", "审计错误："] + [f"- {e['context']}：{e['message']}" for e in audit.errors])
    if audit.warnings:
        lines.extend(["", "审计提示："] + [f"- {e['context']}：{e['message']}" for e in audit.warnings])
    return "\n".join(lines) + "\n"


def make_plot(path, results, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180, layout="constrained",
                             gridspec_kw={"width_ratios": [2.1, 1]})
    colors = {"synthetic_uniform": "#1b88a3", "synthetic_clustered": "#cf7935", "tsplib": "#5b6498"}
    maps = results["by_map"]
    for index, item in enumerate(maps):
        value = item["mean_relative_difference_percent_vs_A"]
        pairs = [p for p in results["paired"] if p["map_id"] == item["map_id"]]
        color = colors.get(item["stratum"], "#596675")
        axes[0].bar(index, value if value is not None else np.nan, .65, color=color, alpha=.8)
        jitter = np.linspace(-.12, .12, len(pairs)) if pairs else []
        axes[0].scatter(index + jitter, [p["relative_difference_percent"] for p in pairs],
                        s=17, color="#192734", zorder=4)
    axes[0].axhline(0, color="#263442", linewidth=.8)
    for limit in (-ENGINEERING_TOLERANCE_PERCENT, ENGINEERING_TOLERANCE_PERCENT):
        axes[0].axhline(limit, color="#7a8491", linestyle="--", linewidth=.8)
    axes[0].set_xticks(range(len(maps)), [r["map_id"] for r in maps], rotation=60, ha="right")
    axes[0].set_ylabel("Final relative difference: B versus A (%)")
    axes[0].set_title("Bars: map means; dots: paired seeds; negative is shorter\nDashed: preregistered engineering tolerance +/-0.01%")
    primary = results["primary"]["bootstrap"]
    if primary["available"]:
        point, lower, upper = primary["mean"], primary["lower"], primary["upper"]
        axes[1].errorbar([point], [0], xerr=[[max(0., point-lower)], [max(0., upper-point)]],
                         fmt="o", color="#1b88a3", capsize=6, markersize=8)
        axes[1].set_yticks([0], ["Equal-map mean"])
        axes[1].axvspan(-.01, .01, color="#d5e3d0", alpha=.6)
        axes[1].axvline(0, color="#263442", linewidth=.8)
        axes[1].set_ylim(-.7, .7)
        axes[1].set_xlabel("B versus A (%)")
        axes[1].set_title("95% stratified map bootstrap interval\n20,000 resamples; seeds stay within maps")
        axes[1].text(.02, .05, f"Mean {point:.6f}%\n95% CI [{lower:.6f}%, {upper:.6f}%]",
                     transform=axes[1].transAxes, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"TSP replication: NN + LKH (A) versus grouped Or-opt + LKH (B); {protocol.get('total_budget_seconds')} s total", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(study):
    study = study.resolve()
    if not study.is_dir():
        raise ValueError("study directory does not exist")
    output = study / "analysis"
    if output.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = study / ("analysis-" + suffix)
    output.mkdir(exist_ok=False)
    audit = Auditor(study)
    protocol = audit.read(study / "frozen_protocol.json") or {}
    manifest = audit.read(study / "source_hashes.json") or {}
    audit.require(bool(manifest), "source_hashes", "source manifest is empty")
    audit.require(set(manifest) == SOURCE_NAMES, "source_hashes", "unexpected frozen source manifest")
    audit.require(protocol.get("methods") == list(METHODS), "protocol", "registered methods must be A/B")
    datasets = audit.read(study / "datasets.json") or {}
    schedule = audit.read(study / "schedule.json") or []
    complete = audit.read(study / "complete.json")
    environment = audit.read(study / "environment.json") or {}
    inheritance = audit.read(study / "engine_inheritance.json") or {}
    inherited = inheritance.get("byte_identical_engine_hashes", {})
    engine_names = SOURCE_NAMES - {"run.py", "prepare_inputs.py"}
    audit.require(set(inherited) == engine_names, "engine_inheritance", "inherited engine file set differs")
    audit.require(inheritance.get("settings_unchanged") is True, "engine_inheritance", "unchanged settings not certified")
    for filename, expected in inherited.items():
        audit.require(manifest.get(filename) == expected, "engine_inheritance", f"{filename} differs from inherited hash")
        previous_path = Path(inheritance.get("previous_source", "")) / filename
        audit.require(previous_path.is_file() and sha(previous_path) == expected,
                      "engine_inheritance", f"{filename} differs from prior experiment source")
    audit.require(environment.get("quantum_execution") is False, "environment", "quantum execution flag unexpected")
    audit.require(protocol.get("quantum_execution") is False, "protocol", "quantum execution flag unexpected")
    audit.require(datasets.get("registered_before_first_formal_job") is True, "datasets", "input registration flag missing")
    for filename, expected in manifest.items():
        path = study / "source" / filename
        if path.is_file():
            actual = sha(path)
            audit.read_hashes[str(path.relative_to(study))] = actual
            audit.require(actual == expected, path, "frozen source hash mismatch")
        else:
            audit.error(path, "frozen source file missing")
    if complete:
        audit.require(complete.get("source_hashes_unchanged") is True, "complete", "source hash stability not certified")
        audit.require(complete.get("jobs") == len(schedule), "complete", "completed count differs from schedule")
    maps = {}
    for metadata in datasets.get("maps", []):
        path = study / "inputs" / (metadata["id"] + ".json")
        contents = audit.read(path)
        if not contents:
            continue
        audit.require(sha(path) == metadata.get("sha256"), path, "frozen input hash mismatch")
        coords = contents.get("coords")
        valid = (isinstance(coords, list) and len(coords) >= 3
                 and all(isinstance(point, list) and len(point) == 2
                         and all(type(v) is int for v in point) for point in coords))
        if not audit.require(valid, path, "coordinates must be pairs of integers"):
            continue
        audit.require(metadata.get("n") == len(coords), path, "input dimension mismatch")
        maps[metadata["id"]] = (metadata, coords)
    map_strata = Counter(stratum(metadata) for metadata, _ in maps.values())
    audit.require(dict(map_strata) == protocol.get("strata"), "datasets", "map strata differ from frozen protocol")
    audit.require(len(maps) == protocol.get("independent_maps"), "datasets", "map count differs from frozen protocol")
    rows = []
    for path in sorted(study.glob("row*.json")):
        row = audit.read(path)
        if row is not None:
            rows.append(row)
    audit.require(len(rows) == len(schedule), "schedule", "row count differs from planned schedule")
    audit.require(len(schedule) == protocol.get("planned_jobs"), "schedule", "planned job count differs from protocol")
    expected_blocks = {(map_id, seed, method) for map_id in maps
                       for seed in protocol.get("seeds", []) for method in METHODS}
    audit.require({(r.get("map_id"), r.get("seed"), r.get("method")) for r in schedule} == expected_blocks,
                  "schedule", "registered map/seed/method blocks differ from full A/B design")
    audit.require(len({r.get("job") for r in rows}) == len(rows), "rows", "duplicate job IDs")
    audit.require(len({(r.get("map_id"), r.get("seed"), r.get("method")) for r in rows}) == len(rows),
                  "rows", "duplicate map/seed/method rows")
    expected_schedule = {entry["job"]: entry for entry in schedule}
    records, nn_cache = [], {}
    for row in rows:
        try:
            planned = expected_schedule.get(row.get("job"), {})
            for key in ("map_id", "method", "seed"):
                audit.require(row.get(key) == planned.get(key), f"job {row.get('job')}", f"schedule {key} mismatch")
            if row.get("map_id") not in maps:
                audit.error(f"job {row.get('job')}", "unknown map")
                continue
            metadata, coords = maps[row["map_id"]]
            key = (row["map_id"], row["seed"])
            if key not in nn_cache:
                nn_route = nearest_neighbor(coords, row["seed"])
                nn_cache[key] = (nn_route, euc_2d_length(coords, nn_route))
            record = audit_row(audit, row, metadata, coords, protocol, nn_cache[key])
            if record is not None:
                records.append(record)
        except Exception as exc:
            audit.error(f"job {row.get('job')}", f"audit could not finish row: {type(exc).__name__}: {exc}")
    for map_id in maps:
        for seed in protocol.get("seeds", []):
            selected = [r for r in records if r["map_id"] == map_id and r["seed"] == seed]
            audit.require({r["method"] for r in selected} == set(METHODS), f"{map_id}/{seed}", "incomplete A/B pairing")
            observed = [r for r in selected if r["worker_ledger_present"]]
            audit.require(len({(r["nn_worker_start_city"], r["nn_worker_length"], r["nn_worker_route_sha256"]) for r in observed}) <= 1,
                          f"{map_id}/{seed}", "paired methods did not share identical NN start/length")
    results = aggregate(records)
    audit.require(len(results["paired"]) == protocol.get("paired_blocks"),
                  "pairs", "paired block count differs from frozen protocol")
    for item in results["by_map"]:
        audit.require(set(item["seed_ids"]) == set(protocol.get("seeds", [])),
                      item["map_id"], "map paired seed set differs from frozen protocol")
    results.update(records=records, audit_passed=not audit.errors,
                   design=dict(maps=len(maps), strata=dict(map_strata),
                               seeds=protocol.get("seeds", []),
                               methods=list(METHODS), source_file_count=len(manifest)),
                   study_directory=str(study), total_budget_seconds=protocol.get("total_budget_seconds"),
                   quantum_execution=False, development_smoke=protocol.get("development_smoke"),
                   scored_source="parent-certified accepted.jsonl; identity fallback; no raw LKH tour recovery",
                   nn_routes=[dict(map_id=key[0], seed=key[1], start_city=value[0][0],
                                   length=value[1], route_sha256=route_hash(value[0]))
                              for key, value in sorted(nn_cache.items())])
    plot_error = None
    try:
        make_plot(output / "comparison.png", results, protocol)
    except Exception as exc:
        plot_error = f"{type(exc).__name__}: {exc}"
        audit.warnings.append(dict(context="plot", message=plot_error))
    original_files_changed = False
    for name, expected in audit.read_hashes.items():
        path = study / name
        if not path.is_file() or sha(path) != expected:
            original_files_changed = True
            audit.error(name, "original input changed during offline analysis")
    results["audit_passed"] = not audit.errors
    audit_result = dict(schema=1, passed=not audit.errors, errors=audit.errors, warnings=audit.warnings,
        independently_recomputed_route_counts=dict(audit.checked_route_counts),
        inputs_and_sources_read_sha256=audit.read_hashes, scored_raw_LKH_tours_read=False,
        audited_rows=len(records), planned_jobs=len(schedule), plot_error=plot_error,
        original_files_changed=original_files_changed,
        analyzer_source_path=str(Path(__file__).resolve()), analyzer_source_sha256=sha(__file__),
        analyzed_at_utc=datetime.now(timezone.utc).isoformat())
    for name, value in (("audit.json", audit_result), ("results.json", results)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    (output / "中文结果说明.txt").write_text(report_text(study, protocol, audit, results), encoding="utf-8")
    return output, audit_result, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_directory", type=Path)
    args = parser.parse_args()
    output, audit, results = analyze(args.study_directory)
    print("ANALYSIS " + str(output), flush=True)
    print(json.dumps(dict(audit_passed=audit["passed"], errors=len(audit["errors"]),
                          warnings=len(audit["warnings"]), overall=results["overall"]), ensure_ascii=False), flush=True)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
