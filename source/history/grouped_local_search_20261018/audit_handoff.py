"""Independently audit a completed study's grouped-search -> LKH handoff.

Usage: python -B audit_handoff.py STUDY_DIR
Only a new handoff_audit.json is written, exclusively. No solver is imported
or launched, no best.tour is read, and no deadline score is reconstructed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
SOURCE_NAMES = {"run.py", "worker.py", "supervisor.py", "resource_v3.py",
                "group_search.py", "test_group_search.py"}
METHODS = ("A", "B", "C")
COMMON_PARAMETERS = {
    "INITIAL_TOUR_FRACTION": "1", "CANDIDATE_SET_TYPE": "ALPHA",
    "MAX_CANDIDATES": "5", "RUNS": "100000", "MAX_TRIALS": "100000",
    "STOP_AT_OPTIMUM": "NO", "TIME_LIMIT": "86400",
    "TOTAL_TIME_LIMIT": "86400", "TRACE_LEVEL": "1",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def route_hash(route):
    return hashlib.sha256(json.dumps(route, separators=(",", ":")).encode("ascii")).hexdigest()


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def close(a, b):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7))


def metrics(values):
    return dict(count=len(values), mean=statistics.fmean(values) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None, total=sum(values))


def integer_tour_cost(coords, route):
    if any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p)
           for p in coords):
        raise ValueError("coordinates must be pairs of integers")
    if (not isinstance(route, list) or any(type(v) is not int for v in route)
            or len(route) != len(coords) or set(route) != set(range(len(coords)))):
        raise ValueError("tour is not a complete Hamiltonian permutation")
    total = 0
    for index, city in enumerate(route):
        other = route[(index + 1) % len(route)]
        dx, dy = coords[city][0] - coords[other][0], coords[city][1] - coords[other][1]
        squared = dx * dx + dy * dy
        lower = math.isqrt(squared)
        total += lower + int(squared - lower * lower > lower)
    return total


def section_document(text, section):
    lines = text.splitlines()
    positions = [i for i, line in enumerate(lines) if line.strip() == section]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one {section}")
    cut = positions[0]
    header = {}
    for line in lines[:cut]:
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        if key in header:
            raise ValueError(f"duplicate header {key}")
        header[key] = value.strip()
    return header, lines[cut + 1:]


def parse_initial(text, n):
    header, lines = section_document(text, "TOUR_SECTION")
    if header.get("TYPE") != "TOUR" or int(header.get("DIMENSION", "-1")) != n:
        raise ValueError("incorrect initial-tour header")
    tokens = " ".join(lines).split()
    if "-1" not in tokens:
        raise ValueError("tour has no terminating -1")
    end = tokens.index("-1")
    if tokens[end + 1:] not in ([], ["EOF"]):
        raise ValueError("unexpected tokens after tour terminator")
    return [int(token) - 1 for token in tokens[:end]]


def parse_problem(text, n):
    header, lines = section_document(text, "NODE_COORD_SECTION")
    if (header.get("TYPE") != "TSP" or header.get("EDGE_WEIGHT_TYPE") != "EUC_2D"
            or int(header.get("DIMENSION", "-1")) != n):
        raise ValueError("incorrect problem header")
    coords, terminated = [], False
    for line in lines:
        if not line.strip():
            continue
        if line.strip() == "EOF":
            terminated = True
            continue
        if terminated:
            raise ValueError("data after problem EOF")
        city, x, y = [int(token) for token in line.split()]
        if city != len(coords) + 1:
            raise ValueError("problem city IDs are not in the declared order")
        coords.append([x, y])
    if len(coords) != n or not terminated:
        raise ValueError("incomplete problem coordinates")
    return coords


def parse_parameters(text):
    values = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        if key in values:
            raise ValueError(f"duplicate LKH parameter {key}")
        values[key] = value.strip()
    return values


class Audit:
    def __init__(self, study):
        self.study = study
        self.errors = []
        self.read_hashes = {}

    def require(self, condition, context, message):
        if not condition:
            self.errors.append(dict(context=str(context), message=message))
        return bool(condition)

    def text(self, path):
        path = Path(path)
        payload = path.read_bytes()
        self.read_hashes[str(path)] = hashlib.sha256(payload).hexdigest()
        return payload.decode("utf-8-sig")

    def json(self, path):
        return json.loads(self.text(path))


def expected_stages(method):
    operations = ("original_or",) if method == "B" else (
        "original_or", "original_three", "shifted_or", "shifted_three")
    return [f"{operation}_r{round_index}" for round_index in (1, 2) for operation in operations]


def audit_job(audit, schedule, protocol, maps, binary):
    job = schedule["job"]
    context = f"j{job:03d}"
    directory = audit.study / context
    row = audit.json(audit.study / f"row{job:03d}.json")
    task = audit.json(directory / "task.json")
    worker = audit.json(directory / "worker_result.json")
    result = audit.json(directory / "result.json")
    method, seed = schedule["method"], schedule["seed"]
    coords = maps[schedule["map_id"]]
    for record_name, record in (("row", row), ("task", task), ("worker", worker), ("result", result)):
        audit.require(record.get("method") == method, context, f"{record_name} method differs from schedule")
    audit.require(row.get("map_id") == schedule["map_id"] and row.get("seed") == seed,
                  context, "row block differs from schedule")
    audit.require(task.get("seed") == seed and task.get("coords") == coords,
                  context, "task seed/coordinates differ from frozen input")
    audit.require(task.get("settings") == protocol["settings"] == worker.get("settings"),
                  context, "settings differ from protocol")
    audit.require(row.get("worker_result") == worker == result.get("worker_result"),
                  context, "worker ledger differs between parent records and standalone file")
    audit.require(Path(task["lkh_binary"]).resolve() == binary, context, "task LKH executable differs")
    budget = task["budget"]
    audit.require(close(budget, protocol["total_budget_seconds"]), context, "budget differs from protocol")
    audit.require(close(task["deadline"] - task["cold_started"], budget), context, "task clock is inconsistent")

    lkh = directory / "lkh"
    initial = lkh / "initial.tour"
    route = parse_initial(audit.text(initial), len(coords))
    cost = integer_tour_cost(coords, route)
    audit.require(type(worker["pre_lkh_length"]) is int and cost == worker["pre_lkh_length"],
                  context, "independent initial-tour cost differs from pre_lkh_length")
    audit.require(parse_problem(audit.text(lkh / "problem.tsp"), len(coords)) == coords,
                  context, "LKH problem coordinates differ from task")
    parameters = parse_parameters(audit.text(lkh / "run.par"))
    expected_keys = set(COMMON_PARAMETERS) | {"PROBLEM_FILE", "OUTPUT_TOUR_FILE", "INITIAL_TOUR_FILE", "SEED"}
    audit.require(set(parameters) == expected_keys, context, "unexpected/missing LKH parameters (including forbidden INPUT_TOUR_FILE/OPTIMUM)")
    for key, value in COMMON_PARAMETERS.items():
        audit.require(parameters.get(key) == value, context, f"LKH {key} differs from common configuration")
    audit.require(parameters.get("SEED") == str(seed), context, "LKH seed differs")
    for key, path in (("PROBLEM_FILE", lkh / "problem.tsp"),
                      ("OUTPUT_TOUR_FILE", lkh / "best.tour"),
                      ("INITIAL_TOUR_FILE", initial)):
        audit.require(key in parameters and Path(parameters[key]).resolve() == path.resolve(),
                      context, f"LKH {key} points elsewhere")
    phase = audit.json(lkh / "phase.json")
    audit.require(worker["lkh"] == phase, context, "LKH phase ledger differs")
    audit.require(phase.get("initial_tour_file_sha256") == sha(initial), context, "initial-tour SHA256 differs")

    nn = worker["nearest_neighbor_length"]
    audit.require(type(nn) is int and nn > 0, context, "NN length is not a positive integer")
    gain = nn - cost
    audit.require(gain >= 0, context, "handoff route is longer than NN")
    if method == "A":
        audit.require(gain == 0, context, "A changes the NN route before LKH")
        audit.require(route_hash(route) == worker["nearest_neighbor_route_sha256"],
                      context, "A input route does not match its NN hash")
        group = {}
        details = []
    else:
        group = worker["grouped_search"]
        details = group["stage_details"]
        stages = expected_stages(method)
        reached = group["stages_reached"]
        audit.require(group.get("mode") == method, context, "grouped mode differs")
        audit.require(group.get("group_size") == protocol["settings"]["group_size"]
                      and group.get("candidate_count") == protocol["settings"]["candidate_count"],
                      context, "grouped settings differ")
        audit.require(reached == stages[:len(reached)] and len(reached) <= len(stages),
                      context, "grouped stage order differs from protocol")
        audit.require([d["stage"] for d in details] == reached, context, "stage detail/reached lists differ")
        for key in ("gain", "or_moves", "three_moves", "window_passes", "rounds_started"):
            audit.require(type(group.get(key)) is int and group[key] >= 0, context, f"invalid grouped {key}")
        audit.require(group["gain"] == gain == sum(d["gain"] for d in details),
                      context, "group gain != NN - pre_LKH != summed stage gains")
        audit.require(group["or_moves"] == sum(d["moves"] for d in details if "_or_" in d["stage"]),
                      context, "Or-opt move totals differ")
        audit.require(group["three_moves"] == sum(d["moves"] for d in details if "_three_" in d["stage"]),
                      context, "3-opt move totals differ")
        audit.require(group["window_passes"] == sum(d["windows"] for d in details),
                      context, "window totals differ")
        for detail in details:
            for key in ("gain", "moves", "windows"):
                audit.require(type(detail.get(key)) is int and detail[key] >= 0,
                              context, f"invalid {detail['stage']} {key}")
            audit.require(type(detail.get("completed")) is bool, context, "stage completion flag is not boolean")
            audit.require(finite_nonnegative(detail.get("elapsed")), context, "stage elapsed is invalid")
            audit.require((detail["moves"] == 0) == (detail["gain"] == 0), context, "stage moves/gain disagree")
            cap_key = "or_moves_per_window" if "_or_" in detail["stage"] else "three_moves_per_window"
            audit.require(detail["moves"] <= detail["windows"] * protocol["settings"][cap_key],
                          context, "stage exceeds registered move cap")
        audit.require(group.get("jit_compilation_included") is True, context, "JIT inclusion flag missing")
        audit.require(finite_nonnegative(group.get("elapsed")) and
                      group["elapsed"] <= worker["group_preprocessing_seconds"] + 1e-7,
                      context, "grouped elapsed exceeds whole preprocessing duration")
        audit.require(close(worker.get("local_deadline_cold_seconds"),
                            budget * protocol["settings"]["preprocessing_fraction_target"]),
                      context, "local target clock differs from protocol")
    for key in ("group_preprocessing_seconds", "pre_lkh_cold_seconds", "nearest_neighbor_seconds"):
        audit.require(finite_nonnegative(worker.get(key)), context, f"invalid {key}")
    kernels = group.get("first_kernel_call_seconds", {})
    audit.require(all(finite_nonnegative(v) for v in kernels.values()), context, "invalid first-call timing")
    audit.require(set(kernels) <= {"or_opt", "candidate_mask", "three_opt"}, context, "unknown kernel timing")
    if group.get("or_moves", 0):
        audit.require("or_opt" in kernels, context, "Or-opt first-call timing missing")
    if group.get("three_moves", 0):
        audit.require({"candidate_mask", "three_opt"} <= set(kernels), context, "3-opt first-call timing missing")
    target = budget * protocol["settings"]["preprocessing_fraction_target"]
    overrun = max(0., worker["pre_lkh_cold_seconds"] - target) if method in ("B", "C") else 0.
    audit.require(close(worker.get("preprocessing_target_overrun_seconds"), overrun),
                  context, "recorded preprocessing target overrun differs from recomputation")
    audit.require(finite_nonnegative(phase.get("preparation_seconds")) and
                  phase["started_cold_seconds"] >= worker["pre_lkh_cold_seconds"],
                  context, "LKH starts before recorded handoff")
    audit.require(close(phase.get("stop_guard_seconds"), .20), context, "LKH stop guard differs")
    stage_map = {d["stage"]: d for d in details}
    all_stages = expected_stages(method) if method != "A" else []
    return dict(job=job, map_id=schedule["map_id"], seed=seed, method=method,
                initial_tour_length=cost, initial_tour_sha256=sha(initial),
                nearest_neighbor_length=nn, nearest_neighbor_start_city=worker["nearest_neighbor_start_city"],
                nearest_neighbor_route_sha256=worker["nearest_neighbor_route_sha256"],
                pre_lkh_gain=gain, pre_lkh_improvement_percent=100. * gain / nn,
                group_preprocessing_seconds=worker["group_preprocessing_seconds"],
                grouped_kernel_and_search_seconds=group.get("elapsed", 0.),
                first_kernel_call_seconds=kernels, first_kernel_calls_sum_seconds=sum(kernels.values()),
                first_kernel_timings_include_jit_and_execution=True,
                pre_lkh_cold_seconds=worker["pre_lkh_cold_seconds"],
                preprocessing_target_cold_seconds=target if method != "A" else None,
                preprocessing_target_overrun_seconds=overrun,
                lkh_started_cold_seconds=phase["started_cold_seconds"],
                remaining_lkh_before_stop_guard_seconds=max(0., budget - .20 - phase["started_cold_seconds"]),
                lkh_stop_guard_return_overrun_seconds=max(0., phase["started_cold_seconds"] + phase["elapsed_seconds"] - (budget - .20)),
                worker_completion_over_total_budget_seconds=max(0., worker["finished_cold_seconds"] - budget),
                parent_cleanup_overrun_seconds=result.get("cleanup_overrun_seconds"),
                grouped_stop_reason=group.get("stop_reason"),
                stage_details=details,
                stage_completion=[dict(stage=name, reached=name in stage_map,
                                       completed=stage_map.get(name, {}).get("completed", False)) for name in all_stages],
                all_expected_stages_completed=all(name in stage_map and stage_map[name]["completed"] for name in all_stages))


def aggregate(rows):
    output = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        stages = defaultdict(lambda: dict(reached_runs=0, completed_runs=0, gain=0, moves=0, windows=0, elapsed_seconds=0.))
        for row in selected:
            for detail in row["stage_details"]:
                total = stages[detail["stage"]]
                total["reached_runs"] += 1
                total["completed_runs"] += int(detail["completed"])
                for key in ("gain", "moves", "windows"):
                    total[key] += detail[key]
                total["elapsed_seconds"] += detail["elapsed"]
        output[method] = dict(
            runs=len(selected),
            pre_lkh_improvement_percent=metrics([r["pre_lkh_improvement_percent"] for r in selected]),
            group_preprocessing_seconds=metrics([r["group_preprocessing_seconds"] for r in selected]),
            grouped_kernel_and_search_seconds=metrics([r["grouped_kernel_and_search_seconds"] for r in selected]),
            first_kernel_calls_sum_seconds=metrics([r["first_kernel_calls_sum_seconds"] for r in selected]),
            first_kernel_call_seconds={name: metrics([r["first_kernel_call_seconds"][name] for r in selected
                                                      if name in r["first_kernel_call_seconds"]])
                                       for name in ("or_opt", "candidate_mask", "three_opt")},
            preprocessing_target_overrun_seconds=metrics([r["preprocessing_target_overrun_seconds"] for r in selected]),
            preprocessing_target_overrun_runs=sum(r["preprocessing_target_overrun_seconds"] > 1e-7 for r in selected),
            all_expected_stages_completed_runs=sum(r["all_expected_stages_completed"] for r in selected),
            stage_totals=dict(stages))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    if not (study / "complete.json").is_file():
        parser.error("study is incomplete; no handoff audit was written")
    output = study / "handoff_audit.json"
    if output.exists():
        parser.error("handoff_audit.json already exists; refusing to overwrite")
    audit = Audit(study)
    complete = audit.json(study / "complete.json")
    protocol = audit.json(study / "frozen_protocol.json")
    schedule = audit.json(study / "schedule.json")
    environment = audit.json(study / "environment.json")
    source_hashes = audit.json(study / "source_hashes.json")
    audit.require(set(source_hashes) == SOURCE_NAMES, "source_hashes", "manifest must contain exactly the six frozen files")
    audit.require(complete.get("source_hashes_unchanged") is True, "complete", "runner did not certify frozen sources")
    source_checks = []
    for name, expected in source_hashes.items():
        for scope, path in (("study/source", study / "source" / name),
                            ("current work", Path(__file__).resolve().parent / name)):
            exists = path.is_file()
            observed = sha(path) if exists else None
            if scope == "study/source" or exists:
                audit.require(observed == expected, str(path), "frozen source SHA256 differs or source is missing")
            source_checks.append(dict(name=name, scope=scope, path=str(path), exists=exists,
                                      expected_sha256=expected, observed_sha256=observed,
                                      matches=observed == expected if exists else None))
    binary = Path(environment["lkh_binary"]).resolve()
    observed_binary = sha(binary) if binary.is_file() else None
    audit.require(observed_binary == environment["lkh_sha256"], "LKH executable", "binary hash differs or executable missing")
    maps = {}
    metadata = audit.json(study / "datasets.json")
    for item in metadata["maps"]:
        path = study / "inputs" / Path(item["path"]).name
        audit.require(sha(path) == item["sha256"], str(path), "frozen dataset SHA256 differs")
        maps[item["id"]] = audit.json(path)["coords"]
    audit.require(complete.get("jobs") == len(schedule), "schedule", "completed job count differs")
    audit.require([item["job"] for item in schedule] == list(range(1, len(schedule) + 1)),
                  "schedule", "job IDs are not unique contiguous integers")
    rows = []
    for item in schedule:
        try:
            rows.append(audit_job(audit, item, protocol, maps, binary))
        except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
            audit.errors.append(dict(context=f"j{item.get('job')}", message=f"audit could not complete: {type(exc).__name__}: {exc}"))
    paired = defaultdict(list)
    for row in rows:
        paired[(row["map_id"], row["seed"])].append(row)
    pair_checks = []
    for (map_id, seed), block in sorted(paired.items()):
        methods = [r["method"] for r in block]
        hashes = {r["nearest_neighbor_route_sha256"] for r in block}
        lengths = {r["nearest_neighbor_length"] for r in block}
        starts = {r["nearest_neighbor_start_city"] for r in block}
        passed = sorted(methods) == list(METHODS) and len(hashes) == len(lengths) == len(starts) == 1
        audit.require(passed, f"{map_id}/{seed}", "paired A/B/C NN hash/length/start mismatch or incomplete block")
        pair_checks.append(dict(map_id=map_id, seed=seed, methods=methods, passed=passed,
                                nn_route_hashes=sorted(hashes), nn_lengths=sorted(lengths)))
    audit.require(len(rows) == len(schedule), "jobs", "not every job handoff could be audited")
    report = dict(schema=1, at=datetime.now(timezone.utc).isoformat(), study_directory=str(study),
                  passed=not audit.errors, errors=audit.errors, audited_jobs=len(rows), expected_jobs=len(schedule),
                  scope="independent LKH input and grouped-gain audit only; no deadline scores are reconstructed",
                  raw_best_tour_files_read=False, solver_imported_or_launched=False,
                  first_kernel_timing_scope="first call includes JIT compilation plus execution; not isolated JIT-only time",
                  stage_completion_scope="not reached/incomplete stages are reported; early convergence or target cutoff is permitted",
                  preprocessing_target_scope="25% cooperative target; all target overruns are recomputed and compared with the worker ledger",
                  source_checks=source_checks, lkh_binary=dict(path=str(binary), expected_sha256=environment["lkh_sha256"], observed_sha256=observed_binary),
                  paired_nn_checks=pair_checks, by_method=aggregate(rows), jobs=rows,
                  input_file_sha256=audit.read_hashes, audit_script_sha256=sha(Path(__file__)))
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), passed=report["passed"], audited_jobs=len(rows), errors=len(audit.errors))))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
