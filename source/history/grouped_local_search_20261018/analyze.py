"""Audit frozen TSP study results and compare A/B/C without changing raw files.

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
METHODS = ("A", "B", "C")
LABELS = {
    "A": "贪心 → LKH",
    "B": "贪心 → 分组 Or-opt → LKH",
    "C": "贪心 → 分组 Or-opt＋候选纯 3-opt＋错位窗口 → LKH",
}


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
    audit.require({k: v for k, v in row.items() if k not in ("job", "map_id", "family")} == result,
                  context, "row differs from original per-job result.json")
    audit.require(row.get("map_id") == metadata.get("id"), context, "map identity mismatch")
    audit.require(row.get("n") == len(coords), context, "dimension mismatch")
    for key in ("method", "seed", "cold_started", "deadline"):
        audit.require(task.get(key) == row.get(key), context, f"task/result {key} mismatch")
    audit.require(task.get("coords") == coords, context, "task coordinates differ from frozen input")
    audit.require(task.get("settings") == protocol.get("settings"), context, "task settings differ from protocol")
    audit.require(row.get("settings") == protocol.get("settings"), context, "result settings differ from protocol")
    audit.require(row.get("family") == metadata.get("family"), context, "map family mismatch")
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
    return dict(job=job, map_id=row["map_id"], family=row.get("family"), seed=row["seed"],
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


def aggregate(records):
    paired = []
    indexed = defaultdict(dict)
    for record in records:
        indexed[(record["map_id"], record["seed"])][record["method"]] = record
    for (map_id, seed), methods in sorted(indexed.items()):
        if "A" not in methods:
            continue
        baseline = methods["A"]["length"]
        for method in ("B", "C"):
            if method not in methods:
                continue
            current = methods[method]
            delta = current["length"] - baseline
            milestone_differences = {}
            for mark, length in current["milestones"].items():
                if mark in methods["A"]["milestones"]:
                    initial = methods["A"]["milestones"][mark]
                    milestone_differences[mark] = dict(delta_length=length-initial,
                        relative_difference_percent=100.*(length-initial)/initial)
            paired.append(dict(map_id=map_id, seed=seed, method=method,
                baseline_A_length=baseline, length=current["length"], delta_length=delta,
                relative_difference_percent=100.*delta/baseline,
                result="win" if delta < 0 else "tie" if delta == 0 else "loss",
                milestones=milestone_differences,
                includes_failed_run=any(methods[m]["status"] == "failed" for m in ("A", method))))
    by_map = []
    for map_id in sorted({r["map_id"] for r in records}):
        item = dict(map_id=map_id, methods={})
        for method in METHODS:
            selected = [r for r in records if r["map_id"] == map_id and r["method"] == method]
            if not selected:
                continue
            pairs = [p for p in paired if p["map_id"] == map_id and p["method"] == method]
            item["methods"][method] = dict(runs=len(selected), seeds=[r["seed"] for r in selected],
                n=selected[0]["n"], family=selected[0]["family"],
                final_length=summary([r["length"] for r in selected]),
                mean_relative_difference_percent_vs_A=mean([p["relative_difference_percent"] for p in pairs]) if method != "A" else 0.,
                mean_pre_lkh_cold_seconds=mean([r["pre_lkh_cold_seconds"] for r in selected]),
                mean_group_preprocessing_seconds=mean([r["group_preprocessing_seconds"] for r in selected]),
                mean_lkh_available_seconds=mean([r["lkh_available_seconds"] for r in selected]))
        by_map.append(item)
    overall = {}
    for method in METHODS:
        selected = [r for r in records if r["method"] == method]
        pairs = [p for p in paired if p["method"] == method]
        outcomes = Counter(p["result"] for p in pairs)
        map_means = [item["methods"][method]["mean_relative_difference_percent_vs_A"]
                     for item in by_map if method in item["methods"]]
        overall[method] = dict(runs=len(selected), paired_runs=len(pairs),
            mean_relative_difference_percent_vs_A=mean([p["relative_difference_percent"] for p in pairs]) if method != "A" else 0.,
            equal_map_weight_mean_relative_difference_percent_vs_A=mean(map_means),
            wins=outcomes["win"], ties=outcomes["tie"], losses=outcomes["loss"],
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
    return dict(by_map=by_map, paired=paired, overall=overall,
                relative_difference_definition="100 * (method_length - A_length) / A_length; negative is shorter",
                units="synthetic coordinate distance units, integer-rounded EUC_2D, closed tours",
                all_failed_and_initial_only_runs_retained=True)


def fmt(value, digits=3):
    return "缺失" if value is None else f"{value:.{digits}f}"


def report_text(study, protocol, audit, results):
    lines = ["TSP 同实例、同冷启动总时间比较", "",
             f"原始目录：{study}",
             f"每次总预算：{protocol.get('total_budget_seconds')} 秒；方法：A、B、C。",
             "A：贪心 → LKH；B：贪心 → 分组 Or-opt → LKH；",
             "C：贪心 → 分组 Or-opt＋候选纯 3-opt＋错位窗口 → LKH。",
             "距离为合成坐标的 EUC_2D 整数单位，闭环包含最后一城返回首城的边。",
             "相对 A 差值 = 100 × (当前方法长度 − A 长度) / A 长度；负值表示更短，不是距最优解的差距。", "",
             f"审计：{'通过' if not audit.errors else '存在错误，下面的比较不可视为有效结论'}；"
             f"错误 {len(audit.errors)} 项，提示 {len(audit.warnings)} 项。",
             f"独立重算路线数：{dict(audit.checked_route_counts)}。",
             "最终成绩和每个里程碑均从截止前完成独立认证的 accepted.jsonl 重建；未读取截止后的原始 LKH 路线补分。", "",
             "各图三种方法的最终长度均值（括号内为相对 A 的配对平均差）："]
    for item in results["by_map"]:
        parts = []
        for method in METHODS:
            entry = item["methods"].get(method)
            if entry:
                parts.append(f"{method} {fmt(entry['final_length']['mean'])}"
                             f"（{fmt(entry['mean_relative_difference_percent_vs_A'])}%）")
        lines.append(item["map_id"] + "：" + "；".join(parts))
    lines.extend(["", "所有配对的最终比较："])
    for method in ("B", "C"):
        item = results["overall"][method]
        lines.append(f"{method}：相对 A 平均 {fmt(item['mean_relative_difference_percent_vs_A'])}%；"
                     f"胜/平/负 = {item['wins']}/{item['ties']}/{item['losses']}"
                     f"（{item['paired_runs']} 个配对，失败运行保留）。")
    lines.extend(["", "每个同图同种子的配对差值（相对 A；负值更短）："])
    for item in results["paired"]:
        lines.append(f"{item['map_id']} seed={item['seed']} {item['method']}："
                     f"A={item['baseline_A_length']}，当前={item['length']}，"
                     f"差={item['delta_length']}，相对差={fmt(item['relative_difference_percent'])}%。")
    lines.extend(["", "每次运行平均耗时（秒）："])
    for method in METHODS:
        item = results["overall"][method]
        lines.append(f"{method}：付费 NN {fmt(item['mean_nearest_neighbor_seconds'])}；"
                     f"组内前处理 {fmt(item['mean_group_preprocessing_seconds'])}；"
                     f"从冷启动到 LKH 准备前 {fmt(item['mean_pre_lkh_cold_seconds'])}；"
                     f"LKH 可用预算 {fmt(item['mean_lkh_available_seconds'])}；"
                     f"超过前处理目标 {fmt(item['mean_preprocessing_target_overrun_seconds'])}。")
    lines.extend(["LKH 可用预算 = 总预算 − LKH 开始时的冷启动累计时间 − 0.20 秒统一停止余量。",
                  "冷启动累计时间包括进程、导入、NN、矩阵、分组、候选构建、Numba JIT 和局部优化；每次用独立空编译缓存。",
                  "B/C 前处理采用总时间 25% 的合作式目标；矩阵、导入和 JIT 可超过目标，超出时间照实扣除，整次硬截止不变。",
                  "输入复制和离线审计在总预算之外；操作系统文件缓存未清空。清理超时单独记录，不能产生新的计分解。", "",
                  "运行状态与异常："])
    for method in METHODS:
        item = results["overall"][method]
        lines.append(f"{method}：状态 {item['status_counts']}；无效 checkpoint {item['invalid_checkpoints']}；"
                     f"晚于截止认证 {item['late_checkpoints']}；仅保底路线 {item['only_initial_runs']}；"
                     f"进程清理失败 {item['cleanup_failures']}；缺失 worker 账本 {item['missing_worker_ledgers']}。")
    lines.extend(["", "适用范围："])
    if protocol.get("development_smoke"):
        lines.append("本次是开发 smoke 检查，仅用于验证程序与计时审计流程，不能判断四张正式图上的算法优势。")
    else:
        lines.append("四张以前研究过的合成图（1000/3000 城市 × 均匀/聚簇），每图三个配对 solver/NN 种子；属于探索性实现比较。")
    lines.extend(["同图同种子的 NN 从同一城市出发；分析独立重建 NN 路线并记录 SHA256，核对各方法 worker 长度与已捕获 NN checkpoint。",
                  "结果不证明广泛算法优势或统计显著性；没有执行量子计算。",
                  "原始文件保持不变；results.json 包含全部配对、各图统计、成本与 NN 路线 hash；audit.json 包含逐项错误和读取文件 hash。"])
    if audit.errors:
        lines.extend(["", "审计错误："])
        lines.extend(f"- {e['context']}：{e['message']}" for e in audit.errors)
    if audit.warnings:
        lines.extend(["", "审计提示："])
        lines.extend(f"- {e['context']}：{e['message']}" for e in audit.warnings)
    return "\n".join(lines) + "\n"


def make_plot(path, results, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    maps = [item["map_id"] for item in results["by_map"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180, layout="constrained")
    colors = {"A": "#596675", "B": "#1b88a3", "C": "#cf7935"}
    centers = np.arange(len(maps))
    for method, offset in (("B", -.17), ("C", .17)):
        values = []
        for item in results["by_map"]:
            values.append(item["methods"].get(method, {}).get("mean_relative_difference_percent_vs_A", np.nan))
        axes[0].bar(centers + offset, values, .30, color=colors[method], label=method, alpha=.85)
        for index, map_id in enumerate(maps):
            pairs = [p for p in results["paired"] if p["map_id"] == map_id and p["method"] == method]
            jitter = np.linspace(-.055, .055, len(pairs)) if pairs else []
            axes[0].scatter(index + offset + jitter,
                            [p["relative_difference_percent"] for p in pairs],
                            s=20, color="#192734", zorder=4)
    axes[0].axhline(0, color="#263442", linewidth=.8)
    axes[0].set_xticks(centers, maps)
    axes[0].set_ylabel("Final length difference versus A (%)")
    axes[0].set_title("Bars: paired means; dots: individual seeds\nNegative values are shorter")
    axes[0].legend(frameon=False)
    pre, available = [], []
    for method in METHODS:
        selected = [r for r in results["records"] if r["method"] == method]
        pre.append(mean([r["lkh_start_cold_seconds"] for r in selected]) or 0.)
        available.append(mean([r["lkh_available_seconds"] for r in selected]) or 0.)
    axes[1].bar(METHODS, pre, label="Cold setup + preprocessing + LKH preparation", color="#b9c5d0")
    axes[1].bar(METHODS, available, bottom=pre, label="Available LKH budget", color="#58758c")
    axes[1].set_ylim(0, protocol.get("total_budget_seconds", 30.) * 1.08)
    axes[1].set_ylabel("Mean seconds per run")
    axes[1].set_title("Same total wall-clock cap; cold JIT included")
    axes[1].legend(frameon=False, fontsize=7, loc="upper center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Exploratory TSP comparison: A / B / C", fontsize=13)
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
    datasets = audit.read(study / "datasets.json") or {}
    schedule = audit.read(study / "schedule.json") or []
    complete = audit.read(study / "complete.json")
    environment = audit.read(study / "environment.json") or {}
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
    rows = []
    for path in sorted(study.glob("row*.json")):
        row = audit.read(path)
        if row is not None:
            rows.append(row)
    audit.require(len(rows) == len(schedule), "schedule", "row count differs from planned schedule")
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
            audit.require({r["method"] for r in selected} == set(METHODS), f"{map_id}/{seed}", "incomplete A/B/C pairing")
            observed = [r for r in selected if r["worker_ledger_present"]]
            audit.require(len({(r["nn_worker_start_city"], r["nn_worker_length"], r["nn_worker_route_sha256"]) for r in observed}) <= 1,
                          f"{map_id}/{seed}", "paired methods did not share identical NN start/length")
    results = aggregate(records)
    results.update(records=records, audit_passed=not audit.errors,
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
