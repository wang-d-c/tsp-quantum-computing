"""Independently audit and summarize the frozen 120-pair A/D replication.

No solver is called and no raw LKH tour is used to recover a score.  Scores are
rebuilt from the parent's deadline-certified incumbent ledger.  The exact
integer EUC_2D implementation below is independent of the float-based runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_seal(study, errors, checks):
    seal_path = study / "freeze_complete.json"
    if not seal_path.is_file():
        errors.append(dict(error="original freeze_complete.json is missing"))
        return
    seal = read(seal_path)
    for relative, expected in seal.get("sealed_file_hashes", {}).items():
        path = (study / relative).resolve()
        if not path.is_relative_to(study):
            errors.append(dict(file=relative, error="sealed path escapes study"))
            continue
        if not path.is_file() or sha(path) != expected:
            errors.append(dict(file=relative, error="original sealed bytes changed or missing"))
        else:
            checks["original_sealed_files_rehashed"] += 1


def execution_assignments(study, original_schedule, errors, checks):
    assignments = {entry["job"]: dict(entry, expected_execution_epoch=1) for entry in original_schedule}
    path = study / "amendments/phase2/execution_amendment.json"
    if not path.is_file():
        return assignments, dict(amendment_present=False)
    amendment = read(path)
    seal_path = study / "amendments/phase2/freeze_complete.json"
    if not seal_path.is_file():
        errors.append(dict(error="phase2 amendment has no freeze manifest"))
        return assignments, dict(amendment_present=True, amendment_path=str(path))
    seal = read(seal_path)
    if sha(study / "freeze_complete.json") != seal.get("original_freeze_sha256"):
        errors.append(dict(error="phase2 original freeze digest differs"))
    for relative, expected in seal.get("sealed_file_hashes", {}).items():
        sealed_path = (study / relative).resolve()
        if not sealed_path.is_relative_to(study) or not sealed_path.is_file() or sha(sealed_path) != expected:
            errors.append(dict(file=relative, error="phase2 sealed file changed/missing or escapes study"))
        else:
            checks["phase2_sealed_files_rehashed"] += 1
    for relative, expected in seal.get("preserved_row_hashes", {}).items():
        row_path = (study / relative).resolve()
        if not row_path.is_relative_to(study) or not row_path.is_file() or sha(row_path) != expected:
            errors.append(dict(file=relative, error="preserved phase1 row changed or missing"))
        else:
            checks["preserved_phase1_rows_rehashed"] += 1
    relative = amendment.get("remaining_schedule", "amendments/phase2/remaining_schedule.json")
    remaining_path = (study / relative).resolve()
    if not remaining_path.is_relative_to(study):
        errors.append(dict(error="remaining schedule escapes study"))
        return assignments, dict(amendment_present=True)
    remaining = read(remaining_path)
    if isinstance(remaining, dict):
        remaining = remaining.get("jobs", remaining.get("schedule", []))
    preserved = set(amendment.get("preserved_phase1_jobs", []))
    remaining_ids = [entry["job"] for entry in remaining]
    if len(remaining_ids) != len(set(remaining_ids)):
        errors.append(dict(error="phase2 remaining schedule repeats a job"))
    if preserved & set(remaining_ids) or preserved | set(remaining_ids) != set(assignments):
        errors.append(dict(error="phase2 preserved/remaining jobs are not an exact partition"))
    recorded_preserved = {int(Path(relative).stem.removeprefix("row"))
                          for relative in seal.get("preserved_row_hashes", {})}
    if recorded_preserved != preserved:
        errors.append(dict(error="phase2 preserved row hashes differ from amendment job set"))
    allowed_cpus = amendment.get("lane_cpu_affinity", [0, 2, 4, 6, 8, 10, 12, 14])
    for entry in remaining:
        original = assignments.get(entry["job"])
        if original is None:
            errors.append(dict(job=entry["job"], error="unregistered phase2 job"))
            continue
        for field in ("job", "map_id", "seed", "method", "block", "position", "block_order"):
            if entry.get(field) != original.get(field):
                errors.append(dict(job=entry["job"], error=f"phase2 changed paired design field {field}"))
        if entry.get("original_cpu") != original.get("cpu") or entry.get("original_lane") != original.get("lane"):
            errors.append(dict(job=entry["job"], error="phase2 original CPU/lane registration differs"))
        if entry.get("cpu") not in allowed_cpus:
            errors.append(dict(job=entry["job"], error="phase2 CPU not in registered affinity set"))
        if entry.get("half_pair_same_original_cpu") and entry.get("cpu") != original.get("cpu"):
            errors.append(dict(job=entry["job"], error="half pair migrated away from original CPU"))
        assignments[entry["job"]] = dict(entry, expected_execution_epoch=2)
    for half in amendment.get("half_completed_pairs", []):
        original = assignments.get(half["completed_job"])
        remaining_entry = assignments.get(half["remaining_job"])
        if not original or not remaining_entry:
            errors.append(dict(error="half-pair amendment refers to unknown job"))
            continue
        if (half["completed_job"] not in preserved or half["remaining_job"] in preserved
                or original.get("block") != remaining_entry.get("block")
                or original.get("cpu") != remaining_entry.get("cpu")
                or remaining_entry.get("cpu") != half.get("cpu")):
            errors.append(dict(error="half-pair amendment did not preserve pair/core"))
    return assignments, dict(amendment_present=True,
        amendment_path=path.relative_to(study).as_posix(), amendment_sha256=sha(path),
        preserved_phase1_jobs=len(preserved), remaining_phase2_jobs=len(remaining),
        half_completed_pairs=len(amendment.get("half_completed_pairs", [])),
        phase2_cpu_affinity=allowed_cpus, execution_epoch=amendment.get("execution_epoch", 2))


def actual_cpu(row):
    value = row.get("actual_cpu_affinity")
    return value[0] if isinstance(value, list) and len(value) == 1 and type(value[0]) is int else None


def execution_epoch(row):
    value = row.get("execution_epoch", 1)
    return {1: "epoch_1_four_lanes", 2: "epoch_2_eight_lanes"}.get(value, str(value))


def execution_summary(rows, pairs, maps):
    by_epoch = defaultdict(list)
    for key, row in rows.items():
        by_epoch[execution_epoch(row)].append(row)
    epochs = {}
    for epoch, group in sorted(by_epoch.items()):
        selected = [pair for pair in pairs if pair["pair_execution_epoch"] == epoch]
        epochs[epoch] = dict(jobs=len(group),
            cpu_affinities=sorted({actual_cpu(row) for row in group if actual_cpu(row) is not None}),
            status_counts=dict(Counter(row.get("status") for row in group)),
            paired_effect_descriptive=descriptive(selected, maps))
    mixed = [pair for pair in pairs if pair["pair_execution_epoch"] == "mixed_epoch"]
    return dict(epochs=epochs, mixed_epoch_pairs=descriptive(mixed, maps),
        same_core_paired_jobs=all(pair["paired_same_cpu"] for pair in pairs),
        concurrency_changed_during_study=len(epochs) > 1,
        scope="execution-epoch subgroups are descriptive; timings may differ with OS/cache/memory contention")


def same_float(left, right, tolerance=2e-7):
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and abs(left - right) <= tolerance)


def distance(first, second):
    squared = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
    base = math.isqrt(squared)
    # Round sqrt(squared) to nearest integer, ties upward.  Because squared is
    # integral, squared > base**2 + base is equivalent to sqrt >= base + .5.
    return base + (squared - base * base > base)


def route_length(coords, candidate):
    route = candidate.get("route")
    size = len(coords)
    if not isinstance(route, list) or len(route) != size:
        raise ValueError("route dimension differs from map")
    if any(type(city) is not int or city < 0 or city >= size for city in route):
        raise ValueError("route contains noninteger or out-of-range city")
    if len(set(route)) != size:
        raise ValueError("route is not a Hamiltonian permutation")
    total = sum(distance(coords[route[index]], coords[route[(index + 1) % size]])
                for index in range(size))
    if type(candidate.get("length")) is not int or candidate["length"] != total:
        raise ValueError(f"length differs: reported={candidate.get('length')} rebuilt={total}")
    return total


def load_jsonl(path):
    # Parent accepted.jsonl is closed before rowNNN.json becomes visible.  A
    # malformed line is evidence damage, unlike the optional raw worker logs.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def classify_lkh(row, job_directory, accepted):
    worker = row.get("worker_result")
    phase = worker.get("lkh") if isinstance(worker, dict) else None
    evidence = "worker_result.lkh"
    if not isinstance(phase, dict) and (job_directory / "lkh/phase.json").is_file():
        try:
            phase = read(job_directory / "lkh/phase.json")
            evidence = "lkh/phase.json"
        except (OSError, ValueError):
            phase = None
    accepted_lkh = sum(record.get("source") == "lkh" for record in accepted)
    if accepted_lkh:
        return True, "parent accepted LKH checkpoint", accepted_lkh
    if isinstance(phase, dict):
        count = phase.get("observed_tour_files")
        if type(count) is int and count >= 0:
            # The unchanged worker observes/publishes only before deadline-.2.
            return count > 0, evidence, count
        if phase.get("skipped") is True:
            return False, evidence + " skipped", 0
    return None, "no intact output ledger; absence is unknown", None


def check_row(row, entry, coords, job_directory, protocol):
    errors = []
    warnings = []
    checks = Counter()

    def ensure(condition, reason):
        if not condition:
            errors.append(reason)

    for field in ("job", "method", "seed", "map_id"):
        ensure(row.get(field) == entry.get(field), f"row/schedule {field} differs")
    ensure(row.get("n") == len(coords), "row dimension differs")
    budget = protocol["total_budget_seconds"]
    ensure(same_float(row.get("budget_seconds"), budget), "row budget differs")
    start, deadline = row.get("cold_started"), row.get("deadline")
    if not isinstance(start, (int, float)) or not isinstance(deadline, (int, float)):
        errors.append("missing start/deadline clock")
        return errors, warnings, checks, None
    ensure(same_float(deadline - start, budget), "deadline minus start differs from total budget")
    initial_time = row.get("initial_validation_cold_seconds")
    ensure(isinstance(initial_time, (int, float)) and 0 <= initial_time < budget,
           "initial certification missing or outside budget")
    if not isinstance(initial_time, (int, float)):
        initial_time = float("inf")
    identity = list(range(len(coords)))
    identity_length = sum(distance(coords[identity[index]], coords[identity[(index + 1) % len(coords)]])
                          for index in range(len(coords)))
    initial = dict(route=identity, length=identity_length, source="initial_identity",
                   validated_at=start + initial_time, cold_seconds=initial_time)
    accepted = []
    try:
        accepted = load_jsonl(job_directory / "accepted.jsonl")
    except (OSError, ValueError) as exc:
        errors.append(f"accepted ledger unreadable: {exc}")
    previous = initial
    for index, record in enumerate(accepted, 1):
        try:
            route_length(coords, record)
            checks["accepted_routes_recomputed"] += 1
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"accepted {index}: {exc}")
        ensure(record.get("sequence") == index, f"accepted {index}: sequence differs")
        cold = record.get("cold_seconds")
        stamp = record.get("validated_at")
        ensure(isinstance(cold, (int, float)) and 0 <= cold < budget,
               f"accepted {index}: certified outside total budget")
        ensure(isinstance(stamp, (int, float)) and start <= stamp < deadline,
               f"accepted {index}: absolute certification outside total budget")
        ensure(same_float(stamp - start if isinstance(stamp, (int, float)) else None, cold),
               f"accepted {index}: clock fields differ")
        if isinstance(stamp, (int, float)):
            ensure(stamp >= previous["validated_at"], f"accepted {index}: time went backward")
        ensure(type(record.get("length")) is int and record["length"] < previous["length"],
               f"accepted {index}: incumbent did not strictly improve")
        if isinstance(stamp, (int, float)) and type(record.get("length")) is int:
            previous = record
    ensure(row.get("accepted_checkpoints") == len(accepted), "accepted count differs from ledger")
    ensure(row.get("only_initial") is (len(accepted) == 0), "only_initial differs from ledger")
    try:
        route_length(coords, row)
        checks["final_routes_recomputed"] += 1
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"final route: {exc}")
    ensure(row.get("route") == previous["route"], "final route differs from certified incumbent")
    ensure(row.get("length") == previous["length"], "final length differs from certified incumbent")
    ensure(row.get("source") == previous["source"], "final source differs from certified incumbent")
    milestones = row.get("milestones", {})
    for seconds in protocol.get("milestones_seconds", (10., budget)):
        key = str(int(seconds)) if float(seconds).is_integer() else str(seconds)
        record = milestones.get(key)
        if not isinstance(record, dict):
            errors.append(f"missing {seconds}s milestone")
            continue
        eligible = [initial] + [item for item in accepted
                                if isinstance(item.get("cold_seconds"), (int, float))
                                and item["cold_seconds"] < seconds]
        expected = min(eligible, key=lambda value: value["length"])
        try:
            route_length(coords, record)
            checks["milestone_routes_recomputed"] += 1
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{seconds}s milestone: {exc}")
        ensure(same_float(record.get("seconds"), seconds), f"{seconds}s milestone target differs")
        ensure(record.get("route") == expected["route"], f"{seconds}s milestone incumbent differs")
        ensure(record.get("length") == expected["length"], f"{seconds}s milestone length differs")
        ensure(record.get("source") == expected["source"], f"{seconds}s milestone source differs")
        cold = record.get("validation_cold_seconds")
        stamp = record.get("validated_at")
        ensure(isinstance(cold, (int, float)) and 0 <= cold < seconds,
               f"{seconds}s milestone certified after target")
        ensure(same_float(stamp - start if isinstance(stamp, (int, float)) else None, cold),
               f"{seconds}s milestone clock fields differ")
        ensure(same_float(stamp, expected["validated_at"]), f"{seconds}s certification stamp differs")
    task_path = job_directory / "task.json"
    try:
        task = read(task_path)
        for field, expected in (("method", row.get("method")), ("seed", row.get("seed")),
                                ("cold_started", start), ("deadline", deadline), ("budget", budget)):
            ensure(task.get(field) == expected, f"task {field} differs")
        ensure(task.get("coords") == coords, "task/map coordinates differ")
    except (OSError, ValueError) as exc:
        errors.append(f"task unreadable: {exc}")
    if not row.get("containment_cleanup_passed"):
        errors.append("process containment/cleanup not confirmed")
    if row.get("status") != "completed":
        warnings.append(f"status={row.get('status')}; incumbent remains in all scores")
    if row.get("invalid_checkpoints", 0):
        warnings.append(f"{row['invalid_checkpoints']} invalid checkpoints rejected online")
    if row.get("late_checkpoints", 0):
        warnings.append(f"{row['late_checkpoints']} late checkpoints rejected online")
    output, evidence, observed = classify_lkh(row, job_directory, accepted)
    return errors, warnings, checks, dict(actual_lkh_output=output,
        lkh_output_evidence=evidence, observed_tour_files=observed,
        initial_identity_length=identity_length, certified_final_time=previous["validated_at"],
        final_validation_cold_seconds=previous["validated_at"] - start)


def percentile(values, proportion):
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def wtl(values):
    return dict(wins=sum(value < 0 for value in values), ties=sum(value == 0 for value in values),
                losses=sum(value > 0 for value in values))


def map_statistics(pairs, maps):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["map_id"]].append(pair)
    result = []
    for map_id in sorted(grouped):
        values = grouped[map_id]
        a = statistics.mean(item["baseline_length"] for item in values)
        d = statistics.mean(item["candidate_length"] for item in values)
        relative = [item["relative_percent"] for item in values]
        at_ten = [item["ten_seconds_relative_percent"] for item in values
                  if item["ten_seconds_relative_percent"] is not None]
        result.append(dict(map_id=map_id, stratum=maps[map_id]["stratum"], n=maps[map_id]["n"],
            pairs=len(values), baseline_mean_length=a, candidate_mean_length=d,
            mean_relative_percent=statistics.mean(relative),
            median_relative_percent=statistics.median(relative), **wtl(relative),
            ten_seconds_mean_relative_percent=statistics.mean(at_ten) if at_ten else None,
            baseline_lkh_outputs=sum(item["baseline_lkh_output"] is True for item in values),
            candidate_lkh_outputs=sum(item["candidate_lkh_output"] is True for item in values),
            both_lkh_outputs=sum(item["stage"] == "both_outputs" for item in values),
            neither_lkh_outputs=sum(item["stage"] == "neither_outputs" for item in values),
            baseline_identity_final=sum(item["baseline_source"] == "initial_identity" for item in values),
            candidate_identity_final=sum(item["candidate_source"] == "initial_identity" for item in values)))
    return result


def descriptive(pairs, maps):
    if not pairs:
        return dict(pairs=0, maps=0, equal_map_mean_percent=None, wins=0, ties=0, losses=0)
    by_map = map_statistics(pairs, maps)
    return dict(pairs=len(pairs), maps=len(by_map),
        equal_map_mean_percent=statistics.mean(item["mean_relative_percent"] for item in by_map),
        median_pair_percent=statistics.median(item["relative_percent"] for item in pairs),
        **wtl([item["relative_percent"] for item in pairs]))


def bootstrap(by_map, draws, seed):
    strata = defaultdict(list)
    for item in by_map:
        strata[item["stratum"]].append(item["mean_relative_percent"])
    random_state = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [random_state.choice(strata[stratum])
                  for stratum in sorted(strata) for _ in strata[stratum]]
        values.append(statistics.mean(sample))
    return dict(mean=statistics.mean(item["mean_relative_percent"] for item in by_map),
        percentile95=[percentile(values, .025), percentile(values, .975)],
        draws=draws, seed=seed, sampling_unit="whole map; paired seeds nested within map",
        strata_counts={name: len(value) for name, value in sorted(strata.items())})


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dump_csv(path, values):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        if not values:
            return
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def build_report(result, audit):
    complete = result["complete"]
    lines = ["# 120 配对种子：深度候选 2-opt 流程与经典 LKH", "",
        f"状态：{'正式数据完整' if complete else '运行中，以下仅为描述性进度'}；"
        f"已保存 {result['observed_jobs']}/{result['planned_jobs']} 次运行，"
        f"已配对 {len(result['pairs'])}/{result['planned_pairs']} 组。", "",
        "A：经典 LKH 正常初始化，不提供外部 NN 初始解；D：付费 NN → 固定深度策略选择候选 2-opt → 路线池 → 同一 LKH。"
        "双方各 30 秒总墙钟，包含全部在线开销；离线训练模型固定复用。没有执行量子计算。", "",
        "本次是已选定版本在既有 12 张图上的种子复验，并非新测试集，也不能证明它是所有版本中的最优方案。", ""]
    if not audit["passed"]:
        lines += ["**独立审计未通过，禁止据此作性能优势结论。** 详细问题见 audit.json。", ""]
    overall = result["overall"]
    if overall["pairs"]:
        lines += [f"全部已完成配对的地图等权平均长度变化：**{overall['equal_map_mean_percent']:+.6f}%**"
                  "（D 相对 A；负值更短）。", "",
                  f"配对胜／平／负：{overall['wins']}／{overall['ties']}／{overall['losses']}。", ""]
    if result.get("bootstrap"):
        low, high = result["bootstrap"]["percentile95"]
        lines += [f"预设整图分层 bootstrap 95% 区间：**[{low:+.6f}%, {high:+.6f}%]**。", ""]
        if low <= 0 <= high:
            lines += ["区间跨零，未确认稳定的平均优势；这也不等于证明完全无差异。", ""]
        elif high < 0:
            lines += ["本配置的截止前可交付路线均值更短；必须结合下面的 LKH 输出情况解释，不能直接归因于深度学习改善了 LKH 精修。", ""]
        else:
            lines += ["本配置的平均最终路线更长。", ""]
    elif not complete:
        lines += ["实验未完成，不计算推断区间，不根据中途结果改参、选模或提前停止。", ""]
    lines += ["## LKH 输出与兜底路线", "",
        "普通组未及时导出 LKH 路线时，会保留计费认证的编号顺序 identity 兜底；D 可能已有 NN/局部改进路线。"
        "这会造成很大的总体差距，属于限时交付机制差异。以下子集是事后机制诊断，不能替代完整主要结果或据此宣布击败 LKH。", "",
        "| 比较范围 | 配对数 | 地图数 | 地图等权平均变化 | 胜／平／负 |",
        "|---|---:|---:|---:|---:|"]
    scopes = [("全部配对", overall)] + [(label, result["stage_diagnostics"][key]) for key, label in (
        ("both_outputs", "双方均有 LKH 输出"), ("neither_outputs", "双方均无 LKH 输出"),
        ("mixed_outputs", "仅一方有 LKH 输出"), ("unknown_outputs", "输出账本缺失／未知"))]
    for label, value in scopes:
        mean = "—" if value["equal_map_mean_percent"] is None else f"{value['equal_map_mean_percent']:+.6f}%"
        lines.append(f"| {label} | {value['pairs']} | {value['maps']} | {mean} | {value['wins']}／{value['ties']}／{value['losses']} |")
    lines += ["", "输出存在不代表最终最优解来源一定是 LKH；等长时可能保留先前来源。未知账本不会当作零次输出。", "",
              "## 逐图最终结果", "",
              "| 地图 | 配对数 | A 平均长度 | D 平均长度 | 配对平均变化 | 胜／平／负 |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in result["by_map"]:
        lines.append(f"| {item['map_id']} | {item['pairs']} | {item['baseline_mean_length']:,.2f} | "
                     f"{item['candidate_mean_length']:,.2f} | {item['mean_relative_percent']:+.6f}% | "
                     f"{item['wins']}／{item['ties']}／{item['losses']} |")
    ten = result["ten_seconds"]
    lines += ["", "## 审计与计时", "",
        f"独立整数重算路线：最终 {audit['checks'].get('final_routes_recomputed', 0)} 条、"
        f"已接收检查点 {audit['checks'].get('accepted_routes_recomputed', 0)} 条、"
        f"里程碑 {audit['checks'].get('milestone_routes_recomputed', 0)} 条。",
        "审计从父进程 accepted.jsonl 重建完整路线、10 秒与 30 秒最短 incumbent，逐条核验认证时间早于截止；不读取原始 best.tour 补分。", ""]
    if ten["pairs"]:
        lines += [f"10 秒里程碑：{ten['pairs']} 个配对，地图等权平均变化 {ten['equal_map_mean_percent']:+.6f}%，"
                  f"胜／平／负 {ten['wins']}／{ten['ties']}／{ten['losses']}。", ""]
    for method in ("A", "D"):
        item = result["method_diagnostics"][method]
        pre = item["mean_pre_lkh_cold_seconds"]
        timing = "未知" if pre is None else f"{pre:.4f} 秒"
        lines += [f"{method}：LKH 完整输出 {item['actual_lkh_outputs']}/{item['jobs']} 次，"
                  f"输出状态未知 {item['unknown_lkh_output_ledgers']} 次；最终来源 {item['final_sources']}；"
                  f"进入 LKH 前平均耗时 {timing}。", ""]
    lines += ["状态、失败、超时和无输出成绩全部保留。区间以地图为单位重采样；120 个配对不是 120 张独立地图。"
              "结果仅适用于这些固定地图、模型及 LKH 配置。", "",
              "## 并行执行环境变更", "",
              "实验最初采用 4 路固定 CPU 并行，后按用户要求为剩余工作改为 8 路。原始封存的算法、地图、模型、种子和 30 秒在线总预算均不变；"
              "已完成成绩保留。半完成配对在原 CPU 上补齐，两条方法串行使用同一逻辑 CPU。并发度、内存和缓存竞争可能变化，"
              "因此本次结果包含两个执行环境，不能将各阶段的耗时差异只归因于算法。", "",
              "| 执行阶段 | 已保存运行 | 使用的逻辑 CPU | 同阶段完整配对 | 配对均值变化（描述性） |",
              "|---|---:|---|---:|---:|"]
    for epoch, value in result["execution"]["epochs"].items():
        effect = value["paired_effect_descriptive"]
        mean = "—" if effect["equal_map_mean_percent"] is None else f"{effect['equal_map_mean_percent']:+.6f}%"
        lines.append(f"| {epoch} | {value['jobs']} | {value['cpu_affinities']} | {effect['pairs']} | {mean} |")
    mixed = result["execution"]["mixed_epoch_pairs"]
    lines += ["", f"跨执行阶段补齐的配对：{mixed['pairs']} 组；各阶段及跨阶段子组只用于描述执行影响，不做独立优势推断。", "",
              "完整记录：results.json、per_pair.csv、per_map.csv、audit.json。", ""]
    return "\n".join(lines)


def analyze(study, partial=False, output=None):
    study = Path(study).resolve()
    protocol = read(study / "frozen_protocol.json")
    dataset_document = read(study / "datasets.json")
    records = dataset_document["maps"] if isinstance(dataset_document, dict) else dataset_document
    maps = {item["id"]: dict(item) for item in records}
    schedule_document = read(study / "schedule.json")
    schedule = schedule_document.get("jobs", schedule_document) if isinstance(schedule_document, dict) else schedule_document
    schedule_by_job = {entry["job"]: entry for entry in schedule}
    errors = []
    warnings = []
    checks = Counter()
    audit_seal(study, errors, checks)
    effective_assignments, execution_registration = execution_assignments(study, schedule, errors, checks)
    if len(maps) != 12:
        errors.append(dict(error="expected twelve unique maps", observed=len(maps)))
    if len(protocol.get("seeds", [])) != 10 or len(set(protocol.get("seeds", []))) != 10:
        errors.append(dict(error="expected ten distinct paired search seeds per map"))
    if len(schedule) != 240 or len(schedule_by_job) != 240:
        errors.append(dict(error="expected 240 distinct scheduled A/D runs"))
    coordinates = {}
    for name, item in maps.items():
        path = study / "inputs" / (name + ".json")
        if not path.is_file():
            path = Path(item["path"])
        dataset = read(path)
        coords = dataset["coords"]
        if len(coords) != item["n"] or any(len(point) != 2 or any(type(v) is not int for v in point) for point in coords):
            errors.append(dict(map_id=name, error="dimension/noninteger coordinates"))
        digest = hashlib.sha256(json.dumps(coords, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        if item.get("coordinate_sha256") and digest != item["coordinate_sha256"]:
            errors.append(dict(map_id=name, error="coordinate hash differs"))
        if item.get("sha256") and sha(path) != item["sha256"]:
            errors.append(dict(map_id=name, error="input file hash differs"))
        coordinates[name] = coords
        checks["maps_rehashed"] += 1
    rows = {}
    diagnostics = {}
    row_paths = sorted(study.glob("row[0-9]*.json"))
    expected_keys = {(entry["map_id"], entry["seed"], entry["method"]) for entry in schedule}
    for path in row_paths:
        try:
            row = read(path)
            key = (row["map_id"], row["seed"], row["method"])
            entry = schedule_by_job[row["job"]]
            if path.name != f"row{row['job']:03d}.json":
                errors.append(dict(file=path.name, error="row filename/job number differs"))
            if key in rows:
                errors.append(dict(file=path.name, error="duplicate map/seed/method"))
                continue
            if key not in expected_keys:
                errors.append(dict(file=path.name, error="unregistered map/seed/method"))
                continue
            job_directory = study / f"j{row['job']:03d}"
            row_errors, row_warnings, row_checks, diagnostic = check_row(
                row, entry, coordinates[row["map_id"]], job_directory, protocol)
            assigned = effective_assignments[row["job"]]
            for field in ("lane", "cpu"):
                if row.get(field) != assigned.get(field):
                    row_errors.append(f"row {field} differs from registered execution assignment")
            if row.get("execution_epoch", 1) != assigned["expected_execution_epoch"]:
                row_errors.append("execution epoch differs from amendment assignment")
            observed_cpu = actual_cpu(row)
            if (observed_cpu != row.get("cpu") or observed_cpu is None
                    or row.get("scheduled_cpu_affinity") != [observed_cpu]):
                row_errors.append("actual/scheduled affinity differs from registered single CPU")
            if assigned["expected_execution_epoch"] == 2:
                for field in ("original_cpu", "original_lane"):
                    if row.get(field) != assigned.get(field):
                        row_errors.append(f"phase2 row {field} differs from amendment")
            checks.update(row_checks)
            errors.extend(dict(file=path.name, error=value) for value in row_errors)
            warnings.extend(dict(file=path.name, warning=value) for value in row_warnings)
            rows[key] = row
            diagnostics[key] = diagnostic or {}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(dict(file=path.name, error=f"cannot audit row: {exc}"))
    planned = protocol.get("planned_jobs", len(schedule))
    planned_pairs = protocol.get("paired_blocks", len(schedule) // 2)
    missing_jobs = sorted(set(schedule_by_job) - {row.get("job") for row in rows.values()})
    complete = len(rows) == len(schedule) == planned and not missing_jobs
    if not partial and not complete:
        errors.append(dict(error="formal report requested before all planned rows exist", missing_jobs=missing_jobs))
    if len(schedule_by_job) != len(schedule):
        errors.append(dict(error="duplicate scheduled job number"))
    if set(protocol.get("methods", [])) != {"A", "D"}:
        errors.append(dict(error="study must predeclare exactly A and D"))
    pairs = []
    for map_id, seed in sorted({(entry["map_id"], entry["seed"]) for entry in schedule}):
        akey, dkey = (map_id, seed, "A"), (map_id, seed, "D")
        if akey not in rows or dkey not in rows:
            continue
        a, d = rows[akey], rows[dkey]
        if not (type(a.get("length")) is int and a["length"] > 0 and type(d.get("length")) is int):
            errors.append(dict(map_id=map_id, seed=seed, error="cannot calculate paired relative difference"))
            continue
        aout = diagnostics[akey].get("actual_lkh_output")
        dout = diagnostics[dkey].get("actual_lkh_output")
        stage = ("unknown_outputs" if aout is None or dout is None else
                 "both_outputs" if aout and dout else "neither_outputs" if not aout and not dout else "mixed_outputs")
        a10 = a.get("milestones", {}).get("10", {}).get("length")
        d10 = d.get("milestones", {}).get("10", {}).get("length")
        ten = 100 * (d10 / a10 - 1) if type(a10) is int and a10 > 0 and type(d10) is int else None
        acpu, dcpu = actual_cpu(a), actual_cpu(d)
        if acpu is None or dcpu is None or acpu != dcpu:
            errors.append(dict(map_id=map_id, seed=seed, error="paired arms did not record the same single logical CPU"))
        for arm in (a, d):
            cpu = actual_cpu(arm)
            if arm.get("cpu") != cpu or arm.get("scheduled_cpu_affinity") != [cpu]:
                errors.append(dict(job=arm["job"], error="recorded CPU and actual/scheduled affinity differ"))
        first, second = sorted((a, d), key=lambda item: item.get("position", 0))
        previous_end = first["cold_started"] + first["elapsed_including_cleanup_seconds"]
        if second["cold_started"] + 1e-6 < previous_end:
            errors.append(dict(map_id=map_id, seed=seed, error="paired arms overlapped instead of serial same-core execution"))
        aepoch, depoch = execution_epoch(a), execution_epoch(d)
        pairs.append(dict(map_id=map_id, seed=seed, stratum=maps[map_id]["stratum"], n=maps[map_id]["n"],
            baseline_job=a["job"], candidate_job=d["job"], baseline_length=a["length"], candidate_length=d["length"],
            relative_percent=100 * (d["length"] / a["length"] - 1),
            baseline_source=a.get("source"), candidate_source=d.get("source"),
            baseline_status=a.get("status"), candidate_status=d.get("status"),
            baseline_lkh_output=aout, candidate_lkh_output=dout, stage=stage,
            baseline_execution_epoch=aepoch, candidate_execution_epoch=depoch,
            pair_execution_epoch=aepoch if aepoch == depoch else "mixed_epoch",
            baseline_cpu=acpu, candidate_cpu=dcpu, paired_same_cpu=acpu == dcpu and acpu is not None,
            baseline_ten_seconds_length=a10, candidate_ten_seconds_length=d10, ten_seconds_relative_percent=ten,
            baseline_pre_lkh_seconds=(a.get("worker_result") or {}).get("pre_lkh_cold_seconds"),
            candidate_pre_lkh_seconds=(d.get("worker_result") or {}).get("pre_lkh_cold_seconds"),
            baseline_final_validation_cold_seconds=diagnostics[akey].get("final_validation_cold_seconds"),
            candidate_final_validation_cold_seconds=diagnostics[dkey].get("final_validation_cold_seconds")))
    by_map = map_statistics(pairs, maps)
    if complete and len(pairs) != planned_pairs:
        errors.append(dict(error="complete rows do not form all planned pairs"))
    full = complete and len(pairs) == planned_pairs
    if full:
        declared_seeds = set(protocol["seeds"])
        for map_id in maps:
            observed = {item["seed"] for item in pairs if item["map_id"] == map_id}
            if observed != declared_seeds:
                errors.append(dict(map_id=map_id, error="paired seeds differ from protocol"))
    overall = descriptive(pairs, maps)
    ten_pairs = [dict(item, relative_percent=item["ten_seconds_relative_percent"])
                 for item in pairs if item["ten_seconds_relative_percent"] is not None]
    stage_diagnostics = {key: descriptive([item for item in pairs if item["stage"] == key], maps)
                         for key in ("both_outputs", "neither_outputs", "mixed_outputs", "unknown_outputs")}
    audit = dict(passed=not errors, complete=full, at=datetime.now(timezone.utc).isoformat(),
        independent_distance="integer isqrt, exact TSPLIB EUC_2D rounding", checks=dict(checks),
        errors=errors, warnings=warnings, missing_jobs=missing_jobs,
        raw_tour_recovery=False, original_scores_retained=True,
        row_sha256={path.name: sha(path) for path in row_paths})
    method_diagnostics = {}
    for method in ("A", "D"):
        arm = [(key, row) for key, row in rows.items() if key[2] == method]
        pre_times = [(row.get("worker_result") or {}).get("pre_lkh_cold_seconds") for _, row in arm]
        pre_times = [value for value in pre_times if isinstance(value, (int, float)) and math.isfinite(value)]
        method_diagnostics[method] = dict(jobs=len(arm),
            actual_lkh_outputs=sum(diagnostics[key].get("actual_lkh_output") is True for key, _ in arm),
            unknown_lkh_output_ledgers=sum(diagnostics[key].get("actual_lkh_output") is None for key, _ in arm),
            final_sources=dict(Counter(row.get("source") for _, row in arm)),
            mean_pre_lkh_cold_seconds=statistics.mean(pre_times) if pre_times else None,
            available_pre_lkh_ledgers=len(pre_times),
            missing_worker_cost_ledgers=sum(not isinstance(row.get("worker_result"), dict) for _, row in arm))
    result = dict(schema=1, study=str(study), complete=full, partial_requested=partial,
        audit_passed=audit["passed"], planned_jobs=planned, observed_jobs=len(rows),
        planned_pairs=planned_pairs, overall=overall, by_map=by_map, pairs=pairs,
        map_outcomes=wtl([item["mean_relative_percent"] for item in by_map]),
        ten_seconds=descriptive(ten_pairs, maps), stage_diagnostics=stage_diagnostics,
        method_diagnostics=method_diagnostics,
        execution=dict(execution_summary(rows, pairs, maps), registration=execution_registration),
        primary_rule="mean paired percent difference within map, equal map weights, all failures retained",
        stage_diagnostic_scope="post-hoc descriptive mechanism subsets; not replacement primary endpoints",
        identity_fallback_limitation="A has only identity incumbent until actual LKH exports; D pays for NN/local improvements",
        statuses={method: dict(Counter(row.get("status") for key, row in rows.items() if key[2] == method))
                  for method in ("A", "D")},
        bootstrap=bootstrap(by_map, protocol.get("bootstrap_draws", 20000), protocol.get("bootstrap_seed", 2026102802))
                  if full and not partial and not errors else None)
    destination = Path(output).resolve() if output else study / ("analysis_partial" if partial else "analysis")
    # New analysis destinations avoid overwriting prior final evidence.  A
    # partial directory is an explicitly replaceable progress snapshot.
    destination.mkdir(parents=True, exist_ok=partial)
    dump_json(destination / "audit.json", audit)
    dump_json(destination / "results.json", result)
    dump_csv(destination / "per_pair.csv", pairs)
    dump_csv(destination / "per_map.csv", by_map)
    (destination / "中文结果报告.md").write_text(build_report(result, audit), encoding="utf-8")
    return result, audit, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--partial", action="store_true", help="descriptive snapshot; no inference or final claims")
    parser.add_argument("--output", type=Path, help="new output directory, default STUDY/analysis[_partial]")
    args = parser.parse_args()
    result, audit, destination = analyze(args.study, args.partial, args.output)
    print(json.dumps(dict(output=str(destination), audit_passed=audit["passed"], complete=result["complete"],
        jobs=result["observed_jobs"], pairs=len(result["pairs"]), errors=len(audit["errors"])), ensure_ascii=False))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
