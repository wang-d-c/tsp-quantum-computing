"""Completion-gated independent arithmetic and paired analysis of internal RL.

No solver is imported or launched and no raw best.tour is read. The analysis
uses only deadline-certified parent incumbents and immutable native ledgers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


METHODS = ('A', 'U', 'D')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def length(coords, route):
    if (not isinstance(route, list) or len(route) != len(coords)
            or any(type(v) is not int for v in route)
            or set(route) != set(range(len(coords)))):
        raise ValueError('not a Hamiltonian permutation')
    total = 0
    for a, b in zip(route, route[1:] + route[:1]):
        squared = sum((coords[a][d] - coords[b][d]) ** 2 for d in (0, 1))
        lower = math.isqrt(squared)
        total += lower + int(squared > lower * lower + lower)
    return total


def quantile(values, probability):
    x = sorted(values)
    position = (len(x) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return x[low] + (position - low) * (x[high] - x[low])


def intervals(map_rows, value_key, replicates, seed):
    strata = defaultdict(list)
    for item in map_rows:
        strata[item['stratum']].append(item[value_key])
    rng, samples = random.Random(seed), []
    for _ in range(replicates):
        total = 0.
        for name in sorted(strata):
            values = strata[name]
            total += sum(values[rng.randrange(len(values))] for _ in values)
        samples.append(total / len(map_rows))
    return dict(mean=statistics.fmean(m[value_key] for m in map_rows),
                percentile95=[quantile(samples, .025), quantile(samples, .975)],
                bonferroni_two_comparisons=[quantile(samples, .0125), quantile(samples, .9875)],
                familywise_confidence_level=.95,
                individual_adjusted_confidence_level=.975,
                replicates=replicates, seed=seed,
                sampling_unit='whole map; mean of paired seed differences',
                strata_counts={k: len(v) for k, v in strata.items()})


def outcomes(values):
    return dict(wins=sum(v < 0 for v in values), ties=sum(v == 0 for v in values),
                losses=sum(v > 0 for v in values))


def native_snapshot(directory):
    path = directory / 'lkh/internal_stats.json'
    latest = None
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
            if not line.endswith(('\n', '\r')):
                break
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    latest = value
            except ValueError:
                pass
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    protocol, schedule = read(root/'frozen_protocol.json'), read(root/'schedule.json')
    datasets, complete = read(root/'datasets.json')['maps'], read(root/'complete.json')
    if complete.get('jobs') != len(schedule) or any(not (root/f'row{e["job"]:03d}.json').is_file() for e in schedule):
        raise ValueError('analysis prohibited until every scheduled job is complete')
    result_dir = root/'analysis'
    result_dir.mkdir(exist_ok=False)
    errors, warnings, evidence_hashes, counts = [], [], {}, Counter()
    for name, expected in read(root/'freeze_complete.json')['sealed_file_hashes'].items():
        observed = sha(root/name)
        evidence_hashes[name] = observed
        if observed != expected:
            errors.append('frozen artifact changed: '+name)
    coordinates = {item['id']: read(root/'inputs'/(item['id']+'.json'))['coords'] for item in datasets}
    records, jobs = {}, []
    for entry in schedule:
        index = entry['job']
        path, directory = root/f'row{index:03d}.json', root/f'j{index:03d}'
        row, task = read(path), read(directory/'task.json')
        evidence_hashes[path.name] = sha(path)
        coords = coordinates[entry['map_id']]
        for key in ('method', 'map_id', 'seed', 'job', 'block', 'position', 'block_order'):
            if row.get(key) != entry[key]:
                errors.append(f'job {index}: schedule {key} differs')
        try:
            if task['coords'] != coords or task['settings'] != protocol['settings']:
                raise ValueError('task coordinates/settings differ')
            if length(coords, row['route']) != row['length']:
                raise ValueError('final length differs')
            counts['final_routes'] += 1
            accepted = []
            initial = dict(route=list(range(len(coords))), source='initial_identity')
            initial['length'] = length(coords, initial['route'])
            for line in (directory/'accepted.jsonl').read_text(encoding='utf-8').splitlines():
                event = json.loads(line)
                if length(coords, event['route']) != event['length']:
                    raise ValueError('accepted length differs')
                if not row['cold_started'] <= event['validated_at'] < row['deadline']:
                    raise ValueError('accepted outside deadline')
                if not math.isclose(event['cold_seconds'], event['validated_at']-row['cold_started'], abs_tol=1e-8):
                    raise ValueError('event clocks differ')
                if event['length'] >= (accepted[-1] if accepted else initial)['length']:
                    raise ValueError('accepted route does not strictly improve incumbent')
                if event['source'] not in ('nearest_neighbor', 'lkh'):
                    raise ValueError('unexpected optimizing preprocessing')
                accepted.append(event)
                counts['accepted_routes'] += 1
            expected = accepted[-1] if accepted else initial
            if any(row[key] != expected[key] for key in ('route', 'length', 'source')):
                raise ValueError('final score differs from accepted incumbent')
            for milestone in row['milestones'].values():
                if length(coords, milestone['route']) != milestone['length']:
                    raise ValueError('milestone length differs')
                eligible = [e for e in accepted if e['cold_seconds'] < milestone['seconds']]
                expected = eligible[-1] if eligible else initial
                if any(milestone[key] != expected[key] for key in ('route', 'length', 'source')):
                    raise ValueError('milestone differs from certified incumbent')
                counts['milestone_routes'] += 1
            if row['budget_seconds'] != 30. or not math.isclose(row['deadline']-row['cold_started'], 30., rel_tol=0, abs_tol=1e-8):
                raise ValueError('not a thirty-second formal job')
            if not row['containment_cleanup_passed']:
                raise ValueError('process containment failed')
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f'job {index}: {exc}')
        worker = row.get('worker_result') or {}
        ledger = worker.get('lkh') or {}
        if worker.get('preprocessing_search') or worker.get('external_route_pool'):
            errors.append(f'job {index}: external optimizer/pool present')
        native = native_snapshot(directory)
        if ledger.get('internal_policy_statistics') is not None and native != ledger['internal_policy_statistics']:
            errors.append(f'job {index}: native ledger differs from worker snapshot')
        if entry['method'] == 'A' and native is not None:
            errors.append(f'job {index}: ordinary solver unexpectedly activated learning')
        if entry['method'] == 'D' and ledger:
            expected_model = protocol['settings']['policy_by_test_seed'][str(entry['seed'])]['D']
            if not ledger.get('skipped') and ledger.get('model_path') != expected_model:
                errors.append(f'job {index}: loaded model path differs')
        key = entry['map_id'], entry['seed'], entry['method']
        if key in records:
            errors.append('duplicate paired record '+str(key))
        records[key] = row
        jobs.append(dict(job=index, map_id=row['map_id'], method=row['method'], seed=row['seed'],
                         length=row['length'], status=row['status'], source=row['source'],
                         lkh_output_observed=bool(ledger.get('observed_tour_files', 0)),
                         native=native, worker=worker))
    seeds = protocol['seeds']
    for item in datasets:
        for seed in seeds:
            workers = [records[(item['id'], seed, m)].get('worker_result') or {} for m in METHODS]
            digests = [w.get('nearest_neighbor_route_sha256') for w in workers]
            if all(digests) and len(set(digests)) != 1:
                errors.append(f'{item["id"]}/{seed}: common independently paid NN differs')
    map_table = []
    for item in datasets:
        record = dict(map_id=item['id'], stratum=item['stratum'], n=item['n'], methods={})
        for method in METHODS:
            rows = [records[(item['id'], seed, method)] for seed in seeds]
            record['methods'][method] = dict(mean_length=statistics.fmean(r['length'] for r in rows),
                                             minimum_length=min(r['length'] for r in rows),
                                             lengths=[r['length'] for r in rows])
        map_table.append(record)
    comparisons = {}
    for method, baseline in (('D', 'A'), ('D', 'U'), ('U', 'A')):
        pairs, maps = [], []
        for item in datasets:
            differences = []
            for seed in seeds:
                candidate, base = (records[(item['id'], seed, m)] for m in (method, baseline))
                value = 100.*(candidate['length']-base['length'])/base['length']
                ledgers = [(r.get('worker_result') or {}).get('lkh') for r in (candidate, base)]
                outputs = [bool((l or {}).get('observed_tour_files', 0)) for l in ledgers]
                stage = ('unknown_output_ledger' if any(l is None for l in ledgers) else
                         'both_outputs' if all(outputs) else 'neither_output' if not any(outputs) else 'mixed_outputs')
                pairs.append(dict(map_id=item['id'], stratum=item['stratum'], seed=seed,
                                  relative_percent=value, baseline_length=base['length'],
                                  candidate_length=candidate['length'], stage=stage))
                differences.append(value)
            maps.append(dict(map_id=item['id'], stratum=item['stratum'], mean_relative_percent=statistics.fmean(differences)))
        diagnostic = {}
        for stage in ('both_outputs', 'neither_output', 'mixed_outputs', 'unknown_output_ledger'):
            subset = [p for p in pairs if p['stage'] == stage]
            grouping = defaultdict(list)
            for pair in subset:
                grouping[pair['map_id']].append(pair['relative_percent'])
            diagnostic[stage] = dict(pairs=len(subset), maps=len(grouping),
                equal_map_mean_percent=statistics.fmean(statistics.fmean(v) for v in grouping.values()) if grouping else None,
                **outcomes([p['relative_percent'] for p in subset]))
        comparisons[method+'_vs_'+baseline] = dict(
            bootstrap=intervals(maps, 'mean_relative_percent', protocol['bootstrap_draws'], protocol['bootstrap_seed']),
            pair_outcomes=outcomes([p['relative_percent'] for p in pairs]),
            map_outcomes=outcomes([p['mean_relative_percent'] for p in maps]),
            median_pair_percent=statistics.median(p['relative_percent'] for p in pairs),
            maps=maps, pairs=pairs, stage_diagnostic=diagnostic,
            by_seed={str(seed):dict(mean_relative_percent=statistics.fmean(p['relative_percent'] for p in pairs if p['seed']==seed),
                                  **outcomes([p['relative_percent'] for p in pairs if p['seed']==seed])) for seed in seeds},
            status='multiplicity_controlled' if method=='D' else 'descriptive_unadjusted')
    overall = {}
    for method in METHODS:
        rows = [r for r in records.values() if r['method']==method]
        subset = [j for j in jobs if j['method']==method]
        workers, natives = [j['worker'] for j in subset], [j['native'] for j in subset if j['native'] is not None]
        def mean_values(values):
            values = [v for v in values if type(v) in (int, float)]
            return statistics.fmean(values) if values else None
        costs = {key:mean_values([n.get(key) for n in natives]) for key in
                 ('model_load_seconds', 'feature_build_seconds', 'inference_seconds', 'rank_seconds')}
        overall[method] = dict(runs=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
            invalid_checkpoints=sum(r['invalid_checkpoints'] for r in rows),
            late_checkpoints=sum(r['late_checkpoints'] for r in rows), missing_worker_ledgers=sum(not w for w in workers),
            final_sources=dict(Counter(r['source'] for r in rows)),
            zero_lkh_outputs=sum(not j['lkh_output_observed'] for j in subset),
            mean_pre_lkh_cold_seconds=mean_values([w.get('pre_lkh_cold_seconds') for w in workers]),
            mean_nearest_neighbor_seconds=mean_values([w.get('nearest_neighbor_seconds') for w in workers]),
            native_ledger_runs=len(natives), missing_native_ledgers=len(rows)-len(natives) if method!='A' else 0,
            native_active_runs=sum(bool(n.get('active')) for n in natives),
            native_model_loaded_runs=sum(bool(n.get('has_model')) for n in natives),
            native_best5opt_proven_runs=sum(n.get('best5opt_calls', 0)>0 for n in natives),
            native_mean_phase_seconds_conditional_on_ledger=costs,
            native_snapshot_note='last complete append-only snapshot; call counters can be lower bounds if stopped mid-run; absent costs are unknown, not zero',
            mean_cleanup_overrun_seconds=statistics.fmean(r['cleanup_overrun_seconds'] for r in rows))
    audit = dict(passed=not errors, errors=errors, warnings=warnings, route_counts=dict(counts),
                 expected_jobs=len(schedule), observed_jobs=len(jobs), raw_lkh_tour_files_read=False, evidence_hashes=evidence_hashes)
    results = dict(audit_passed=audit['passed'], comparisons=comparisons, by_map=map_table,
                   overall=overall, jobs=jobs,
                   primary_rule='Bonferroni interval upper endpoint <0 supports shorter average; intended comparisons D-A and D-U; U-A descriptive',
                   estimator='equal map means of paired seed-relative differences; negative is shorter',
                   stage_diagnostic_scope='descriptive only; every formal score remains included')
    lines = ['LKH内部深度强化学习：固定30秒对比', '',
             'A=普通LKH；U=内部均匀随机候选；D=内部深度强化学习候选。',
             '三组都独立付费计算相同NN兜底，但不把该路线交给LKH作初始解；没有外部2-opt或外部路线池。',
             '12张独立地图×5个配对种子×3方法=180次，每次冷启动总预算30秒。',
             '相对差=100×(候选长度−基线长度)/基线长度，负数更短。先对同图种子求均值，再对地图等权平均。',
             f'截止计分审计={audit["passed"]}；合法路线重算={dict(counts)}。', '']
    for label, data in comparisons.items():
        estimate = data['bootstrap']
        interval = (f'两比较Bonferroni97.5%区间{estimate["bonferroni_two_comparisons"]}' if label.startswith('D_') else
                    f'描述性95%区间{estimate["percentile95"]}，不作确认性结论')
        lines += [f'{label}: 均值{estimate["mean"]:+.6f}%；{interval}；',
                  f'配对胜平负{data["pair_outcomes"]}；图级胜平负{data["map_outcomes"]}；配对中位数{data["median_pair_percent"]:+.6f}%。',
                  f'种子描述{data["by_seed"]}；输出阶段描述{data["stage_diagnostic"]}。', '']
    lines += ['逐图五种子平均最终长度（原始EUC_2D单位）：', '图名                       A              U              D']
    for item in map_table:
        lines.append(f'{item["map_id"]:<24}'+' '.join(f'{item["methods"][m]["mean_length"]:14.2f}' for m in METHODS))
    lines += ['', '在线成本：候选生成、模型加载、特征、推理、排序、内部路线维护、返回与在线验算均计入30秒；无量子选择。']
    for method, value in overall.items():
        lines.append(f'{method}: 无LKH输出{value["zero_lkh_outputs"]}次；内部记录{value["native_ledger_runs"]}次；确证进入Best5Opt搜索{value["native_best5opt_proven_runs"]}次；')
        lines.append(f'有内部记录运行的分项平均秒数={value["native_mean_phase_seconds_conditional_on_ledger"]}；缺失时间不按0填补。')
    lines += ['', '网络12→64→32→1，两层ReLU隐藏层；在C内一次评分，选择并排序实际LKH候选边。',
              'ALPHA5与kNN16取并集，保留前两条ALPHA边，模型选其余三条并排序；模型不在每次LK移动中重新推理。',
              '训练回报来自完整定试次数LKH最终路线。离线训练/验证时间另报；本方法不是论文的完整复现。',
              '统计结论仅适用于本实现、这些地图、固定种子与30秒预算。全部失败与无输出运行均保留。']
    (result_dir/'中文结果说明.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 4.5))
        for axis, label in zip(axes, ('D_vs_A', 'D_vs_U')):
            value = comparisons[label]['bootstrap']
            low, high = value['bonferroni_two_comparisons']
            axis.errorbar(value['mean'], 0, xerr=[[value['mean']-low], [high-value['mean']]], fmt='o', capsize=5)
            axis.axvline(0, color='black', linewidth=.8)
            axis.set_yticks([0], [label])
            axis.grid(axis='x', alpha=.2)
            axis.set_xlabel('Relative tour length difference (%); negative is shorter')
        fig.suptitle('30 s cold total; Bonferroni-adjusted stratified map-bootstrap intervals')
        fig.tight_layout()
        fig.savefig(result_dir/'comparison.png', dpi=180)
        plt.close(fig)
    except ImportError:
        warnings.append('matplotlib unavailable; numeric reports complete')
    write(result_dir/'audit.json', audit)
    write(result_dir/'results.json', results)
    print(json.dumps(dict(audit_passed=audit['passed'], comparisons={k:dict(bootstrap=v['bootstrap'], pair_outcomes=v['pair_outcomes']) for k,v in comparisons.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
