"""Independent deadline-score audit and paired, map-clustered analysis.

No solver is imported and no raw LKH best.tour is read. Numeric reports are
derived only from immutable parent records and independently rebuilt lengths.
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


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def length(coords, route):
    if not isinstance(route, list) or len(route) != len(coords):
        raise ValueError('route size differs')
    if any(type(v) is not int for v in route) or set(route) != set(range(len(coords))):
        raise ValueError('route is not a Hamiltonian permutation')
    return sum(math.floor(math.hypot(coords[a][0] - coords[b][0],
                                    coords[a][1] - coords[b][1]) + .5)
               for a, b in zip(route, route[1:] + route[:1]))


def quantile(values, probability):
    x = sorted(values)
    position = (len(x) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return x[low] + (position - low) * (x[high] - x[low])


def intervals(map_rows, value_key, replicates=20000, seed=20261019050002):
    strata = defaultdict(list)
    for item in map_rows:
        strata[item['stratum']].append(item[value_key])
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        total = 0.
        for name in sorted(strata):
            values = strata[name]
            total += sum(values[rng.randrange(len(values))] for _ in values)
        samples.append(total / len(map_rows))
    return dict(mean=statistics.fmean(m[value_key] for m in map_rows),
                percentile95=[quantile(samples, .025), quantile(samples, .975)],
                bonferroni_two_comparisons=[quantile(samples, .05 / 4),
                                              quantile(samples, 1 - .05 / 4)],
                familywise_confidence_level=.95,
                individual_adjusted_confidence_level=1 - .05 / 2,
                replicates=replicates, seed=seed,
                sampling_unit='whole map; mean of paired seed differences',
                strata_counts={k:len(v) for k,v in strata.items()})


def outcomes(values):
    return dict(wins=sum(v < 0 for v in values), ties=sum(v == 0 for v in values),
                losses=sum(v > 0 for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    protocol = read(root/'frozen_protocol.json')
    datasets = read(root/'datasets.json')['maps']
    schedule = read(root/'schedule.json')
    complete = read(root/'complete.json')
    result_dir = root/'analysis'
    result_dir.mkdir(exist_ok=False)
    errors, warnings = [], []
    coords_by_map = {}
    evidence_hashes = {}
    counts = Counter()
    for name, expected in read(root/'source_hashes.json').items():
        observed = sha(root/'source'/name)
        evidence_hashes['source/'+name] = observed
        if observed != expected:
            errors.append('source hash changed: '+name)
    for item in datasets:
        path = root/'inputs'/(item['id']+'.json')
        evidence_hashes['inputs/'+path.name] = sha(path)
        if evidence_hashes['inputs/'+path.name] != item['sha256']:
            errors.append('input hash changed: '+item['id'])
        content = read(path)
        coords_by_map[item['id']] = content['coords']
    records = {}
    jobs = []
    by_id = {item['id']:item for item in datasets}
    expected_count = len(schedule)
    if complete.get('jobs') != expected_count:
        errors.append('completion count mismatch')
    for entry in schedule:
        index = entry['job']
        path = root/f'row{index:03d}.json'
        row = read(path)
        evidence_hashes[path.name] = sha(path)
        for key in ('method','map_id','seed','job'):
            if row.get(key) != entry[key]:
                errors.append(f'job {index}: schedule {key} mismatch')
        directory = root/f'j{index:03d}'
        task = read(directory/'task.json')
        coords = coords_by_map[entry['map_id']]
        try:
            if task['coords'] != coords:
                raise ValueError('task coordinates mismatch')
            if length(coords, row['route']) != row['length']:
                raise ValueError('final length mismatch')
            counts['final_routes'] += 1
            accepted = []
            for line in (directory/'accepted.jsonl').read_text(encoding='utf-8').splitlines():
                event = json.loads(line)
                if length(coords, event['route']) != event['length']:
                    raise ValueError('accepted length mismatch')
                if not event['validated_at'] < row['deadline']:
                    raise ValueError('late accepted event')
                if not math.isclose(event['cold_seconds'], event['validated_at'] - row['cold_started'], abs_tol=1e-8):
                    raise ValueError('event clocks differ')
                if accepted and event['length'] >= accepted[-1]['length']:
                    raise ValueError('accepted incumbent is not strictly improving')
                accepted.append(event)
                counts['accepted_routes'] += 1
            initial = list(range(len(coords)))
            initial_length = length(coords, initial)
            final_event = accepted[-1] if accepted else dict(route=initial,length=initial_length)
            if final_event['length'] != row['length'] or final_event['route'] != row['route']:
                raise ValueError('final score differs from accepted incumbent')
            for mark, value in row['milestones'].items():
                if length(coords, value['route']) != value['length']:
                    raise ValueError('milestone length mismatch')
                eligible = [e for e in accepted if e['cold_seconds'] < value['seconds']]
                expected = eligible[-1] if eligible else dict(route=initial,length=initial_length)
                if value['length'] != expected['length'] or value['route'] != expected['route']:
                    raise ValueError('milestone differs from deadline-certified incumbent')
                counts['milestone_routes'] += 1
            if row['budget_seconds'] != 30.:
                raise ValueError('not a 30-second formal job')
            if row['deadline'] - row['cold_started'] != 30.:
                raise ValueError('deadline changed')
            if not row['containment_cleanup_passed']:
                raise ValueError('process cleanup not contained')
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f'job {index}: {exc}')
        worker = row.get('worker_result') or {}
        search = worker.get('preprocessing_search') or {}
        if not search.get('skipped') and row['method']!='A' and search:
            if search.get('gain') != search.get('best_gain_sum'):
                errors.append(f'job {index}: search global reward does not telescope')
            if search.get('best_length') != worker.get('pre_lkh_length'):
                errors.append(f'job {index}: local best differs from LKH handoff')
            if search.get('accounted_seconds',0) > search.get('elapsed',0) + .001:
                errors.append(f'job {index}: overlapping time categories')
            if row['method'] in ('U','D') and search.get('evaluations',0) > search.get('steps',0):
                errors.append(f'job {index}: D exact-evaluation quota differs from training')
        if row['method']=='D' and worker.get('policy_model_path'):
            if sha(worker['policy_model_path']) != worker.get('policy_model_file_sha256'):
                errors.append(f'job {index}: loaded model differs from ledger')
        key = (entry['map_id'],entry['seed'],entry['method'])
        if key in records:
            errors.append('duplicate pairing '+str(key))
        records[key] = row
        jobs.append(dict(job=index,map_id=row['map_id'],method=row['method'],seed=row['seed'],
                         length=row['length'],status=row['status'],source=row['source'],
                         lkh_output_observed=bool((worker.get('lkh') or {}).get('observed_tour_files',0)),
                         worker=worker))
    comparisons = {}
    map_table = []
    seeds = protocol['seeds']
    for item in datasets:
        for seed in seeds:
            workers=[records[(item['id'],seed,m)].get('worker_result') or {} for m in ('U','D')]
            digests=[w.get('nearest_neighbor_route_sha256') for w in workers]
            if all(digests) and len(set(digests))!=1:
                errors.append(f'{item["id"]}/{seed}: independently paid NN initialization differs across arms')
    for item in datasets:
        record = dict(map_id=item['id'],stratum=item['stratum'],n=item['n'],methods={})
        for method in ('A','U','D'):
            rows = [records[(item['id'],seed,method)] for seed in seeds]
            record['methods'][method] = dict(mean_length=statistics.fmean(r['length'] for r in rows),
                                            minimum_length=min(r['length'] for r in rows),
                                            lengths=[r['length'] for r in rows])
        map_table.append(record)
    for method, baseline in (('D','A'),('D','U'),('U','A')):
        label = method+'_vs_'+baseline
        pairs = []
        maps = []
        for item in datasets:
            diffs = []
            for seed in seeds:
                candidate = records[(item['id'],seed,method)]
                base = records[(item['id'],seed,baseline)]
                value = 100. * (candidate['length'] - base['length']) / base['length']
                candidate_ledger = (candidate.get('worker_result') or {}).get('lkh')
                base_ledger = (base.get('worker_result') or {}).get('lkh')
                candidate_out = bool((candidate_ledger or {}).get('observed_tour_files',0))
                base_out = bool((base_ledger or {}).get('observed_tour_files',0))
                unknown_stage = candidate_ledger is None or base_ledger is None
                pairs.append(dict(map_id=item['id'],stratum=item['stratum'],seed=seed,
                                  relative_percent=value,baseline_length=base['length'],
                                  candidate_length=candidate['length'],
                                  stage='unknown_output_ledger' if unknown_stage else
                                        'both_outputs' if candidate_out and base_out else
                                        'neither_output' if not candidate_out and not base_out else 'mixed_outputs'))
                diffs.append(value)
            maps.append(dict(map_id=item['id'],stratum=item['stratum'],mean_relative_percent=statistics.fmean(diffs)))
        diagnostic = {}
        for stage in ('both_outputs','neither_output','mixed_outputs','unknown_output_ledger'):
            subset = [r for r in pairs if r['stage']==stage]
            grouping = defaultdict(list)
            for r in subset:
                grouping[r['map_id']].append(r['relative_percent'])
            diagnostic[stage] = dict(pairs=len(subset),maps=len(grouping),
                equal_map_mean_percent=statistics.fmean(statistics.fmean(v) for v in grouping.values()) if grouping else None,
                **outcomes([r['relative_percent'] for r in subset]))
        estimate = intervals(maps,'mean_relative_percent',
                             replicates=protocol.get('bootstrap_draws',20000),
                             seed=protocol.get('bootstrap_seed',20261019020002))
        by_seed = {str(seed):dict(mean_relative_percent=statistics.fmean(p['relative_percent'] for p in pairs if p['seed']==seed),
                                  **outcomes([p['relative_percent'] for p in pairs if p['seed']==seed])) for seed in seeds}
        comparisons[label] = dict(bootstrap=estimate,pair_outcomes=outcomes([p['relative_percent'] for p in pairs]),
                                 map_outcomes=outcomes([p['mean_relative_percent'] for p in maps]),
                                 median_pair_percent=statistics.median(p['relative_percent'] for p in pairs),
                                 maps=maps,pairs=pairs,by_seed=by_seed,stage_diagnostic=diagnostic,
                                 status='multiplicity_controlled' if method=='D' else 'descriptive_unadjusted')
    overall = {}
    for method in ('A','U','D'):
        rows = [r for r in records.values() if r['method']==method]
        workers = [r.get('worker_result') or {} for r in rows]
        def mean_values(key):
            values=[w[key] for w in workers if isinstance(w.get(key),(int,float))]
            return statistics.fmean(values) if values else None
        search_cost_keys=set().union(*[set((w.get('preprocessing_search') or {}).get('ledger',{})) for w in workers])
        mean_search_costs={key:statistics.fmean((w.get('preprocessing_search') or {}).get('ledger',{}).get(key,0.) for w in workers) for key in search_cost_keys}
        overall[method] = dict(runs=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
                              invalid_checkpoints=sum(r['invalid_checkpoints'] for r in rows),
                              late_checkpoints=sum(r['late_checkpoints'] for r in rows),
                              missing_worker_ledgers=sum(not w for w in workers),
                              final_sources=dict(Counter(r['source'] for r in rows)),
                              zero_lkh_outputs=sum(not bool((w.get('lkh') or {}).get('observed_tour_files',0)) for w in workers),
                              mean_pre_lkh_cold_seconds=mean_values('pre_lkh_cold_seconds'),
                              mean_policy_load_seconds=mean_values('policy_load_seconds'),
                              mean_distance_matrix_seconds=mean_values('distance_matrix_seconds'),
                              mean_group_preprocessing_seconds=mean_values('group_preprocessing_seconds'),
                              mean_search_costs=mean_search_costs,
                              mean_cleanup_overrun_seconds=statistics.fmean(r['cleanup_overrun_seconds'] for r in rows),
                              pool_and_policy_ledgers=[dict(job=r['job'],search=(r.get('worker_result') or {}).get('preprocessing_search')) for r in rows])
    audit = dict(passed=not errors,errors=errors,warnings=warnings,route_counts=dict(counts),
                 expected_jobs=expected_count,observed_jobs=len(jobs),raw_lkh_tour_files_read=False,
                 evidence_hashes=evidence_hashes)
    results = dict(audit_passed=audit['passed'],comparisons=comparisons,by_map=map_table,
                   overall=overall,jobs=jobs,primary_rule='Bonferroni interval upper endpoint <0 supports shorter average; two intended comparisons D-A and D-U; U-A descriptive only',
                   stage_diagnostic_scope='predeclared descriptive output-stage grouping; no score exclusions; output presence does not prove final source is LKH',
                   estimator='equal map means of paired seed-relative differences; negative is shorter')
    lines = ['深度强化学习kNN候选TSP实验：脚本生成结果','',
             'A：普通LKH，无外部初始路线；U：NN+均匀随机候选选择+路线池+LKH；D：NN+深度RL候选选择+相同路线池+LKH。',
             '每次30秒冷启动总预算；相对差=100×(方法长度−基线长度)/基线长度，负数更短。',
             '先对同图配对种子求均值，再对地图等权平均；种子不作为独立地图。',
             f'截止计分独立审计：{audit["passed"]}；合法路线重算计数：{dict(counts)}。','']
    for label,data in comparisons.items():
        b=data['bootstrap']
        lines += [f'{label}: 均值{b["mean"]:+.6f}%；95%区间{b["percentile95"]}；',
                  f'两比较Bonferroni97.5%区间{b["bonferroni_two_comparisons"]}（U_vs_A仅描述，不作确认性结论）；',
                  f'配对胜平负{data["pair_outcomes"]}；图级胜平负{data["map_outcomes"]}；配对中位数{data["median_pair_percent"]:+.6f}%；',
                  f'输出阶段诊断{data["stage_diagnostic"]}。',
                  f'按配对测试/训练种子分层的描述性结果{data["by_seed"]}。','']
    lines += ['逐图五种子平均最终长度（原始EUC_2D单位）：','图名                 A              U              D']
    for item in map_table:
        lines.append(f'{item["map_id"]:<18} '+ ' '.join(f'{item["methods"][m]["mean_length"]:14.2f}' for m in ('A','U','D')))
    lines += ['','每方法60次状态与平均进入LKH前耗时：']
    for method,data in overall.items():
        lines.append(f'{method}: 状态{data["status_counts"]}；前处理累计{data["mean_pre_lkh_cold_seconds"]}秒；无LKH输出{data["zero_lkh_outputs"]}次。')
        lines.append(f'模型加载{data["mean_policy_load_seconds"]}秒；搜索分项均值{data["mean_search_costs"]}。')
    lines += ['','结论仅针对本实现、固定30秒预算及固定普通LKH及学习前处理配置。',
              '所有失败和无LKH输出运行均保留；完整阶段成本与推理/池维护账本见results.json。',
              '离线训练与验证成本另报；测试模型加载、特征、推理、候选、池维护均在预算内。']
    (result_dir/'中文结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4))
        for index,label in enumerate(('D_vs_A','D_vs_U')):
            data=comparisons[label]['bootstrap']
            low,high=data['bonferroni_two_comparisons']
            ax.errorbar(data['mean'],index,xerr=[[data['mean']-low],[high-data['mean']]],fmt='o',capsize=5,label=label)
        ax.axvline(0,color='black',linewidth=.8)
        ax.set_yticks(range(2),['D deep RL vs A stock LKH','D deep RL vs U uniform'])
        ax.set_xlabel('Paired relative tour length difference (%); negative is shorter')
        ax.set_title('30 s total cold budget; equal-map means\nBonferroni-adjusted stratified map-bootstrap intervals')
        ax.grid(axis='x',alpha=.2)
        fig.tight_layout()
        fig.savefig(result_dir/'comparison.png',dpi=180)
        plt.close(fig)
    except ImportError:
        warnings.append('matplotlib unavailable; numeric report complete')
    write(result_dir/'audit.json',audit)
    write(result_dir/'results.json',results)
    print(json.dumps(dict(audit_passed=audit['passed'],comparisons={k:{'bootstrap':v['bootstrap'],'pair_outcomes':v['pair_outcomes']} for k,v in comparisons.items()}),ensure_ascii=False))


if __name__=='__main__':
    main()
