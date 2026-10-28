"""Explain scored routes using phase ledgers; never recover a raw best.tour."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys


def average(values):
    return statistics.fmean(values) if values else None


def main():
    root = Path(sys.argv[1]).resolve()
    if not (root/'complete.json').is_file():
        raise RuntimeError('wait for the complete fixed schedule')
    indexed = defaultdict(dict)
    records = []
    for path in sorted(root.glob('row*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        worker = row.get('worker_result') or {}
        lkh = worker.get('lkh') or {}
        exports = lkh.get('observed_tour_files')
        record = dict(job=row['job'], map_id=row['map_id'], stratum=row['stratum'],
                      method=row['method'], seed=row['seed'], length=row['length'],
                      final_source=row.get('source'), status=row['status'],
                      observed_LKH_tour_files=exports,
                      observed_any_LKH_tour=None if exports is None else exports>0,
                      final_incumbent_from_LKH=row.get('source')=='lkh',
                      pre_lkh_length=worker.get('pre_lkh_length'),
                      nearest_neighbor_length=worker.get('nearest_neighbor_length'),
                      pre_lkh_cold_seconds=worker.get('pre_lkh_cold_seconds'))
        records.append(record)
        indexed[(row['map_id'],row['seed'])][row['method']] = record
    pairs = []
    for (map_id,seed), methods in sorted(indexed.items()):
        if set(methods) != {'A','B'}:
            raise RuntimeError('incomplete paired run')
        a,b = methods['A'],methods['B']
        flags = a['observed_any_LKH_tour'],b['observed_any_LKH_tour']
        group = ('ledger_missing' if None in flags else
                 'both_LKH_tours_observed' if all(flags) else
                 'neither_LKH_tour_observed' if not any(flags) else
                 'only_one_LKH_tour_observed')
        pairs.append(dict(map_id=map_id,stratum=a['stratum'],seed=seed,
                          diagnostic_group=group,
                          relative_difference_percent=100.*(b['length']-a['length'])/a['length'],
                          A_source=a['final_source'], B_source=b['final_source'],
                          A_observed_LKH_tours=a['observed_LKH_tour_files'],
                          B_observed_LKH_tours=b['observed_LKH_tour_files']))
    groups = {}
    for group in sorted({p['diagnostic_group'] for p in pairs}):
        selected = [p for p in pairs if p['diagnostic_group']==group]
        per_map = defaultdict(list)
        for p in selected:
            per_map[p['map_id']].append(p['relative_difference_percent'])
        values = [p['relative_difference_percent'] for p in selected]
        counts = Counter('win' if x<0 else 'loss' if x>0 else 'tie' for x in values)
        groups[group] = dict(pairs=len(selected),maps=len(per_map),
            paired_mean_relative_difference_percent=average(values),
            equal_map_mean_relative_difference_percent=average([average(v) for v in per_map.values()]),
            wins=counts['win'],ties=counts['tie'],losses=counts['loss'],
            per_map={m:dict(pairs=len(v),mean_relative_difference_percent=average(v))
                     for m,v in per_map.items()})
    by_method = {}
    for method in ('A','B'):
        selected = [r for r in records if r['method']==method]
        by_method[method] = dict(runs=len(selected),
            zero_LKH_output_runs=sum(r['observed_any_LKH_tour'] is False for r in selected),
            missing_phase_ledgers=sum(r['observed_any_LKH_tour'] is None for r in selected),
            final_incumbent_from_LKH=sum(r['final_incumbent_from_LKH'] for r in selected),
            final_sources=dict(Counter(r['final_source'] for r in selected)))
    payload = dict(scope='additional phase-availability diagnostic, not a replacement primary endpoint',
        reason='early8000-node paired runs exported no complete LKH tour before scoring cutoff',
        interpretation='strict-budget scores remain valid; when no LKH tour was observed, scores assess paid construction/incumbent availability, not a measured benefit after LKH refinement',
        selection='all120registered runs retained; this explanatory subdivision was added after observing initial8000-node outputs and is descriptive',
        raw_best_tour_read=False, phase_counts=by_method,
        paired_diagnostic_groups=groups,pairs=pairs,records=records,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (root/'stage_diagnostics.json').open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2)
    lines = ['阶段解释：截止前 LKH 是否实际产出了可用路线','',
        '此文件是解释性补充，不能替代预先冻结的12图等权主要终点。',
        '它在最初8000城市配对没有LKH路线输出后增加；组别描述不作独立确认性检验。',
        '没有读取截止后的 best.tour 补分，全部原始成绩保持不变。','']
    for method,item in by_method.items():
        lines.append(f"{method}：{item['runs']}次运行，零LKH输出{item['zero_LKH_output_runs']}次，"
                     f"最终最好路线来自LKH {item['final_incumbent_from_LKH']}次；来源{item['final_sources']}。")
    lines.append('')
    for group,item in groups.items():
        lines.append(f"{group}：{item['pairs']}个配对/{item['maps']}张图，"
                     f"图等权相对差{item['equal_map_mean_relative_difference_percent']:+.6f}%，"
                     f"胜/平/负{item['wins']}/{item['ties']}/{item['losses']}。")
        for name,entry in item['per_map'].items():
            lines.append(f"  {name}：{entry['pairs']}配对，相对差{entry['mean_relative_difference_percent']:+.6f}%。")
    lines.extend(['','没有观察到LKH输出时，计分的是此前已付费构造/组内改进得到的可用路线。',
                  '这种总预算收益有实际意义，但不能称为“LKH已精修后仍更短”的证据。',
                  '判断完整搜索优势，需要同时查看已有LKH输出的实例，以及原生初始化/候选准备更适合短预算的强对照。'])
    (root/'阶段解释.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(phase_counts=by_method,paired_diagnostic_groups=groups),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
