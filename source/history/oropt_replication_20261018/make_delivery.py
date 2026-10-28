"""Render audited replication metrics and archive the complete fixed study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    root = Path(sys.argv[1]).resolve()
    if not (root/'complete.json').is_file():
        raise RuntimeError('fixed schedule incomplete')
    results = json.loads((root/'analysis/results.json').read_text(encoding='utf-8'))
    audit = json.loads((root/'analysis/audit.json').read_text(encoding='utf-8'))
    handoff = json.loads((root/'handoff_audit.json').read_text(encoding='utf-8'))
    diagnostics = json.loads((root/'stage_diagnostics.json').read_text(encoding='utf-8'))
    if not (audit['passed'] and handoff['passed']):
        raise RuntimeError('audit failed; do not publish a successful result readout')
    primary = results['primary']['bootstrap']
    b = results['overall']['B']
    lines = ['轻量分组 Or-opt：扩大实例与种子后的结果速读','',
        '结论：上一轮很小的精修收益没有稳定复现。总体均值改善主要集中在两张8000城市图，',
        '这些运行在截止前双方都没有LKH路线输出；收益来自更好的贪心/Or-opt可用解。',
        '在双方观测到LKH路线输出的十张图上，B平均略长，不能称为稳定改善LKH精修。','',
        'A：最近邻贪心 → LKH。B：最近邻贪心 → 固定连续分组 Or-opt → LKH。',
        '12张实例 × 5个配对种子 × 2种方法 = 120次独立冷启动，每次30秒总墙钟预算。',
        '八张新合成图覆盖1000/3000/5000/8000城市的均匀与聚簇分布，另有四个公开TSPLIB实例。',
        '上轮算法、计时与资源隔离引擎逐字保留；组大小64、最多两轮、每窗口32次改善不调整。',
        '全部构造、矩阵、导入、Numba首次编译、分组、Or-opt、LKH准备/搜索及在线核验均计费。','',
        '主要统计：先平均每图五个配对相对长度差，再对十二张图等权平均。',
        '相对差 = 100×(B长度−A长度)/A长度；负数更短。',
        f"均值：{primary['mean']:+.6f}%；分层地图bootstrap95%区间："
        f"[{primary['lower']:+.6f}%, {primary['upper']:+.6f}%]。",
        f"配对胜/平/负：{b['wins']}/{b['ties']}/{b['losses']}；"
        f"图级胜/平/负：{b['map_wins']}/{b['map_ties']}/{b['map_losses']}。",
        f"配对差中位数：{b['median_paired_relative_difference_percent']:+.6f}%；"
        f"地图均值中位数：{b['median_map_mean_relative_difference_percent']:+.6f}%。"]
    if primary['upper'] < 0:
        lines.append('此样本的主要区间支持在固定30秒冷预算下存在平均改善。')
    elif primary['lower'] > 0:
        lines.append('此样本的主要区间支持在固定30秒冷预算下平均变长。')
    else:
        lines.append('主要区间跨过零，不能确认平均优势；这也不是对严格无差异的证明。')
    lines += [primary['interpretation']['text'],
              'Bootstrap在三层内重采样整张地图，各层四图；20000次，固定种子202610182399。',
              '五个种子是同图内重复，不当作独立地图；预设工程容差为±0.01%。','',
              '每图最终长度均值（五个种子；各实例原坐标EUC_2D整数单位，越小越好）：',
              '地图                       A均值              B均值         配对平均差(%)']
    for row in sorted(results['by_map'],key=lambda r:(r['stratum'],r['n'],r['map_id'])):
        a_mean = row['methods']['A']['final_length']['mean']
        b_mean = row['methods']['B']['final_length']['mean']
        lines.append(f"{row['map_id']:<18} {a_mean:17,.2f} {b_mean:17,.2f} "
                     f"{row['mean_relative_difference_percent_vs_A']:+.6f}")
    lines += ['','按图族等权汇总：']
    for name,row in results['by_stratum'].items():
        lines.append(f"{name}：{row['equal_map_weight_mean_relative_difference_percent']:+.6f}%；"
                     f"图级结果{row['map_outcomes']}。")
    lines += ['','成绩来源诊断（观察到最初8000城市无LKH输出后增加，属于事后解释，不替代主要终点）：']
    for method,row in diagnostics['phase_counts'].items():
        lines.append(f"{method}：{row['runs']}次中，无LKH路线输出{row['zero_LKH_output_runs']}次；"
                     f"最终最优路线来自LKH {row['final_incumbent_from_LKH']}次。")
    for name,row in diagnostics['paired_diagnostic_groups'].items():
        lines.append(f"{name}：{row['pairs']}配对/{row['maps']}图；"
                     f"此子集图等权平均差{row['equal_map_mean_relative_difference_percent']:+.6f}%；"
                     f"胜/平/负{row['wins']}/{row['ties']}/{row['losses']}。")
    lines += ['没有LKH输出时，成绩保留了预算内付费生成的贪心/Or-opt路线；',
              '该收益属于限时构造和可用解质量，不能直接叫作“LKH精修后仍更好”。',
              '来源标记补充：pr2392的十次成绩均为378032，最终来源标记initial_identity。',
              '这是计费的共同编号顺序保底路线；LKH确有完整路线输出且达到同长度，',
              '父进程遇到等长路线保留先前来源，因此该标记不表示LKH未运行或未输出。',
              '“双方观测到LKH路线输出”只描述输出是否出现，不代表最终成绩必然由LKH改善。','',
              '平均耗时（秒）：']
    for method in ('A','B'):
        row = results['overall'][method]
        lines.append(f"{method}：进入LKH准备前累计{row['mean_pre_lkh_cold_seconds']:.3f}；"
                     f"组内前处理{row['mean_group_preprocessing_seconds']:.3f}；"
                     f"LKH可用预算{row['mean_lkh_available_seconds']:.3f}。")
    lines += ['LKH可用预算不表示其必然已经输出路线，须结合上述阶段诊断阅读。','',
              '审计与范围：',
              f"两项独立审计均通过；独立重算路线计数{audit['independently_recomputed_route_counts']}。",
              f"A状态：{results['overall']['A']['status_counts']}；B状态：{b['status_counts']}。",
              '具体错误、晚于截止认证、清理和缺失账本计数见完整审计，原始负面结果全部保留。',
              '此结果针对固定NN+LKH-3.0.14 ALPHA5配置和30秒冷预算，不能外推为击败调好的全部LKH版本。',
              '共同数据准备/封存和离线分析不计时，OS文件缓存没有清空，原始记录从未覆盖。',
              '没有执行量子计算。','',
              '研究方向预测：30秒短预算优先验证原生POPMUSIC/候选边与重启调度；',
              '中长预算更看好多样路线池、成熟EAX/GPX2重组和局部搜索的组合。',
              '这些是文献支持下的方向判断，尚未在本轮执行；详见可能超过LKH的方向.txt。','',
              '完整逐次结果：analysis/中文结果说明.txt、analysis/results.json。',
              '阶段解释：阶段解释.txt、stage_diagnostics.json。',
              '核验：analysis/audit.json、handoff_audit.json。',
              '源码、数据、协议、原始日志、依赖版本及本机复验命令均保留在本目录。']
    (root/'结果速读.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive = root.parent/'轻量分组Or-opt_扩展验证_结果与代码.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                package.write(path,root.name+'/'+path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as package:
        bad = package.testzip()
        if bad:
            raise RuntimeError('archive checksum failure: '+bad)
        count = len(package.infolist())
    record = dict(path=str(archive),bytes=archive.stat().st_size,files=count,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),crc_verified=True)
    with (root.parent/'轻量分组Or-opt_扩展验证_压缩包校验.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(archive=record,primary=primary,overall_B=b),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
