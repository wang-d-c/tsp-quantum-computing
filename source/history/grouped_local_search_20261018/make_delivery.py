"""Generate the short Chinese readout from audited metrics, then archive outputs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import zipfile


def main():
    study = Path(sys.argv[1]).resolve()
    results = json.loads((study/'analysis/results.json').read_text(encoding='utf-8'))
    audit = json.loads((study/'analysis/audit.json').read_text(encoding='utf-8'))
    handoff = json.loads((study/'handoff_audit.json').read_text(encoding='utf-8'))
    if not (audit['passed'] and handoff['passed']):
        raise RuntimeError('refusing a success readout for an invalid study')
    lines = ['三流程 TSP 同总时间预算：结果速读', '',
        '本轮结论：轻量分组 Or-opt 的平均收益极小；加入候选 3-opt 和错位分组后，整体结果更差。',
        '尚未证明分组预处理具有稳定、普遍的优势。', '',
        'A：最近邻贪心 → LKH。',
        'B：最近邻贪心 → 分组 Or-opt → LKH。',
        'C：最近邻贪心 → 分组 Or-opt＋候选纯 3-opt＋错位分组 → LKH。', '',
        '4 张既有合成图 × 3 个配对种子 × 3 条流程，共 36 次独立冷启动。',
        '每次 30 秒总墙钟预算，包括任务准备、启动与导入、贪心、距离矩阵、分组、',
        'Numba 首次编译、候选构建、组内优化、LKH 准备/搜索以及在线路线核验。',
        'LKH 相同配置，独立计时、相同配对贪心起点，不共享免费路线或编译结果。', '',
        '最终长度：每图三个种子的算术均值，越小越好；合成坐标 EUC_2D 单位。',
        '地图                 A                 B                 C']
    by_map = {m['map_id']: m for m in results['by_map']}
    for name in ('u1000', 'c1000', 'u3000', 'c3000'):
        values = [by_map[name]['methods'][m]['final_length']['mean'] for m in 'ABC']
        lines.append(name.ljust(8) + ''.join(f'{value:18,.2f}' for value in values))
    lines += ['', '相对 A 的配对平均长度差（负数更短）：']
    for method in 'BC':
        item = results['overall'][method]
        lines.append(f"{method}：{item['mean_relative_difference_percent_vs_A']:+.6f}%；"
                     f"胜/平/负 {item['wins']}/{item['ties']}/{item['losses']}。")
    lines += ['', '每次进入 LKH 准备前的冷启动累计耗时：']
    for method in 'ABC':
        item = results['overall'][method]
        lines.append(f"{method}：{item['mean_pre_lkh_cold_seconds']:.3f} 秒；"
                     f"LKH 平均可用预算 {item['mean_lkh_available_seconds']:.3f} 秒。")
    lines += ['', '组内模块确实缩短了初始贪心路线：']
    for method in 'BC':
        item = handoff['by_method'][method]
        lines.append(f"{method}：进入 LKH 前相对贪心平均缩短 "
                     f"{item['pre_lkh_improvement_percent']['mean']:.3f}%；"
                     f"组内前处理平均 {item['group_preprocessing_seconds']['mean']:.3f} 秒。")
    lines += ['这些初始路线改善没有转化成同等幅度的最终收益；本轮没有拆分时间减少与搜索起点改变的各自影响。']
    milestone_summary = {}
    lines += ['', '10 秒中间成绩（同样从总冷启动起计时）：']
    for method in 'BC':
        items = [p for p in results['paired'] if p['method'] == method]
        values = [p['milestones']['10.0']['relative_difference_percent'] for p in items]
        counts = Counter('win' if v < 0 else 'loss' if v > 0 else 'tie' for v in values)
        milestone_summary[method] = dict(mean_relative_difference_percent=statistics.fmean(values),
                                        wins=counts['win'], ties=counts['tie'], losses=counts['loss'])
        lines.append(f"{method}：相对 A 平均 {statistics.fmean(values):+.6f}%；"
                     f"胜/平/负 {counts['win']}/{counts['tie']}/{counts['loss']}。")
    lines += ['', '检验结果：',
        '6 项算子正确性测试通过；36 次运行全部完成。',
        f"独立重算 {sum(audit['independently_recomputed_route_counts'].values())} 份最终/中间/检查点路线。",
        '路线/截止审计与 LKH 输入/组内收益审计均通过，0 错误。',
        '无非法路线、无晚于截止计入的成绩、无进程清理失败。', '',
        '参数：每组最多 64 城市、对称全局 12 近邻候选、最多两轮，具体限制见协议。',
        '前处理停止目标为总时间 25%，不可中断操作可超目标，实际总预算仍硬截止；本轮没有目标超时。',
        '输入封存与离线审计不计入求解预算，操作系统文件缓存未清空。',
        '这些结果仅涉及四张既有合成图、三个种子和本轮参数/冷启动实现；不构成广泛优势或统计显著性结论。',
        '本轮没有执行量子计算。', '',
        '详细逐次对比：analysis/中文结果说明.txt、analysis/results.json。',
        '完整审计：analysis/audit.json、handoff_audit.json。',
        '输入、原始日志、源码与复现说明保留在本目录。']
    (study/'结果速读.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (study/'analysis/milestone_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(milestone_summary, stream, ensure_ascii=False, indent=2)
    archive = study.parent/'三流程TSP_同预算实验_结果与复现代码.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(study.rglob('*')):
            if path.is_file():
                package.write(path, arcname=study.name+'/'+path.relative_to(study).as_posix())
    with zipfile.ZipFile(archive) as package:
        bad = package.testzip()
        if bad:
            raise RuntimeError('archive CRC failure: '+bad)
        count = len(package.infolist())
    metadata = dict(archive=str(archive), files=count, bytes=archive.stat().st_size,
                    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), crc_verified=True)
    with (study.parent/'三流程TSP_压缩包校验.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    print(json.dumps(milestone_summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
