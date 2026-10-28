"""Standalone paired comparison, separating route availability from length."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs' / 'seed120_20261028'
ANALYSIS = STUDY / 'analysis'


def main():
    result = json.loads((ANALYSIS / 'results.json').read_text(encoding='utf-8'))
    if not result['complete']:
        raise RuntimeError('Full experiment required')
    with (ANALYSIS / 'per_map.csv').open(encoding='utf-8-sig', newline='') as stream:
        maps = list(csv.DictReader(stream))
    with (ANALYSIS / 'per_pair.csv').open(encoding='utf-8-sig', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    order = sorted(maps, key=lambda x: (x['stratum'], int(x['n']), x['map_id']))
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
    fig, (coverage, length) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.3]})
    y = np.arange(len(order))
    a = np.array([int(item['baseline_lkh_outputs']) for item in order])
    d = np.array([int(item['candidate_lkh_outputs']) for item in order])
    coverage.barh(y - .18, a, height=.34, color='#808b99', label='LKH')
    coverage.barh(y + .18, d, height=.34, color='#207a9e', label='深度 2-opt + LKH')
    coverage.set_yticks(y, [item['map_id'] for item in order])
    coverage.set_xlim(0, 11)
    coverage.set_xticks(range(0, 11, 2))
    coverage.set_xlabel('截止前观察到 LKH 路线输出的次数 / 10', fontproperties=font)
    coverage.set_title('30 秒内的 LKH 路线输出', fontproperties=font, pad=34)
    coverage.legend(prop=font, loc='upper left', bbox_to_anchor=(0, 1.045), ncol=2, frameon=False)
    values = []
    for item in order:
        subset = [float(pair['relative_percent']) for pair in pairs
                  if pair['map_id'] == item['map_id'] and pair['stage'] == 'both_outputs']
        values.append(float(np.mean(subset)) if subset else float('nan'))
    for index, value in enumerate(values):
        if np.isfinite(value):
            length.plot([0, value], [index, index], color='#9ba6b0', lw=2)
            length.scatter([value], [index], s=55, color='#207a9e' if value <= 0 else '#bd653e', zorder=3)
            length.annotate(f'{value:+.4f}%', (value, index), xytext=(6 if value >= 0 else -6, 0),
                            textcoords='offset points', va='center', ha='left' if value >= 0 else 'right', fontsize=9)
        else:
            length.text(.04, index, '无双方均输出配对', transform=length.get_yaxis_transform(),
                        va='center', color='#68727a', fontproperties=font)
    length.axvline(0, color='#424b54', lw=1)
    length.set_yticks(y, [])
    length.set_title('双方均有输出时的长度差', fontproperties=font, pad=12)
    length.set_xlabel('每图配对均值；负值表示混合方法更短（%）', fontproperties=font)
    length.margins(x=.55)
    for axis in (coverage, length):
        axis.invert_yaxis()
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle('120 配对种子：深度候选 2-opt 与原版 LKH', fontproperties=font, fontsize=16, y=.98)
    fig.text(.5, .015, '12 张旧测试图 × 10 个新配对种子；各 30 秒总预算。右图为条件性诊断，不能单独证明整体优势。',
             ha='center', fontproperties=font, fontsize=10, color='#555f68')
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(ANALYSIS / 'comparison.png', dpi=170)
    fig.savefig(ANALYSIS / 'comparison.svg')
    print(str(ANALYSIS / 'comparison.png'))


if __name__ == '__main__':
    main()
