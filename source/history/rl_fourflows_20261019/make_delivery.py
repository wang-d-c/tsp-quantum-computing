"""Package a complete audited study; never rewrites raw observations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    root=Path(sys.argv[1]).resolve()
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    if not (root/'complete.json').is_file():
        raise RuntimeError('formal schedule incomplete')
    result=read('analysis/results.json')
    if not result['audit_passed']:
        raise RuntimeError('scoring audit did not pass')
    training=read('models/training_report.json')
    lines=['四流程TSP强化学习验证：结果速读','',
           '12张测试地图×5个配对种子×4条流程=240次冷启动；每次30秒总墙钟上限。',
           'A：NN→LKH；B：固定规则短窗口与路线池→LKH；C：RL选择相同窗口/算子与路线池→LKH；D：RL选择候选2-opt→LKH。',
           'B/C/D也都先独立计费生成NN初始路线；LKH为同一3.0.14 ALPHA5配置。',
           '在线开销含模型加载、特征、推理、候选构造、路线池维护及全部精修。',
           '模型为轻量线性softmax策略，使用真实顺序rollout和REINFORCE训练；不是深度神经网络。',
           '训练160张、验证40张；五组独立模型与五个测试种子预先一一映射。',
           '训练与测试核心使用同一候选/算子/路线池语义，训练24步、测试最多128步；训练城市128..1000，测试含更大规模及公开图。','',
           '相对差=100×(方法长度−基线长度)/基线长度，负数更短；先平均每张图的配对种子差，再对图等权平均。',
           '三个vsA比较采用Bonferroni98.333%地图bootstrap区间控制多重比较；CvsB是次要机制对照。','']
    for method in ('B','C','D'):
        item=result['comparisons'][method+'_vs_A']
        b=item['bootstrap']
        lo,hi=b['bonferroni_three_comparisons']
        direction='支持样本中的平均缩短方向' if hi<0 else '支持样本中的平均变长方向' if lo>0 else '区间跨零，未确认稳定平均优势'
        lines += [f'{method}相对A：{b["mean"]:+.6f}%；95%区间{b["percentile95"]}；',
                  f'多比较区间[{lo:+.6f}%, {hi:+.6f}%]，{direction}。',
                  f'配对胜平负{item["pair_outcomes"]}；地图胜平负{item["map_outcomes"]}。',
                  f'预设输出阶段诊断（不剔除任何成绩）：{item["stage_diagnostic"]}。','']
    item=result['comparisons']['C_vs_B']
    lines += [f'C相对B（学习选择器的次要对照）：{item["bootstrap"]["mean"]:+.6f}%；95%区间{item["bootstrap"]["percentile95"]}。',
              '若学习流程超过A但未超过B，不能把共同局部搜索/路线池的收益全部归因于强化学习。','',
              '在线阶段成本（每方法60次均值，秒）：']
    for method,item in result['overall'].items():
        lines.append(f'{method}: 进入LKH前累计{item["mean_pre_lkh_cold_seconds"]}；模型加载{item["mean_policy_load_seconds"]}；分项{item["mean_search_costs"]}。')
        lines.append(f'状态{item["status_counts"]}；无LKH路线输出{item["zero_lkh_outputs"]}次。')
    lines += ['',f'离线训练全过程（含训练/验证输入准备）{training.get("end_to_end_seconds")}秒；',
              f'其中训练rollout/更新{training["training_seconds"]:.3f}秒，选checkpoint验证{training["validation_seconds"]:.3f}秒（完整时间还含uniform验证和日志等）。',
              '所有训练日志、验证记录、模型、坐标、原始计分与成本账本均保存。',
              '无量子计算；不能外推为超过调参后的所有LKH配置或否定其他强化学习架构。','',
              '完整逐图/逐种子与统计：analysis/中文结果说明.txt、analysis/results.json。',
              '计分核验：analysis/audit.json；训练/验证/测试分离：train_test_separation.json。',
              '复验：保持登记的LKH可执行文件与environment.json相同；安装记录的依赖版本。',
              'python -B STUDY/source/run.py --model-directory STUDY/models --input-directory STUDY/inputs --output NEW_STUDY --freeze-only',
              'python -B STUDY/source/run.py --execute-frozen NEW_STUDY',
              'python -B STUDY/source/analyze.py NEW_STUDY',
              '重训：python -B STUDY/source/train.py --output-dir NEW_MODELS',
              'NEW_STUDY和NEW_MODELS必须使用尚不存在的新位置，原始实验不得覆盖。']
    (root/'结果速读.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=root.parent/'四流程强化学习TSP_30秒同预算_结果与代码.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                stream.write(path,root.name+'/'+path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        count=len(stream.infolist())
    record=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=count,crc_verified=True)
    with (root.parent/'四流程强化学习TSP_压缩包校验.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    main()
