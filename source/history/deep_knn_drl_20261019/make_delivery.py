"""Package an audited immutable study, excluding offline training dependencies."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    study = Path(sys.argv[1]).resolve()
    read = lambda p: json.loads((study / p).read_text(encoding='utf-8'))
    complete = read('complete.json')
    result = read('analysis/results.json')
    if complete['jobs'] != 180 or not result['audit_passed']:
        raise RuntimeError('incomplete or unaudited formal study')
    for gate in ('prospective_design_audit.json', 'models/training_independent_audit.json',
                 'independent_review/independent_audit.json'):
        if not read(gate).get('passed'):
            raise RuntimeError('required independent gate did not pass: ' + gate)
    training = read('models/training_report.json')
    lines = ['深度强化学习kNN候选选择：30秒总预算实验', '',
             'A普通原版LKH，正常WALK初始化，不提供外部路线。',
             'U先付费生成NN路线，再均匀随机选择候选2-opt、维护小路线池，最后LKH。',
             'D与U搜索环境相同，用独立训练的深度策略选择候选；最后LKH。',
             '学习发生在LKH之前，没有修改LKH内部候选集/搜索。无量子计算。',
             '两层64维4头稀疏kNN注意力、非线性actor、学习式critic，共124290参数。',
             '4步actor-critic、gamma=.99、Adam，128张训练图、32张验证图，五个训练种子；验证集选检查点，测试集不参与训练或选择。',
             '12张独立测试图×5配对种子×3流程=180次冷启动；每次总墙钟30秒。',
             '预算含进程启动、NN、模型加载、矩阵、kNN、编码、动态特征/推理、路线池、路线核验与LKH。',
             '负相对差表示路线更短；图内平均配对种子差后，对12张地图等权平均。',
             'D-A与D-U采用整图分层bootstrap的Bonferroni97.5%区间；U-A仅描述。', '']
    for label in ('D_vs_A', 'D_vs_U', 'U_vs_A'):
        item = result['comparisons'][label]
        boot = item['bootstrap']
        interval = boot['bonferroni_two_comparisons'] if label.startswith('D') else boot['percentile95']
        lines += [f'{label}: 平均长度变化{boot["mean"]:+.6f}%；区间{interval}。',
                  f'配对胜平负{item["pair_outcomes"]}；地图胜平负{item["map_outcomes"]}。',
                  f'输出阶段描述（不剔除任何成绩）{item["stage_diagnostic"]}。', '']
    lines += ['普通LKH无完整路线输出时，共同合法identity兜底可能明显弱于NN；因此D-A不能单独证明RL收益。',
              'D-U才控制共同的NN、候选、算子和路线池。无LKH输出与有输出结果必须分开解释，条件子组不替代主要终点。',
              '五个种子同时绑定训练模型、NN起点及LKH随机性，地图区间条件于这五个固定组合。', '', '在线平均成本（秒）：']
    for method, cost in result['overall'].items():
        lines += [f'{method}: 进入LKH前{cost["mean_pre_lkh_cold_seconds"]}；加载{cost["mean_policy_load_seconds"]}；搜索分项{cost["mean_search_costs"]}；',
                  f'状态{cost["status_counts"]}；无LKH路线输出{cost["zero_lkh_outputs"]}次；最终来源{cost["final_sources"]}。']
    lines += ['', f'离线训练/验证全流程{training["end_to_end_seconds"]:.3f}秒，单独记录；训练{training["training_seconds"]:.3f}秒，检查点验证{training["validation_seconds"]:.3f}秒。',
              '这是论文启发的稀疏改造：共享静态几何编码器、动态当前路线解码、小路线池、只接受改进动作、有限128步目标。',
              '原论文全图注意力、位置编码、允许变差动作、训练规模等不同，不能称完整复现，也不能据此否定所有深度RL或所有LKH配置。',
              '完整逐图原始长度、逐种子、统计和在线成本在analysis/；全部训练日志、权重、输入在models/。',
              '复验使用新的空输出目录，命令与依赖见实验设置与复现.txt。训练需torch2.14.1+cpu；在线仅需记录版本的NumPy与psutil。']
    (study / '结果速读.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    archive = study.parent / '深度强化学习kNN_TSP_30秒_结果与代码.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for p in sorted(study.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                stream.write(p, study.name + '/' + p.relative_to(study).as_posix())
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        count = len(stream.infolist())
    record = dict(path=str(archive), bytes=archive.stat().st_size,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  files=count, crc_verified=True)
    target = study.parent / '深度强化学习kNN_TSP_压缩包校验.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=True))


if __name__ == '__main__':
    main()
