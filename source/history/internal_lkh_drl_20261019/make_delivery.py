"""Package completed, independently verified evidence without runtime dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--independent-directory', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    study, independent, archive = args.study.resolve(), args.independent_directory.resolve(), args.archive.resolve()
    if archive.is_relative_to(study) or archive.exists() or archive.with_suffix('.verification.json').exists():
        raise ValueError('archive must be new and outside the evidence directory')
    if any((study/name).exists() for name in ('简单结果.txt', 'delivery_manifest.json')):
        raise ValueError('refusing to overwrite existing delivery evidence')
    results = read(study/'analysis/results.json')
    review = read(independent/'supplemental_results.json')
    independent_audit = read(independent/'independent_audit.json')
    if (read(study/'complete.json')['jobs'] != 180 or not results['audit_passed']
            or read(study/'analysis/audit.json')['passed'] is not True
            or independent_audit['passed'] is not True
            or review['audit_passed'] is not True):
        raise ValueError('complete, passing primary and independent audits required')
    for ledger in (read(study/'freeze_complete.json')['sealed_file_hashes'],
                   read(study/'analysis/audit.json')['evidence_hashes'],
                   independent_audit['evidence_sha256']):
        for relative, expected in ledger.items():
            path = Path(relative)
            if not path.is_absolute():
                path = study/path
            if sha(path) != expected:
                raise ValueError('evidence changed after audit: '+relative)
    for label in ('D_vs_A', 'D_vs_U', 'U_vs_A'):
        first, second = results['comparisons'][label], review['final_comparisons'][label]
        for key in ('mean', 'percentile95'):
            values = [first['bootstrap'][key], second['bootstrap'][key]]
            if key == 'mean':
                values = [[v] for v in values]
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(*values)):
                raise ValueError('independent numerical disagreement: '+label+'/'+key)
        if label.startswith('D_'):
            if first['bootstrap']['bonferroni_two_comparisons'] != second['bootstrap']['bonferroni_two_comparisons']:
                raise ValueError('adjusted interval disagreement: '+label)
        if first['pair_outcomes'] != second['pair_outcomes']:
            raise ValueError('independent win/tie/loss disagreement: '+label)
    if not independent.is_relative_to(study):
        target = study/'independent_review'
        shutil.copytree(independent, target)
    else:
        target = independent
    method = Path(__file__).parent/'方法与复现说明.txt'
    if not method.is_file():
        method = study/'方法与复现说明.txt'
    if not method.is_file():
        raise ValueError('method notes missing')
    destination = study/method.name
    if destination.is_file():
        if sha(method) != sha(destination):
            raise ValueError('method notes differ from frozen evidence')
    else:
        shutil.copyfile(method, destination)
    training = read(study/'models/training_report.json')
    main_result = results['comparisons']['D_vs_A']
    estimate = main_result['bootstrap']
    low, high = estimate['bonferroni_two_comparisons']
    verdict = ('本次实现稳定超过普通LKH。' if high < 0 else
               '本次实现比普通LKH更差。' if low > 0 else '本次实现未能稳定超过普通LKH。')
    summary = ['简单结果：'+verdict, '',
               f'独立测试：12个实例×5个配对随机种子，每次总预算30秒，共180次。',
               f'深度RL内部候选相对普通LKH：路线长度平均变化{estimate["mean"]:+.4f}%。',
               f'调整后区间：[{low:+.4f}%, {high:+.4f}%]，负数更短。',
               f'60次配对：胜{main_result["pair_outcomes"]["wins"]}，平{main_result["pair_outcomes"]["ties"]}，负{main_result["pair_outcomes"]["losses"]}。',
               '模型直接在LKH内部选择和排序候选边；模型加载、推理、候选维护和在线验算全部计时。',
               f'五个模型的离线训练、基线准备及验证总耗时{training["end_to_end_seconds"]/60:.2f}分钟，另计。',
               '已完成独立审计，未剔除失败、超时或无LKH输出运行。', '',
               '详细逐图结果：analysis/中文结果说明.txt；原始统计：analysis/results.json。',
               '独立候选边与截止计分核验：'+str(target.relative_to(study))+'/独立核验与统计补充.txt。',
               '代码、模型、训练轨迹、原始运行记录、原生源代码与二进制均在本包。',
               '结论限于本次小型深度网络、静态候选策略、固定实例与30秒预算。']
    (study/'简单结果.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8')
    entries = sorted(p for p in study.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
    manifest = {p.relative_to(study).as_posix():dict(bytes=p.stat().st_size, sha256=sha(p)) for p in entries}
    write(study/'delivery_manifest.json', dict(at=datetime.now(timezone.utc).isoformat(),
        files=manifest, excludes='this manifest itself; Python bytecode caches',
        independent_numerical_agreement=True))
    entries.append(study/'delivery_manifest.json')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for path in entries:
            bundle.write(path, path.relative_to(study).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        damaged = bundle.testzip()
        if damaged is not None:
            raise ValueError('archive CRC failed: '+damaged)
        if set(bundle.namelist()) != set(manifest)|{'delivery_manifest.json'}:
            raise ValueError('archive members differ from manifest')
        for name, expected in manifest.items():
            digest = hashlib.sha256()
            with bundle.open(name) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError('archived evidence hash differs: '+name)
    report = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                  members=len(entries), crc_passed=True, all_manifest_hashes_passed=True,
                  independent_numerical_agreement=True)
    write(archive.with_suffix('.verification.json'), report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
