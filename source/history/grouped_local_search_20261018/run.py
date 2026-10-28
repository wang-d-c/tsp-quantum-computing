"""Serial, independently certified, equal cold-wall-clock TSP comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from uuid import uuid4

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LKH = Path('D:/My project/outputs/tsp_research/lkh3_baseline/LKH-3.exe')
LKH_SHA256 = 'b44414b7c9aa111b4e782ba13fd2cf4684bf3ad505ba1f5d65822b8ad162c044'
PREVIOUS_MAPS = Path('C:/Users/31226/Documents/Codex/2026-10-16/phd-3-0-985-phd-tsp-2/outputs/python_eax_v10_20261018T074000Z/runs/pilot-075837-0a84d137')
METHODS = ('A', 'B', 'C')
SETTINGS = dict(group_size=64, candidate_count=12, preprocessing_fraction_target=.25,
                max_rounds=2, or_moves_per_window=32, three_moves_per_window=8)
SOURCES = ('run.py', 'worker.py', 'supervisor.py', 'resource_v3.py',
           'group_search.py', 'test_group_search.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return dict(at=utc(), python=sys.version, executable=sys.executable,
                platform=platform.platform(), processor=platform.processor(),
                cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),
                logical_cpu_count=os.cpu_count(), lkh_binary=str(LKH),
                lkh_sha256=sha(LKH),
                versions={p: importlib.metadata.version(p) for p in
                          ('numpy', 'numba', 'psutil', 'pytest')},
                quantum_execution=False, revision='SHA256 source manifest; no git revision')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--input-directory', type=Path, default=PREVIOUS_MAPS)
    args = parser.parse_args()
    budget = args.seconds if args.seconds is not None else (12. if args.smoke else 30.)
    if budget < 3 or budget > 3600:
        parser.error('seconds must be in [3,3600]')
    if not LKH.is_file() or sha(LKH) != LKH_SHA256:
        raise RuntimeError('LKH binary absent or changed')
    output = (args.output or HERE / ('smoke-' + uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print('OUTPUT ' + str(output), flush=True)
    from supervisor import run_one

    sources = {name: sha(HERE / name) for name in SOURCES}
    package = output / 'source'
    package.mkdir()
    for name in SOURCES:
        shutil.copyfile(HERE / name, package / name)
    write_new(output / 'source_hashes.json', sources)
    write_new(output / 'environment.json', environment())
    datasets = []
    input_dir = output / 'inputs'
    input_dir.mkdir()
    if args.smoke:
        rng = random.Random(202610180987)
        coords = [[rng.randrange(100001), rng.randrange(100001)] for _ in range(256)]
        path = input_dir / 'smoke256.json'
        write_new(path, dict(id='smoke256', n=256, family='uniform', coords=coords))
        datasets.append(dict(id='smoke256', n=256, family='uniform', coords=coords,
                             path=str(path), sha256=sha(path)))
        seeds = [123]
    else:
        for label in ('u1000', 'c1000', 'u3000', 'c3000'):
            original = args.input_directory / (label + '.json')
            path = input_dir / original.name
            shutil.copyfile(original, path)
            item = json.loads(path.read_text(encoding='utf-8'))
            datasets.append(dict(id=label, n=len(item['coords']), family=item['family'],
                                 coords=item['coords'], path=str(path),
                                 sha256=sha(path), original_source=str(original),
                                 coordinate_seed=item.get('coordinate_seed')))
        seeds = [1, 2, 3]
    protocol = dict(schema=1, frozen_at=utc(), development_smoke=args.smoke,
        methods=dict(A='paid nearest-neighbor -> global LKH',
                     B='paid nearest-neighbor -> grouped Or-opt -> global LKH',
                     C='paid nearest-neighbor -> grouped Or-opt + candidate pure 3-opt + half-shifted windows -> global LKH'),
        seeds=seeds, total_budget_seconds=budget,
        milestones_seconds=sorted(set((min(10., budget), budget))), settings=SETTINGS,
        local_order='up to two rounds; unshifted Or-opt; C unshifted 3-opt, shifted Or-opt, shifted 3-opt',
        local_stopping='32 Or-opt improvements/window and 8 pure-3-opt improvements/window maximum; stop earlier at no admissible improvement or preprocessing cutoff',
        preprocessing_target='cooperative target at 25% of total cold time including common setup/NN, dense matrix, imports, JIT, candidate construction and local operations; indivisible matrix/import/JIT/kernel calls may overrun the target and all overrun is charged to the same hard total budget; actual overrun recorded; early convergence transfers remainder immediately to LKH',
        shared_nn='independently computed in each run; start random.Random(seed).randrange(n); EUC_2D nearest, city ID tie-break; no free route, matrix or compiled kernels',
        local_endpoints='fixed endpoints of each open path window; offset floor(group_size/2) with cyclic wrap; all moves strictly improving',
        lkh='same LKH-3.0.14 SHA256 and ALPHA5 config; INITIAL_TOUR_FILE, INITIAL_TOUR_FRACTION=1; no INPUT_TOUR_FILE or optimum',
        timing='cold supervisor origin before per-run setup/task IO/process startup/imports/NN/JIT/matrix/grouping/search/exports/parent validation; identical 0.20s stop guard; Windows Job hard total cutoff',
        score='complete Hamiltonian tours independently validated by parent before milestone; post-cutoff tour recovery prohibited',
        threading='serial jobs, single-thread BLAS/Numba settings; Windows Job process-tree cleanup verified',
        schedule_seed=202610181930, schedule='randomized paired map+seed blocks, randomized methods within each block',
        primary='paired final-length difference versus A at 30s; report wins/ties/losses, map-level means and all negative results',
        limitations='four previously studied synthetic maps; three paired solver/NN seeds per map; exploratory implementation comparison, not held-out or broad statistical proof',
        common_exclusions='copying/fixing common datasets and offline result audit; OS file cache not flushed; process cleanup overrun reported separately and produces no scored new solution',
        quantum_execution=False)
    write_new(output / 'frozen_protocol.json', protocol)
    write_new(output / 'datasets.json', dict(maps=[{k: v for k, v in m.items() if k != 'coords'}
                                                for m in datasets], registered_before_first_formal_job=True))
    blocks = [(item, seed) for item in datasets for seed in seeds]
    rng = random.Random(protocol['schedule_seed'])
    rng.shuffle(blocks)
    schedule = []
    for item, seed in blocks:
        methods = list(METHODS)
        rng.shuffle(methods)
        for method in methods:
            schedule.append((item, seed, method))
    write_new(output / 'schedule.json', [dict(job=i+1, map_id=item['id'], seed=seed, method=method)
                                         for i, (item, seed, method) in enumerate(schedule)])
    started = time.perf_counter()
    for index, (item, seed, method) in enumerate(schedule, 1):
        if any(sha(HERE / name) != value for name, value in sources.items()):
            raise RuntimeError('frozen source changed; refusing mixed-version study')
        if any(sha(m['path']) != m['sha256'] for m in datasets):
            raise RuntimeError('frozen input changed')
        print(f'START {index}/{len(schedule)} {item["id"]} seed={seed} method={method}', flush=True)
        row = run_one(method, item['coords'], seed, budget, output / f'j{index:03d}', str(LKH),
                      marks=protocol['milestones_seconds'], settings=SETTINGS)
        row.update(map_id=item['id'], family=item['family'], job=index)
        write_new(output / f'row{index:03d}.json', row)
        worker = row.get('worker_result') or {}
        print(f'END {index}/{len(schedule)} status={row["status"]} length={row["length"]}'
              f' pre={worker.get("pre_lkh_length")} pre_seconds={worker.get("pre_lkh_cold_seconds")}'
              f' invalid={row["invalid_checkpoints"]} cleanup={row["containment_cleanup_passed"]}', flush=True)
        if not row['containment_cleanup_passed']:
            raise RuntimeError('uncertified cleanup; stopping sequential batch')
    write_new(output / 'complete.json', dict(at=utc(), jobs=len(schedule),
              batch_elapsed_seconds=time.perf_counter()-started,
              source_hashes_unchanged=all(sha(HERE / n) == h for n, h in sources.items())))
    print('COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    main()
