"""Prospectively frozen A/B replication; unchanged solver and timing engines."""
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

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LKH = Path('D:/My project/outputs/tsp_research/lkh3_baseline/LKH-3.exe')
LKH_SHA256 = 'b44414b7c9aa111b4e782ba13fd2cf4684bf3ad505ba1f5d65822b8ad162c044'
METHODS = ('A', 'B')
SEEDS = (11, 23, 37, 53, 71)
SETTINGS = dict(group_size=64, candidate_count=12, preprocessing_fraction_target=.25,
                max_rounds=2, or_moves_per_window=32, three_moves_per_window=8)
ENGINE = ('worker.py', 'supervisor.py', 'resource_v3.py', 'group_search.py', 'test_group_search.py')
SOURCES = ('run.py', *ENGINE, 'prepare_inputs.py')
BASELINE = HERE.parent.parent / 'outputs/grouped_local_search_20261018/source'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-directory', type=Path, default=HERE/'datasets')
    parser.add_argument('--baseline-source', type=Path, default=BASELINE)
    args = parser.parse_args()
    budget = 30.
    if sha(LKH) != LKH_SHA256:
        raise RuntimeError('LKH binary changed')
    # The previous complete experiment is preserved. No algorithm retuning here.
    engine_hashes = {name: sha(HERE/name) for name in ENGINE}
    if any(sha(args.baseline_source/name) != value for name, value in engine_hashes.items()):
        raise RuntimeError('replication solver/timing engine differs from previous experiment')
    inputs = args.input_directory.resolve()
    manifest = json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
    registered = manifest['maps']
    if len(registered) != 12 or len({m['id'] for m in registered}) != 12:
        raise RuntimeError('requires the fixed twelve input instances')
    from collections import Counter
    if Counter(m['stratum'] for m in registered) != dict(synthetic_uniform=4, synthetic_clustered=4, tsplib=4):
        raise RuntimeError('unexpected input strata')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print('OUTPUT '+str(output), flush=True)
    source = output/'source'
    source.mkdir()
    hashes = {name: sha(HERE/name) for name in SOURCES}
    for name in SOURCES:
        shutil.copyfile(HERE/name, source/name)
    write_new(output/'source_hashes.json', hashes)
    write_new(output/'engine_inheritance.json', dict(previous_source=str(args.baseline_source.resolve()),
              byte_identical_engine_hashes=engine_hashes, settings_unchanged=True,
              current_analysis_not_used_to_retune=True))
    write_new(output/'environment.json', dict(at=utc(), python=sys.version, executable=sys.executable,
        platform=platform.platform(), processor=platform.processor(),
        cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER'), logical_cpu_count=os.cpu_count(),
        lkh_binary=str(LKH), lkh_sha256=sha(LKH),
        versions={p: importlib.metadata.version(p) for p in
                  ('numpy','numba','llvmlite','psutil','pytest','matplotlib')},
        quantum_execution=False, revision='SHA256 source manifest; no git revision'))
    packaged_inputs = output/'inputs'
    packaged_inputs.mkdir()
    raw = inputs/'raw_tsp'
    if raw.is_dir():
        shutil.copytree(raw, packaged_inputs/'raw_tsp')
    shutil.copyfile(inputs/'manifest.json', output/'input_generation_manifest.json')
    datasets = []
    for original in registered:
        item = dict(original)
        path = inputs/(item['id']+'.json')
        if sha(path) != item['sha256']:
            raise RuntimeError('registered input hash changed')
        destination = packaged_inputs/path.name
        shutil.copyfile(path, destination)
        content = json.loads(destination.read_text(encoding='utf-8'))
        if content['n'] != item['n'] or len(content['coords']) != item['n']:
            raise RuntimeError('input dimensions differ')
        item.update(path=str(destination), coords=content['coords'], original_source=str(path))
        datasets.append(item)
    protocol = dict(schema=2, frozen_at=utc(), development_smoke=False,
        methods=list(METHODS),
        method_descriptions=dict(A='paid nearest-neighbor -> global LKH',
                                 B='paid nearest-neighbor -> grouped Or-opt -> global LKH'),
        seeds=list(SEEDS), total_budget_seconds=budget, milestones_seconds=[10.,30.],
        settings=SETTINGS, independent_maps=12, paired_blocks=60, planned_jobs=120,
        data_status='all twelve maps new relative to previous four-map comparison; eight new synthetic draws and four established public benchmarks, not claimed unseen in all historical projects',
        strata=dict(synthetic_uniform=4,synthetic_clustered=4,tsplib=4),
        algorithm_change='none; worker, supervisor, resource helper, grouped search and operator tests byte-identical to previous experiment',
        local_order='B up to two unshifted Or-opt passes, strings1..3, fixed open-window endpoints, 32 strict improvements/window/pass; no 3-opt or shifted windows',
        preprocessing_target='cooperative stop target at 25% total cold time, including common startup/NN, matrix, imports, cold JIT and local search; indivisible calls may overrun and all overrun debits the same hard total budget',
        shared_nn='paid independent initialization in every run; start random.Random(seed).randrange(n), EUC_2D nearest and city-ID tie-break; same paired NN and solver seeds; no shared free route/matrix/JIT',
        lkh='unchanged LKH-3.0.14 ALPHA5 configuration; INITIAL_TOUR_FILE with INITIAL_TOUR_FRACTION=1, no INPUT_TOUR_FILE, no known optimum input',
        timing='unchanged cold supervisor origin before per-run setup/input IO/launch/imports/NN/matrix/JIT/grouping/Or-opt/LKH preparation/search and online independent certification; Windows Job hard total cutoff; same0.20s stop guard',
        score='only full routes independently certified before milestone; no post-cutoff best.tour recovery',
        schedule_seed=20261018239901,
        schedule='serial; shuffled map+seed blocks, A/B adjacent, exactly30AB and30BA orders randomly assigned',
        primary='mean paired percent length difference B-A within each map, then equal-weight mean across12maps; negative is shorter',
        uncertainty='20000 fixed-seed map-level stratified bootstrap draws (seed202610182399), retain four maps per each of three strata; report percentile95% interval; five seeds are within-map repetitions, not independent maps',
        engineering_margin_percent=.01,
        interpretation='report statistical direction separately from a predeclared engineering margin+-0.01%; CI crossing0 means inconclusive, not proof of equivalence; interval wholly within margin supports descriptive practical equivalence under this threshold, not a formal TOST claim',
        secondary='paired/map wins-ties-losses, median,10s checkpoints, initialization gain and all phase costs; no adaptation or early stopping based on intermediate results',
        failures='keep failures and incumbent fallback scores; no selective reruns, deletion or extra seeds after viewing outcomes; stop only if process containment fails',
        threading='serial solvers, single-thread BLAS/Numba environment, no affinity retuning',
        common_exclusions='input generation/conversion/sealing, offline analysis and cleanup after scoring; OS file cache not flushed; cleanup overrun disclosed',
        scope='fixed implementation/configuration cold30s replication; cannot establish superiority over tuned native LKH2/LKH3 configurations or all TSP distributions',
        quantum_execution=False)
    write_new(output/'frozen_protocol.json', protocol)
    write_new(output/'datasets.json', dict(maps=[{k:v for k,v in m.items() if k!='coords'} for m in datasets],
              registered_before_first_formal_job=True))
    rng = random.Random(protocol['schedule_seed'])
    blocks = [(item,seed) for item in datasets for seed in SEEDS]
    rng.shuffle(blocks)
    orders = [METHODS]*30+[METHODS[::-1]]*30
    rng.shuffle(orders)
    schedule = [(item,seed,method) for (item,seed),order in zip(blocks,orders) for method in order]
    write_new(output/'schedule.json', [dict(job=i+1,map_id=item['id'],seed=seed,method=method)
                                       for i,(item,seed,method) in enumerate(schedule)])
    from supervisor import run_one
    started = time.perf_counter()
    for index,(item,seed,method) in enumerate(schedule,1):
        if any(sha(HERE/name)!=value for name,value in hashes.items()):
            raise RuntimeError('frozen source changed')
        if any(sha(m['path'])!=m['sha256'] for m in datasets):
            raise RuntimeError('frozen input changed')
        print(f'START {index}/120 {item["id"]} seed={seed} method={method}',flush=True)
        row = run_one(method,item['coords'],seed,budget,output/f'j{index:03d}',str(LKH),
                      marks=(10.,30.),settings=SETTINGS)
        row.update(map_id=item['id'],family=item['family'],stratum=item['stratum'],job=index)
        write_new(output/f'row{index:03d}.json',row)
        worker = row.get('worker_result') or {}
        print(f'END {index}/120 status={row["status"]} length={row["length"]}'
              f' pre={worker.get("pre_lkh_length")} pre_seconds={worker.get("pre_lkh_cold_seconds")}'
              f' invalid={row["invalid_checkpoints"]} cleanup={row["containment_cleanup_passed"]}',flush=True)
        if not row['containment_cleanup_passed']:
            raise RuntimeError('uncertified cleanup, stopping batch')
    write_new(output/'complete.json',dict(at=utc(),jobs=len(schedule),
              batch_elapsed_seconds=time.perf_counter()-started,
              source_hashes_unchanged=all(sha(HERE/n)==h for n,h in hashes.items())))
    print('COMPLETE '+str(output),flush=True)


if __name__=='__main__':
    main()
