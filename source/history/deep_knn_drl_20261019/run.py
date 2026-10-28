"""Freeze and run ordinary LKH and paired candidate-search flows without test tuning.

Formal: --model-directory MODELS --output NEW_STUDY [--freeze-only]
After review: --execute-frozen FROZEN_STUDY
Development: add --smoke [--seconds 12 --smoke-cities 256].
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASELINE = (HERE.parent / 'baseline_reference' if (HERE.parent / 'baseline_reference').is_dir()
            else HERE.parent.parent / 'outputs/oropt_replication_20261018/source')
LKH = Path('D:/My project/outputs/tsp_research/lkh3_baseline/LKH-3.exe')
LKH_SHA256 = 'b44414b7c9aa111b4e782ba13fd2cf4684bf3ad505ba1f5d65822b8ad162c044'
METHODS = ('A', 'U', 'D')
SEEDS = (101, 211, 307, 401, 503)
LATIN_ORDERS = tuple(itertools.permutations(METHODS))
SETTINGS = dict(preprocessing_fraction_target=.25, local_max_steps=128,
                pool_size=3, d_candidate_count=192, d_exact_quota=1, knn=16,
                audit_decisions=128)
REQUIRED_SOURCE = ('run.py','worker.py','prepare_inputs.py','supervisor.py',
                   'resource_v3.py','search_core.py','deep_search.py','neural_model.py','train_deep.py')



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate_sha(coords):
    return hashlib.sha256(json.dumps(coords, separators=(',', ':'),
                                     ensure_ascii=True).encode('ascii')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def files_under(directory):
    return sorted(path for path in directory.rglob('*') if path.is_file()
                  and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'))


def collect_coordinate_hashes(value):
    found = set()
    if isinstance(value, dict):
        if isinstance(value.get('coordinate_sha256'), str):
            found.add(value['coordinate_sha256'])
        for child in value.values():
            found.update(collect_coordinate_hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_coordinate_hashes(child))
    return found


def safe_relative(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('model path escapes its root or is missing: ' + str(value))
    return path.relative_to(root.resolve()).as_posix()


def model_registration(directory, seeds, formal):
    registry_path = directory / 'model_registry.json'
    if registry_path.is_file():
        registry = read(registry_path)
        mapping = registry['by_test_seed']
    else:
        if formal:
            raise RuntimeError('formal study requires model_registry.json with five independently trained policy seeds')
        registry = dict(schema=1, development_shared_models=True)
        mapping = {str(seed):dict(D='model_D.json') for seed in seeds}
    selected = {}
    for seed in seeds:
        record = mapping[str(seed)]
        selected[str(seed)] = {method:safe_relative(directory,record[method]) for method in ('D',)}
    models = {record['D'] for record in selected.values()}
    if formal and len(models) != len(SEEDS):
        raise RuntimeError('formal study requires five distinct trained deep policy models')
    if formal:
        training_seeds = [mapping[str(seed)].get('training_seed') for seed in seeds]
        if (any(type(seed) is not int for seed in training_seeds)
                or len(set(training_seeds)) != len(SEEDS)):
            raise RuntimeError('formal study requires five recorded distinct training seeds')
        if len({sha(directory / model) for model in models}) != len(SEEDS):
            raise RuntimeError('independently trained model bytes must be distinct')
    manifests = sorted(path for path in files_under(directory)
                       if path.name in ('train_manifest.json','validation_manifest.json'))
    train_hashes, validation_hashes = set(), set()
    for path in manifests:
        values = collect_coordinate_hashes(read(path))
        (train_hashes if path.name == 'train_manifest.json' else validation_hashes).update(values)
    if not train_hashes or not validation_hashes:
        raise RuntimeError('training and validation canonical-coordinate manifests are required')
    if train_hashes & validation_hashes:
        raise RuntimeError('training and validation map hashes overlap')
    reports = [path for path in files_under(directory) if path.name == 'training_report.json']
    if not reports:
        raise RuntimeError('training_report.json missing; training costs must be disclosed')
    return dict(registry=registry, selected=selected, independent_policy_models=len(models),
                train_coordinate_sha256=sorted(train_hashes), validation_coordinate_sha256=sorted(validation_hashes),
                manifest_files=[p.relative_to(directory).as_posix() for p in manifests],
                training_report_files=[p.relative_to(directory).as_posix() for p in reports])


def development_maps(n):
    rng = random.Random(20261019098901)
    coords = [[rng.randrange(100001),rng.randrange(100001)] for _ in range(n)]
    return [dict(id='development_u'+str(n), n=n, family='uniform',stratum='development',
                 coordinate_seed=20261019098901,coordinate_sha256=coordinate_sha(coords),
                 edge_weight_type='EUC_2D',coords=coords,
                 development_only=True,not_a_formal_test_map=True)]


def registered_maps(directory):
    manifest = read(directory/'manifest.json')
    registered = manifest['maps']
    if len(registered)!=12 or len({item['id'] for item in registered})!=12:
        raise RuntimeError('formal study requires exactly twelve unique frozen maps')
    if Counter(item['stratum'] for item in registered) != dict(synthetic_uniform=4,synthetic_clustered=4,tsplib=4):
        raise RuntimeError('formal strata must each contain four maps')
    result = []
    for item in registered:
        path = directory/(item['id']+'.json')
        if sha(path)!=item['sha256']:
            raise RuntimeError('registered dataset bytes changed')
        value = read(path)
        coords = value['coords']
        if value['n']!=len(coords) or value['n']!=item['n']:
            raise RuntimeError('dataset dimension differs')
        if any(len(p)!=2 or any(type(v) is not int for v in p) for p in coords):
            raise RuntimeError('dataset has noninteger coordinates')
        digest = coordinate_sha(coords)
        if digest!=item['coordinate_sha256'] or value.get('coordinate_sha256')!=digest:
            raise RuntimeError('canonical coordinate hash differs')
        result.append(dict(item,coords=coords,original_source=str(path.resolve())))
    expected_public={'rl1304','rl1323','rl1889','u1432'}
    if {item['id'] for item in registered if item['stratum']=='tsplib'}!=expected_public:
        raise RuntimeError('public instances differ from prospectively selected integer EUC_2D inputs')
    for family in ('uniform','clustered'):
        if sorted(item['n'] for item in registered if item['family']==family)!=[1000,3000,5000,8000]:
            raise RuntimeError('synthetic city counts differ from the fixed design')
    return result,manifest


def freeze(args):
    formal = not args.smoke
    budget = 30. if formal else (12. if args.seconds is None else args.seconds)
    if formal and args.seconds is not None and args.seconds!=30.:
        raise ValueError('formal budget is fixed at 30 seconds')
    if not math.isfinite(budget) or budget<3 or budget>3600:
        raise ValueError('budget must be finite and in [3,3600]')
    if args.smoke_cities<16 or args.smoke_cities>512:
        raise ValueError('development graph size must be in [16,512]')
    if sha(LKH)!=LKH_SHA256:
        raise RuntimeError('registered LKH executable changed')
    for name in REQUIRED_SOURCE:
        if not (HERE/name).is_file():
            raise RuntimeError('required source missing: '+name)
    engine_hashes={name:sha(HERE/name) for name in ('supervisor.py','resource_v3.py')}
    if any(sha(BASELINE/name)!=value for name,value in engine_hashes.items()):
        raise RuntimeError('containment/timing engine differs from prior validated implementation')
    seeds = list(SEEDS) if formal else [101]
    model_directory = args.model_directory.resolve()
    registration = model_registration(model_directory,seeds,formal)
    if formal:
        datasets,input_manifest = registered_maps(args.input_directory.resolve())
    else:
        datasets=development_maps(args.smoke_cities)
        input_manifest=dict(development_only=True,no_formal_inputs_used=True,maps=[{k:v for k,v in d.items() if k!='coords'} for d in datasets])
    test_hashes={item['coordinate_sha256'] for item in datasets}
    training_hashes=set(registration['train_coordinate_sha256'])
    validation_hashes=set(registration['validation_coordinate_sha256'])
    if test_hashes & (training_hashes|validation_hashes):
        raise RuntimeError('test map canonical-coordinate hashes overlap training or validation')
    if len(test_hashes)!=len(datasets):
        raise RuntimeError('duplicate test coordinates')
    output=args.output.resolve()
    if output.is_relative_to(model_directory):
        raise ValueError('output cannot be inside the training model directory')
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source'
    source.mkdir()
    source_hashes={path.name:sha(path) for path in sorted(HERE.glob('*.py'))}
    for name in source_hashes:
        shutil.copyfile(HERE/name,source/name)
    write_new(output/'source_hashes.json',source_hashes)
    write_new(output/'engine_inheritance.json',dict(previous_source=str(BASELINE),byte_identical_engine_hashes=engine_hashes))
    baseline_reference=output/'baseline_reference'
    baseline_reference.mkdir()
    for name in engine_hashes:
        shutil.copyfile(BASELINE/name,baseline_reference/name)
    frozen_models=output/'models'
    frozen_models.mkdir()
    model_hashes={}
    for original in files_under(model_directory):
        relative=original.relative_to(model_directory)
        destination=frozen_models/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
        model_hashes[relative.as_posix()]=sha(destination)
    write_new(output/'model_hashes.json',model_hashes)
    policy_mapping={seed:{method:str((frozen_models/path).resolve()) for method,path in entry.items()}
                    for seed,entry in registration['selected'].items()}
    settings=dict(SETTINGS,policy_by_test_seed=policy_mapping)
    write_new(output/'train_test_separation.json',dict(
        coordinate_hash_rule="sha256(json.dumps(coords,separators=(',',':'),ensure_ascii=True).encode('ascii'))",
        train_coordinate_sha256=sorted(training_hashes),validation_coordinate_sha256=sorted(validation_hashes),
        test_coordinate_sha256=sorted(test_hashes),train_validation_overlap=[],train_test_overlap=[],validation_test_overlap=[],
        manifest_files=registration['manifest_files'],training_report_files=registration['training_report_files'],
        independent_policy_models=registration['independent_policy_models'],model_registry=registration['registry'],
        checked_before_first_formal_job=True))
    packaged=output/'inputs'
    packaged.mkdir()
    write_new(packaged/'manifest.json',input_manifest)
    if formal and (args.input_directory/'raw_tsp').is_dir():
        shutil.copytree(args.input_directory/'raw_tsp',packaged/'raw_tsp')
    for item in datasets:
        destination=packaged/(item['id']+'.json')
        if formal:
            shutil.copyfile(item['original_source'],destination)
        else:
            write_new(destination,item)
        item.update(path=str(destination),sha256=sha(destination))
    write_new(output/'input_generation_manifest.json',input_manifest)
    write_new(output/'datasets.json',dict(maps=[{k:v for k,v in item.items() if k!='coords'} for item in datasets],registered_before_first_formal_job=True))
    write_new(output/'environment.json',dict(at=utc(),python=sys.version,executable=sys.executable,
        platform=platform.platform(),processor=platform.processor(),cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),
        logical_cpu_count=os.cpu_count(),lkh_binary=str(LKH),lkh_sha256=sha(LKH),
        versions={name:importlib.metadata.version(name) for name in ('numpy','psutil')},
        quantum_execution=False))
    protocol=dict(schema=4,frozen_at=utc(),development_smoke=args.smoke,
        methods=list(METHODS),method_descriptions=dict(
            A='ordinary stock LKH normal initialization; no NN and no supplied initial tour; certified identity fallback only if no better complete route is exported in time',
            U='paid nearest neighbor -> uniform selection of kNN candidate 2-opt with bounded diverse full-route pool -> LKH',
            D='paid nearest neighbor -> trained deep attention RL selection from the identical U kNN candidate 2-opt and full-route pool implementation -> LKH'),
        seeds=seeds,total_budget_seconds=budget,milestones_seconds=sorted(set((min(10.,budget),budget))),
        settings=settings,independent_maps=len(datasets),paired_blocks=len(datasets)*len(seeds),planned_jobs=len(datasets)*len(seeds)*len(METHODS),
        training='offline deep actor-critic policy; train and validation maps synthetic and coordinate-hash disjoint from tests; five independently initialized/trained policies mapped prospectively to five test/NN/LKH seeds; checkpoints selected only on validation; all training costs disclosed separately',
        test_status='eight fresh synthetic draws and rl1304/rl1323/rl1889/u1432; public inputs are established external benchmarks, distinct from all previous formal maps in this workspace and not claimed globally unseen',
        prospective_public_replacements='fl1400/vm1748/u1817/u1060/d1291 failed exact-integer Decimal validation and were rejected before any solver without rounding/scaling',
        u_d_control='same candidate generator, step cap, exact-evaluation quota, move acceptance, parent selection and diverse pool; U samples uniformly and D samples trained-policy probabilities; model load/features/inference debit D; proposals and selector RNG streams separated',
        preprocessing_target='cooperative absolute target at 25% total cold time including launch/import/NN, model load, matrix, KNN, features, inference, pool and local operations; indivisible overrun debits hard total budget',
        shared_nn='U and D independently pay random.Random(seed).randrange(n), EUC_2D distance and city-ID ties; A uses stock LKH normal initialization and does not receive a paid/free NN route',
        lkh='same byte-registered stock LKH-3.0.14 ALPHA5 and shared fixed search settings; A has NO INITIAL_TOUR_FILE or INPUT_TOUR_FILE; U/D use INITIAL_TOUR_FILE and INITIAL_TOUR_FRACTION=1; no known optimum; common0.20-second return guard',
        timing='30-second cold supervisor origin before task creation/IO/import/startup/NN/model/matrix/KNN/features/search/pool/LKH and parent online certification; Windows Job hard total cutoff, serial single-thread jobs',
        score='only complete Hamiltonian routes independently validated by parent strictly before milestone; no post-cutoff raw best.tour recovery; initial identity fallback is paid and identically available to every arm',
        schedule_seed=20261019050001,schedule='randomized paired map+seed blocks; all6 permutations of3 methods appear10times across60blocks, balancing positions and directed adjacencies',
        primary='D-A: per-map mean paired percentage length difference then equal weight over12maps; negative is shorter',
        uncertainty='20000 fixed-seed whole-map stratified bootstrap draws, preserve4maps in each synthetic-uniform/synthetic-clustered/TSPLIB stratum; report95% and Bonferroni97.5% percentile intervals for two intended claims D-A and D-U; use latter for stable-direction claims',
        bootstrap_seed=20261019050002,bootstrap_draws=20000,engineering_margin_percent=.01,
        secondary='D-U isolates learned selection versus uniform in the identical candidate/pool search; U-A is descriptive;10-second checkpoints, per-map/per-pair wins ties losses, five joint-seed effects, online model load/inference/pool costs; both/neither/mixed LKH-output diagnostics without excluding any formal score; seed repeats are nested within maps and not independent test instances',
        failures='preserve failure/timeout incumbents, no selective reruns or added seeds; stop batch for containment failure or frozen integrity failure',
        common_exclusions='offline training, common input generation/parsing/copying/sealing and offline analysis; OS file cache not flushed; cleanup overrun reported without additional scored solutions',
        limitations='fixed cold30s implementation comparison with stock LKH normal initialization; does not attribute D-A solely to RL; D-U is selector ablation; no tuning/early stopping on formal outcomes; reference-paper inspired architecture, not claimed exact paper reproduction',
        quantum_execution=False)
    write_new(output/'frozen_protocol.json',protocol)
    rng=random.Random(protocol['schedule_seed'])
    blocks=[(item,seed) for item in datasets for seed in seeds]
    rng.shuffle(blocks)
    orders=[LATIN_ORDERS[i%len(LATIN_ORDERS)] for i in range(len(blocks))]
    rng.shuffle(orders)
    schedule=[dict(job=index+1,map_id=item['id'],seed=seed,method=method,block=block_index+1,
                   block_order=''.join(order),position=position+1)
              for index,(item,seed,method,block_index,order,position) in enumerate(
                  (item,seed,method,block_index,order,position)
                  for block_index,((item,seed),order) in enumerate(zip(blocks,orders))
                  for position,method in enumerate(order))]
    write_new(output/'schedule.json',schedule)
    sealed_file_hashes={path.relative_to(output).as_posix():sha(path) for path in files_under(output)}
    write_new(output/'freeze_complete.json',dict(at=utc(),planned_jobs=len(schedule),source_hashes=source_hashes,
              sealed_file_hashes=sealed_file_hashes,
              model_file_count=len(model_hashes),all_inputs_models_schedule_frozen_before_first_job=True))
    print('FROZEN '+str(output),flush=True)
    return output


def assert_frozen(output,source_hashes,model_hashes,datasets,sealed_file_hashes):
    if sha(LKH)!=LKH_SHA256:
        raise RuntimeError('registered solver binary changed')
    for name,expected in source_hashes.items():
        if sha(HERE/name)!=expected or sha(output/'source'/name)!=expected:
            raise RuntimeError('frozen method source changed: '+name)
    for name,expected in model_hashes.items():
        if sha(output/'models'/name)!=expected:
            raise RuntimeError('frozen policy/training artifact changed: '+name)
    for item in datasets:
        if sha(item['path'])!=item['sha256']:
            raise RuntimeError('frozen test bytes changed')
    for relative,expected in sealed_file_hashes.items():
        if sha(output/relative)!=expected:
            raise RuntimeError('frozen study metadata/input/model/source changed: '+relative)


def execute(output):
    if not (output/'freeze_complete.json').is_file():
        raise RuntimeError('study has not completed its freeze')
    if (output/'complete.json').exists() or any(output.glob('j[0-9][0-9][0-9]')) or any(output.glob('row[0-9][0-9][0-9].json')):
        raise RuntimeError('refusing to rerun or overwrite an already started study')
    protocol=read(output/'frozen_protocol.json')
    source_hashes=read(output/'source_hashes.json')
    model_hashes=read(output/'model_hashes.json')
    datasets=read(output/'datasets.json')['maps']
    by_id={item['id']:dict(item,coords=read(item['path'])['coords']) for item in datasets}
    schedule=read(output/'schedule.json')
    sealed_file_hashes=read(output/'freeze_complete.json')['sealed_file_hashes']
    assert_frozen(output,source_hashes,model_hashes,datasets,sealed_file_hashes)
    from supervisor import run_one
    started=time.perf_counter()
    for item in schedule:
        assert_frozen(output,source_hashes,model_hashes,datasets,sealed_file_hashes)
        index=item['job']
        instance=by_id[item['map_id']]
        print(f'START {index}/{len(schedule)} {instance["id"]} seed={item["seed"]} method={item["method"]}',flush=True)
        row=run_one(item['method'],instance['coords'],item['seed'],protocol['total_budget_seconds'],
                    output/f'j{index:03d}',str(LKH),marks=protocol['milestones_seconds'],settings=protocol['settings'])
        row.update(map_id=instance['id'],family=instance['family'],stratum=instance['stratum'],job=index,
                   block=item['block'],position=item['position'],block_order=item['block_order'])
        write_new(output/f'row{index:03d}.json',row)
        worker=row.get('worker_result') or {}
        print(f'END {index}/{len(schedule)} status={row["status"]} length={row["length"]}'
              f' source={row.get("source")} pre={worker.get("pre_lkh_length")} pre_seconds={worker.get("pre_lkh_cold_seconds")}'
              f' invalid={row["invalid_checkpoints"]} cleanup={row["containment_cleanup_passed"]}',flush=True)
        if not row['containment_cleanup_passed']:
            raise RuntimeError('containment cleanup unconfirmed; stopping serial batch')
    assert_frozen(output,source_hashes,model_hashes,datasets,sealed_file_hashes)
    write_new(output/'complete.json',dict(at=utc(),jobs=len(schedule),
              batch_elapsed_seconds=time.perf_counter()-started,source_hashes_unchanged=True,
              frozen_models_inputs_unchanged=True))
    print('COMPLETE '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--model-directory',type=Path)
    parser.add_argument('--input-directory',type=Path,default=HERE/'datasets_final')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--smoke-cities',type=int,default=256)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--execute-frozen',type=Path)
    args=parser.parse_args()
    if args.execute_frozen is not None:
        if args.output is not None or args.model_directory is not None or args.freeze_only or args.smoke:
            parser.error('--execute-frozen is a standalone execution command')
        execute(args.execute_frozen.resolve())
        return
    if args.output is None or args.model_directory is None:
        parser.error('--output and --model-directory are required to freeze a study')
    output=freeze(args)
    if not args.freeze_only:
        execute(output)


if __name__=='__main__':
    main()
