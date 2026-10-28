"""Frozen, resumable paired replication using the unchanged 2026-10-19 engine.

prepare writes a new immutable study, smoke uses an unrelated small graph,
run executes only a frozen study, and status never starts a solver.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKSPACE = Path('C:/Users/31226/Documents/Codex/2026-10-28/2opt-2')
UPSTREAM = Path('C:/Users/31226/Documents/Codex/2026-10-18/v10-k-k-i-j-2opt/outputs/deep_knn_drl_20261019')
DEFAULT_STUDY = WORKSPACE/'outputs/seed120_20261028'
LKH_ORIGINAL = Path('D:/My project/outputs/tsp_research/lkh3_baseline/LKH-3.exe')
LKH_SHA256 = 'b44414b7c9aa111b4e782ba13fd2cf4684bf3ad505ba1f5d65822b8ad162c044'
SEEDS = (1009,1013,1019,1021,1031,1033,1039,1049,1051,1061)
SETTINGS = dict(preprocessing_fraction_target=.25,local_max_steps=128,pool_size=3,
                d_candidate_count=192,d_exact_quota=1,knn=16,audit_decisions=128)
ENGINE_FILES = ('worker.py','supervisor.py','resource_v3.py','search_core.py','deep_search.py','neural_model.py')
THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def coordinate_sha(coords):
    return hashlib.sha256(json.dumps(coords,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()

def set_threads():
    for key in THREAD_ENV:
        os.environ[key]='1'
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'

def make_schedule(maps,seeds,cpus,schedule_seed):
    """Each seed appears on every lane equally; each map has near-equal lanes.

    Within each map five AB/five BA; within each lane equally AB/BA.
    """
    rng=random.Random(schedule_seed)
    k=len(cpus)
    blocks=[dict(map_id=item['id'],seed=seed,lane=(mi+si)%k,cpu=cpus[(mi+si)%k])
            for mi,item in enumerate(maps) for si,seed in enumerate(seeds)]
    if len(seeds)%2:
        raise ValueError('even number of seeds required for within-map AB/BA balance')
    for attempt in range(200000):
        for mi in range(len(maps)):
            orders=['AD']*(len(seeds)//2)+['DA']*(len(seeds)//2)
            rng.shuffle(orders)
            for si,order in enumerate(orders):
                blocks[mi*len(seeds)+si]['block_order']=order
        counts=Counter((b['lane'],b['block_order']) for b in blocks)
        if all(counts[lane,'AD']==counts[lane,'DA'] for lane in range(k)):
            break
    else:
        raise RuntimeError('balanced order randomization failed')
    rng.shuffle(blocks)
    result=[]
    lane_positions=Counter()
    for block_index,block in enumerate(blocks,1):
        lane_positions[block['lane']]+=1
        for position,method in enumerate(block['block_order'],1):
            result.append(dict(block,job=len(result)+1,block=block_index,
                               position=position,method=method,lane_block=lane_positions[block['lane']]))
    return result

def prepare(study,cpus):
    import psutil
    study=study.resolve()
    if study.exists():
        raise FileExistsError('prepare requires a new output directory: '+str(study))
    if len(set(cpus))!=len(cpus) or len(cpus) not in (1,2,4):
        raise ValueError('choose one, two or four distinct CPU indices')
    if not set(cpus)<=set(psutil.Process().cpu_affinity()):
        raise ValueError('CPU affinity is outside the available set')
    if sha(LKH_ORIGINAL)!=LKH_SHA256:
        raise RuntimeError('upstream registered stock executable changed')
    old_sources=read(UPSTREAM/'source_hashes.json')
    old_models=read(UPSTREAM/'model_hashes.json')
    old_sealed=read(UPSTREAM/'freeze_complete.json')['sealed_file_hashes']
    for name in ENGINE_FILES:
        if sha(HERE/name)!=old_sources[name] or sha(UPSTREAM/'source'/name)!=old_sources[name]:
            raise RuntimeError('upstream method/timing module differs: '+name)
    for rel in ('frozen_protocol.json','train_test_separation.json','inputs/manifest.json','models/model_registry.json'):
        if sha(UPSTREAM/rel)!=old_sealed[rel]:
            raise RuntimeError('upstream frozen metadata changed: '+rel)
    original_protocol=read(UPSTREAM/'frozen_protocol.json')
    for key,value in SETTINGS.items():
        if original_protocol['settings'][key]!=value:
            raise RuntimeError('algorithm setting differs from upstream: '+key)
    manifest=read(UPSTREAM/'inputs/manifest.json')
    maps=[]
    for item in manifest['maps']:
        path=UPSTREAM/'inputs'/(item['id']+'.json')
        data=read(path)
        if sha(path)!=item['sha256'] or coordinate_sha(data['coords'])!=item['coordinate_sha256']:
            raise RuntimeError('upstream input integrity failed: '+item['id'])
        if any(type(v) is not int for xy in data['coords'] for v in xy):
            raise RuntimeError('noninteger coordinate')
        maps.append(dict(item,coords=data['coords']))
    if len(maps)!=12 or Counter(m['stratum'] for m in maps)!=dict(synthetic_uniform=4,synthetic_clustered=4,tsplib=4):
        raise RuntimeError('expected the twelve upstream maps and three equal strata')
    separation=read(UPSTREAM/'train_test_separation.json')
    hashes={m['coordinate_sha256'] for m in maps}
    if hashes & (set(separation['train_coordinate_sha256'])|set(separation['validation_coordinate_sha256'])):
        raise RuntimeError('test/train or test/validation overlap')
    registry=read(UPSTREAM/'models/model_registry.json')['by_test_seed']
    model_records=[registry[str(seed)] for seed in sorted(map(int,registry))]
    if len(model_records)!=5 or len({r['training_seed'] for r in model_records})!=5:
        raise RuntimeError('expected five fixed independent trained models')
    study.mkdir(parents=True,exist_ok=False)
    for name in ('source','inputs','models','bin','lanes'):
        (study/name).mkdir()
    for name in (*ENGINE_FILES,'runner.py'):
        shutil.copyfile(HERE/name,study/'source'/name)
    shutil.copyfile(LKH_ORIGINAL,study/'bin/LKH-3.exe')
    shutil.copyfile(UPSTREAM/'frozen_protocol.json',study/'upstream_protocol.json')
    shutil.copyfile(UPSTREAM/'train_test_separation.json',study/'upstream_train_test_separation.json')
    shutil.copyfile(UPSTREAM/'inputs/manifest.json',study/'inputs/manifest.json')
    datasets=[]
    for item in maps:
        target=study/'inputs'/(item['id']+'.json')
        shutil.copyfile(UPSTREAM/'inputs'/target.name,target)
        datasets.append({k:v for k,v in item.items() if k!='coords'})
        datasets[-1].update(path=str(target),sha256=sha(target),original_source=str(UPSTREAM/'inputs'/target.name))
    mapping={}
    for index,seed in enumerate(SEEDS):
        record=model_records[index%5]
        rel=record['D']
        if sha(UPSTREAM/'models'/rel)!=old_models[rel]:
            raise RuntimeError('upstream registered model changed: '+rel)
        target=study/'models'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copyfile(UPSTREAM/'models'/rel,target)
        mapping[str(seed)]=dict(D=str(target.resolve()),training_seed=record['training_seed'],sha256=sha(target))
    if len({m['sha256'] for m in mapping.values()})!=5:
        raise RuntimeError('fixed policy bytes not independently distinct')
    write_new(study/'models/model_registry.json',dict(by_test_seed=mapping,training_reused=True,
              original_registry=str(UPSTREAM/'models/model_registry.json')))
    write_new(study/'datasets.json',dict(maps=datasets,registered_before_first_formal_job=True))
    settings=dict(SETTINGS,policy_by_test_seed={s:dict(D=m['D']) for s,m in mapping.items()})
    schedule_seed=20261028012001
    protocol=dict(schema=5,frozen_at=utc(),development_smoke=False,methods=['A','D'],seeds=list(SEEDS),
        method_descriptions=dict(A='ordinary registered stock LKH, normal initialization; no nearest neighbor or supplied initial tour',
                                 D='paid nearest neighbor -> unchanged fixed pretrained deep-RL candidate 2-opt and route pool -> stock LKH'),
        total_budget_seconds=30.,milestones_seconds=[10.,30.],settings=settings,
        independent_maps=12,paired_blocks=120,planned_jobs=240,
        selection='D is a fixed reproducible candidate from the named pinned chat; differing historical maps/baselines do not establish a universal best version',
        reused_test_maps=True,new_search_seeds=True,training='five original pretrained policies reused without tuning/retraining; two search seeds per model, twelve maps per search seed',
        interpretation='120 paired runs means twelve maps times ten search seeds, not 120 search seeds per map or 120 independently trained models',
        schedule_seed=schedule_seed,concurrent_lanes=len(cpus),lane_cpu_affinity=cpus,
        schedule='paired arms serial on the same fixed logical CPU; shuffled blocks; five AD/five DA per map and equal AD/DA per lane; seed/lane and map/lane balance; four concurrent lanes differ from the original serial timing',
        parallel_change='wall-clock budgets are equal within new paired runs; cross-lane memory/cache/OS contention may change performance versus the old serial study',
        resource_scheduler='whole paired blocks reserve a fixed predicted RSS allowance before either cold timing origin; n>=8000:420 MB, n>=5000:230 MB, n>=3000:160 MB, otherwise125 MB; total allowance750 MB; blocks with n>=5000 are globally mutually exclusive; waits are scheduling, not algorithm time; no outcome-dependent scheduling',
        memory_allowance_mb=750,memory_weights_mb=dict(n8000=420,n5000=230,n3000=160,small=125),
        threading='all numerical library thread limits are one; subprocesses inherit lane affinity; no affinity adjustment after the clock starts',
        timing=original_protocol['timing'].replace('serial single-thread jobs','separate fixed-CPU paired lanes; single-thread jobs'),
        score=original_protocol['score'],lkh=original_protocol['lkh'],
        primary='D-A paired percentage final certified incumbent length, mean within each map and then equal-weight mean over all twelve maps; negative is shorter; identity-fallback effects must be reported separately and do not demonstrate superiority over LKH solutions',
        diagnostics='report LKH route-output coverage, both-LKH-output paired effects and neither/mixed-output groups alongside all-map results; subsets are conditional mechanism diagnostics, not unbiased global superiority estimates',
        uncertainty='20000 fixed-seed whole-map stratified percentile bootstrap draws preserving four maps in each of three strata; 95% CI for the single D-A primary comparison; seed repetitions are nested within maps',
        bootstrap_seed=20261028012002,bootstrap_draws=20000,engineering_margin_percent=.01,
        failures='no outcome-adaptive reruns or added seeds; finished result rows are immutable; interrupted arm directories are sealed as failed, never rerun; stop on containment/integrity failure',
        quantum_execution=False,model_mapping=mapping)
    write_new(study/'frozen_protocol.json',protocol)
    schedule=make_schedule(maps,SEEDS,cpus,schedule_seed)
    write_new(study/'schedule.json',schedule)
    environment=dict(at=utc(),python=sys.version,executable=sys.executable,platform=platform.platform(),
        cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),logical_cpu_count=os.cpu_count(),
        available_cpu_affinity=psutil.Process().cpu_affinity(),memory=psutil.virtual_memory()._asdict(),
        versions={name:importlib.metadata.version(name) for name in ('numpy','psutil')},
        upstream=str(UPSTREAM),lkh_original=str(LKH_ORIGINAL),lkh_sha256=LKH_SHA256)
    write_new(study/'environment.json',environment)
    write_new(study/'source_hashes.json',{name:sha(study/'source'/name) for name in (*ENGINE_FILES,'runner.py')})
    sealed={p.relative_to(study).as_posix():sha(p) for p in sorted(study.rglob('*')) if p.is_file()}
    write_new(study/'freeze_complete.json',dict(at=utc(),planned_jobs=240,sealed_file_hashes=sealed,
              all_inputs_models_schedule_frozen_before_first_job=True))
    print(json.dumps(dict(frozen=str(study),jobs=240,lanes=len(cpus),cpu_affinity=cpus,
                         nominal_minutes=120/len(cpus),schedule_counts=dict(Counter(x['block_order'] for x in schedule[::2])))),flush=True)

def assert_frozen(study):
    for rel,expected in read(study/'freeze_complete.json')['sealed_file_hashes'].items():
        if sha(study/rel)!=expected:
            raise RuntimeError('frozen study artifact changed: '+rel)
    expected=read(study/'source_hashes.json')
    for name,digest in expected.items():
        if sha(HERE/name)!=digest:
            raise RuntimeError('execute the exact frozen source; module changed: '+name)

def decorate(row,item,started,affinity,gate_wait):
    row.update({k:item[k] for k in ('map_id','job','block','position','block_order','lane','cpu','lane_block')})
    row.update(scheduled_cpu_affinity=[item['cpu']],actual_cpu_affinity=affinity,
               scheduler_wait_seconds=gate_wait,formal_arm_started_at=started,
               restored_from_existing_result=False)
    return row

def seal_interrupted(study,item):
    """Preserve an interrupted observation without rerunning its solver.

    The independent final scorer also reconstructs accepted.jsonl and milestones.
    This conservative failure row uses only the already certified identity tour;
    any prior accepted events remain available as raw evidence for the audit.
    """
    directory=study/f'j{item["job"]:03d}'
    task=read(directory/'task.json')
    from supervisor import independent_check,Milestones
    route,length=independent_check(task['coords'],dict(route=task['initial_route'],
        length=sum(int(math.hypot(task['coords'][a][0]-task['coords'][b][0],task['coords'][a][1]-task['coords'][b][1])+.5)
                   for a,b in zip(task['initial_route'],task['initial_route'][1:]+task['initial_route'][:1]))))
    began=task['cold_started']
    initial_at=began
    start_meta=read(study/f'arm{item["job"]:03d}_start.json')
    initial_at=start_meta.get('initial_validated_at',began)
    bank=Milestones((10.,30.),route,length,began,initial_at)
    best_source='initial_identity'
    accepted=0
    events=directory/'accepted.jsonl'
    if events.exists():
        for line in events.read_text(encoding='utf-8').splitlines():
            try:
                event=json.loads(line)
            except ValueError:
                continue
            candidate,cost=independent_check(task['coords'],event)
            when=event['validated_at']
            if when<task['deadline'] and cost<length:
                route,length=candidate,cost
                best_source=event.get('source')
                bank.accept(route,length,when,best_source)
                accepted+=1
    row=dict(method=item['method'],seed=item['seed'],n=len(route),budget_seconds=30.,
        status='failed',error=dict(reason='interrupted arm sealed without solver rerun'),
        route=route,length=length,source=best_source,milestones=bank.values,settings=task['settings'],
        cold_started=began,deadline=task['deadline'],only_initial=accepted==0,
        accepted_checkpoints=accepted,invalid_checkpoints=0,late_checkpoints=0,
        observed_checkpoints=accepted,initial_validation_cold_seconds=0.,
        containment_cleanup_passed=False,worker_result=None,missing_worker_cost_ledger=True,
        interrupted_recovery=True,scoring_rule='conservative replay of immutable online accepted events; no solver rerun')
    decorate(row,item,start_meta['started_at'],start_meta['actual_cpu_affinity'],start_meta.get('scheduler_wait_seconds',0.))
    row['restored_from_existing_result']=True
    return row

def memory_weight(n):
    return 420 if n>=8000 else 230 if n>=5000 else 160 if n>=3000 else 125

def acquire_block(condition,used,big,weight,is_big,abort):
    began=time.perf_counter()
    with condition:
        while used.value+weight>750 or (is_big and big.value):
            if abort.is_set():
                return None
            condition.wait(timeout=.2)
        if abort.is_set():
            return None
        used.value+=weight
        if is_big:
            big.value=1
    return time.perf_counter()-began

def release_block(condition,used,big,weight,is_big):
    with condition:
        used.value-=weight
        if is_big:
            big.value=0
        condition.notify_all()

def lane_worker(study_text,lane,condition,used,big,abort):
    import psutil
    set_threads()
    study=Path(study_text)
    write_new(study/'lanes'/f'lane{lane}_identity_{time.time_ns()}.json',dict(
        at=utc(),lane=lane,pid=os.getpid(),create_time=psutil.Process().create_time(),controller_pid=os.getppid()))
    protocol=read(study/'frozen_protocol.json')
    cpu=protocol['lane_cpu_affinity'][lane]
    psutil.Process().cpu_affinity([cpu])
    affinity=psutil.Process().cpu_affinity()
    if affinity!=[cpu]:
        raise RuntimeError('lane affinity was not applied')
    maps={m['id']:m for m in read(study/'datasets.json')['maps']}
    schedule=[i for i in read(study/'schedule.json') if i['lane']==lane]
    from supervisor import run_one
    active_block=None
    reserved=None
    wait_seconds=0.
    try:
        for item in schedule:
            if abort.is_set():
                return
            instance=maps[item['map_id']]
            if active_block!=item['block']:
                if reserved is not None:
                    release_block(condition,used,big,*reserved)
                    reserved=None
                weight=memory_weight(instance['n'])
                is_big=instance['n']>=5000
                wait_seconds=acquire_block(condition,used,big,weight,is_big,abort)
                if wait_seconds is None:
                    return
                reserved=(weight,is_big)
                active_block=item['block']
            row_path=study/f'row{item["job"]:03d}.json'
            if row_path.exists():
                prior=read(row_path)
                if any(prior.get(k)!=item[k] for k in ('job','method','seed','map_id')):
                    raise RuntimeError('existing row does not match frozen schedule')
                if prior.get('containment_cleanup_passed') is not True:
                    raise RuntimeError('existing arm lacks confirmed containment cleanup; recovery requires explicit containment review')
                continue
            assert_frozen(study)
            directory=study/f'j{item["job"]:03d}'
            if directory.exists():
                if (directory/'result.json').exists():
                    row=read(directory/'result.json')
                    meta=read(study/f'arm{item["job"]:03d}_start.json')
                    decorate(row,item,meta['started_at'],meta['actual_cpu_affinity'],meta.get('scheduler_wait_seconds',0.))
                    row['restored_from_existing_result']=True
                else:
                    row=seal_interrupted(study,item)
                row.update(family=instance['family'],stratum=instance['stratum'])
                write_new(row_path,row)
                if not row['containment_cleanup_passed']:
                    raise RuntimeError('interrupted arm requires containment review; no rerun allowed')
                continue
            if abort.is_set():
                return
            stamp=utc()
            if (study/f'arm{item["job"]:03d}_start.json').exists():
                raise RuntimeError('arm start ledger exists without its task directory; preserved for interrupted-start review, no solver rerun')
            write_new(study/f'arm{item["job"]:03d}_start.json',dict(started_at=stamp,
                job=item['job'],method=item['method'],seed=item['seed'],map_id=item['map_id'],
                scheduled_cpu_affinity=[cpu],actual_cpu_affinity=affinity,
                scheduler_wait_seconds=wait_seconds,reserved_block_rss_mb=weight,
                whole_big_block_exclusion=is_big,memory=psutil.virtual_memory()._asdict()))
            coords=read(instance['path'])['coords']
            print(f'START lane={lane} job={item["job"]}/240 {item["map_id"]} seed={item["seed"]} method={item["method"]}',flush=True)
            row=run_one(item['method'],coords,item['seed'],30.,directory,str(study/'bin/LKH-3.exe'),
                        marks=(10.,30.),settings=protocol['settings'])
            decorate(row,item,stamp,affinity,wait_seconds)
            row.update(family=instance['family'],stratum=instance['stratum'],reserved_block_rss_mb=weight,
                       whole_big_block_exclusion=is_big)
            write_new(row_path,row)
            print(f'END lane={lane} job={item["job"]}/240 status={row["status"]} source={row["source"]} length={row["length"]}',flush=True)
            if not row['containment_cleanup_passed']:
                raise RuntimeError('containment cleanup not confirmed')
    except BaseException:
        abort.set()
        error=dict(at=utc(),lane=lane,traceback=traceback.format_exc())
        destination=study/'lanes'/f'lane{lane}_failure_{time.time_ns()}.json'
        write_new(destination,error)
        print(json.dumps(error),flush=True)
        raise
    finally:
        if reserved is not None:
            release_block(condition,used,big,*reserved)

def reject_live_prior_processes(study,previous_controller=None):
    import psutil
    identities={}
    controllers=set()
    for path in (study/'lanes').glob('launch_*.json'):
        record=read(path)
        if 'create_time' in record:
            identities[record['pid']]=record['create_time']
        controllers.add(record['pid'])
    if previous_controller:
        identities[previous_controller['pid']]=previous_controller['create_time']
        controllers.add(previous_controller['pid'])
    for path in (study/'lanes').glob('lane*_identity_*.json'):
        record=read(path)
        identities[record['pid']]=record['create_time']
    for directory in study.glob('j[0-9][0-9][0-9]'):
        ledger=directory/'tree_resources.jsonl'
        if ledger.exists():
            for line in ledger.open(encoding='utf-8'):
                try:
                    sample=json.loads(line)
                except ValueError:
                    continue
                for process in sample.get('processes',[]):
                    identities[process['pid']]=process['create_time']
    live=[]
    for pid,created in identities.items():
        try:
            process=psutil.Process(pid)
            if abs(process.create_time()-created)<.01:
                live.append(pid)
        except psutil.Error:
            pass
    # This also catches a spawned lane before its first identity ledger write.
    for process in psutil.process_iter(['pid','ppid']):
        if process.info['ppid'] in controllers:
            live.append(process.info['pid'])
    if live:
        raise RuntimeError('refusing recovery while prior registered study processes remain alive: '+str(sorted(set(live))))

def run(study):
    import psutil
    study=study.resolve()
    set_threads()
    assert_frozen(study)
    if (study/'complete.json').exists():
        print('Already complete; no solver rerun.',flush=True)
        return
    lock=study/'run_lock.json'
    previous=None
    if lock.exists():
        previous=read(lock)
        try:
            process=psutil.Process(previous['pid'])
            alive=abs(process.create_time()-previous['create_time'])<.01
        except psutil.Error:
            alive=False
        if alive:
            raise RuntimeError('study already has a live controller')
        reject_live_prior_processes(study,previous)
        lock.unlink()
    else:
        reject_live_prior_processes(study)
    write_new(lock,dict(pid=os.getpid(),create_time=psutil.Process().create_time(),at=utc()))
    protocol=read(study/'frozen_protocol.json')
    began=time.perf_counter()
    ctx=mp.get_context('spawn')
    condition=ctx.Condition()
    used=ctx.Value('i',0)
    big=ctx.Value('i',0)
    abort=ctx.Event()
    processes=[]
    try:
        write_new(study/'lanes'/f'launch_{time.time_ns()}.json',dict(at=utc(),pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            memory=psutil.virtual_memory()._asdict(),concurrent_lanes=protocol['concurrent_lanes']))
        for lane in range(protocol['concurrent_lanes']):
            process=ctx.Process(target=lane_worker,args=(str(study),lane,condition,used,big,abort),name=f'seed120_lane{lane}')
            process.start()
            processes.append(process)
        while any(p.is_alive() for p in processes):
            for process in processes:
                process.join(timeout=.5)
            if any(p.exitcode not in (0,None) for p in processes):
                abort.set()
        codes=[p.exitcode for p in processes]
        if any(c!=0 for c in codes):
            raise RuntimeError('lane failed; preserved all rows and stopped new arms: '+str(codes))
        rows=list(study.glob('row[0-9][0-9][0-9].json'))
        if len(rows)!=protocol['planned_jobs']:
            raise RuntimeError('formal batch missing result rows')
        assert_frozen(study)
        write_new(study/'complete.json',dict(at=utc(),jobs=len(rows),
                  batch_elapsed_seconds=time.perf_counter()-began,all_frozen_artifacts_unchanged=True,
                  methods=['A','D'],lane_exit_codes=codes))
        print('COMPLETE '+str(study),flush=True)
    finally:
        abort.set()
        for process in processes:
            process.join(timeout=35)
        lock.unlink(missing_ok=True)

def smoke(directory,cpu,seconds,cities,methods):
    import psutil
    set_threads()
    directory=directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    psutil.Process().cpu_affinity([cpu])
    rng=random.Random(20261028011999)
    coords=[[rng.randrange(100001),rng.randrange(100001)] for _ in range(cities)]
    registry=read(UPSTREAM/'models/model_registry.json')['by_test_seed']
    model=UPSTREAM/'models'/registry['101']['D']
    settings=dict(SETTINGS,policy_by_test_seed={'1009':dict(D=str(model))})
    from supervisor import run_one
    rows=[]
    for method in methods:
        row=run_one(method,coords,1009,seconds,directory/method,str(LKH_ORIGINAL),
                    marks=(min(10.,seconds),) if seconds<=10 else (10.,seconds),settings=settings)
        rows.append({k:row.get(k) for k in ('method','status','error','length','source','containment_cleanup_passed',
                                         'invalid_checkpoints','paired_peak_rss_bytes','worker_result')})
    write_new(directory/'summary.json',dict(development_only=True,no_formal_maps_used=True,rows=rows))
    print(json.dumps(dict(smoke=str(directory),rows=[{k:v for k,v in row.items() if k!='worker_result'} for row in rows])),flush=True)
    if any(row['status']=='failed' or not row['containment_cleanup_passed'] or row['invalid_checkpoints'] for row in rows):
        raise RuntimeError('smoke failed')

def status(study):
    protocol=read(study/'frozen_protocol.json')
    rows=[read(p) for p in sorted(study.glob('row[0-9][0-9][0-9].json'))]
    byblock=Counter(r['block'] for r in rows)
    result=dict(study=str(study),planned=protocol['planned_jobs'],finished=len(rows),
        full_pairs=sum(n==2 for n in byblock.values()),status_counts=dict(Counter(r['status'] for r in rows)),
        method_counts=dict(Counter(r['method'] for r in rows)),
        lanes=dict(Counter(str(r['lane']) for r in rows)),complete=(study/'complete.json').exists())
    print(json.dumps(result,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    p.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    p.add_argument('--cpus',type=int,nargs='+',default=[0,2,4,6])
    p=commands.add_parser('smoke')
    p.add_argument('--directory',type=Path,default=WORKSPACE/'work/seed120_smoke')
    p.add_argument('--cpu',type=int,default=0)
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--cities',type=int,default=256)
    p.add_argument('--methods',nargs='+',choices=['A','D'],default=['A','D'])
    for name in ('run','status'):
        p=commands.add_parser(name)
        p.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.study,args.cpus)
    elif args.command=='smoke':
        smoke(args.directory,args.cpu,args.seconds,args.cities,args.methods)
    elif args.command=='run':
        run(args.study)
    else:
        status(args.study)

if __name__=='__main__':
    main()
