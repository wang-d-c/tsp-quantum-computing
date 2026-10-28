"""Append-only execution amendment: preserve completed arms and expand CPU lanes.

The original protocol, schedule, code and all completed rows remain untouched.
Only unstarted arms are executed; a half-finished pair keeps its original CPU.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
import phase1_runner as old
DEFAULT_STUDY=old.DEFAULT_STUDY
EPOCH_REL='amendments/phase2'
SOURCE_NAMES=(*old.ENGINE_FILES,'phase1_runner.py','runner.py')

def epoch(study):
    return study/EPOCH_REL

def original_integrity(study):
    for rel,expected in old.read(study/'freeze_complete.json')['sealed_file_hashes'].items():
        if old.sha(study/rel)!=expected:
            raise RuntimeError('original frozen artifact changed: '+rel)

def assert_integrity(study,baseline=False):
    original_integrity(study)
    data=old.read(epoch(study)/'freeze_complete.json')
    if old.sha(study/'freeze_complete.json')!=data['original_freeze_sha256']:
        raise RuntimeError('original freeze manifest bytes changed')
    for rel,expected in data['sealed_file_hashes'].items():
        if old.sha(study/rel)!=expected:
            raise RuntimeError('phase2 frozen artifact changed: '+rel)
    for name,expected in old.read(epoch(study)/'source_hashes.json').items():
        if old.sha(HERE/name)!=expected:
            raise RuntimeError('execute the registered phase2 source: '+name)
    if baseline:
        for rel,expected in data['preserved_row_hashes'].items():
            if old.sha(study/rel)!=expected:
                raise RuntimeError('preserved phase1 result changed: '+rel)

def prepare(study,cpus):
    import psutil
    study=study.resolve()
    if len(cpus)!=8 or len(set(cpus))!=8 or not set(cpus)<=set(psutil.Process().cpu_affinity()):
        raise ValueError('phase2 requires eight distinct available CPU indices')
    if epoch(study).exists():
        raise FileExistsError('phase2 amendment already exists')
    if (study/'complete.json').exists():
        raise RuntimeError('original study is already complete')
    if not (study/'phase1_boundary_pause.json').exists():
        raise RuntimeError('boundary pause and restored manifest record required')
    original_integrity(study)
    previous=old.read(study/'run_lock.json') if (study/'run_lock.json').exists() else None
    old.reject_live_prior_processes(study,previous)
    original_schedule=old.read(study/'schedule.json')
    by_job={item['job']:item for item in original_schedule}
    rows={}
    for path in study.glob('row[0-9][0-9][0-9].json'):
        row=old.read(path)
        item=by_job[row['job']]
        if any(row.get(k)!=item[k] for k in ('job','method','seed','map_id')):
            raise RuntimeError('existing row identity differs from original schedule')
        if row.get('containment_cleanup_passed') is not True:
            raise RuntimeError('existing row lacks confirmed containment cleanup')
        rows[row['job']]=row
    for item in original_schedule:
        if item['job'] not in rows:
            if (study/f'j{item["job"]:03d}').exists() or (study/f'arm{item["job"]:03d}_start.json').exists():
                raise RuntimeError('unsealed started arm exists; explicit recovery review needed, no rerun: '+str(item['job']))
    if any(item['job'] in rows and rows[item['job']].get('actual_cpu_affinity')!=[item['cpu']] for item in original_schedule):
        raise RuntimeError('phase1 actual CPU assignment differs from its schedule')
    blocks=defaultdict(list)
    for item in original_schedule:
        blocks[item['block']].append(item)
    rng=random.Random(20261028012003)
    half=[]
    untouched=[]
    for number,items in blocks.items():
        pending=[item for item in items if item['job'] not in rows]
        if not pending:
            continue
        record=dict(block=number,items=pending,map_id=items[0]['map_id'],seed=items[0]['seed'])
        if len(pending)==1:
            completed=next(item for item in items if item['job'] in rows)
            cpu=rows[completed['job']]['actual_cpu_affinity'][0]
            if cpu not in cpus:
                raise RuntimeError('half pair CPU absent from phase2 available set')
            record.update(cpu=cpu,lane=cpus.index(cpu),half_pair=True,completed_job=completed['job'])
            half.append(record)
        else:
            record['half_pair']=False
            untouched.append(record)
    rng.shuffle(half)
    rng.shuffle(untouched)
    lane_jobs=Counter()
    lane_maps=Counter()
    lane_seeds=Counter()
    assigned=[]
    for record in half+untouched:
        if not record['half_pair']:
            candidates=list(range(8))
            rng.shuffle(candidates)
            lane=min(candidates,key=lambda lane:(lane_jobs[lane],lane_maps[lane,record['map_id']],lane_seeds[lane,record['seed']]))
            record.update(lane=lane,cpu=cpus[lane])
        lane=record['lane']
        lane_jobs[lane]+=len(record['items'])
        lane_maps[lane,record['map_id']]+=1
        lane_seeds[lane,record['seed']]+=1
        assigned.append(record)
    remaining=[]
    lane_block=Counter()
    for record in assigned:
        lane=record['lane']
        lane_block[lane]+=1
        for item in record['items']:
            remaining.append(dict(item,original_lane=item['lane'],original_cpu=item['cpu'],lane=lane,
                cpu=record['cpu'],lane_block=lane_block[lane],execution_epoch=2,
                half_pair_same_original_cpu=record['half_pair']))
    if len(rows)+len(remaining)!=240:
        raise RuntimeError('completed and pending counts do not cover the registered 240 arms')
    destination=epoch(study)
    (destination/'source').mkdir(parents=True,exist_ok=False)
    for name in SOURCE_NAMES:
        if name in old.ENGINE_FILES and old.sha(HERE/name)!=old.sha(study/'source'/name):
            raise RuntimeError('phase2 changes an algorithm or timing module: '+name)
        shutil.copyfile(HERE/name,destination/'source'/name)
    old.write_new(destination/'remaining_schedule.json',remaining)
    old.write_new(destination/'source_hashes.json',{name:old.sha(destination/'source'/name) for name in SOURCE_NAMES})
    amendment=dict(schema=1,execution_epoch=2,at=old.utc(),authorized_by='user request to split work across more CPU cores',
        request_file='execution_amendment_request.json',boundary_pause_file='phase1_boundary_pause.json',
        original_protocol_sha256=old.sha(study/'frozen_protocol.json'),original_schedule_sha256=old.sha(study/'schedule.json'),
        original_freeze_sha256=old.sha(study/'freeze_complete.json'),
        preserved_phase1_jobs=sorted(rows),preserved_phase1_rows=len(rows),remaining_jobs=len(remaining),
        remaining_schedule='amendments/phase2/remaining_schedule.json',
        concurrent_lanes=8,lane_cpu_affinity=cpus,memory_allowance_mb=1400,
        memory_weights_mb=dict(n8000=420,n5000=230,n3000=160,small=125),
        max_simultaneous_n8000_blocks=2,no_global_n5000_exclusion=True,
        scheduler='reserve predicted memory for an entire paired block; at most1400 MB total; at most two n>=8000 blocks; fixed per-lane order; no result-dependent tuning',
        assignment='four-to-eight-lane change affects only remaining unstarted arms; half-completed pairs keep their original CPU; untouched blocks are deterministically balanced by remaining job counts then map/seed counts',
        no_algorithm_change=True,no_training_or_model_change=True,no_budget_change=True,no_reruns=True,
        search_seeds=old.read(study/'frozen_protocol.json')['seeds'],
        half_completed_pairs=[dict(block=r['block'],completed_job=r['completed_job'],remaining_job=r['items'][0]['job'],cpu=r['cpu'],lane=r['lane']) for r in half],
        lane_remaining_jobs={str(k):lane_jobs[k] for k in range(8)},
        limitations='execution amendment is a user-requested hardware change after some outcomes existed; phase2 uses mixed physical core classes and different memory/cache contention; all pairs retain matched CPU, and execution epochs must be reported separately as a diagnostic',
        memory_snapshot=psutil.virtual_memory()._asdict(),quantum_execution=False)
    old.write_new(destination/'execution_amendment.json',amendment)
    sealed={p.relative_to(study).as_posix():old.sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    for name in ('execution_amendment_request.json','phase1_boundary_pause.json'):
        sealed[name]=old.sha(study/name)
    preserved={f'row{j:03d}.json':old.sha(study/f'row{j:03d}.json') for j in sorted(rows)}
    old.write_new(destination/'freeze_complete.json',dict(at=old.utc(),sealed_file_hashes=sealed,
        preserved_row_hashes=preserved,original_freeze_sha256=old.sha(study/'freeze_complete.json'),
        all_remaining_execution_settings_frozen_before_phase2=True))
    print(json.dumps(dict(prepared=str(destination),preserved=len(rows),remaining=len(remaining),
                         half_pairs=len(half),lane_jobs=amendment['lane_remaining_jobs'])),flush=True)

def acquire(condition,used,large,weight,is_large,abort):
    began=time.perf_counter()
    with condition:
        while used.value+weight>1400 or (is_large and large.value>=2):
            if abort.is_set():
                return None
            condition.wait(timeout=.2)
        if abort.is_set():
            return None
        used.value+=weight
        if is_large:
            large.value+=1
    return time.perf_counter()-began

def release(condition,used,large,weight,is_large):
    with condition:
        used.value-=weight
        if is_large:
            large.value-=1
        condition.notify_all()

def lane_worker(study_text,lane,condition,used,large,abort):
    import psutil
    old.set_threads()
    study=Path(study_text)
    amendment=old.read(epoch(study)/'execution_amendment.json')
    protocol=old.read(study/'frozen_protocol.json')
    cpu=amendment['lane_cpu_affinity'][lane]
    psutil.Process().cpu_affinity([cpu])
    actual=psutil.Process().cpu_affinity()
    if actual!=[cpu]:
        raise RuntimeError('phase2 affinity not applied')
    old.write_new(study/'lanes'/f'lane{lane}_identity_{time.time_ns()}.json',dict(at=old.utc(),
        pid=os.getpid(),create_time=psutil.Process().create_time(),controller_pid=os.getppid(),lane=lane,execution_epoch=2))
    maps={item['id']:item for item in old.read(study/'datasets.json')['maps']}
    schedule=[item for item in old.read(epoch(study)/'remaining_schedule.json') if item['lane']==lane]
    from supervisor import run_one
    active=None
    reserved=None
    wait_seconds=0.
    try:
        for item in schedule:
            if abort.is_set():
                return
            instance=maps[item['map_id']]
            if active!=item['block']:
                if reserved is not None:
                    release(condition,used,large,*reserved)
                    reserved=None
                weight=old.memory_weight(instance['n'])
                is_large=instance['n']>=8000
                wait_seconds=acquire(condition,used,large,weight,is_large,abort)
                if wait_seconds is None:
                    return
                active=item['block']
                reserved=(weight,is_large)
            assert_integrity(study)
            row_path=study/f'row{item["job"]:03d}.json'
            if row_path.exists():
                row=old.read(row_path)
                if any(row.get(key)!=item[key] for key in ('job','method','seed','map_id')) or row.get('containment_cleanup_passed') is not True:
                    raise RuntimeError('existing phase2 row is mismatched or lacks cleanup confirmation')
                continue
            directory=study/f'j{item["job"]:03d}'
            start_path=study/f'arm{item["job"]:03d}_start.json'
            if directory.exists() or start_path.exists():
                raise RuntimeError('started phase2 arm lacks a sealed row; no automatic rerun')
            stamp=old.utc()
            old.write_new(start_path,dict(started_at=stamp,job=item['job'],method=item['method'],
                seed=item['seed'],map_id=item['map_id'],scheduled_cpu_affinity=[cpu],actual_cpu_affinity=actual,
                scheduler_wait_seconds=wait_seconds,reserved_block_rss_mb=weight,execution_epoch=2,
                original_cpu=item['original_cpu'],half_pair_same_original_cpu=item['half_pair_same_original_cpu'],
                memory=psutil.virtual_memory()._asdict()))
            coords=old.read(instance['path'])['coords']
            print(f'START epoch2 lane={lane} cpu={cpu} job={item["job"]}/240 {item["map_id"]} seed={item["seed"]} method={item["method"]}',flush=True)
            row=run_one(item['method'],coords,item['seed'],30.,directory,str(study/'bin/LKH-3.exe'),
                        marks=(10.,30.),settings=protocol['settings'])
            old.decorate(row,item,stamp,actual,wait_seconds)
            row.update(family=instance['family'],stratum=instance['stratum'],execution_epoch=2,
                original_cpu=item['original_cpu'],original_lane=item['original_lane'],
                half_pair_same_original_cpu=item['half_pair_same_original_cpu'],
                reserved_block_rss_mb=weight,execution_amendment='amendments/phase2/execution_amendment.json')
            old.write_new(row_path,row)
            print(f'END epoch2 lane={lane} job={item["job"]}/240 status={row["status"]} source={row["source"]} length={row["length"]}',flush=True)
            if row.get('containment_cleanup_passed') is not True:
                raise RuntimeError('containment cleanup not confirmed')
    except BaseException:
        abort.set()
        record=dict(at=old.utc(),lane=lane,execution_epoch=2,traceback=traceback.format_exc())
        old.write_new(study/'lanes'/f'epoch2_lane{lane}_failure_{time.time_ns()}.json',record)
        print(json.dumps(record),flush=True)
        raise
    finally:
        if reserved is not None:
            release(condition,used,large,*reserved)

def run(study):
    import psutil
    old.set_threads()
    study=study.resolve()
    if (study/'complete.json').exists():
        print('Already complete; no solver rerun.',flush=True)
        return
    assert_integrity(study,baseline=True)
    previous=old.read(study/'run_lock_v2.json') if (study/'run_lock_v2.json').exists() else None
    old.reject_live_prior_processes(study,previous)
    lock=study/'run_lock_v2.json'
    lock.unlink(missing_ok=True)
    old.write_new(lock,dict(at=old.utc(),pid=os.getpid(),create_time=psutil.Process().create_time(),execution_epoch=2))
    ctx=mp.get_context('spawn')
    condition=ctx.Condition()
    used=ctx.Value('i',0)
    large=ctx.Value('i',0)
    abort=ctx.Event()
    children=[]
    began=time.perf_counter()
    try:
        old.write_new(study/'lanes'/f'launch_{time.time_ns()}.json',dict(at=old.utc(),pid=os.getpid(),
            create_time=psutil.Process().create_time(),execution_epoch=2,concurrent_lanes=8,
            memory=psutil.virtual_memory()._asdict()))
        for lane in range(8):
            process=ctx.Process(target=lane_worker,args=(str(study),lane,condition,used,large,abort),name=f'seed120_epoch2_lane{lane}')
            process.start()
            children.append(process)
        while any(p.is_alive() for p in children):
            for process in children:
                process.join(timeout=.3)
            if any(p.exitcode not in (None,0) for p in children):
                abort.set()
        codes=[p.exitcode for p in children]
        if any(code!=0 for code in codes):
            raise RuntimeError('phase2 lane failure; all completed arms preserved: '+str(codes))
        allrows=list(study.glob('row[0-9][0-9][0-9].json'))
        if len(allrows)!=240:
            raise RuntimeError('formal result set is incomplete')
        assert_integrity(study,baseline=True)
        old.write_new(study/'complete.json',dict(at=old.utc(),jobs=240,execution_epochs=2,
            final_execution_amendment='amendments/phase2/execution_amendment.json',
            phase2_batch_elapsed_seconds=time.perf_counter()-began,lane_exit_codes=codes,
            original_frozen_artifacts_unchanged=True,preserved_phase1_rows_unchanged=True,
            phase2_frozen_artifacts_unchanged=True))
        print('COMPLETE '+str(study),flush=True)
    finally:
        abort.set()
        for process in children:
            process.join(timeout=35)
        lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','status'):
        command=subs.add_parser(name)
        command.add_argument('--study',type=Path,default=DEFAULT_STUDY)
        if name=='prepare':
            command.add_argument('--cpus',type=int,nargs='+',default=[0,2,4,6,8,10,12,14])
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.study,args.cpus)
    elif args.command=='run':
        run(args.study)
    else:
        old.status(args.study)

if __name__=='__main__':
    main()
