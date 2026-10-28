"""Cold-budget serial experiments with independent online tour certification.

The budget begins before per-job setup. Only tours fully checked by the parent
before the shared deadline can affect the score. Raw final tours are never
read after cutoff. WindowsJob is mandatory and no containment fallback exists.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from uuid import uuid4

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def independent_check(coords,candidate):
    route=candidate.get('route')
    n=len(coords)
    if not isinstance(route,list) or any(type(v) is not int for v in route):
        raise ValueError('checkpoint city IDs must be integers')
    if len(route)!=n or set(route)!=set(range(n)):
        raise ValueError('not a Hamiltonian permutation')
    total=0
    for index in range(n):
        x,y=coords[route[index]]
        u,v=coords[route[(index+1)%n]]
        total+=int(math.hypot(int(x)-int(u),int(y)-int(v))+.5)
    if type(candidate.get('length')) is not int or candidate['length']!=total:
        raise ValueError('reported length differs from independently rebuilt EUC_2D')
    return route,total


class Milestones:
    def __init__(self,marks,initial,length,started,initial_validated_at):
        self.started=started
        self.values={str(int(mark)) if float(mark).is_integer() else str(mark):
                     {'seconds':float(mark),'route':list(initial),'length':length,
                      'validated_at':initial_validated_at,'only_initial':True,
                      'source':'initial_identity',
                      'validation_cold_seconds':initial_validated_at-started}
                     for mark in marks}

    def accept(self,route,length,validated_at,source):
        for record in self.values.values():
            if validated_at-self.started<record['seconds'] and length<record['length']:
                record.update(route=list(route),length=length,validated_at=validated_at,
                              validation_cold_seconds=validated_at-self.started,
                              only_initial=False,source=source)


def run_one(method,coords,seed,budget,directory,lkh_binary,marks=(10.,30.),settings=None):
    # This origin precedes all per-arm setup, task IO, imports, JIT and children.
    began=time.perf_counter()
    parent_cpu_start=time.process_time()
    if not math.isfinite(budget) or budget<=0:
        raise ValueError('budget must be finite and positive')
    if len(coords)<3:
        raise ValueError('a TSP instance needs at least three cities')
    marks=tuple(float(mark) for mark in marks)
    if any(not math.isfinite(mark) or mark<=0 or mark>budget for mark in marks):
        raise ValueError('milestones must be finite, positive and inside the total budget')
    if len(set(marks))!=len(marks):
        raise ValueError('milestones must be distinct')
    settings={} if settings is None else dict(settings)
    directory=Path(directory).resolve()
    deadline=began+budget
    directory.mkdir(parents=True,exist_ok=False)
    # Numba's hashed cache subdirectories can exceed legacy Windows MAX_PATH.
    # This short, unique, initially empty cache remains inside writable roots.
    cache=HERE/'cache'/uuid4().hex[:12]
    cache.mkdir(parents=True,exist_ok=False)
    initial=list(range(len(coords)))
    length=0
    for i,a in enumerate(initial):
        b=initial[(i+1)%len(initial)]
        length+=int(math.hypot(coords[a][0]-coords[b][0],coords[a][1]-coords[b][1])+.5)
    initial,checked_length=independent_check(coords,{'route':initial,'length':length})
    initial_validated_at=time.perf_counter()
    if initial_validated_at>=deadline:
        raise TimeoutError('cold budget expired before initial route certification')
    bank=Milestones(marks,initial,checked_length,began,initial_validated_at)
    task={'method':method,'coords':coords,'seed':seed,'budget':budget,'directory':str(directory),
          'cold_started':began,'deadline':deadline,'initial_route':initial,'lkh_binary':lkh_binary,
          'numba_cache_directory':str(cache),'settings':settings}
    write_new(directory/'task.json',task)
    from resource_v3 import WindowsJob,ProcessMonitor
    if os.name!='nt':raise RuntimeError('registered supervisor requires Windows Job containment')
    import psutil
    parent=psutil.Process()
    env=dict(os.environ,NUMBA_CACHE_DIR=str(cache),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    process=job=monitor=timer=None
    best_route,best_length=initial,length
    best_source='initial_identity'
    peak=None
    accepted=invalid=late=reads=samples=0
    seen=None
    failure=None
    final_accounting=None
    containment=False
    watchdog={}
    ready=None
    observation_finished=None
    validation_seconds=0.
    poll_seconds=.005

    def expire():
        if watchdog:return
        watchdog['fired_at']=time.perf_counter()
        try:watchdog['termination']=job.terminate()
        except BaseException:watchdog['error']=traceback.format_exc()

    with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr, \
         (directory/'accepted.jsonl').open('x',encoding='utf-8',buffering=65536) as events, \
         (directory/'parent_resources.jsonl').open('x',encoding='utf-8',buffering=65536) as resources:
        try:
            if time.perf_counter()>=deadline:
                raise TimeoutError('cold budget expired during supervisor setup')
            process=subprocess.Popen([sys.executable,'-B',str(HERE/'worker.py'),str(directory/'task.json')],
                stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,cwd=HERE,env=env,
                creationflags=subprocess.CREATE_NO_WINDOW)
            identity=psutil.Process(process.pid).create_time()
            job=WindowsJob(process.pid,expected_create_time=identity)
            monitor=ProcessMonitor(process.pid,interval_seconds=.05,sample_path=directory/'tree_resources.jsonl')
            timer=threading.Timer(max(0.,deadline-time.perf_counter()),expire)
            timer.daemon=True
            timer.start()
            while time.perf_counter()<min(deadline,began+10.):
                try:
                    ready=json.loads((directory/'ready.json').read_text(encoding='utf-8'))
                    break
                except (OSError,ValueError):pass
                if process.poll() is not None:raise RuntimeError('worker exited before READY')
                time.sleep(.005)
            if ready is None:raise TimeoutError('READY was not received inside cold cap')
            if (ready.get('deadline')!=deadline or ready.get('cold_started')!=began
                or ready.get('method')!=method or ready.get('before_any_child') is not True):
                raise ValueError('READY changed registered method/clock')
            if time.perf_counter()>=deadline:
                raise TimeoutError('cold budget expired before the GO gate')
            process.stdin.write((json.dumps({'go':True,'deadline':deadline})+'\n').encode('ascii'))
            process.stdin.flush()
            next_sample=0.
            while time.perf_counter()<deadline:
                stamp=time.perf_counter()
                if stamp>=next_sample:
                    sample=monitor.sample()
                    try:parent_rss=int(parent.memory_info().rss)
                    except psutil.Error:parent_rss=None
                    tree_rss=sample['rss_total_bytes'] if sample['rss_complete'] else None
                    paired=None if parent_rss is None or tree_rss is None else parent_rss+tree_rss
                    if paired is not None:peak=paired if peak is None else max(peak,paired)
                    resources.write(json.dumps({'at':stamp,'cold_seconds':stamp-began,'tree_rss_bytes':tree_rss,
                        'parent_rss_bytes':parent_rss,'paired_rss_bytes':paired})+'\n')
                    samples+=1
                    next_sample=stamp+.05
                if time.perf_counter()>=deadline:
                    break
                before=time.perf_counter()
                try:
                    path=directory/'checkpoint.json'
                    stat=path.stat()
                    signature=(stat.st_mtime_ns,stat.st_size)
                    if signature!=seen:
                        if stat.st_size>128*len(coords)+16384:raise ValueError('oversized checkpoint')
                        if time.perf_counter()>=deadline:
                            break
                        candidate=json.loads(path.read_text(encoding='ascii'))
                        route,cost=independent_check(coords,candidate)
                        certified=time.perf_counter()
                        seen=signature
                        reads+=1
                        if certified>=deadline:late+=1
                        elif cost<best_length:
                            best_route,best_length=list(route),cost
                            best_source=candidate.get('source')
                            bank.accept(route,cost,certified,best_source)
                            accepted+=1
                            events.write(json.dumps({'route':route,'length':cost,'validated_at':certified,
                                'cold_seconds':certified-began,'source':candidate.get('source'),
                                'sequence':accepted},separators=(',',':'))+'\n')
                except (OSError,UnicodeError):pass
                except (ValueError,TypeError,KeyError):invalid+=1
                validation_seconds+=time.perf_counter()-before
                if (directory/'done.json').exists():break
                if process.poll() is not None:
                    if watchdog:break
                    raise RuntimeError(f'worker disappeared without DONE ({process.returncode})')
                time.sleep(min(poll_seconds,max(0.,deadline-time.perf_counter())))
            if (directory/'error.json').exists():
                failure=json.loads((directory/'error.json').read_text(encoding='utf-8'))
        except BaseException:
            failure={'traceback':traceback.format_exc(),'previous':failure}
        finally:
            observation_finished=time.perf_counter()
            if timer is not None:
                timer.cancel()
                timer.join(timeout=.2)
            if job is not None:
                try:
                    if job.accounting()['active_processes']:
                        job.terminate()
                    if process is not None:process.wait(timeout=5)
                    cleanup_deadline=time.perf_counter()+5
                    while job.accounting()['active_processes']:
                        if time.perf_counter()>=cleanup_deadline:raise RuntimeError('contained tree did not exit')
                        time.sleep(.005)
                    final_accounting=job.accounting()
                    containment=True
                except BaseException:
                    failure={'traceback':traceback.format_exc(),'previous':failure}
                finally:job.close()
            elif process is not None:
                # Worker has not received GO: no descendant creation is permitted.
                if process.poll() is None:process.terminate()
                process.wait(timeout=5)
            if monitor is not None:monitor.close()
    ended=time.perf_counter()
    worker_result=None
    worker_result_read_error=None
    if (directory/'worker_result.json').exists():
        try:
            worker_result=json.loads((directory/'worker_result.json').read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:
            worker_result_read_error=f'{type(exc).__name__}: {exc}'
    if not containment:failure=failure or {'traceback':'containment not confirmed'}
    budget_cutoff=bool(watchdog) or observation_finished>=deadline
    if not budget_cutoff and worker_result is None:
        failure=failure or {'traceback':'worker completed without an intact result ledger',
                            'read_error':worker_result_read_error}
    result={'method':method,'seed':seed,'n':len(coords),'budget_seconds':budget,
            'status':'failed' if failure else ('budget_cutoff' if budget_cutoff else 'completed'),
            'error':failure,'route':best_route,'length':best_length,'source':best_source,
            'milestones':bank.values,'settings':settings,
            'only_initial':accepted==0,'accepted_checkpoints':accepted,'invalid_checkpoints':invalid,
            'late_checkpoints':late,'observed_checkpoints':reads,'cold_started':began,'deadline':deadline,
            'elapsed_including_cleanup_seconds':ended-began,'cleanup_overrun_seconds':max(0,ended-deadline),
            'parent_validation_seconds':validation_seconds,'watchdog':watchdog,
            'initial_validation_cold_seconds':initial_validated_at-began,
            'scoring_rule':'parent independent EUC_2D validation completed strictly before deadline; no post-cutoff raw-tour read',
            'containment_cleanup_passed':containment,'job_final_accounting':final_accounting,
            'paired_peak_rss_bytes':peak,'resource_samples':samples,
            'parent_plus_job_cpu_seconds':None if final_accounting is None else
                time.process_time()-parent_cpu_start+final_accounting['cpu_seconds'],
            'resource_scope':'sampled parent+descendant RSS, lower bound, shared pages double counted; CPU includes cleanup',
            'worker_result':worker_result,'missing_worker_cost_ledger':worker_result is None,
            'worker_result_read_error':worker_result_read_error,'numba_cache_directory':str(cache)}
    write_new(directory/'result.json',result)
    return result
