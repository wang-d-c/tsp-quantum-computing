"""Completion-gated independent score and native candidate-policy audit.

python -B verify_native_study.py STUDY_DIRECTORY
No solver/training imports or launches. Never reads raw LKH best.tour. All
reported lengths are reconstructed with integer arithmetic from parent routes.
The audit is prospectively registered before formal outcomes are inspected.
"""
from __future__ import annotations

import argparse
import csv
import io
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
METHODS = ("A", "U", "D")
COMPARISONS = (("D", "A"), ("D", "U"), ("U", "A"))
STRATA = ("synthetic_uniform", "synthetic_clustered", "tsplib")
FIXED_PARAMETERS = {"CANDIDATE_SET_TYPE": "ALPHA", "MAX_CANDIDATES": "5",
    "RUNS": "100000", "MAX_TRIALS": "100000", "STOP_AT_OPTIMUM": "NO",
    "TIME_LIMIT": "86400", "TOTAL_TIME_LIMIT": "86400", "TRACE_LEVEL": "1"}


def route_hash(route):
    return hashlib.sha256(json.dumps(route, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def exact_length(coords, route):
    if (not isinstance(route, list) or len(route) != len(coords)
            or any(type(city) is not int for city in route)
            or set(route) != set(range(len(coords)))):
        raise ValueError("route is not a Hamiltonian permutation")
    total = 0
    for i, city in enumerate(route):
        other = route[(i + 1) % len(route)]
        dx, dy = coords[city][0] - coords[other][0], coords[city][1] - coords[other][1]
        squared = dx * dx + dy * dy
        lower = math.isqrt(squared)
        # sqrt(integer) cannot be an exact half integer. This is exact EUC_2D.
        total += lower + int(squared - lower * lower > lower)
    return total


def independent_nn(coords, seed):
    import numpy as np
    points = np.asarray(coords, dtype=np.int64)
    span = points.max(axis=0) - points.min(axis=0)
    if sum(int(v) ** 2 for v in span) > np.iinfo(np.int64).max // 4:
        raise ValueError("unsafe coordinates for vectorized independent NN")
    used = np.zeros(len(coords), dtype=bool)
    city = random.Random(seed).randrange(len(coords))
    route = []
    for position in range(len(coords)):
        route.append(city)
        used[city] = True
        if position + 1 == len(coords):
            break
        delta = points - points[city]
        squared = np.sum(delta * delta, axis=1)
        lower = np.sqrt(squared).astype(np.int64)
        lower -= (lower * lower > squared).astype(np.int64)
        lower += ((lower + 1) * (lower + 1) <= squared).astype(np.int64)
        costs = lower + (squared - lower * lower > lower).astype(np.int64)
        costs[used] = np.iinfo(np.int64).max
        city = int(np.argmin(costs))
    return route


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(first, second):
    return finite(first) and finite(second) and math.isclose(first, second, rel_tol=0, abs_tol=1e-7)


def outcomes(values):
    return dict(wins=sum(v < 0 for v in values), ties=sum(v == 0 for v in values),
                losses=sum(v > 0 for v in values))


def percentile(ordered, probability):
    p = (len(ordered) - 1) * probability
    low, high = math.floor(p), math.ceil(p)
    return ordered[low] + (ordered[high] - ordered[low]) * (p - low)


def bootstrap(map_rows, draws, seed, confirmatory):
    groups = {name: [r["mean_relative_percent"] for r in map_rows if r["stratum"] == name]
              for name in sorted(STRATA)}
    if not all(groups.values()) or sum(map(len, groups.values())) != len(map_rows):
        raise ValueError("bootstrap strata incomplete")
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        total = 0.0
        for values in groups.values():
            total += sum(values[rng.randrange(len(values))] for _ in values)
        samples.append(total / len(map_rows))
    samples.sort()
    result = dict(mean=statistics.fmean(r["mean_relative_percent"] for r in map_rows),
                  percentile95=[percentile(samples, .025), percentile(samples, .975)],
                  draws=draws, seed=seed, strata_counts={k: len(v) for k, v in groups.items()},
                  sampling_unit="whole map; five paired seeds remain inside each map mean",
                  inference_scope="conditional on five fixed trained-policy/NN/LKH seed combinations")
    if confirmatory:
        result.update(bonferroni_two_comparisons=[percentile(samples, .0125), percentile(samples, .9875)],
                      family_size=2, individual_nominal_confidence=.975, family_nominal_confidence=.95)
    return result


class Auditor:
    def __init__(self, root):
        self.root, self.errors, self.warnings = root, [], []
        self.hashes, self.paths, self.counts = {}, {}, Counter()

    def require(self, condition, context, message):
        if not condition:
            self.errors.append(dict(context=str(context), message=message))
        return bool(condition)

    def warn(self, context, message):
        self.warnings.append(dict(context=str(context), message=message))

    def payload(self, path):
        path = Path(path).resolve()
        relative=path.relative_to(self.root) if path.is_relative_to(self.root) else None
        if (path.name.lower() == "best.tour" and relative is not None
                and len(relative.parts[0])==4 and relative.parts[0].startswith('j')
                and relative.parts[0][1:].isdigit()):
            raise ValueError("formal raw best.tour reads are prohibited")
        data = path.read_bytes()
        key = path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else str(path)
        self.paths[key], self.hashes[key] = path, hashlib.sha256(data).hexdigest()
        return data

    def text(self, path):
        return self.payload(path).decode("utf-8-sig")

    def read(self, path):
        return json.loads(self.text(path))

    def check_route(self, coords, record, context, kind):
        value = exact_length(coords, record["route"])
        self.counts[kind] += 1
        self.require(type(record.get("length")) is int and record["length"] == value,
                     context, "exact integer EUC_2D length differs")
        return value


def section(text, name):
    lines = text.splitlines()
    positions = [i for i, line in enumerate(lines) if line.strip() == name]
    if len(positions) != 1:
        raise ValueError("missing/duplicate " + name)
    header = {}
    for line in lines[:positions[0]]:
        if line.strip():
            key, value = line.split(":", 1)
            if key.strip().upper() in header:
                raise ValueError("duplicate header")
            header[key.strip().upper()] = value.strip()
    return header, lines[positions[0] + 1:]


def initial_tour(text, dimension):
    header, lines = section(text, "TOUR_SECTION")
    if header.get("TYPE") != "TOUR" or int(header.get("DIMENSION", "-1")) != dimension:
        raise ValueError("initial-tour type/dimension mismatch")
    tokens = " ".join(lines).split()
    stop = tokens.index("-1")
    if tokens[stop + 1:] not in ([], ["EOF"]):
        raise ValueError("initial-tour trailing tokens")
    return [int(value) - 1 for value in tokens[:stop]]


def problem_coords(text, dimension):
    header, lines = section(text, "NODE_COORD_SECTION")
    if (header.get("TYPE") != "TSP" or header.get("EDGE_WEIGHT_TYPE") != "EUC_2D"
            or int(header.get("DIMENSION", "-1")) != dimension):
        raise ValueError("LKH input type/dimension mismatch")
    coords, terminated = [], False
    for line in lines:
        if not line.strip():
            continue
        if line.strip() == "EOF":
            terminated = True
            continue
        if terminated:
            raise ValueError("data after EOF")
        city, x, y = map(int, line.split())
        if city != len(coords) + 1:
            raise ValueError("LKH input node IDs changed")
        coords.append([x, y])
    if len(coords) != dimension or not terminated:
        raise ValueError("incomplete LKH input")
    return coords


def comparisons(records, metadata, seeds, protocol, endpoint="final"):
    indexed = {(r["map_id"], r["seed"], r["method"]): r for r in records}
    output = {}
    for candidate, baseline in COMPARISONS:
        pairs, maps = [], []
        for item in metadata:
            values = []
            for seed in seeds:
                first, second = indexed[(item["id"], seed, candidate)], indexed[(item["id"], seed, baseline)]
                a = first["length"] if endpoint == "final" else first["milestones"][endpoint]
                b = second["length"] if endpoint == "final" else second["milestones"][endpoint]
                difference = 100.0 * (a - b) / b
                flags = first["lkh_output_observed"], second["lkh_output_observed"]
                stage = ("unknown_output_ledger" if None in flags else "both_outputs" if all(flags)
                         else "neither_output" if not any(flags) else "mixed_outputs")
                pairs.append(dict(map_id=item["id"], stratum=item["stratum"], joint_seed=seed,
                                  relative_percent=difference, candidate_length=a, baseline_length=b,
                                  candidate_final_source=first["final_source"], baseline_final_source=second["final_source"], stage=stage))
                values.append(difference)
            maps.append(dict(map_id=item["id"], stratum=item["stratum"], mean_relative_percent=statistics.fmean(values)))
        confirmatory = endpoint == "final" and candidate == "D"
        estimate = bootstrap(maps, protocol["bootstrap_draws"], protocol["bootstrap_seed"], confirmatory)
        interval = estimate["bonferroni_two_comparisons"] if confirmatory else estimate["percentile95"]
        diagnostics = {}
        for stage in ("both_outputs", "neither_output", "mixed_outputs", "unknown_output_ledger"):
            groups = defaultdict(list)
            selected = [pair for pair in pairs if pair["stage"] == stage]
            for pair in selected:
                groups[pair["map_id"]].append(pair["relative_percent"])
            diagnostics[stage] = dict(pairs=len(selected), maps=len(groups),
                equal_map_mean_percent=statistics.fmean(statistics.fmean(v) for v in groups.values()) if groups else None,
                **outcomes([pair["relative_percent"] for pair in selected]))
        output[candidate + "_vs_" + baseline] = dict(endpoint=endpoint,
            role="confirmatory family of two" if confirmatory else "descriptive unadjusted",
            bootstrap=estimate, direction="shorter" if interval[1] < 0 else "longer" if interval[0] > 0 else "inconclusive",
            exceeds_engineering_gain=interval[1] < -protocol["engineering_margin_percent"],
            pair_outcomes=outcomes([pair["relative_percent"] for pair in pairs]),
            map_outcomes=outcomes([item["mean_relative_percent"] for item in maps]), maps=maps, pairs=pairs,
            stage_diagnostic=diagnostics,
            by_joint_seed={str(seed): dict(mean_relative_percent=statistics.fmean(p["relative_percent"] for p in pairs if p["joint_seed"] == seed),
                **outcomes([p["relative_percent"] for p in pairs if p["joint_seed"] == seed])) for seed in seeds})
    return output


def native_snapshots(audit, path):
    snapshots=[]
    for line in audit.text(path).splitlines(keepends=True):
        if not line.endswith(('\n','\r')):break
        try:value=json.loads(line)
        except ValueError:
            audit.require(False,path,'invalid complete native statistics line')
            continue
        audit.require(isinstance(value,dict),path,'native statistics snapshot is not an object')
        if isinstance(value,dict):snapshots.append(value)
    return snapshots


def independent_knn(coords):
    """Continuous geometric kNN, with exact squared-distance/city-ID ties."""
    import numpy as np
    points=np.asarray(coords,dtype=np.int64)
    n,nk=len(points),min(16,len(points)-1)
    result=[]
    means,minima=[],[]
    for city in range(n):
        delta=points-points[city]
        squared=np.sum(delta*delta,axis=1)
        squared[city]=np.iinfo(np.int64).max
        threshold=np.partition(squared,nk-1)[nk-1]
        choices=np.flatnonzero(squared<=threshold)
        order=choices[np.lexsort((choices,squared[choices]))][:nk]
        d=[math.sqrt(int(squared[target])) for target in order]
        result.append([int(v)+1 for v in order])
        # Match the mathematical feature, independently of C loop rounding.
        means.append(math.fsum(d)/nk)
        minima.append(d[0])
    return dict(targets=result,means=means,minima=minima)


def load_native_actor_independently(audit,path):
    import numpy as np
    words=audit.text(path).split()
    if words[:5]!=['IRL_MLP_V1','12','64','32','1']:
        raise ValueError('native actor architecture/header mismatch')
    flat=np.asarray([float(v) for v in words[5:]],dtype=np.float64)
    if len(flat)!=2945 or not np.isfinite(flat).all():raise ValueError('invalid native actor parameters')
    sizes=((64,12),(64,),(32,64),(32,),(1,32),(1,))
    arrays=[]
    offset=0
    for shape in sizes:
        size=math.prod(shape)
        arrays.append(flat[offset:offset+size].reshape(shape))
        offset+=size
    return arrays


def actor_forward(features,arrays):
    import numpy as np
    w1,b1,w2,b2,w3,b3=arrays
    hidden=np.maximum(features@w1.T+b1,0.)
    hidden=np.maximum(hidden@w2.T+b2,0.)
    return np.clip((hidden@w3.T+b3).reshape(-1),-10.,10.)


def uniform_scores(seed,count):
    state=seed or 1
    mask=(1<<64)-1
    values=[]
    for _ in range(count):
        state^=state>>12
        state^=(state<<25)&mask
        state^=state>>27
        state&=mask
        values.append((((state*2685821657736338717)&mask)>>11)/9007199254740992.)
    return values


def audit_native(audit,directory,task,worker,coords,knn,context):
    import numpy as np
    folder=directory/'lkh'
    stats_path=folder/'internal_stats.json'
    scores_path=folder/'scored_candidates.csv'
    selected_path=folder/'selected_candidates.csv'
    if task['method']=='A':
        audit.require(not any(p.exists() for p in (stats_path,scores_path,selected_path)),context,
                      'ordinary binary produced intervention artifacts')
        return dict(hook_reached=False,ordinary=True)
    snapshots=native_snapshots(audit,stats_path) if stats_path.is_file() else []
    active=[v for v in snapshots if v.get('active') is True]
    phase=worker.get('lkh') or {}
    if not active:
        audit.require(not phase.get('observed_tour_files',0),context,'LKH emitted complete tour without active internal-hook proof')
        audit.counts['internal_hook_not_reached_before_cutoff']+=1
        return dict(hook_reached=False,ordinary=False,snapshot_count=len(snapshots))
    latest=active[-1]
    audit.require(latest.get('dimension')==len(coords) and latest.get('features')==12
                  and latest.get('hidden')==[64,32],context,'native architecture/dimension differs')
    audit.require(latest.get('has_model') is (task['method']=='D'),context,'native model activation differs')
    audit.require(phase.get('internal_policy_statistics')==snapshots[-1],context,'worker native ledger differs from final complete snapshot')
    previous=None
    for snap in snapshots:
        if previous:
            for key in ('scored_candidates','selected_candidates','rerank_calls','best5opt_calls','cost_checks'):
                audit.require(snap.get(key,-1)>=previous.get(key,-1),context,'native counter decreases: '+key)
        previous=snap
    for key in ('model_load_seconds','feature_build_seconds','inference_seconds','rank_seconds'):
        audit.require(finite(latest.get(key)) and latest[key]>=0,context,'invalid paid native timing: '+key)
    audit.require(sum(latest[k] for k in ('model_load_seconds','feature_build_seconds','inference_seconds','rank_seconds'))
                  <=phase.get('elapsed_seconds',0)+.02,context,'native nonoverlapping stage time exceeds full LKH phase')

    expected_score_header=['node','target','score','selected','selected_rank','alpha','alpha_rank','knn_rank','protected']
    reader=csv.DictReader(io.StringIO(audit.text(scores_path)))
    audit.require(reader.fieldnames==expected_score_header,context,'full candidate-score schema differs')
    scored=[]
    by_node=defaultdict(list)
    for raw in reader:
        row={key:(float(raw[key]) if key=='score' else int(raw[key])) for key in expected_score_header}
        audit.require(finite(row['score']) and 1<=row['node']<=len(coords) and 1<=row['target']<=len(coords)
                      and row['node']!=row['target'],context,'invalid candidate endpoint/score')
        scored.append(row)
        by_node[row['node']].append(row)
    n=len(coords)
    audit.require(set(by_node)==set(range(1,n+1)),context,'candidate rows omit nodes')
    audit.require(latest.get('scored_candidates')==len(scored),context,'native scoring count differs from full union trace')
    audit.require(latest.get('selected_candidates')==latest.get('cost_checks')==5*n,context,'native selected/cost-check count differs')
    if not knn:knn=independent_knn(coords)
    extent=max(1,max(x for x,y in coords)-min(x for x,y in coords),max(y for x,y in coords)-min(y for x,y in coords))
    features=[]
    for node in range(1,n+1):
        rows=by_node[node]
        audit.require(16<=len(rows)<=21 and len({r['target'] for r in rows})==len(rows),context,'invalid ALPHA5-union-kNN16 size/duplicate target')
        alpha=[r for r in rows if r['alpha_rank']>=0]
        # MAX_CANDIDATES=5 is an upper bound; stock MaxAlpha filtering can
        # leave 2..4 valid ALPHA candidates at a node before the union.
        audit.require(2<=len(alpha)<=5 and [r['alpha_rank'] for r in alpha]==list(range(len(alpha))),
                      context,'bounded ALPHA insertion rank differs')
        audit.require(all(r['alpha_rank'] in (-1,0,1,2,3,4) and -1<=r['knn_rank']<16
                          and (r['alpha_rank']>=0 or r['knn_rank']>=0) for r in rows),
                      context,'candidate outside registered ALPHA5 union kNN16')
        audit.require(sum(r['protected'] for r in rows)==2 and all(r['protected']==int(0<=r['alpha_rank']<2) for r in rows),
                      context,'first two ALPHA edges not protected exactly')
        nearest={r['knn_rank']:r['target'] for r in rows if r['knn_rank']>=0}
        audit.require(nearest==dict(enumerate(knn['targets'][node-1])),context,'independent geometric 16-NN differs')
        audit.require(sum(r['knn_rank']>=0 for r in rows)==16
                      and all(r['knn_rank']<0 or knn['targets'][node-1][r['knn_rank']]==r['target'] for r in rows),
                      context,'duplicate or misassigned kNN rank')
        for row in rows:
            target=row['target']
            dx=coords[target-1][0]-coords[node-1][0]
            dy=coords[target-1][1]-coords[node-1][1]
            ar,kr=row['alpha_rank'],row['knn_rank']
            audit.require(ar>=0 or row['alpha']==2147483646,context,'new kNN edge ALPHA sentinel differs')
            features.append([dx/extent,dy/extent,math.hypot(dx,dy)/extent,
                1. if ar<0 else min(1.,max(0.,math.log1p(max(0,row['alpha'])/100.)/math.log1p(extent))),
                1. if ar<0 else ar/5.,1. if kr<0 else kr/16.,float(ar>=0),float(kr>=0),
                knn['means'][node-1]/extent,knn['means'][target-1]/extent,
                knn['minima'][node-1]/extent,knn['minima'][target-1]/extent])
    audit.require([r['node'] for r in scored]==sorted(r['node'] for r in scored),context,'candidate rows not source-node ordered')
    if task['method']=='D':
        arrays=load_native_actor_independently(audit,task['settings']['policy_by_test_seed'][str(task['seed'])]['D'])
        predicted=actor_forward(np.asarray(features,dtype=np.float64),arrays)
    else:predicted=np.asarray(uniform_scores(task['seed'],len(scored)))
    observed=np.asarray([r['score'] for r in scored])
    error=float(np.max(np.abs(predicted-observed)))
    audit.require(error<=1e-8,context,'independent reconstructed actor/uniform scores differ')

    selected_reader=csv.DictReader(io.StringIO(audit.text(selected_path)))
    selected_header=['node','rank','target','cost','alpha','protected','score','pi_from','pi_to']
    audit.require(selected_reader.fieldnames==selected_header,context,'selected candidate schema differs')
    selected=[]
    chosen_by_node=defaultdict(list)
    for raw in selected_reader:
        row={key:(float(raw[key]) if key=='score' else int(raw[key])) for key in selected_header}
        selected.append(row)
        chosen_by_node[row['node']].append(row)
    audit.require(len(selected)==5*n and set(chosen_by_node)==set(range(1,n+1)),context,'selected trace must contain five per node')
    source_pi={node:rows[0]['pi_from'] for node,rows in chosen_by_node.items()}
    choices={}
    for node in range(1,n+1):
        ranked=sorted(by_node[node],key=lambda r:(-r['score'],r['target']))
        decoded=[]
        optional=0
        for row in ranked:
            if row['protected'] or optional<3:
                decoded.append(row)
                optional+=not row['protected']
        actual=chosen_by_node[node]
        audit.require([r['rank'] for r in actual]==list(range(5)) and len(decoded)==5,context,'selected ranks/count differ')
        audit.require([r['target'] for r in actual]==[r['target'] for r in decoded],context,'actual selection/order differs from protected top-score decoder')
        choices[node]=[r['target'] for r in actual]
        expected_by_target={r['target']:r for r in decoded}
        for row in by_node[node]:
            rank=choices[node].index(row['target']) if row['target'] in choices[node] else -1
            audit.require(row['selected']==int(rank>=0) and row['selected_rank']==rank,context,'full score trace selected flags differ')
        for row in actual:
            original=expected_by_target[row['target']]
            audit.require(all(row[k]==original[k] for k in ('alpha','protected','score')),context,'selected metadata differs from scored union')
            target=row['target']
            dx=coords[node-1][0]-coords[target-1][0]
            dy=coords[node-1][1]-coords[target-1][1]
            square=dx*dx+dy*dy
            lower=math.isqrt(square)
            distance=lower+int(square-lower*lower>lower)
            audit.require(row['pi_from']==source_pi[node] and row['pi_to']==source_pi[target]
                          and row['cost']==distance*100+row['pi_from']+row['pi_to'],context,'transformed candidate Cost/Precision/Pi differs')
    audit.counts['native_complete_hook_traces']+=1
    audit.counts['native_candidate_scores_recomputed']+=len(scored)
    audit.counts['native_selected_costs_recomputed']+=len(selected)
    if latest.get('best5opt_calls',0)>0:audit.counts['native_best5opt_consumption_proved']+=1
    return dict(hook_reached=True,ordinary=False,snapshot_count=len(snapshots),
                latest_statistics=latest,max_score_absolute_error=error,
                scored_edges=len(scored),selected_edges=len(selected),
                choices=choices,
                union_sha256=route_hash([[r[k] for k in ('node','target','alpha','alpha_rank','knn_rank','protected')] for r in scored]))


def audit_parameters(audit,directory,task,worker,coords,context):
    folder=directory/'lkh'
    if not (folder/'run.par').is_file():
        audit.warn(context,'LKH parameter file absent; certified fallback remains scored')
        return
    audit.require(problem_coords(audit.text(folder/'problem.tsp'),len(coords))==coords,context,'LKH input coordinates differ')
    parameters={}
    for line in audit.text(folder/'run.par').splitlines():
        if not line.strip():continue
        key,value=line.split('=',1)
        key=key.strip().upper()
        audit.require(key not in parameters,context,'duplicate LKH parameter')
        parameters[key]=value.strip()
    values=dict(FIXED_PARAMETERS,SEED=str(task['seed']))
    paths=dict(PROBLEM_FILE=folder/'problem.tsp',OUTPUT_TOUR_FILE=folder/'best.tour')
    audit.require(set(parameters)==set(values)|set(paths),context,'unexpected or missing LKH parameter')
    audit.require(not (folder/'initial.tour').exists() and not any('INITIAL_TOUR' in key or 'INPUT_TOUR' in key for key in parameters),
                  context,'external initial or input tour was supplied')
    for key,value in values.items():audit.require(parameters.get(key)==value,context,'LKH setting differs: '+key)
    for key,value in paths.items():audit.require(Path(parameters[key]).resolve()==value.resolve(),context,'LKH file path differs: '+key)
    phase_path=folder/'phase.json'
    if phase_path.is_file():
        phase=audit.read(phase_path)
        audit.require(phase==worker.get('lkh'),context,'standalone and worker LKH ledgers differ')
        audit.require(phase.get('initial_tour_enabled') is False and phase.get('external_route_pool') is False,
                      context,'external preprocessing or route-pool isolation flag differs')
        audit.require(phase.get('internal_mode')=={'A':'ordinary','U':'uniform','D':'deep'}[task['method']],
                      context,'native intervention mode differs')
        expected=(task['settings']['policy_by_test_seed'][str(task['seed'])]['D'] if task['method']=='D' else None)
        audit.require(phase.get('model_path')==expected,context,'native model path differs')
    audit.counts['ordinary_initialization_parameter_checks']+=1


def audit_job(audit,entry,protocol,metadata,coords,nn,knn):
    job,context=entry['job'],f'j{entry["job"]:03d}'
    directory=audit.root/context
    row=audit.read(audit.root/f'row{job:03d}.json')
    result=audit.read(directory/'result.json')
    task=audit.read(directory/'task.json')
    for key in ('job','method','map_id','seed','block','position','block_order'):
        audit.require(row.get(key)==entry.get(key),context,'schedule mismatch: '+key)
    for key,value in result.items():audit.require(row.get(key)==value,context,'row/result mismatch: '+key)
    permitted={'job','map_id','family','stratum','block','position','block_order'}
    audit.require(not set(row)-set(result)-permitted,context,'unexpected driver-added row fields')
    audit.require(row.get('stratum')==metadata['stratum'] and row.get('family')==metadata['family'],context,'family/stratum differs')
    audit.require(row.get('n')==metadata['n']==len(coords),context,'dimension differs')
    for key in ('method','seed','cold_started','deadline'):
        audit.require(task.get(key)==row.get(key),context,'task/result mismatch: '+key)
    audit.require(task.get('coords')==coords,context,'task coordinates differ')
    audit.require(task.get('settings')==protocol['settings']==row.get('settings'),context,'registered settings differ')
    expected_binary=protocol['settings']['binary_by_method'][row['method']]
    audit.require(Path(task['lkh_binary']).resolve()==Path(expected_binary).resolve(),context,'per-method binary differs')
    budget,origin,deadline=protocol['total_budget_seconds'],row['cold_started'],row['deadline']
    audit.require(finite(origin) and finite(deadline) and close(deadline-origin,budget),context,'cold budget differs')
    audit.require(row.get('budget_seconds')==budget==task.get('budget'),context,'budget field differs')
    initial=list(range(len(coords)))
    initial_cost=exact_length(coords,initial)
    initial_cold=row.get('initial_validation_cold_seconds')
    audit.require(finite(initial_cold) and 0<=initial_cold<budget,context,'initial certification outside budget')
    audit.require(task.get('initial_route')==initial,context,'common identity emergency fallback differs')
    incumbent=dict(route=initial,length=initial_cost,source='initial_identity',validated_at=origin+initial_cold)
    events=[]
    for index,line in enumerate((v for v in audit.text(directory/'accepted.jsonl').splitlines() if v.strip()),1):
        event=json.loads(line)
        cost=audit.check_route(coords,event,f'{context}/accepted/{index}','accepted_routes')
        stamp=event.get('validated_at')
        audit.require(finite(stamp) and origin<=stamp<deadline,context,'accepted route certified after cutoff')
        audit.require(stamp>=incumbent['validated_at'] and close(event.get('cold_seconds'),stamp-origin),context,'accepted clock differs')
        audit.require(event.get('sequence')==index and cost<incumbent['length'],context,'accepted sequence or strict improvement differs')
        audit.require(event.get('source') in ('nearest_neighbor','lkh'),context,'unexpected external optimizing source')
        if event.get('source')=='nearest_neighbor':audit.require(event['route']==nn['route'],context,'common independently paid NN differs')
        events.append(event)
        incumbent=event
    audit.check_route(coords,row,context,'final_routes')
    for key in ('route','length','source'):
        audit.require(row.get(key)==incumbent.get(key),context,'final incumbent reconstruction differs: '+key)
    audit.require(row.get('accepted_checkpoints')==len(events) and row.get('only_initial')==(not events),context,'accepted/initial flags differ')
    marks,mark_scores=[],{}
    for mark in row.get('milestones',{}).values():
        seconds=mark['seconds']
        marks.append(seconds)
        eligible=[event for event in events if event['validated_at']<origin+seconds]
        expected=eligible[-1] if eligible else dict(route=initial,length=initial_cost,source='initial_identity',validated_at=origin+initial_cold)
        audit.check_route(coords,mark,context+'/milestone','milestone_routes')
        for key in ('route','length','source','validated_at'):
            audit.require(mark.get(key)==expected.get(key),context,'milestone reconstruction differs: '+key)
        audit.require(mark['validated_at']<origin+seconds,context,'milestone route certified too late')
        audit.require(close(mark.get('validation_cold_seconds'),expected['validated_at']-origin),context,'milestone cold clock differs')
        mark_scores[str(float(seconds))]=mark['length']
        if seconds==budget:audit.require(mark['route']==row['route'] and mark['length']==row['length'],context,'final milestone differs')
    audit.require(sorted(marks)==sorted(protocol['milestones_seconds']),context,'milestone set differs')
    audit.require(row.get('containment_cleanup_passed') is True and (row.get('job_final_accounting') or {}).get('active_processes')==0,
                  context,'contained process cleanup failed')
    audit.require(row.get('invalid_checkpoints')==0 and row.get('late_checkpoints')==0,context,'invalid or late checkpoints were observed')
    path=directory/'worker_result.json'
    worker=audit.read(path) if path.is_file() else None
    audit.require(worker==row.get('worker_result'),context,'standalone worker ledger differs')
    if worker is None:
        audit.warn(context,'worker cost ledger absent; no cost invented')
        worker={}
    else:
        audit.require(worker.get('method')==row['method'] and worker.get('settings')==protocol['settings'],context,'worker method/settings differ')
        for key,value in (('nearest_neighbor_length',nn['length']),('nearest_neighbor_start_city',nn['route'][0]),
                          ('nearest_neighbor_route_sha256',nn['sha256'])):
            audit.require(worker.get(key)==value,context,'independent NN differs: '+key)
        audit.require(worker.get('pre_lkh_length')==nn['length'],context,'external search modified NN before LKH')
        audit.require(worker.get('ordinary_lkh_normal_initialization') is True and worker.get('external_route_pool') is False,
                      context,'normal initialization or external pool flags differ')
        audit.require('preprocessing_search' not in worker and worker.get('group_preprocessing_seconds')==0.,context,'external local optimizer executed')
        audit_parameters(audit,directory,task,worker,coords,context)
    native=audit_native(audit,directory,task,worker,coords,knn,context)
    observed=(worker.get('lkh') or {}).get('observed_tour_files')
    return dict(job=job,map_id=row['map_id'],stratum=metadata['stratum'],seed=row['seed'],method=row['method'],length=row['length'],
                route_sha256=route_hash(row['route']),milestones=mark_scores,final_source=row['source'],status=row.get('status'),
                lkh_output_observed=observed>0 if type(observed) is int else None,observed_lkh_tour_files=observed,
                only_initial=row.get('only_initial'),invalid_checkpoints=row.get('invalid_checkpoints'),late_checkpoints=row.get('late_checkpoints'),
                cleanup_overrun_seconds=row.get('cleanup_overrun_seconds'),pre_lkh_cold_seconds=worker.get('pre_lkh_cold_seconds'),native=native)


def report_text(audit,results):
    lines=['LKH内部深度强化学习：独立核验与统计补充','',
           f'核验通过：{not audit.errors}；错误{len(audit.errors)}；提示{len(audit.warnings)}。',
           'A=普通LKH；U=内部均匀候选选择；D=内部深度RL候选选择。',
           '三组都独立付费计算相同NN兜底，均不向LKH注入初始路线；没有外部2-opt或额外路线池。',
           '每次30秒冷启动，候选构建、模型读取、特征、推理、排序、核验CSV输出和内部LKH路线维护全部计时。',
           '负差表示更短；先在图内平均五个配对种子，再等权平均12张图。',
           'D-A和D-U两个预设比较使用Bonferroni97.5%区间，U-A为描述性比较。','']
    for label,result in results['final_comparisons'].items():
        estimate=result['bootstrap']
        ci=estimate.get('bonferroni_two_comparisons',estimate['percentile95'])
        lines.extend([f'{label}: 均值{estimate["mean"]:+.6f}%；区间[{ci[0]:+.6f}%, {ci[1]:+.6f}%]；{result["direction"]}。',
                      f'配对胜平负{result["pair_outcomes"]}；图级胜平负{result["map_outcomes"]}。',
                      f'LKH输出阶段描述：{result["stage_diagnostic"]}。',''])
    lines.extend(['内部决策的独立证据：',str(results['paired_internal_influence']),
                  f'逐项重算计数：{dict(audit.counts)}。','',
                  '边界：模型是两层ReLU边评分器，在LKH候选构建后计算分数并在ResetCandidateSet后恢复排序；不是每次LK动作重新调用网络。',
                  '训练使用实际LKH终局长度的策略梯度；验证集选择检查点，测试集只用于这次固定比较。不是所给论文注意力网络的完整复现。',
                  '所有成绩来自截止前父进程已验证的完整路线；未读取正式运行的raw best.tour补分。',
                  '没有完整LKH输出时，相同NN兜底会使两组成绩相同；阶段分组仅解释机制，不排除任何正式测试成绩。',
                  '五个模型与求解种子配对，区间条件于这五个已训练模型；12张图、30秒和固定LKH配置限制结论外推。'])
    if audit.errors:lines+=['','错误：']+[str(v) for v in audit.errors]
    if audit.warnings:lines+=['','提示：']+[str(v) for v in audit.warnings]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args()
    root=args.study.resolve()
    # Gate before reading any formal outcome, route, task or candidate artifact.
    if not (root/'complete.json').is_file():parser.error('study incomplete; no outcomes read')
    output=root/'independent_native_review'
    if output.exists():parser.error('independent_native_review exists; refusing overwrite')
    audit=Auditor(root)
    protocol=audit.read(root/'frozen_protocol.json')
    complete=audit.read(root/'complete.json')
    freeze=audit.read(root/'freeze_complete.json')
    schedule=audit.read(root/'schedule.json')
    datasets=audit.read(root/'datasets.json')['maps']
    environment=audit.read(root/'environment.json')
    registry=audit.read(root/'models/model_registry.json')['by_test_seed']
    audit.require(protocol.get('methods')==list(METHODS) and protocol.get('quantum_execution') is False,'protocol','method set or quantum status differs')
    audit.require(protocol.get('development_smoke') is args.development,'protocol','formal/development mismatch')
    if not args.development:
        audit.require(protocol.get('total_budget_seconds')==30. and protocol.get('planned_jobs')==180,'protocol','formal budget/job count differs')
    audit.require(freeze.get('all_inputs_models_schedule_frozen_before_first_job') is True,'freeze','prospective freeze absent')
    audit.require(complete.get('source_hashes_unchanged') is True and complete.get('frozen_models_inputs_unchanged') is True,
                  'completion','unchanged-artifact completion proof absent')
    for relative,expected in freeze['sealed_file_hashes'].items():
        audit.require(hashlib.sha256(audit.payload(root/relative)).hexdigest()==expected,relative,'sealed artifact changed')
    for name,expected in audit.read(root/'source_hashes.json').items():
        audit.require(hashlib.sha256(audit.payload(root/'source'/name)).hexdigest()==expected,name,'frozen source hash differs')
    for kind in ('models','binaries'):
        inventory=audit.read(root/('model_hashes.json' if kind=='models' else 'binary_hashes.json'))
        actual={p.relative_to(root/kind).as_posix() for p in (root/kind).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        audit.require(actual==set(inventory),kind,'frozen artifact inventory differs')
        for relative,expected in inventory.items():
            audit.require(hashlib.sha256(audit.payload(root/kind/relative)).hexdigest()==expected,relative,'frozen artifact SHA256 differs')
    for method,path in protocol['settings']['binary_by_method'].items():
        audit.require(hashlib.sha256(audit.payload(path)).hexdigest()==environment['lkh_binary_sha256'][method],method,'solver binary SHA256 differs')
    prospective_path=root/'prospective_design_audit.json'
    if prospective_path.is_file():
        prospective=audit.read(prospective_path)
        read_count=prospective.get('formal_outcomes_read',prospective.get('formal_outcome_files_read'))
        audit.require(prospective.get('passed') is True and prospective.get('outcome_blind') is True and read_count==0,
                      'prospective audit','outcome-blind pre-test design audit did not pass')
    elif not args.development:audit.require(False,'prospective audit','pre-test design audit missing')
    seeds=protocol['seeds']
    if not args.development:audit.require(seeds==[101,211,307,401,503],'seeds','formal seed set differs')
    audit.require(set(registry)=={str(seed) for seed in seeds},'registry','registered model seed set differs')
    for seed in seeds:
        expected=(root/'models'/registry[str(seed)]['D']).resolve()
        actual=Path(protocol['settings']['policy_by_test_seed'][str(seed)]['D']).resolve()
        audit.require(expected==actual,'registry','native model mapping differs')
    metadata,coords_by_map={},{}
    for item in datasets:
        metadata[item['id']]=item
        path=root/'inputs'/(item['id']+'.json')
        value=audit.read(path)
        coords=value['coords']
        audit.require(hashlib.sha256(audit.payload(path)).hexdigest()==item['sha256'],item['id'],'input file hash differs')
        audit.require(len(coords)==item['n'] and all(len(p)==2 and all(type(v) is int for v in p) for p in coords),item['id'],'input coordinate schema differs')
        audit.require(route_hash(coords)==item['coordinate_sha256'],item['id'],'canonical input coordinate hash differs')
        coords_by_map[item['id']]=coords
    audit.require(len(datasets)==protocol['independent_maps'],'datasets','map count differs')
    if not args.development:
        audit.require(len(datasets)==12 and Counter(x['stratum'] for x in datasets)==dict.fromkeys(STRATA,4),'datasets','formal 12-map strata differ')
    expected={(item['id'],seed,method) for item in datasets for seed in seeds for method in METHODS}
    actual=[(e['map_id'],e['seed'],e['method']) for e in schedule]
    audit.require(set(actual)==expected and len(actual)==len(expected),'schedule','paired design differs')
    audit.require(complete.get('jobs')==len(schedule)==protocol['planned_jobs'],'completion','completed job count differs')
    audit.require([e['job'] for e in schedule]==list(range(1,len(schedule)+1)),'schedule','job indices differ')
    orders=Counter()
    blocks=defaultdict(list)
    for entry in schedule:blocks[entry['block']].append(entry)
    for block in blocks.values():
        jobs=sorted(block,key=lambda e:e['position'])
        order=''.join(e['method'] for e in jobs)
        audit.require(len(jobs)==3 and [e['position'] for e in jobs]==[1,2,3]
                      and len({(e['map_id'],e['seed']) for e in jobs})==1,'schedule','block positions/map/seed differ')
        audit.require(all(e['block_order']==order for e in jobs),'schedule','block order metadata differs')
        orders[order]+=1
    if not args.development:audit.require(orders==Counter({''.join(p):10 for p in itertools.permutations(METHODS)}),'schedule','all-six-order balance differs')

    records,nn_cache,knn_cache=[],{},{}
    for entry in schedule:
        try:
            key=entry['map_id'],entry['seed']
            coords=coords_by_map[key[0]]
            if key not in nn_cache:
                route=independent_nn(coords,key[1])
                nn_cache[key]=dict(route=route,length=exact_length(coords,route),sha256=route_hash(route))
                audit.counts['independent_nn_routes']+=1
            if entry['method'] in ('U','D') and key[0] not in knn_cache:
                knn_cache[key[0]]=independent_knn(coords)
                audit.counts['independent_knn_instances']+=1
            records.append(audit_job(audit,entry,protocol,metadata[key[0]],coords,nn_cache[key],knn_cache.get(key[0])))
        except Exception as exc:
            audit.errors.append(dict(context=f'job{entry.get("job")}',message=f'{type(exc).__name__}: {exc}'))
    indexed={(r['map_id'],r['seed'],r['method']):r for r in records}
    paired_influence=[]
    for item in datasets:
        for seed in seeds:
            u=indexed.get((item['id'],seed,'U'))
            d=indexed.get((item['id'],seed,'D'))
            if not u or not d or not u['native']['hook_reached'] or not d['native']['hook_reached']:continue
            un,dn=u['native'],d['native']
            audit.require(un['union_sha256']==dn['union_sha256'],item['id'],'paired uniform/deep original candidate unions differ')
            uc,dc=un['choices'],dn['choices']
            paired_influence.append(dict(map_id=item['id'],seed=seed,n=item['n'],
                nodes_with_different_order=sum(uc[node]!=dc[node] for node in uc),
                nodes_with_different_selected_set=sum(set(uc[node])!=set(dc[node]) for node in uc)))
    for record in records:record['native'].pop('choices',None)
    final,at10={},{}
    if len(records)==len(expected) and not args.development:
        final=comparisons(records,datasets,seeds,protocol)
        if 10. in protocol['milestones_seconds']:at10=comparisons(records,datasets,seeds,protocol,'10.0')
    elif len(records)!=len(expected):audit.require(False,'statistics','not all jobs audited; no statistics generated')
    main_result=root/'analysis/results.json'
    if main_result.is_file() and final:
        primary=audit.read(main_result)
        for label,result in final.items():
            other=primary['comparisons'][label]['bootstrap']
            current=result['bootstrap']
            audit.require(close(current['mean'],other['mean']),label,'independent mean differs from main analyzer')
            audit.require(all(close(x,y) for x,y in zip(current['percentile95'],other['percentile95'])),
                          label,'independent 95% interval differs from main analyzer')
            if label.startswith('D_'):
                audit.require(all(close(x,y) for x,y in zip(current['bonferroni_two_comparisons'],other['bonferroni_two_comparisons'])),
                              label,'independent adjusted 97.5% interval differs from main analyzer')
    for key,expected_hash in list(audit.hashes.items()):
        audit.require(hashlib.sha256(audit.paths[key].read_bytes()).hexdigest()==expected_hash,key,'evidence changed during audit')
    result=dict(schema=1,audit_passed=not audit.errors,at=datetime.now(timezone.utc).isoformat(),
                development_only=args.development,final_comparisons=final,descriptive10seconds=at10,
                jobs=records,paired_internal_influence=paired_influence,
                source_and_raw_files_modified=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    summary=dict(passed=not audit.errors,errors=audit.errors,warnings=audit.warnings,counts=dict(audit.counts),
                 evidence_sha256=audit.hashes,formal_raw_best_tour_files_read=False,
                 solver_or_training_imported_or_launched=False,script_sha256=result['script_sha256'])
    output.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    for name,value in (('independent_audit.json',summary),('supplemental_results.json',result)):
        with (output/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    (output/'独立核验与统计补充.txt').write_text(report_text(audit,result),encoding='utf-8')
    print(json.dumps(dict(passed=not audit.errors,errors=len(audit.errors),warnings=len(audit.warnings),counts=dict(audit.counts),output=str(output)),ensure_ascii=False))
    return 0 if not audit.errors else 1


if __name__=='__main__':raise SystemExit(main())


