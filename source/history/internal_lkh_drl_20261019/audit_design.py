"""Outcome-blind audit of sealed internal-LKH experiment inputs and protocol.

Never reads solver outcomes or runs a solver. Training-specific RL audits are
reported by the training pipeline and independently reviewed before execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

METHODS=('A','U','D')
SEEDS=(101,211,307,401,503)
TRAIN_SEEDS=(3101,3211,3307,3401,3503)
PUBLIC={'pr299':299,'pr439':439,'rat783':783,'rl5915':5915}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def coordinates_digest(coords):
    return digest(json.dumps(coords,separators=(',',':'),ensure_ascii=True).encode('ascii'))


def confined(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('missing or unconfined registered file: '+str(relative))
    return path


def tour_cost(a,path,coords):
    text=a.text(path)
    if 'TOUR_SECTION' not in text:raise ValueError('missing training tour section')
    route=[]
    for token in text.split('TOUR_SECTION',1)[1].split():
        if token in ('-1','EOF'):break
        route.append(int(token)-1)
    if sorted(route)!=list(range(len(coords))):raise ValueError('invalid training Hamiltonian tour')
    total=0
    for first,last in zip(route,route[1:]+route[:1]):
        dx=coords[first][0]-coords[last][0]
        dy=coords[first][1]-coords[last][1]
        total+=math.floor(math.sqrt(dx*dx+dy*dy)+.5)
    return total


class Audit:
    def __init__(self):
        self.checks=[]
        self.errors=[]
        self.opened={}

    def check(self,condition,name,detail=None):
        item=dict(name=name,passed=bool(condition))
        if detail is not None:item['detail']=detail
        self.checks.append(item)
        if not condition:self.errors.append(item)

    def bytes(self,path):
        data=Path(path).read_bytes()
        self.opened[str(Path(path).resolve())]=digest(data)
        return data

    def json(self,path):
        return json.loads(self.bytes(path).decode('utf-8-sig'))

    def text(self,path):
        return self.bytes(path).decode('utf-8-sig')


def audit(study,a):
    seal=a.json(study/'freeze_complete.json')
    a.check(seal['all_inputs_models_schedule_frozen_before_first_job'] is True,'prospective_freeze')
    for relative,expected in seal['sealed_file_hashes'].items():
        a.check(digest(a.bytes(confined(study,relative)))==expected,'sealed_'+relative)
    protocol=a.json(study/'frozen_protocol.json')
    a.check(tuple(protocol['methods'])==METHODS and tuple(protocol['seeds'])==SEEDS,'three_arms_five_seeds')
    a.check(protocol['total_budget_seconds']==30 and protocol['planned_jobs']==180,'thirty_seconds_180_jobs')
    a.check(protocol['development_smoke'] is False,'formal_not_smoke')
    a.check(protocol['quantum_execution'] is False,'no_quantum')
    a.check(protocol['bootstrap_draws']==20000 and '97.5%' in protocol['uncertainty'],'prospective_adjusted_bootstrap')
    verifier_registration=a.json(study/'independent_verifier_registration.json')
    a.check(verifier_registration['formal_outcomes_read']==0 and verifier_registration['formal_solver_jobs_read']==0,
            'independent_verifier_registered_without_formal_outcomes')
    a.check(verifier_registration['verifier_sha256']==digest(a.bytes(study/'source/verify_native_study.py')),
            'registered_independent_verifier_unchanged')
    a.check(datetime.fromisoformat(verifier_registration['registered_at_utc'])<=datetime.fromisoformat(protocol['frozen_at']),
            'independent_verifier_registered_before_freeze')
    settings=protocol['settings']
    a.check(settings['external_initial_tour'] is False and settings['external_route_pool'] is False,
            'no_external_initial_tour_or_route_pool')
    a.check(settings['common_paid_nearest_neighbor_fallback'] is True,'same_paid_fallback')
    a.check((settings['knn'],settings['active_candidates'],settings['retain_alpha_candidates'])==(16,5,2),
            'fixed_internal_candidate_sizes')

    source=study/'source'
    worker=a.text(source/'worker.py')
    runner=a.text(source/'run.py')
    for key in ('INITIAL_TOUR_FILE','INPUT_TOUR_FILE','INITIAL_TOUR_FRACTION'):
        a.check(key not in worker,'worker_has_no_'+key)
    a.check("task['deadline'] - .20" in worker,'common_point20_return_guard')
    a.check("route = nearest_neighbor(task['coords'], task['seed'])" in worker,'worker_pays_nn')
    a.check("not key.upper().startswith('INTERNAL_RL_')" in worker,'clear_inherited_intervention_environment')
    a.check("env['INTERNAL_RL_MODE'] = 'uniform'" in worker and "env['INTERNAL_RL_MODEL']" in worker,
            'uniform_and_trained_modes_inside_lkh')
    a.check('from deep_search' not in worker and 'from neural_model' not in worker,'no_external_local_optimizer')
    a.check('policy_by_test_seed' in runner and 'binary_by_method' in runner,'registered_model_binary_per_arm')
    inherited=a.json(study/'engine_inheritance.json')['byte_identical_engine_hashes']
    for name,expected in inherited.items():
        a.check(digest(a.bytes(source/name))==digest(a.bytes(study/'baseline_reference'/name))==expected,
                'unchanged_clock_containment_'+name)
    binary_registry=a.json(study/'binaries/binary_registry.json')
    build_manifest=a.json(study/'binaries/build_manifest.json')
    a.check(build_manifest['builds_complete'] is True,'both_builds_complete')
    a.check(build_manifest['compiler_sha256']==binary_registry['compiler_sha256']
            and build_manifest['flags']==binary_registry['flags'],'same_registered_compiler_and_flags')
    a.check(build_manifest['common_portability_change']==binary_registry['common_portability_change'],
            'common_portability_patch_disclosed')
    by_method=binary_registry['by_method']
    a.check(by_method['U']==by_method['D'] and by_method['A']!=by_method['D'],'matched_internal_binary_separate_ordinary')
    for method,path in by_method.items():
        actual=confined(study/'binaries',path)
        a.check(actual==Path(settings['binary_by_method'][method]).resolve(),'binary_settings_'+method)
        a.check(actual.suffix=='.exe' and actual.stat().st_size>10000,'executable_exists_'+method)
        a.check(digest(a.bytes(actual))==binary_registry['sha256'][path], 'registered_executable_'+method)
    environment=a.json(study/'environment.json')
    for method,path in settings['binary_by_method'].items():
        a.check(digest(a.bytes(path))==environment['lkh_binary_sha256'][method],'binary_hash_'+method)

    records=a.json(study/'datasets.json')['maps']
    manifest=a.json(study/'inputs/manifest.json')
    a.check(len(records)==12 and len({x['id'] for x in records})==12,'twelve_unique_maps')
    a.check(Counter(x['stratum'] for x in records)==Counter(synthetic_uniform=4,synthetic_clustered=4,tsplib=4),
            'four_maps_each_stratum')
    a.check({x['id']:x['n'] for x in records if x['stratum']=='tsplib'}==PUBLIC,'fresh_public_selection')
    a.check(manifest['source_script_sha256']==digest(a.bytes(source/'prepare_inputs.py')),'frozen_generator_used')
    a.check(manifest['no_solver_work_performed'] is True,'input_preparation_no_solver')
    old=set(manifest['prior_formal_coordinate_hashes'])
    test_hashes=set()
    from decimal import Decimal
    for item in records:
        value=a.json(study/'inputs'/(item['id']+'.json'))
        coords=value['coords']
        h=coordinates_digest(coords)
        a.check(len(coords)==item['n'] and all(len(p)==2 and all(type(v) is int for v in p) for p in coords),
                'integer_coordinates_'+item['id'])
        a.check(h==item['coordinate_sha256']==value['coordinate_sha256'],'coordinate_hash_'+item['id'])
        a.check(h not in old,'disjoint_prior_formal_tests_'+item['id'])
        test_hashes.add(h)
        if item['stratum']=='tsplib':
            provenance=value['source_provenance']
            raw=a.bytes(study/'inputs'/provenance['raw_tsp_path'])
            a.check(digest(raw)==provenance['raw_tsp_sha256'],'raw_public_hash_'+item['id'])
            inside=False
            rows={}
            for line in raw.decode('ascii').splitlines():
                line=line.strip()
                if line=='NODE_COORD_SECTION':inside=True
                elif inside and line not in ('','EOF'):
                    fields=line.split()
                    vals=[Decimal(v) for v in fields[1:]]
                    a.check(len(vals)==2 and all(v.is_finite() and v==v.to_integral_value() for v in vals),
                            'public_exact_integer_'+item['id'])
                    rows[int(fields[0])]=[int(v) for v in vals]
            a.check([rows[i+1] for i in range(item['n'])]==coords,'no_public_coordinate_transform_'+item['id'])
    a.check(len(test_hashes)==12,'distinct_test_coordinates')
    for family in ('uniform','clustered'):
        a.check(sorted(x['n'] for x in records if x['family']==family)==[1000,3000,5000,8000],
                'fixed_test_sizes_'+family)

    separation=a.json(study/'train_test_separation.json')
    train=set(separation['train_coordinate_sha256'])
    validation=set(separation['validation_coordinate_sha256'])
    a.check(len(train)==128 and len(validation)==32,'fresh_128train_32validation')
    a.check(not train&validation and not test_hashes&(train|validation),'all_splits_coordinate_disjoint')
    map_records={}
    for split,expected in (('train',train),('validation',validation)):
        rows=a.json(study/'models'/(split+'_manifest.json'))
        actual=set()
        for row in rows:
            record=a.json(confined(study/'models',row['path']))
            coords=record['coords']
            h=coordinates_digest(coords)
            a.check(len(coords)==row['n'] and all(len(p)==2 and all(type(v) is int for v in p) for p in coords),
                    'actual_'+split+'_integer_coordinates')
            a.check(h==row['coordinate_sha256'],'actual_'+split+'_coordinate_hash')
            actual.add(h)
            map_records[row['id']]=dict(row,coords=coords)
        a.check(actual==expected and len(rows)==len(expected),'actual_'+split+'_split_contents')
    registry=a.json(study/'models/model_registry.json')['by_test_seed']
    a.check(set(registry)=={str(v) for v in SEEDS},'five_registered_models')
    a.check(tuple(registry[str(s)]['training_seed'] for s in SEEDS)==TRAIN_SEEDS,'fixed_training_seed_mapping')
    model_hashes=[]
    for seed in SEEDS:
        record=registry[str(seed)]
        path=confined(study/'models',record['D'])
        a.check(a.text(path).startswith('IRL_MLP_V1'),'native_deep_runtime_export_'+str(seed))
        model_hashes.append(digest(a.bytes(path)))
        a.check(Path(settings['policy_by_test_seed'][str(seed)]['D']).resolve()==path,'selected_model_mapping_'+str(seed))
    a.check(len(set(model_hashes))==5,'five_distinct_policy_exports')
    report=a.json(study/'models/training_report.json')
    config=a.json(study/'models/frozen_training_config.json')
    a.check(report['pilot'] is False and report['actual_reinforcement_learning_inside_lkh'] is True,
            'formal_actual_internal_lkh_reinforcement_training')
    a.check(report['deep_neural_network'] is True and report['test_data_read'] is False,
            'deep_training_has_no_test_exposure')
    a.check(report['validation_only_checkpoint_selection'] is True and report['validation_gradients_computed'] is False,
            'checkpoint_selection_only_validation_no_gradients')
    a.check(config['actor_architecture']==[12,64,32,1] and config['critic_architecture']==[25,32,1],
            'two_hidden_layer_actor_nonlinear_critic')
    a.check(config['epochs']==2 and config['test_data_read'] is False,'fixed_two_training_passes')
    for key in ('end_to_end_seconds','baseline_preparation_seconds','training_seconds','validation_seconds'):
        a.check(math.isfinite(report[key]) and report[key]>0,'disclosed_offline_cost_'+key)
    for name,expected in report['source_sha256'].items():
        a.check(digest(a.bytes(source/name))==expected,'training_source_unchanged_'+name)
    baselines={}
    for map_id,entry in map_records.items():
        leaf=study/'models/baselines'/map_id
        cost=tour_cost(a,leaf/'best.tour',entry['coords'])
        episode=a.json(leaf/'episode.json')
        a.check(cost==episode['length'],'recomputed_training_baseline_'+map_id)
        baselines[map_id]=cost
    by_training_seed={r['training_seed']:r for r in report['models']}
    for test_seed in SEEDS:
        entry=registry[str(test_seed)]
        trained=by_training_seed[entry['training_seed']]
        folder=(study/'models'/entry['D']).parent
        a.check(trained['actor_parameter_count']==2945 and trained['parameter_count']==3810,
                'deep_parameter_count_'+str(test_seed))
        selected=max(trained['checkpoint_scores'],key=lambda r:r['mean_improvement_fraction'])
        a.check(trained['selected_checkpoint']==selected['episode'],'validation_selected_best_earliest_'+str(test_seed))
        a.check(digest(a.bytes(folder/'model_D.txt'))==trained['model_file_sha256'],
                'native_model_matches_training_report_'+str(test_seed))
        a.check(all(math.isfinite(v) and v>0 for v in trained['selected_parameter_change_l2'].values()),
                'both_actor_hidden_layers_and_critic_trained_'+str(test_seed))
        raw=a.text(folder/'training_episodes.jsonl')
        episodes=[json.loads(line) for line in raw.splitlines() if line.strip()]
        a.check(len(episodes)==256 and trained['episodes']==256,'two128map_training_passes_'+str(test_seed))
        map_counts=Counter()
        for ep in episodes:
            map_id=ep['map_id']
            actual=map_records[map_id]
            map_counts[map_id]+=1
            a.check(ep['coordinate_sha256'] in train and ep['coordinate_sha256']==actual['coordinate_sha256'],
                    'training_episode_only_train_coordinates')
            leaf=folder/'training'/f'episode_{ep["episode"]:04d}'/'lkh'
            cost=tour_cost(a,leaf/'best.tour',actual['coords'])
            native=a.json(leaf/'episode.json')
            a.check(cost==ep['policy_length']==native['length'] and ep['baseline_length']==baselines[map_id],
                    'actual_native_training_return_recomputed')
            reward=config['reward_scale']*(baselines[map_id]-cost)/baselines[map_id]
            a.check(math.isclose(reward,ep['terminal_reward'],rel_tol=1e-12,abs_tol=1e-12),
                    'actual_lkh_terminal_reward_recomputed')
            a.check(all(math.isfinite(ep[k]) for k in ('loss','actor_loss','critic_loss','gradient_norm_before_clip')),
                    'finite_actual_return_actor_critic_gradients')
            a.check(ep['return_is_actual_native_lkh_final_tour'] is True,'native_training_not_external_twoopt')
        a.check(len(map_counts)==128 and set(map_counts.values())=={2},'every_training_map_twice_'+str(test_seed))
        for score in trained['checkpoint_scores']:
            leaf=folder/f'checkpoint_{score["episode"]:04d}'/'validation'
            summary=a.json(leaf/'validation_summary.json')
            a.check(summary['gradients'] is False and len(summary['records'])==32,
                    'validation_32_maps_without_gradients')
            improvements=[]
            seen=set()
            for record in summary['records']:
                actual=map_records[record['id']]
                seen.add(record['coordinate_sha256'])
                cost=tour_cost(a,leaf/record['id']/'best.tour',actual['coords'])
                base=baselines[record['id']]
                a.check(record['coordinate_sha256'] in validation and cost==record['policy_length']
                        and base==record['baseline_length'],'validation_native_return_recomputed')
                improvement=(base-cost)/base
                a.check(math.isclose(improvement,record['improvement_fraction'],rel_tol=1e-12,abs_tol=1e-12),
                        'validation_map_score_recomputed')
                improvements.append(improvement)
            mean=sum(improvements)/len(improvements)
            a.check(seen==validation and math.isclose(mean,summary['mean_improvement_fraction'],rel_tol=1e-12,abs_tol=1e-12)
                    and math.isclose(mean,score['mean_improvement_fraction'],rel_tol=1e-12,abs_tol=1e-12),
                    'heldout_checkpoint_score_rebuilt_from_routes')
            a.check(score['export_parity']['passed'] is True
                    and score['export_parity']['torch_numpy_native_max_absolute_logit_error']<=2e-5,
                    'training_native_export_forward_parity')

    schedule=a.json(study/'schedule.json')
    expected={(r['id'],s,m) for r in records for s in SEEDS for m in METHODS}
    counts=Counter((j['map_id'],j['seed'],j['method']) for j in schedule)
    a.check(set(counts)==expected and set(counts.values())=={1},'all180_paired_jobs_once')
    a.check([j['job'] for j in schedule]==list(range(1,181)),'sequential_jobs')
    orders,positions,adjacencies=Counter(),Counter(),Counter()
    for block in range(60):
        jobs=schedule[block*3:block*3+3]
        order=''.join(j['method'] for j in jobs)
        a.check(len({(j['map_id'],j['seed']) for j in jobs})==1,'paired_map_seed_block')
        a.check(all(j['block']==block+1 and j['position']==i+1 and j['block_order']==order for i,j in enumerate(jobs)),
                'schedule_block_metadata')
        orders[order]+=1
        positions.update((j['method'],j['position']) for j in jobs)
        adjacencies.update(zip(order,order[1:]))
    a.check(orders==Counter({''.join(p):10 for p in itertools.permutations(METHODS)}),'six_orders_ten_each')
    a.check(set(positions.values())=={20},'balanced_positions')
    a.check(len(adjacencies)==6 and set(adjacencies.values())=={20},'balanced_adjacencies')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    study,output=args.study.resolve(),args.output.resolve()
    if output.is_relative_to(study):raise ValueError('audit output must be outside study')
    a=Audit()
    try:audit(study,a)
    except BaseException as exc:
        import traceback
        a.check(False,'audit_exception',dict(error=str(exc),traceback=traceback.format_exc()))
    result=dict(schema=1,at=datetime.now(timezone.utc).isoformat(),passed=not a.errors,
                errors=a.errors,warnings=[],check_count=len(a.checks),checks=a.checks,
                opened_file_sha256=a.opened,outcome_blind=True,formal_outcomes_read=0,solver_jobs_launched=0,
                limitations='Static protocol/input/model-export audit; training gradients and selected checkpoint reconstruction are separately reviewed; runtime internal-call proof is checked in formal results.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(a.checks),errors=len(a.errors),output=str(output))))
    if a.errors:raise SystemExit(1)


if __name__=='__main__':main()
