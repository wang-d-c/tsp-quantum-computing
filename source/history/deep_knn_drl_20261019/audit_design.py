"""Outcome-blind audit of the prospective deep-RL versus ordinary-LKH study.

Only sealed source, models, train/validation inputs/logs, test input bytes and
the prospective schedule are opened. No row, job, tour or formal outcome is
opened. Reports belong outside the frozen directory until the root archives
them after the audit. This does not launch a solver or change a model.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
METHODS = ('A', 'U', 'D')
TEST_SEEDS = (101, 211, 307, 401, 503)
TRAIN_SEEDS = (2101, 2211, 2307, 2401, 2503)
PUBLIC = {'rl1304': 1304, 'rl1323': 1323, 'rl1889': 1889, 'u1432': 1432}
ENGINE = {'supervisor.py': 'a7a4893d64fc9e678f129b091841e3719c3e435dd7357977638464bfcf8729cb',
          'resource_v3.py': '00e5778da91378fb59fea263786e8eb3b87bd7a21414b528f167041bac45ffbb'}
CORE_SHA = '9e9a87ba1c5ce6be1192428fd59814aa60c9c12138bab5072e2fd9f9c4ea7447'
LKH_SHA = 'b44414b7c9aa111b4e782ba13fd2cf4684bf3ad505ba1f5d65822b8ad162c044'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def coord_digest(coords):
    return digest(json.dumps(coords, ensure_ascii=True,
                             separators=(',', ':')).encode('ascii'))


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('missing or unconfined sealed file: ' + str(relative))
    return path


class Audit:
    def __init__(self):
        self.checks, self.errors, self.read_sha256 = [], [], {}
        self.details = {}

    def check(self, condition, code, details=None):
        record = dict(code=code, passed=bool(condition), details=details)
        self.checks.append(record)
        if not condition:
            self.errors.append(record)
        return bool(condition)

    def bytes(self, path):
        path = Path(path).resolve()
        value = path.read_bytes()
        self.read_sha256[str(path)] = digest(value)
        return value

    def json(self, path):
        return json.loads(self.bytes(path).decode('utf-8-sig'))

    def hash(self, path):
        return digest(self.bytes(path))


def load_module(source, name):
    spec = importlib.util.spec_from_file_location('deep_audit_' + name, source / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_checks(a, source):
    names = ('run.py', 'worker.py', 'supervisor.py', 'resource_v3.py', 'prepare_inputs.py',
             'search_core.py', 'deep_search.py', 'neural_model.py', 'train_deep.py')
    sources = {name: a.bytes(source / name).decode('utf-8-sig') for name in names}
    trees = {name: ast.parse(text, filename=name) for name, text in sources.items()}
    for name, expected in ENGINE.items():
        a.check(a.hash(source / name) == expected, 'unchanged_validated_engine_' + name)
    a.check(a.hash(source / 'search_core.py') == CORE_SHA, 'unchanged_candidate_pool_core')
    functions = lambda tree: {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    calls = lambda node: [ast.unparse(x.func) for x in ast.walk(node) if isinstance(x, ast.Call)]
    run_fns, worker_fns = functions(trees['run.py']), functions(trees['worker.py'])
    a.check('assert_frozen' in calls(run_fns['execute']), 'seal_rechecked_before_every_job')
    a.check('run_one' in calls(run_fns['execute']), 'uses_inherited_cold_supervisor')
    a.check("('A', 'U', 'D')" in sources['run.py'], 'three_registered_arms')
    a.check('itertools.permutations(METHODS)' in sources['run.py'], 'all_six_execution_orders')
    lkh_strings = {x.value for x in ast.walk(worker_fns['lkh_search'])
                   if isinstance(x, ast.Constant) and isinstance(x.value, str)}
    required = {'CANDIDATE_SET_TYPE = ALPHA', 'MAX_CANDIDATES = 5', 'RUNS = 100000',
                'MAX_TRIALS = 100000', 'STOP_AT_OPTIMUM = NO', 'TIME_LIMIT = 86400',
                'TOTAL_TIME_LIMIT = 86400', 'TRACE_LEVEL = 1', 'INITIAL_TOUR_FRACTION = 1'}
    a.check(required <= lkh_strings, 'common_registered_lkh_parameters')
    a.check(not any(x.startswith(('INPUT_TOUR_FILE', 'OPTIMUM =')) for x in lkh_strings),
            'no_reference_tour_or_known_optimum')
    a.check("if task['method'] in ('U', 'D'):" in sources['worker.py']
            and 'if initial is not None:' in sources['worker.py'],
            'external_initial_tour_only_uniform_and_deep')
    a.check("if task['method'] == 'A':" in sources['worker.py']
            and "route = task['initial_route']" in sources['worker.py']
            and "publisher(route, 'initial_identity')" in sources['worker.py'],
            'ordinary_lkh_identity_fallback_explicit')
    a.check("task['deadline'] - .20" in sources['worker.py'], 'common_return_guard_point20')
    a.check('read_text' not in calls(run_fns['execute']), 'driver_no_postdeadline_tour_read')
    a.check('from neural_model import load_policy' in sources['worker.py'], 'deep_model_load_in_paid_worker')
    a.check('from deep_search import optimize' in sources['worker.py'], 'same_paid_uniform_deep_driver')
    a.check('policy.prepare(coords, knn)' in sources['deep_search.py'], 'static_attention_encoder_charged_online')
    a.check('policy.logits(' in sources['deep_search.py'], 'dynamic_decoder_charged_online')
    a.check("config['d_exact_quota'] != 1" in sources['deep_search.py']
            and "result = evaluate_d_action(parent['route'], distances, action, 1," in sources['deep_search.py'],
            'one_exact_evaluation_after_selection')
    a.check('optimizer is not None' in sources['train_deep.py']
            and 'loss.backward()' in sources['train_deep.py'], 'genuine_gradient_actor_critic_training')
    a.check('with torch.set_grad_enabled(training):' in sources['train_deep.py'],
            'validation_does_not_compute_gradients')
    return sources


def input_checks(a, study):
    records = a.json(study / 'datasets.json')['maps']
    manifest = a.json(study / 'inputs/manifest.json')
    a.check(len(records) == 12 and len({x['id'] for x in records}) == 12, 'twelve_unique_test_maps')
    a.check(Counter(x['stratum'] for x in records)
            == Counter(synthetic_uniform=4, synthetic_clustered=4, tsplib=4), 'four_maps_per_stratum')
    a.check({x['id']: x['n'] for x in records if x['stratum'] == 'tsplib'} == PUBLIC,
            'prospective_fresh_public_selection')
    a.check(manifest['no_solver_work_performed'] is True, 'test_sealing_has_no_solver_precomputation')
    a.check(manifest['source_script_sha256'] == a.hash(study / 'source/prepare_inputs.py'),
            'test_generator_matches_frozen_source')
    hashes = set()
    for item in records:
        path = study / 'inputs' / (item['id'] + '.json')
        record = a.json(path)
        coords = record['coords']
        valid = len(coords) == item['n'] and all(len(p) == 2 and all(type(v) is int for v in p) for p in coords)
        a.check(valid, 'integer_coordinate_test_' + item['id'])
        h = coord_digest(coords)
        a.check(h == item['coordinate_sha256'] == record['coordinate_sha256'], 'test_coordinate_hash_' + item['id'])
        a.check(a.hash(path) == item['sha256'], 'test_file_hash_' + item['id'])
        hashes.add(h)
        if item['stratum'] == 'tsplib':
            provenance = record['source_provenance']
            raw = a.bytes(study / 'inputs' / provenance['raw_tsp_path'])
            a.check(digest(raw) == provenance['raw_tsp_sha256'], 'public_raw_bytes_' + item['id'])
            rows, inside = {}, False
            for rawline in raw.decode('ascii').splitlines():
                line = rawline.strip()
                if line == 'NODE_COORD_SECTION':
                    inside = True
                elif inside and line not in ('', 'EOF'):
                    fields = line.split()
                    values = [Decimal(q) for q in fields[1:]]
                    a.check(len(values) == 2 and all(v.is_finite() and v == v.to_integral_value() for v in values),
                            'public_exact_integer_' + item['id'])
                    rows[int(fields[0])] = [int(v) for v in values]
            a.check([rows[i + 1] for i in range(item['n'])] == coords, 'public_no_transform_' + item['id'])
    a.check(len(hashes) == 12, 'distinct_test_coordinate_hashes')
    for family in ('uniform', 'clustered'):
        a.check(sorted(x['n'] for x in records if x['family'] == family) == [1000, 3000, 5000, 8000],
                'prospective_sizes_' + family)
    return records, hashes


def episode_checks(a, path, allowed, training, expected_count, max_steps):
    count = steps = updates = positive = parent_only = 0
    checkpoint_values, map_counts = {}, Counter()
    for line in a.bytes(path).decode('utf-8').splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        count += 1
        map_counts[entry['coordinate_sha256']] += 1
        a.check(entry['coordinate_sha256'] in allowed, 'episode_uses_only_expected_split')
        a.check(entry['reward_identity_passed'] is True, 'episode_reward_identity_flag')
        gain = entry['initial_length'] - entry['final_length']
        a.check(math.isclose(entry['improvement_fraction'], gain / entry['initial_length'],
                             rel_tol=1e-12, abs_tol=1e-12), 'episode_improvement_fraction_recomputed')
        checkpoint_values.setdefault(str(entry.get('checkpoint_episode', 'training')), []).append(
            gain / entry['initial_length'])
        step_records = entry['steps']
        a.check(len(step_records) <= max_steps, 'episode_within_step_cap')
        a.check(gain == sum(x['best_gain'] for x in step_records) == entry['total_exact_gain'],
                'episode_incumbent_gain_telescopes')
        a.check(math.isclose(sum(x['reward'] for x in step_records),
                             gain / entry['reward_scale_initial_mean_edge'], rel_tol=1e-9, abs_tol=1e-9),
                'episode_reward_uses_global_incumbent')
        a.check(entry['selection_seed'] == entry['proposal_seed'] ^ 0x5E1EC7,
                'independent_proposal_selection_rng_streams')
        a.check(all(x['best_gain'] >= 0 and (x['best_gain'] != 0 or x['reward'] == 0) for x in step_records),
                'no_reward_for_parent_only_gain')
        a.check(all(math.isfinite(x['value']) and 0 <= x['chosen_probability'] <= 1 for x in step_records),
                'finite_learned_values_and_probabilities')
        if training:
            a.check(len(entry['updates']) > 0 and all(math.isfinite(x['loss'])
                    and math.isfinite(x['gradient_norm_before_clip'])
                    and x['encoder_gradient_norm_after_clip'] >= 0 for x in entry['updates']),
                    'finite_actor_critic_and_encoder_updates')
        else:
            a.check(not entry['updates'], 'validation_has_no_gradient_updates')
        steps += len(step_records)
        updates += len(entry['updates'])
        positive += sum(x['best_gain'] > 0 for x in step_records)
        parent_only += sum(x['parent_gain'] > 0 and x['best_gain'] == 0 for x in step_records)
    a.check(count == expected_count, 'episode_count_' + path.name, dict(actual=count, expected=expected_count))
    a.check(set(map_counts) == allowed and set(map_counts.values()) == {2},
            'two_complete_passes_over_expected_maps_' + path.name)
    return dict(episodes=count, steps=steps, updates=updates, incumbent_improvement_steps=positive,
                parent_only_improvement_steps=parent_only,
                checkpoints={key: dict(episodes=len(values), mean_improvement_fraction=sum(values) / len(values))
                             for key, values in checkpoint_values.items()})


def model_checks(a, study, test_hashes):
    root = study / 'models'
    report = a.json(root / 'training_report.json')
    config = a.json(root / 'frozen_training_config.json')
    registry = a.json(root / 'model_registry.json')['by_test_seed']
    a.check(report['pilot'] is False and report['deep_neural_network'] is True, 'formal_deep_training_not_pilot')
    a.check(report['test_data_read'] is False and report['validation_gradients_computed'] is False,
            'no_test_exposure_or_validation_gradients')
    a.check(report['train_map_count'] == 128 and report['validation_map_count'] == 32,
            'fixed_128train_32validation')
    a.check(config['steps'] == 128 and config['test_step_cap'] == 128 and config['epochs'] == 2,
            'fixed_training_steps_and_two_passes')
    a.check(config['knn'] == 16 and config['candidate_count'] == 192
            and config['exact_quota'] == 1 and config['pool_size'] == 3, 'matched_training_action_pool_settings')
    for key in ('end_to_end_seconds', 'training_seconds', 'validation_seconds', 'preparation_seconds'):
        a.check(math.isfinite(report[key]) and report[key] > 0, 'disclosed_offline_cost_' + key)
    split_hashes = {}
    for split, expected in (('train', 128), ('validation', 32)):
        records = a.json(root / (split + '_manifest.json'))
        a.check(len(records) == expected, 'training_split_count_' + split)
        values = set()
        for entry in records:
            record = a.json(confined(root, entry['path']))
            value = coord_digest(record['coords'])
            a.check(value == record['coordinate_sha256'] == entry['coordinate_sha256'],
                    'actual_coordinate_digest_' + split)
            values.add(value)
        a.check(len(values) == expected, 'unique_training_maps_' + split)
        split_hashes[split] = values
    a.check(not split_hashes['train'] & split_hashes['validation'], 'actual_train_validation_disjoint')
    a.check(not test_hashes & (split_hashes['train'] | split_hashes['validation']),
            'actual_test_disjoint_from_both_training_splits')
    for name, expected in report['source_sha256'].items():
        a.check(a.hash(study / 'source' / name) == expected, 'training_used_frozen_source_' + name)
    a.check(set(registry) == {str(s) for s in TEST_SEEDS}, 'five_test_model_mappings')
    a.check(tuple(registry[str(s)]['training_seed'] for s in TEST_SEEDS) == TRAIN_SEEDS,
            'fixed_independent_training_seed_mapping')
    neural = load_module(study / 'source', 'neural_model')
    import numpy as np
    selected_hashes, all_log_stats = set(), {}
    by_train_seed = {x['training_seed']: x for x in report['models']}
    for seed in TEST_SEEDS:
        entry = registry[str(seed)]
        model_path = confined(root, entry['D'])
        model_json = a.json(model_path)
        policy = neural.load_policy(model_path)
        model_report = by_train_seed[entry['training_seed']]
        selected_hashes.add(a.hash(model_path))
        a.check(policy.parameter_count == 124290, 'deep_parameter_count')
        a.check(policy.architecture['embedding_dim'] == 64 and policy.architecture['encoder_layers'] == 2
                and policy.architecture['heads'] == 4, 'two_layer_four_head_sparse_attention')
        a.check(model_json['metadata']['training_seed'] == entry['training_seed'], 'model_seed_metadata')
        a.check(model_json['metadata']['deep_neural_network'] is True, 'model_is_deep_rl')
        parity = model_json['metadata']['export_parity']
        a.check(parity['passed'] is True and max(parity['max_logit_absolute_error'],
                parity['max_embedding_absolute_error'], parity['value_absolute_error']) <= 1e-4,
                'training_torch_numpy_export_parity')
        a.check(a.hash(model_path) == model_report['model_file_sha256'], 'selected_model_file_matches_training_report')
        for part in ('encoder', 'actor', 'critic'):
            a.check(model_report['selected_' + part + '_parameter_change_l2'] > 0,
                    'trained_parameters_changed_' + part)
        scores = model_report['checkpoint_scores']
        selected = max(scores, key=lambda x: x['mean_improvement_fraction'])
        a.check(model_report['selected_checkpoint'] == selected['checkpoint_episode'],
                'checkpoint_selected_only_by_validation_with_earliest_tie')
        folder = model_path.parent
        initial = neural.load_policy(folder / 'initial_model_D.json')
        groups = dict(encoder=lambda key: key.startswith(('block', 'input')),
                      actor=lambda key: key.startswith('actor'), critic=lambda key: key.startswith('critic'))
        for group, include in groups.items():
            change = float(sum(np.square(policy.parameters[key] - initial.parameters[key]).sum()
                               for key in policy.parameters if include(key)) ** .5)
            reported = model_report['selected_' + group + '_parameter_change_l2']
            a.check(change > 0 and math.isclose(change, reported, rel_tol=1e-5, abs_tol=1e-6),
                    'actual_selected_parameter_change_' + group)
        logs = dict(train=episode_checks(a, folder / 'training_D.jsonl', split_hashes['train'], True, 256, 128),
                    validation=episode_checks(a, folder / 'validation_D.jsonl', split_hashes['validation'], False, 64, 128))
        for score in scores:
            actual = logs['validation']['checkpoints'][str(score['checkpoint_episode'])]
            a.check(actual['episodes'] == 32 and math.isclose(actual['mean_improvement_fraction'],
                    score['mean_improvement_fraction'], rel_tol=1e-10, abs_tol=1e-12),
                    'validation_checkpoint_score_rebuilt_from_raw_episodes')
        all_log_stats[str(seed)] = logs
    a.check(len(selected_hashes) == 5, 'five_distinct_trained_model_files')
    a.details['training_log_counts'] = all_log_stats
    a.details['offline_costs_seconds'] = {key: report[key] for key in (
        'end_to_end_seconds', 'training_seconds', 'validation_seconds', 'preparation_seconds')}


def schedule_checks(a, study, records):
    protocol, schedule = a.json(study / 'frozen_protocol.json'), a.json(study / 'schedule.json')
    a.check(protocol['development_smoke'] is False, 'formal_not_smoke')
    a.check(tuple(protocol['methods']) == METHODS and tuple(protocol['seeds']) == TEST_SEEDS,
            'formal_methods_and_paired_seeds')
    a.check(protocol['total_budget_seconds'] == 30 and protocol['planned_jobs'] == 180,
            'thirty_seconds_180_jobs')
    a.check(protocol['quantum_execution'] is False, 'no_quantum_execution')
    config = protocol['settings']
    for key, value in dict(preprocessing_fraction_target=.25, local_max_steps=128, pool_size=3,
                           d_candidate_count=192, d_exact_quota=1, knn=16).items():
        a.check(config[key] == value, 'frozen_setting_' + key)
    a.check(protocol['bootstrap_draws'] == 20000, 'twenty_thousand_whole_map_bootstrap_draws')
    a.check('97.5%' in protocol['uncertainty'] and 'D-A' in protocol['primary']
            and 'D-U' in protocol['secondary'], 'prospective_claims_and_adjusted_intervals')
    triples = Counter((x['map_id'], x['seed'], x['method']) for x in schedule)
    expected = {(item['id'], seed, method) for item in records for seed in TEST_SEEDS for method in METHODS}
    a.check(set(triples) == expected and set(triples.values()) == {1}, 'all_180_unique_paired_runs')
    a.check([x['job'] for x in schedule] == list(range(1, 181)), 'sequential_registered_job_ids')
    orders, positions, adjacencies = Counter(), Counter(), Counter()
    for block in range(60):
        items = schedule[block * 3:(block + 1) * 3]
        order = ''.join(x['method'] for x in items)
        a.check(len({(x['map_id'], x['seed']) for x in items}) == 1, 'paired_block_same_map_and_seed')
        a.check(all(x['block'] == block + 1 and x['position'] == i + 1 and x['block_order'] == order
                    for i, x in enumerate(items)), 'block_schedule_metadata_consistent')
        orders[order] += 1
        positions.update((x['method'], x['position']) for x in items)
        adjacencies.update(zip(order, order[1:]))
    a.check(orders == Counter({''.join(x): 10 for x in itertools.permutations(METHODS)}), 'six_orders_each_ten_blocks')
    a.check(set(positions.values()) == {20}, 'balanced_method_positions')
    a.check(len(adjacencies) == 6 and set(adjacencies.values()) == {20}, 'balanced_directed_adjacencies')
    a.details['schedule_order_counts'] = dict(orders)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    study, output = args.study.resolve(), args.output.resolve()
    if output.is_relative_to(study):
        raise ValueError('audit output must be outside the frozen study')
    a = Audit()
    try:
        freeze = a.json(study / 'freeze_complete.json')
        a.check(freeze['all_inputs_models_schedule_frozen_before_first_job'] is True, 'freeze_completed_before_jobs')
        for name, expected in freeze['sealed_file_hashes'].items():
            a.check(a.hash(confined(study, name)) == expected, 'sealed_file_unchanged_' + name)
        source_checks(a, study / 'source')
        environment = a.json(study / 'environment.json')
        a.check(environment['lkh_sha256'] == LKH_SHA
                and a.hash(Path(environment['lkh_binary'])) == LKH_SHA, 'registered_stock_lkh_binary')
        records, test_hashes = input_checks(a, study)
        model_checks(a, study, test_hashes)
        schedule_checks(a, study, records)
    except BaseException as exc:
        import traceback
        a.check(False, 'audit_exception', dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
    value = dict(schema=1, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 study=str(study), passed=not a.errors, errors=a.errors, warnings=[],
                 check_count=len(a.checks), checks=a.checks, details=a.details,
                 opened_file_sha256=a.read_sha256,
                 outcome_blind=True, formal_outcome_files_read=0, solver_jobs_launched=0,
                 limitations='Static source checks and sealed training/input/schedule audit; independent final outcome/route audit remains separate.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=value['passed'], checks=len(a.checks), errors=len(a.errors), output=str(output))))
    if a.errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
