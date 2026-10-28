"""Independent, outcome-blind preflight audit of the four-flow RL experiment.

Usage: python -B audit_design.py [--study FROZEN_ROOT] [--output REPORT.json]

Reads source, frozen inputs, training artifacts, models and prospective schedule.
Never opens row*.json, job ledgers, LKH results or other formal test outcomes.
Pure-function fixtures use a new 96-city synthetic graph, never a formal map.
Reports are written under work by default, outside the frozen study/seal.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
METHODS = ('A', 'B', 'C', 'D')
TEST_SEEDS = (101, 211, 307, 401, 503)
TRAIN_SEEDS = (1101, 1211, 1307, 1401, 1503)
ORDERS = ('ABDC', 'BCAD', 'CDBA', 'DACB')
PUBLIC = {'pcb1173': 1173, 'pr1002': 1002, 'u2319': 2319, 'pcb3038': 3038}
LKH_SHA256 = 'b44414b7c9aa111b4e782ba13fd2cf4684bf3ad505ba1f5d65822b8ad162c044'


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def coordinate_digest(coords):
    return digest_bytes(json.dumps(coords, separators=(',', ':'),
                                   ensure_ascii=True).encode('ascii'))


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks = []
        self.read_sha256 = {}
        self.details = {}

    def check(self, condition, code, details=None):
        self.checks.append(dict(code=code, passed=bool(condition), details=details))
        if not condition:
            self.errors.append(dict(code=code, details=details))
        return bool(condition)

    def warn(self, code, details):
        self.warnings.append(dict(code=code, details=details))

    def read_bytes(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        self.read_sha256[str(path)] = digest_bytes(data)
        return data

    def read_json(self, path):
        return json.loads(self.read_bytes(path).decode('utf-8-sig'))

    def hash(self, path):
        return digest_bytes(self.read_bytes(path))


def confined_file(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f'path escapes its declared root or is missing: {relative}')
    return candidate


def functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def function_dump(tree, name):
    return ast.dump(functions(tree)[name], include_attributes=False)


def literals(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def strings(node):
    return {value.value for value in ast.walk(node)
            if isinstance(value, ast.Constant) and isinstance(value.value, str)}


def call_names(node):
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            names.append(ast.unparse(child.func))
    return names


def source_checks(audit, source):
    parsed = {}
    text = {}
    for filename in ('run.py', 'worker.py', 'supervisor.py', 'resource_v3.py',
                     'search.py', 'policy.py', 'train.py', 'prepare_inputs.py'):
        text[filename] = audit.read_bytes(source / filename).decode('utf-8-sig')
        parsed[filename] = ast.parse(text[filename], filename=filename)
    constants = literals(parsed['run.py'])
    audit.check(constants.get('METHODS') == METHODS, 'source_four_flows')
    audit.check(constants.get('SEEDS') == TEST_SEEDS, 'source_five_test_seeds')
    audit.check(constants.get('LATIN_ORDERS') == tuple(tuple(order) for order in ORDERS),
                'source_williams_orders')
    run_functions = functions(parsed['run.py'])
    for needed in ('freeze', 'execute', 'assert_frozen', 'model_registration'):
        audit.check(needed in run_functions, 'source_function_' + needed)
    audit.check('assert_frozen' in call_names(run_functions['execute']), 'execution_checks_seal')
    audit.check('run_one' in call_names(run_functions['execute']), 'execution_uses_paid_supervisor')
    audit.check('rglob' not in call_names(run_functions['execute']), 'execution_has_no_recursive_outcome_selection')

    baseline = HERE.parent.parent / 'outputs/oropt_replication_20261018/source'
    for filename in ('supervisor.py', 'resource_v3.py'):
        audit.check(audit.hash(source / filename) == audit.hash(baseline / filename),
                    'inherited_' + filename)
    baseline_worker = ast.parse(audit.read_bytes(baseline / 'worker.py').decode('utf-8-sig'))
    for name in ('nearest_neighbor', 'distance_matrix', 'lkh_search', 'parse_tour', 'scalar_length'):
        audit.check(function_dump(parsed['worker.py'], name) == function_dump(baseline_worker, name),
                    'unchanged_worker_function_' + name)
    worker_functions = functions(parsed['worker.py'])
    lkh_strings = strings(worker_functions['lkh_search'])
    required_parameters = {'INITIAL_TOUR_FRACTION = 1', 'CANDIDATE_SET_TYPE = ALPHA',
                           'MAX_CANDIDATES = 5', 'RUNS = 100000', 'MAX_TRIALS = 100000',
                           'STOP_AT_OPTIMUM = NO', 'TIME_LIMIT = 86400',
                           'TOTAL_TIME_LIMIT = 86400', 'TRACE_LEVEL = 1'}
    audit.check(required_parameters <= lkh_strings, 'lkh_registered_parameters',
                sorted(required_parameters - lkh_strings))
    audit.check(not any(value.startswith(('INPUT_TOUR_FILE', 'OPTIMUM =')) for value in lkh_strings),
                'lkh_no_optimum_or_input_tour')
    audit.check('.20' in text['worker.py'] and "task['deadline'] - .20" in text['worker.py'],
                'lkh_uniform_stop_guard')
    supervisor = functions(parsed['supervisor.py'])['run_one']
    first_statement = supervisor.body[0]
    audit.check(isinstance(first_statement, ast.Assign)
                and ast.unparse(first_statement.value) == 'time.perf_counter()',
                'cold_clock_first_supervisor_statement')
    audit.check('NUMBA_CACHE_DIR' in text['supervisor.py']
                and "cache.mkdir(parents=True,exist_ok=False)" in text['supervisor.py'],
                'independent_initially_empty_compilation_cache')
    top_imports = [ast.unparse(n) for n in parsed['worker.py'].body
                   if isinstance(n, (ast.Import, ast.ImportFrom))]
    audit.check(not any(any(name in statement for name in ('numpy', 'numba', 'policy', 'search', 'torch'))
                        for statement in top_imports), 'worker_no_unpaid_heavy_module_imports')
    worker_main = ast.unparse(worker_functions['main'])
    audit.check("origin + task['budget'] * settings['preprocessing_fraction_target']" in worker_main,
                'preprocessing_deadline_uses_total_cold_origin')
    audit.check('load_policy(model_path)' in worker_main and 'distance_matrix' in worker_main,
                'model_and_matrix_inside_worker_budget')

    search_functions = functions(parsed['search.py'])
    for name in ('enumerate_actions', 'enumerate_2opt_candidates'):
        calls = call_names(search_functions[name])
        audit.check(not any(value in calls for value in ('evaluate_action', 'evaluate_d_action',
                                                        'evaluate_d_candidates', '_policy_scores')),
                    'no_candidate_gain_or_policy_in_feature_generation_' + name)
    optimize = ast.unparse(search_functions['optimize'])
    audit.check("mode in ('B', 'C')" in optimize and 'enumerate_actions' in optimize
                and 'evaluate_action' in optimize, 'b_c_shared_environment_branch')
    audit.check('config[\'d_exact_quota\'] != 1' in optimize
                and 'evaluate_d_action' in optimize, 'd_one_chosen_true_gain_evaluation')
    rng_calls = sum(value == 'np.random.default_rng' for value in call_names(search_functions['optimize']))
    helper_rng_calls = sum(value == 'np.random.default_rng'
                           for value in call_names(search_functions['make_rng_streams'])) if 'make_rng_streams' in search_functions else 0
    separate_streams = (rng_calls >= 2 or
                        ('make_rng_streams' in call_names(search_functions['optimize'])
                         and helper_rng_calls >= 2
                         and 'proposal_rng' in optimize and 'selection_rng' in optimize))
    audit.check(separate_streams, 'test_proposal_and_selection_streams_independent')
    if not separate_streams:
        audit.warn('candidate_and_policy_random_streams_coupled_at_test',
                   'Training uses separate candidate/action RNGs; test optimize currently shares a stream. '
                   'This changes stochastic coupling and weakens B/C matched proposal randomness.')
    train_functions = functions(parsed['train.py'])
    rollout = ast.unparse(train_functions['rollout'])
    audit.check('search.TourPool' in rollout and 'search.enumerate_actions' in rollout
                and 'search.evaluate_action' in rollout and 'search.evaluate_d_action' in rollout,
                'training_reuses_test_search_environment')
    audit.check('policy.sample' in rollout and 'pool.update' in rollout,
                'training_samples_actions_then_observes_state_transition')
    audit.check('initial_length - final_length' in rollout and 'reward_scale' in rollout,
                'training_true_tour_reward_telescope_assertion')
    train_one = ast.unparse(train_functions['train_one'])
    audit.check('discounted_returns' in train_one and 'reinforce_gradient' in train_one
                and 'baseline[t]' in train_one and 'policy.weights +=' in train_one,
                'genuine_sequential_reinforce_weight_update')
    validation = ast.unparse(train_functions['validation'])
    audit.check('collect_gradient=True' not in validation and 'reinforce_gradient' not in validation,
                'validation_does_not_update_gradients')
    main_train = ast.unparse(train_functions['main'])
    if main_train.find('make_manifest') < main_train.find('started = time.perf_counter()'):
        audit.warn('training_report_main_timer_excludes_input_preparation',
                   'training_report.total_seconds starts after train/validation input generation. '
                   'Disclose that boundary or provide launch-to-exit external wall time.')
    audit.details['manual_source_review'] = {
        'B_C': 'One optimize implementation; shared action enumeration, fixed-endpoint windows, exact quota, pool parent schedule and retention. B fixed selector versus C learned selector.',
        'D': 'Candidate features use existing tour edges and prebuilt KNN membership. A softmax-trained action is chosen before evaluating one complete 2-opt delta.',
        'training': 'REINFORCE acts sequentially on the same TourPool/search evaluators, uses actual pool-global-best full-tour decrease, and selects checkpoints only by validation.',
        'known_external_generalization': 'Training uses smaller 128..1000-city synthetic maps and 24-step episodes; tests allow 128 steps and up to 8000 cities. Cluster distributions also differ.'}


def pure_function_checks(audit, source):
    import numpy as np
    import time
    loaded = {}
    for name in ('policy', 'search'):
        spec = importlib.util.spec_from_file_location('_independent_audit_' + name, source / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    policy, search = loaded['policy'], loaded['search']
    audit.details['gradient_finite_difference'] = policy.finite_difference_check()
    rng = np.random.default_rng(20261019029811)
    points = rng.integers(0, 10001, size=(96, 2))
    difference = points[:, None, :] - points[None, :, :]
    distances = np.floor(np.hypot(difference[:, :, 0], difference[:, :, 1]) + .5).astype(np.int32)
    route = rng.permutation(len(points)).astype(np.int64)
    original = route.copy()
    initial_length = search.tour_length(route, distances)
    evaluated = 0
    for operator in ('2opt', 'or1', 'or2', 'or3'):
        for seed in range(8):
            action = dict(start=73, size=48, operator=operator, seed=seed)
            result = search.evaluate_action(route, distances, action, budget=128)
            child = result['route']
            audit.check(np.array_equal(route, original), f'fixture_no_parent_mutation_{operator}_{seed}')
            audit.check(sorted(child.tolist()) == list(range(len(route))),
                        f'fixture_hamiltonian_{operator}_{seed}')
            indices = (np.arange(48) + 73) % len(route)
            audit.check(child[indices[0]] == route[indices[0]] and child[indices[-1]] == route[indices[-1]],
                        f'fixture_fixed_endpoints_{operator}_{seed}')
            audit.check(initial_length - search.tour_length(child, distances) == result['gain']
                        and result['gain'] >= 0, f'fixture_actual_gain_{operator}_{seed}')
            evaluated += 1
    knn = search.build_knn(distances, 16)
    candidates, features = search.enumerate_2opt_candidates(route, distances, np.random.default_rng(81),
                                                           192, knn)
    for candidate in candidates:
        result = search.evaluate_d_action(route, distances, candidate, budget=1)
        audit.check(result['evaluations'] <= 1
                    and initial_length - search.tour_length(result['route'], distances) == result['gain'],
                    'fixture_d_selected_candidate_actual_gain')
    altered = distances.copy()
    old_edge_mask = np.zeros(distances.shape, dtype=bool)
    old_edge_mask[route, np.roll(route, -1)] = True
    old_edge_mask[np.roll(route, -1), route] = True
    altered[~old_edge_mask] += 100000
    actions_a, features_a = search.enumerate_actions(route, distances, np.random.default_rng(82))
    actions_b, features_b = search.enumerate_actions(route, altered, np.random.default_rng(82))
    audit.check(actions_a == actions_b and np.array_equal(features_a, features_b),
                'fixture_B_C_features_invariant_to_all_nonold_edge_costs')
    actions_a, features_a = search.enumerate_2opt_candidates(route, distances, np.random.default_rng(83), 192, knn)
    actions_b, features_b = search.enumerate_2opt_candidates(route, altered, np.random.default_rng(83), 192, knn)
    audit.check(actions_a == actions_b and np.array_equal(features_a, features_b),
                'fixture_D_features_invariant_to_nonold_costs_with_fixed_paid_KNN')

    class ForceBRulePolicy:
        # Finite peaked logits still consume a C categorical RNG draw. With
        # separated streams this cannot disturb subsequent proposal randomness.
        def __init__(self):
            self.step = 0

        def score(self, current_features):
            scores = current_features[:, 6] + .5 * current_features[:, 5] + .25 * current_features[:, 7]
            scores = scores + 100. * current_features[:, self.step % 4]
            self.step += 1
            logits = np.full(len(scores), -1000.)
            logits[int(np.argmax(scores))] = 1000.
            return logits

    fixture_settings = dict(local_max_steps=24, pool_size=3, candidate_quota=128,
                            action_count=12, window_sizes=[24, 48, 64], audit_decisions=128)
    route_b, route_c = route.copy(), route.copy()
    stats_b = search.optimize(route_b, distances, 'B', time.perf_counter() + 10., None,
                              seed=20261019029812, settings=fixture_settings)
    stats_c = search.optimize(route_c, distances, 'C', time.perf_counter() + 10., None,
                              policy=ForceBRulePolicy(), seed=20261019029812, settings=fixture_settings)
    audit.check(stats_b['steps'] == stats_c['steps'] == 24,
                'fixture_forced_selector_control_full_horizon')
    audit.check([item['selected_action'] for item in stats_b['decisions']]
                == [item['selected_action'] for item in stats_c['decisions']],
                'fixture_forced_same_selector_same_proposals_despite_C_random_draws')
    audit.check(np.array_equal(route_b, route_c)
                and stats_b['gain'] == stats_c['gain']
                and stats_b['pool_final'] == stats_c['pool_final'],
                'fixture_forced_same_selector_same_tours_and_pool')
    for mode, stats in [('B', stats_b), ('C', stats_c)]:
        audit.check(stats['gain'] == stats['best_gain_sum']
                    == sum(item['best_gain'] for item in stats['decisions']),
                    'fixture_reward_telescope_' + mode)
        audit.check(stats['accounted_seconds'] <= stats['elapsed'] + .001,
                    'fixture_exclusive_time_ledger_' + mode)
    audit.details['pure_fixtures'] = dict(cities=96, local_move_cases=evaluated,
                                         D_candidate_cases=len(candidates),
                                         matched_B_C_forced_selector_steps=24,
                                         formal_maps_used=False)


def canonical_model_digest(value):
    clean = dict(value)
    clean.pop('sha256', None)
    return digest_bytes(json.dumps(clean, ensure_ascii=True, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode('ascii'))


def split_manifest(audit, model_root, split, relative_files):
    entries = []
    for relative in relative_files:
        if Path(relative).name == split + '_manifest.json':
            entries.extend(audit.read_json(confined_file(model_root, relative)))
    audit.check(bool(entries), split + '_manifest_present')
    hashes = set()
    for entry in entries:
        value = audit.read_json(confined_file(model_root, entry['path']))
        coords = value['coords']
        digest = coordinate_digest(coords)
        audit.check(len(coords) == entry['n'] == value['n'], split + '_map_dimension', entry['id'])
        audit.check(value.get('split') == entry.get('split') == split, split + '_declared_split', entry['id'])
        audit.check(digest == entry['coordinate_sha256'] == value['coordinate_sha256'],
                    split + '_actual_coordinate_hash', entry['id'])
        audit.check(digest not in hashes, split + '_no_duplicate_actual_map', entry['id'])
        hashes.add(digest)
    return entries, hashes


def public_raw_checks(audit, study, metadata, coords):
    provenance = metadata['source_provenance']
    raw = confined_file(study / 'inputs', provenance['raw_tsp_path'])
    raw_bytes = audit.read_bytes(raw)
    audit.check(digest_bytes(raw_bytes) == metadata['raw_tsp_sha256'] == provenance['raw_tsp_sha256'],
                'public_raw_byte_hash', metadata['id'])
    in_coords = False
    actual = {}
    headers = {}
    for line in raw_bytes.decode('ascii').splitlines():
        line = line.strip()
        if not line:
            continue
        if line == 'EOF':
            break
        if line == 'NODE_COORD_SECTION':
            in_coords = True
            continue
        if not in_coords:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()
        else:
            city, x, y = line.split()
            values = [Decimal(x), Decimal(y)]
            if any(not v.is_finite() or v != v.to_integral_value() for v in values):
                raise ValueError('public coordinates are not exact integers: ' + metadata['id'])
            actual[int(city)] = [int(v) for v in values]
    audit.check(headers.get('TYPE') == 'TSP' and headers.get('EDGE_WEIGHT_TYPE') == 'EUC_2D',
                'public_raw_euc2d', metadata['id'])
    audit.check(set(actual) == set(range(1, len(coords) + 1))
                and [actual[i] for i in sorted(actual)] == coords,
                'public_exact_untransformed_coordinates', metadata['id'])


def frozen_checks(audit, study):
    audit.check(not any(study.glob('row[0-9][0-9][0-9].json'))
                and not any(study.glob('j[0-9][0-9][0-9]')),
                'formal_execution_not_started', 'Only names checked; no outcome file opened.')
    protocol = audit.read_json(study / 'frozen_protocol.json')
    freeze = audit.read_json(study / 'freeze_complete.json')
    audit.check(protocol.get('development_smoke') is False, 'formal_protocol_not_smoke')
    audit.check(protocol.get('methods') == list(METHODS), 'protocol_four_flows')
    audit.check(protocol.get('seeds') == list(TEST_SEEDS), 'protocol_five_seeds')
    for field, expected in (('total_budget_seconds', 30.), ('independent_maps', 12),
                            ('paired_blocks', 60), ('planned_jobs', 240),
                            ('bootstrap_seed', 20261019020002), ('bootstrap_draws', 20000)):
        audit.check(protocol.get(field) == expected, 'protocol_' + field, protocol.get(field))
    audit.check(protocol.get('milestones_seconds') == [10., 30.], 'protocol_milestones')
    audit.check(protocol.get('quantum_execution') is False, 'protocol_no_quantum_claim')
    audit.check(freeze.get('planned_jobs') == 240
                and freeze.get('all_inputs_models_schedule_frozen_before_first_job') is True,
                'freeze_complete_240_prospective_jobs')
    settings = protocol['settings']
    for field, expected in dict(preprocessing_fraction_target=.25, local_max_steps=128,
                                 action_count=12, window_sizes=[24, 48, 64], candidate_quota=128,
                                 pool_size=3, d_candidate_count=192, d_exact_quota=1, knn=16).items():
        audit.check(settings.get(field) == expected, 'settings_' + field, settings.get(field))
    for relative, expected in freeze['sealed_file_hashes'].items():
        audit.check(audit.hash(confined_file(study, relative)) == expected, 'sealed_file_hash', relative)
    source_hashes = audit.read_json(study / 'source_hashes.json')
    audit.check(source_hashes == freeze['source_hashes'], 'freeze_source_hash_manifest')
    for relative, expected in source_hashes.items():
        audit.check(audit.hash(confined_file(study / 'source', relative)) == expected,
                    'frozen_source_hash', relative)
        audit.check(audit.hash(confined_file(HERE, relative)) == expected,
                    'current_source_still_frozen', relative)
    inheritance = audit.read_json(study / 'engine_inheritance.json')
    for filename in ('supervisor.py', 'resource_v3.py'):
        audit.check(audit.hash(confined_file(study / 'baseline_reference', filename))
                    == inheritance['byte_identical_engine_hashes'][filename]
                    == source_hashes[filename], 'packaged_baseline_reference_' + filename)
    model_hashes = audit.read_json(study / 'model_hashes.json')
    model_root = study / 'models'
    actual_files = {path.relative_to(model_root).as_posix() for path in model_root.rglob('*')
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')}
    audit.check(actual_files == set(model_hashes), 'model_hash_manifest_exact_coverage')
    for relative, expected in model_hashes.items():
        audit.check(audit.hash(confined_file(model_root, relative)) == expected, 'frozen_model_artifact_hash', relative)
    environment = audit.read_json(study / 'environment.json')
    audit.check(environment.get('lkh_sha256') == LKH_SHA256, 'registered_solver_environment_hash')
    audit.check(audit.hash(Path(environment['lkh_binary'])) == LKH_SHA256, 'actual_solver_executable_hash')
    separation = audit.read_json(study / 'train_test_separation.json')
    train_entries, train_hashes = split_manifest(audit, model_root, 'train', separation['manifest_files'])
    validation_entries, validation_hashes = split_manifest(audit, model_root, 'validation', separation['manifest_files'])
    audit.check(train_hashes == set(separation['train_coordinate_sha256']), 'separation_train_actual_hashes')
    audit.check(validation_hashes == set(separation['validation_coordinate_sha256']), 'separation_validation_actual_hashes')
    datasets = audit.read_json(study / 'datasets.json')
    maps = datasets['maps']
    generation_manifest = audit.read_json(study / 'input_generation_manifest.json')
    audit.check(audit.read_json(study / 'inputs/manifest.json') == generation_manifest,
                'packaged_input_generation_manifest_identical')
    audit.check(datasets.get('registered_before_first_formal_job') is True, 'test_maps_registered_prospectively')
    audit.check(len(maps) == 12 and len({item['id'] for item in maps}) == 12, 'twelve_unique_map_ids')
    audit.check(Counter(item['stratum'] for item in maps) == dict(synthetic_uniform=4, synthetic_clustered=4, tsplib=4),
                'three_equal_four_map_strata')
    audit.check({item['id']: item['n'] for item in maps if item['stratum'] == 'tsplib'} == PUBLIC,
                'prospective_public_instances')
    test_hashes = set()
    for item in maps:
        actual_path = confined_file(study / 'inputs', item['id'] + '.json')
        audit.check(Path(item['path']).resolve() == actual_path, 'frozen_test_path_confined', item['id'])
        value = audit.read_json(actual_path)
        coords = value['coords']
        digest = coordinate_digest(coords)
        audit.check(audit.hash(actual_path) == item['sha256'], 'frozen_test_byte_hash', item['id'])
        audit.check(len(coords) == value['n'] == item['n'], 'test_dimension', item['id'])
        audit.check(all(isinstance(point, list) and len(point) == 2
                        and all(type(v) is int for v in point) for point in coords), 'test_integer_coordinates', item['id'])
        audit.check(digest == item['coordinate_sha256'] == value['coordinate_sha256'],
                    'test_canonical_coordinate_hash', item['id'])
        audit.check(digest not in test_hashes, 'no_duplicate_test_coordinates', item['id'])
        test_hashes.add(digest)
        if item['stratum'] == 'tsplib':
            public_raw_checks(audit, study, item, coords)
    audit.check(test_hashes == set(separation['test_coordinate_sha256']), 'separation_test_actual_hashes')
    audit.check(not train_hashes & validation_hashes, 'no_actual_train_validation_overlap')
    audit.check(not train_hashes & test_hashes, 'no_actual_train_test_overlap')
    audit.check(not validation_hashes & test_hashes, 'no_actual_validation_test_overlap')
    audit.check(separation.get('checked_before_first_formal_job') is True, 'separation_checked_before_testing')

    registry = audit.read_json(model_root / 'model_registry.json')
    audit.check(registry == separation['model_registry'], 'registry_matches_separation_record')
    mapping = registry['by_test_seed']
    audit.check(set(mapping) == {str(seed) for seed in TEST_SEEDS}, 'five_exact_model_test_seed_mappings')
    individual_paths = defaultdict(set)
    individual_hashes = defaultdict(set)
    config = audit.read_json(model_root / 'frozen_training_config.json')
    for test_seed, training_seed in zip(TEST_SEEDS, TRAIN_SEEDS):
        entry = mapping[str(test_seed)]
        audit.check(entry.get('training_seed') == training_seed, 'model_mapping_training_seed', str(test_seed))
        for method in ('C', 'D'):
            model_path = confined_file(model_root, entry[method])
            value = audit.read_json(model_path)
            metadata = value['metadata']
            audit.check(value.get('model_type') == 'linear_softmax_reinforce'
                        and value.get('feature_count') == 12, 'model_linear_reinforce_type', entry[method])
            audit.check(value.get('sha256') == canonical_model_digest(value), 'model_internal_content_hash', entry[method])
            audit.check(metadata.get('kind') == method and metadata.get('training_seed') == training_seed,
                        'model_metadata_method_seed', entry[method])
            audit.check(metadata.get('actual_rng_seed') == training_seed + (1000000 if method == 'D' else 0),
                        'model_independent_rng_seed', entry[method])
            audit.check(metadata.get('training_config') == config, 'model_frozen_training_config', entry[method])
            audit.check(Path(settings['policy_by_test_seed'][str(test_seed)][method]).resolve() == model_path,
                        'worker_policy_path_matches_registry', entry[method])
            individual_paths[method].add(model_path)
            individual_hashes[method].add(value['sha256'])
    for method in ('C', 'D'):
        audit.check(len(individual_paths[method]) == len(individual_hashes[method]) == 5,
                    'five_distinct_' + method + '_trained_models')
    for key, expected in dict(candidate_budget=128, candidate_count=192, C_window_sizes=[24, 48, 64],
                               C_action_count=12, pool_size=3, D_knn=16, D_exact_evaluations_per_action=1).items():
        audit.check(config.get(key) == expected, 'training_test_environment_' + key, config.get(key))
    reports = [audit.read_json(confined_file(model_root, relative)) for relative in separation['training_report_files']]
    audit.check(bool(reports), 'training_cost_report_present')
    for report in reports:
        audit.check(report.get('pilot') is False, 'formal_models_not_pilot')
        audit.check(report.get('test_data_read') is False
                    and report.get('training_completed_before_formal_test') is True,
                    'training_report_prospective_no_test_read')
        audit.check(report.get('validation_only_checkpoint_selection') is True
                    and report.get('validation_gradients_computed') is False,
                    'training_report_validation_only_selection')
        audit.check(report.get('training_seeds') == list(TRAIN_SEEDS), 'training_report_five_seeds')
        audit.check(report.get('train_map_count') == len(train_entries)
                    and report.get('validation_map_count') == len(validation_entries), 'training_report_map_counts')
        for field, filename in (('train_source_sha256', 'train.py'), ('policy_source_sha256', 'policy.py'),
                                ('search_source_sha256', 'search.py')):
            audit.check(report.get(field) == source_hashes.get(filename), 'trained_source_matches_frozen_' + filename)
        audit.check(all(isinstance(report.get(field), (int, float)) and report[field] >= 0
                        for field in ('training_seconds', 'validation_seconds', 'total_seconds')),
                    'training_costs_numeric_and_disclosed')
        audit.check(len(report.get('models', [])) == 10, 'training_report_ten_models')
        for model in report.get('models', []):
            audit.check(audit.hash(confined_file(model_root, model['model_path'])) == model['model_file_sha256'],
                        'training_report_model_file_hash', model['model_path'])

    schedule = audit.read_json(study / 'schedule.json')
    audit.check(len(schedule) == 240 and [item['job'] for item in schedule] == list(range(1, 241)),
                'schedule_240_sequential_jobs')
    expected = {(item['id'], seed, method) for item in maps for seed in TEST_SEEDS for method in METHODS}
    audit.check({(item['map_id'], item['seed'], item['method']) for item in schedule} == expected
                and len(schedule) == len(expected), 'schedule_complete_four_flow_pairing')
    blocks = defaultdict(list)
    for item in schedule:
        blocks[item['block']].append(item)
    audit.check(set(blocks) == set(range(1, 61)), 'schedule_sixty_blocks')
    order_counts, positions, directed = Counter(), Counter(), Counter()
    for block, items in blocks.items():
        order = ''.join(item['method'] for item in items)
        order_counts[order] += 1
        audit.check(len(items) == 4 and order in ORDERS
                    and all(item['block_order'] == order for item in items)
                    and [item['position'] for item in items] == [1, 2, 3, 4]
                    and len({(item['map_id'], item['seed']) for item in items}) == 1
                    and items[-1]['job'] - items[0]['job'] == 3,
                    'williams_block_' + str(block))
        for item in items:
            positions[(item['method'], item['position'])] += 1
        directed.update(zip(order, order[1:]))
    audit.check(order_counts == {order: 15 for order in ORDERS}, 'williams_four_orders_each_fifteen')
    audit.check(positions == {(method, pos): 15 for method in METHODS for pos in range(1, 5)},
                'williams_each_method_position_fifteen')
    audit.check(directed == {(a, b): 15 for a in METHODS for b in METHODS if a != b},
                'williams_directed_adjacent_pairs_fifteen')
    audit.details['frozen_study_summary'] = dict(maps=len(maps), train_maps=len(train_entries),
        validation_maps=len(validation_entries), models_C=len(individual_paths['C']),
        models_D=len(individual_paths['D']), jobs=len(schedule), seconds_per_job=30,
        source_files=len(source_hashes), training_artifact_files=len(model_hashes),
        train_validation_test_coordinate_overlap=False, outcomes_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    study = None if args.study is None else args.study.resolve()
    source = HERE if study is None else study / 'source'
    output = args.output or HERE / ('design_audit_preflight.json' if study is None
                                   else 'design_audit_' + study.name + '.json')
    output = output.resolve()
    if study is not None and output.is_relative_to(study):
        parser.error('audit report must remain outside frozen study')
    audit = Audit()
    for name, operation in [('source_review', lambda: source_checks(audit, source)),
                            ('pure_function_fixtures', lambda: pure_function_checks(audit, source)),
                            ('frozen_study', lambda: frozen_checks(audit, study) if study is not None else None)]:
        try:
            operation()
        except Exception as exc:
            audit.check(False, name + '_exception', type(exc).__name__ + ': ' + str(exc))
    report = dict(schema=1, audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  passed=not audit.errors, scope='source_only_preflight' if study is None else 'prospective_frozen_study',
                  study_directory=None if study is None else str(study),
                  formal_outcome_files_read=False, formal_solvers_started=False,
                  audit_script_sha256=digest_bytes(Path(__file__).read_bytes()),
                  errors=audit.errors, warnings=audit.warnings, checks=audit.checks,
                  details=audit.details, read_file_sha256=audit.read_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], checks=len(audit.checks),
                          errors=audit.errors, warnings=audit.warnings, report=str(output)),
                     ensure_ascii=True), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
