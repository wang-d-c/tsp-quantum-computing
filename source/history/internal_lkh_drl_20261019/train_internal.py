"""Train deep candidate permutations using actual native LKH episode returns.

No tour-improvement surrogate, supervised optimal-edge labels, previous 2-opt
network, or test instance is used.  A model action is a full Plackett--Luce
permutation of each node's ALPHA+kNN candidate union.  The C engine decodes
that action into its internal candidate arrays and performs ordinary LK moves.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np

from edge_model import (FEATURE_NAMES, build_torch_model, initialize_policy,
                        load_native_actor, native_reference_logits,
                        read_candidate_graph, read_native_stats, sample_plackett_luce,
                        write_actions)

TRAINING_SEEDS = [3101, 3211, 3307, 3401, 3503]
TEST_SEEDS = [101, 211, 307, 401, 503]
SIZES = [128, 256, 512, 1000]
TRAIN_MAP_BASE = 2110000
VAL_MAP_BASE = 2210000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True,
                                   allow_nan=False), encoding='ascii')


def coordinate_sha(coords):
    return hashlib.sha256(json.dumps(coords, separators=(',', ':'),
                                    ensure_ascii=True).encode('ascii')).hexdigest()


def make_map(split, index):
    seed = (TRAIN_MAP_BASE if split == 'train' else VAL_MAP_BASE) + index
    n = SIZES[index % len(SIZES)]
    distribution = ['uniform', 'clustered'][(index // len(SIZES)) % 2]
    rng = np.random.default_rng(seed)
    if distribution == 'uniform':
        xy = rng.uniform(0, 100000, (n, 2))
    else:
        centers = rng.uniform(15000, 85000, (8, 2))
        labels = rng.integers(0, 8, n)
        xy = np.clip(centers[labels] + rng.normal(0, 4500, (n, 2)), 0, 100000)
    coords = np.rint(xy).astype(np.int64).tolist()
    return {'id': f'{split}_{index:04d}', 'split': split, 'seed': seed,
            'n': n, 'distribution': distribution, 'coords': coords,
            'solver_seed': (410000 if split == 'train' else 420000) + index,
            'coordinate_sha256': coordinate_sha(coords)}


def make_manifest(output, split, count):
    directory = output / 'maps' / split
    directory.mkdir(parents=True)
    entries = []
    for i in range(count):
        data = make_map(split, i)
        relative = f'maps/{split}/{data["id"]}.json'
        dump(output / relative, data)
        entries.append({k: v for k, v in data.items() if k != 'coords'} | {'path': relative})
    dump(output / f'{split}_manifest.json', entries)
    return entries


def parse_tour(path, n):
    text = Path(path).read_text(encoding='ascii')
    if 'TOUR_SECTION' not in text:
        raise ValueError('Native LKH did not publish TOUR_SECTION')
    values = []
    for token in text.split('TOUR_SECTION', 1)[1].split():
        if token in ('-1', 'EOF'):
            break
        values.append(int(token) - 1)
    if sorted(values) != list(range(n)):
        raise ValueError('Native LKH published an invalid tour')
    return values


def exact_length(coords, route):
    total = 0
    for a, b in zip(route, route[1:] + route[:1]):
        dx, dy = coords[a][0] - coords[b][0], coords[a][1] - coords[b][1]
        total += math.floor(math.sqrt(dx * dx + dy * dy) + .5)
    return total


def write_tsp(path, coords):
    with Path(path).open('w', encoding='ascii', newline='\n') as stream:
        stream.write(f'NAME : internal_candidate_training\nTYPE : TSP\nDIMENSION : {len(coords)}\n'
                     'EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n')
        for i, xy in enumerate(coords, 1):
            stream.write(f'{i} {xy[0]} {xy[1]}\n')
        stream.write('EOF\n')


def episode(output, entry, job, config, mode, *, actions=None, model=None, feature_dump=None):
    """Run one completed, fixed-trial native LKH episode and recompute its return."""
    if digest(config['lkh_binary']) != config['lkh_binary_sha256']:
        raise RuntimeError('Native engine binary changed before an offline episode')
    job.mkdir(parents=True, exist_ok=False)
    source = json.loads((output / entry['path']).read_text(encoding='ascii'))
    tsp = output / 'tsp' / (entry['id'] + '.tsp')
    if not tsp.exists():
        tsp.parent.mkdir(exist_ok=True)
        write_tsp(tsp, source['coords'])
    tour = job / 'best.tour'
    par = job / 'run.par'
    par.write_text('\n'.join([
        'PROBLEM_FILE = ' + tsp.as_posix(),
        'OUTPUT_TOUR_FILE = ' + tour.as_posix(),
        'CANDIDATE_SET_TYPE = ALPHA', 'MAX_CANDIDATES = 5',
        'RUNS = 1', 'MAX_TRIALS = ' + str(config['max_trials']),
        'STOP_AT_OPTIMUM = NO', 'SEED = ' + str(entry['solver_seed']),
        'TIME_LIMIT = 86400', 'TOTAL_TIME_LIMIT = 86400', 'TRACE_LEVEL = 1',
    ]) + '\n', encoding='ascii')
    env = {k: v for k, v in os.environ.items() if not k.startswith('INTERNAL_RL_')}
    env['INTERNAL_RL_MODE'] = mode
    env['INTERNAL_RL_STATS'] = str(job / 'native_stats.json')
    if actions is not None:
        env['INTERNAL_RL_ACTIONS'] = str(actions)
    if model is not None:
        env['INTERNAL_RL_MODEL'] = str(model)
    if feature_dump is not None:
        env['INTERNAL_RL_DUMP'] = str(feature_dump)
    started = time.perf_counter()
    with (job / 'stdout.log').open('wb') as stdout, (job / 'stderr.log').open('wb') as stderr:
        process = subprocess.Popen([config['lkh_binary'], str(par)], cwd=job, env=env,
                                   stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            code = process.wait(timeout=config['episode_timeout'])
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            dump(job / 'episode_timeout.json', {'timeout': True, 'seconds': time.perf_counter() - started})
            raise RuntimeError('Offline episode exceeded fixed-trial safety timeout: ' + str(job))
    elapsed = time.perf_counter() - started
    if code != 0:
        raise RuntimeError(f'Native LKH failed with exit {code}: {job}')
    route = parse_tour(tour, entry['n'])
    length = exact_length(source['coords'], route)
    log = (job / 'stdout.log').read_text(encoding='ascii', errors='replace')
    costs = re.findall(r'Cost\.min\s*=\s*([0-9.eE+\-]+)', log)
    if costs and round(float(costs[-1])) != length:
        raise AssertionError('LKH stdout cost differs from independently evaluated tour')
    native_stats = read_native_stats(job / 'native_stats.json')
    result = {'length': length, 'route_sha256': hashlib.sha256(
        json.dumps(route, separators=(',', ':')).encode('ascii')).hexdigest(),
        'tour_file_sha256': digest(tour), 'elapsed_seconds': elapsed,
        'solver_seed': entry['solver_seed'], 'max_trials': config['max_trials'],
        'native_stats': native_stats, 'mode': mode,
        'actual_return_source': 'complete native LKH final tour, independently recomputed integer EUC_2D'}
    dump(job / 'episode.json', result)
    return result


def prepare_baselines(output, entries, config):
    cached = {}
    for i, entry in enumerate(entries, 1):
        job = output / 'baselines' / entry['id']
        result = episode(output, entry, job, config, 'dump', feature_dump=job / 'features.csv')
        graph = read_candidate_graph(job / 'features.csv')
        if graph.n != entry['n']:
            raise AssertionError('Native candidate dump dimension differs from instance')
        cached[entry['id']] = (result, graph)
        if i % 8 == 0 or i == len(entries):
            print(json.dumps({'phase': 'ordinary_baselines', 'completed': i, 'total': len(entries),
                              'last_seconds': result['elapsed_seconds']}), flush=True)
    return cached


def export_parity(model, graph, path):
    import torch
    policy = model.export()
    policy.export_native(path)
    features = torch.from_numpy(graph.features)
    critic = torch.from_numpy(graph.critic_features())
    with torch.no_grad():
        logits, value = model(features, critic)
    reference = native_reference_logits(graph.features, load_native_actor(path))
    error = float(np.max(np.abs(logits.detach().numpy() - reference)))
    if error > 2e-5:
        raise AssertionError('Autograd versus native model forward disagreement')
    return {'torch_numpy_native_max_absolute_logit_error': error,
            'critic_max_absolute_error': abs(float(value) - policy.value(graph.critic_features())),
            'passed': True}


def validation(output, entries, cache, config, folder, model_path):
    records = []
    started = time.perf_counter()
    for entry in entries:
        result = episode(output, entry, folder / entry['id'], config, 'model', model=model_path)
        base = cache[entry['id']][0]['length']
        records.append({'id': entry['id'], 'coordinate_sha256': entry['coordinate_sha256'],
                        'baseline_length': base, 'policy_length': result['length'],
                        'improvement_fraction': (base - result['length']) / base,
                        'seconds': result['elapsed_seconds']})
    summary = {'mean_improvement_fraction': float(np.mean([v['improvement_fraction'] for v in records])),
               'instances': len(records), 'seconds': time.perf_counter() - started,
               'records': records, 'selection': 'deterministic actor ordering inside native LKH',
               'gradients': False}
    dump(folder / 'validation_summary.json', summary)
    return summary


def train_one(output, train, validation_entries, cache, config, seed):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed ^ 0xA7710)
    initial = initialize_policy(seed)
    model = build_torch_model(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    folder = output / f'seed_{seed}'
    folder.mkdir()
    initial.save(folder / 'initial_model.json')
    selected, best_score, reports, episode_number = None, -float('inf'), [], 0
    training_seconds, validation_seconds = 0., 0.
    log_path = folder / 'training_episodes.jsonl'
    with log_path.open('x', encoding='ascii') as log:
        for epoch in range(config['epochs']):
            for train_index in rng.permutation(len(train)):
                entry = train[int(train_index)]
                baseline, graph = cache[entry['id']]
                episode_number += 1
                started = time.perf_counter()
                features = torch.from_numpy(graph.features)
                critic_features = torch.from_numpy(graph.critic_features())
                index, valid = graph.padded_indices()
                index_t, valid_t = torch.from_numpy(index), torch.from_numpy(valid)
                logits, value = model(features, critic_features)
                order, log_probability, entropy = sample_plackett_luce(logits, index_t, valid_t, generator)
                job = folder / 'training' / f'episode_{episode_number:04d}'
                job.mkdir(parents=True)
                actions = job / 'actions.csv'
                write_actions(actions, graph, order.detach().numpy())
                # episode() requires a fresh leaf, preserving the sampled action next to it.
                result = episode(output, entry, job / 'lkh', config, 'actions', actions=actions)
                reward = config['reward_scale'] * (baseline['length'] - result['length']) / baseline['length']
                advantage = torch.tensor(reward, dtype=torch.float32) - value
                actor_loss = -advantage.detach() * log_probability
                critic_loss = torch.nn.functional.smooth_l1_loss(value, torch.tensor(reward, dtype=torch.float32))
                loss = actor_loss + config['critic_weight'] * critic_loss - config['entropy_coefficient'] * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise AssertionError('Nonfinite actual-return policy gradient')
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']))
                optimizer.step()
                elapsed = time.perf_counter() - started
                training_seconds += elapsed
                record = {'episode': episode_number, 'epoch': epoch + 1, 'training_seed': seed,
                          'map_id': entry['id'], 'coordinate_sha256': entry['coordinate_sha256'],
                          'n': entry['n'], 'solver_seed': entry['solver_seed'],
                          'baseline_length': baseline['length'], 'policy_length': result['length'],
                          'terminal_reward': reward, 'critic_value': float(value.detach()),
                          'advantage': float(advantage.detach()),
                          'mean_joint_log_probability': float(log_probability.detach()),
                          'joint_log_probability': float(log_probability.detach()) * int(valid.sum()),
                          'decision_count': int(valid.sum()), 'first_choice_entropy': float(entropy.detach()),
                          'loss': float(loss.detach()), 'actor_loss': float(actor_loss.detach()),
                          'critic_loss': float(critic_loss.detach()), 'gradient_norm_before_clip': grad_norm,
                          'actions_sha256': digest(actions), 'tour_file_sha256': result['tour_file_sha256'],
                          'elapsed_seconds': elapsed, 'lkh_seconds': result['elapsed_seconds'],
                          'return_is_actual_native_lkh_final_tour': True}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                if episode_number % 16 == 0:
                    print(json.dumps({'phase': 'actual_lkh_reinforce', 'seed': seed,
                                      'episode': episode_number, 'reward': reward,
                                      'training_seconds': training_seconds}), flush=True)
            checkpoint = folder / f'checkpoint_{episode_number:04d}'
            checkpoint.mkdir()
            parity = export_parity(model, cache[train[0]['id']][1], checkpoint / 'model_D.txt')
            policy = model.export({'training_seed': seed, 'checkpoint_episode': episode_number})
            policy.save(checkpoint / 'model_D.json')
            score = validation(output, validation_entries, cache, config,
                               checkpoint / 'validation', checkpoint / 'model_D.txt')
            validation_seconds += score['seconds']
            reports.append({'episode': episode_number, 'epoch': epoch + 1,
                            'mean_improvement_fraction': score['mean_improvement_fraction'],
                            'validation_seconds': score['seconds'], 'export_parity': parity})
            if score['mean_improvement_fraction'] > best_score:
                best_score = score['mean_improvement_fraction']
                selected = model.export({
                    'training_seed': seed, 'checkpoint_episode': episode_number,
                    'validation_mean_improvement_fraction': best_score,
                    'selection_rule': 'maximize mean held-out validation LKH tour improvement; earliest checkpoint wins ties',
                    'algorithm': 'terminal-return combinatorial contextual-bandit REINFORCE with learned nonlinear value critic, Adam and entropy',
                    'environment': 'native LKH candidate selection and ranking followed by actual LK search',
                    'reward': '10000*(same-seed ordinary LKH length - sampled-candidate LKH length)/ordinary length',
                    'test_data_read': False, 'native_runtime': 'actor only, C forward inside LKH',
                    'training_config': config,
                })
            print(json.dumps({'phase': 'validation', 'seed': seed, 'episode': episode_number,
                              'mean_improvement_fraction': score['mean_improvement_fraction']}), flush=True)
    if selected is None:
        raise AssertionError('No validation-selected checkpoint')
    selected.save(folder / 'model_D.json')
    selected.export_native(folder / 'model_D.txt')
    changes = {}
    for prefix in ('actor0', 'actor1', 'actor2', 'critic'):
        changes[prefix] = float(sum(np.square(selected.parameters[k] - initial.parameters[k]).sum()
                                    for k in selected.parameters if k.startswith(prefix)) ** .5)
    if min(changes.values()) <= 0:
        raise AssertionError('Both nonlinear actor layers, output and critic must train')
    report = {'kind': 'D', 'training_seed': seed, 'episodes': episode_number,
              'actor_parameter_count': selected.actor_parameter_count,
              'parameter_count': selected.parameter_count,
              'selected_checkpoint': selected.metadata['checkpoint_episode'],
              'selected_parameter_change_l2': changes,
              'checkpoint_scores': reports, 'training_seconds': training_seconds,
              'validation_seconds': validation_seconds,
              'model_path': f'seed_{seed}/model_D.txt',
              'json_model_path': f'seed_{seed}/model_D.json',
              'model_file_sha256': digest(folder / 'model_D.txt'),
              'archival_model_file_sha256': digest(folder / 'model_D.json')}
    dump(folder / 'training_report_D.json', report)
    return report


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--lkh-binary', type=Path, required=True)
    parser.add_argument('--dependency-dir', type=Path)
    parser.add_argument('--train-maps', type=int, default=128)
    parser.add_argument('--validation-maps', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max-trials', type=int, default=100)
    parser.add_argument('--episode-timeout', type=float, default=10.)
    parser.add_argument('--training-seeds', nargs='+', type=int, default=TRAINING_SEEDS)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'deep_rl_benchmark' / 'deps'))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit('Training output must be empty; refusing overwrite')
    config = {'actor_architecture': [12, 64, 32, 1], 'critic_architecture': [25, 32, 1],
              'feature_names': FEATURE_NAMES, 'max_trials': args.max_trials,
              'episode_timeout': args.episode_timeout, 'lkh_binary': str(args.lkh_binary.resolve()),
              'lkh_binary_sha256': digest(args.lkh_binary), 'epochs': args.epochs,
              'learning_rate': args.learning_rate, 'reward_scale': 10000.,
              'entropy_coefficient': .01, 'critic_weight': .5, 'gradient_clip': 1.,
              'optimizer': 'Adam', 'training_device': 'CPU', 'torch_num_threads': 1,
              'actor': 'two ReLU hidden layers and scalar clipped edge logit',
              'training_actions': 'full Plackett-Luce candidate permutation per node',
              'decision_horizon': 'one static candidate assignment, followed by a complete fixed-trial native LKH episode and terminal reward; no dynamic intra-move policy updates',
              'action_decoder': 'retain first two ALPHA edges; select first three unprotected union edges; order all chosen edges by permutation',
              'gradient_scaling': 'exact full joint log probability divided by fixed episode candidate count; positive action-independent scaling',
              'validation_and_test_actions': 'deterministic decreasing actor score inside native LKH',
              'environment_return': 'independently recomputed complete native LKH final integer EUC_2D tour',
              'reward_baseline': 'same instance and fixed map-specific LKH seed, ordinary ALPHA5, identical fixed trial cap',
              'critic_inputs': 'mean and standard deviation of 12 edge features, and log(n)/10; no final cost input',
              'test_data_read': False}
    sources = {name: digest(Path(__file__).with_name(name))
               for name in ('train_internal.py', 'edge_model.py')}
    started_utc = datetime.now(timezone.utc).isoformat()
    dump(output / 'source_seal_start.json', {'created_utc': started_utc, 'source_sha256': sources,
                                           'binary_sha256': config['lkh_binary_sha256']})
    dump(output / 'frozen_training_config.json', config)
    train = make_manifest(output, 'train', args.train_maps)
    validation_entries = make_manifest(output, 'validation', args.validation_maps)
    if {e['coordinate_sha256'] for e in train} & {e['coordinate_sha256'] for e in validation_entries}:
        raise AssertionError('Train and validation coordinate hashes overlap')
    preparation_started = time.perf_counter()
    cache = prepare_baselines(output, train + validation_entries, config)
    preparation_seconds = time.perf_counter() - preparation_started
    reports = [train_one(output, train, validation_entries, cache, config, seed)
               for seed in args.training_seeds]
    registry = {'schema': 3, 'model_type': 'internal_lkh_deep_edge_actor_critic', 'by_test_seed': {}}
    for seed in args.training_seeds:
        test_seed = TEST_SEEDS[TRAINING_SEEDS.index(seed)] if seed in TRAINING_SEEDS else seed
        registry['by_test_seed'][str(test_seed)] = {
            'training_seed': seed, 'D': f'seed_{seed}/model_D.txt',
            'json_model': f'seed_{seed}/model_D.json'}
    dump(output / 'model_registry.json', registry)
    for name, expected in sources.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise RuntimeError('Training source changed during run: ' + name)
    if digest(args.lkh_binary) != config['lkh_binary_sha256']:
        raise RuntimeError('Native LKH binary changed during training')
    report = {'schema_version': 3, 'pilot': args.pilot, 'deep_neural_network': True,
              'actual_reinforcement_learning_inside_lkh': True,
              'training_completed_before_formal_test': True, 'test_data_read': False,
              'started_utc': started_utc, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'training_seeds': args.training_seeds, 'test_seed_mapping': registry['by_test_seed'],
              'train_map_count': len(train), 'validation_map_count': len(validation_entries),
              'train_seed_domain': [TRAIN_MAP_BASE, TRAIN_MAP_BASE + len(train) - 1],
              'validation_seed_domain': [VAL_MAP_BASE, VAL_MAP_BASE + len(validation_entries) - 1],
              'coordinate_hash_spec': 'SHA256 compact ASCII JSON original city-order integer coordinates',
              'map_sizes': SIZES, 'distributions': ['uniform', 'clustered'],
              'validation_only_checkpoint_selection': True, 'validation_gradients_computed': False,
              'config': config, 'models': reports, 'source_sha256': sources,
              'end_to_end_seconds': time.perf_counter() - start,
              'baseline_preparation_seconds': preparation_seconds,
              'training_seconds': sum(v['training_seconds'] for v in reports),
              'validation_seconds': sum(v['validation_seconds'] for v in reports),
              'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
              'platform': platform.platform(),
              'online_dependency': 'Native C actor; no Python/NumPy/Torch model inference',
              'limitations': ['Small offline training scale and fixed short LKH episode cap',
                              'Static local geometric edge scorer; not the paper attention architecture',
                              'Train sizes128..1000; larger test sizes assess transfer',
                              'Training stochastic candidate ordering; validation/test use deterministic logits',
                              'Two ALPHA backbone edges retained, limiting the learned action space',
                              'One map-specific solver seed during training; test solver seeds are separate']}
    dump(output / 'training_report.json', report)
    print(json.dumps({'completed': True, 'output': str(output),
                      'seconds': report['end_to_end_seconds']}), flush=True)


if __name__ == '__main__':
    main()
