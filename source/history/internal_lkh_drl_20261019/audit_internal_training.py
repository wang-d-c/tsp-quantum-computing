"""Independent artifact checks for actual LKH-return policy training."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from edge_model import load_native_actor, load_policy, read_candidate_graph


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coord_sha(coords):
    return hashlib.sha256(json.dumps(coords, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


def independent_tour_cost(path, coords):
    text = Path(path).read_text(encoding='ascii')
    if 'TOUR_SECTION' not in text:
        raise AssertionError('Missing tour section')
    route = []
    for token in text.split('TOUR_SECTION', 1)[1].split():
        if token in ('-1', 'EOF'):
            break
        route.append(int(token) - 1)
    if len(route) != len(coords) or set(route) != set(range(len(coords))):
        raise AssertionError('Incomplete/nonpermutation tour')
    total = 0
    for i in range(len(route)):
        x1, y1 = coords[route[i]]
        x2, y2 = coords[route[(i + 1) % len(route)]]
        total += math.floor(math.hypot(x1 - x2, y1 - y2) + .5)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report = json.loads((directory / 'training_report.json').read_text(encoding='ascii'))
    seal = json.loads((directory / 'source_seal_start.json').read_text(encoding='ascii'))
    errors, checks = [], 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    for name, expected in seal['source_sha256'].items():
        check(sha(Path(__file__).with_name(name)) == expected, 'Training source hash: ' + name)
    check(report['source_sha256'] == seal['source_sha256'], 'Start/final source hash consistency')
    check(report['test_data_read'] is False, 'Test files were excluded from training')
    check(report['validation_only_checkpoint_selection'] is True, 'Validation-only selection')
    check(report['validation_gradients_computed'] is False, 'No validation gradients')
    check(sha(report['config']['lkh_binary']) == seal['binary_sha256'], 'Engine hash preserved')
    records, coordinates, baseline, graphs = {}, {}, {}, {}
    for split in ('train', 'validation'):
        entries = json.loads((directory / f'{split}_manifest.json').read_text(encoding='ascii'))
        for entry in entries:
            actual = json.loads((directory / entry['path']).read_text(encoding='ascii'))
            coords = actual['coords']
            check(coord_sha(coords) == entry['coordinate_sha256'], 'Coordinate hash: ' + entry['id'])
            check(actual['coordinate_sha256'] == entry['coordinate_sha256'], 'Map metadata hash')
            check(actual['split'] == split, 'Map split metadata')
            check(actual['n'] == len(coords) == entry['n'], 'Map dimensions')
            records[entry['id']] = entry
            coordinates[entry['id']] = coords
            folder = directory / 'baselines' / entry['id']
            episode = json.loads((folder / 'episode.json').read_text(encoding='ascii'))
            actual_cost = independent_tour_cost(folder / 'best.tour', coords)
            check(actual_cost == episode['length'], 'Ordinary baseline cost recomputation')
            check(episode['solver_seed'] == entry['solver_seed'], 'Baseline solver seed')
            check(episode['max_trials'] == report['config']['max_trials'], 'Baseline trial cap')
            check(episode['native_stats']['active'] is False, 'Ordinary baseline candidate arrays unmodified')
            baseline[entry['id']] = actual_cost
            graphs[entry['id']] = read_candidate_graph(folder / 'features.csv')
    hashes = [entry['coordinate_sha256'] for entry in records.values()]
    check(len(hashes) == len(set(hashes)), 'All training/validation maps disjoint')
    reward_signs = Counter()
    episode_count, validation_count, action_edges = 0, 0, 0
    for model_report in report['models']:
        seed = model_report['training_seed']
        folder = directory / f'seed_{seed}'
        selected = load_policy(folder / 'model_D.json')
        native = load_native_actor(folder / 'model_D.txt')
        for key, value in native.items():
            check(np.array_equal(value.astype(np.float32), selected.parameters[key]), 'Native/archival actor identity')
        check(all(v > 0 for v in model_report['selected_parameter_change_l2'].values()),
              'Both actor hidden layers and critic updated')
        scores = model_report['checkpoint_scores']
        best = max(scores, key=lambda v: (v['mean_improvement_fraction'], -v['episode']))
        check(best['episode'] == model_report['selected_checkpoint'], 'Validation selected checkpoint')
        episodes = [json.loads(line) for line in (folder / 'training_episodes.jsonl').read_text(encoding='ascii').splitlines()]
        check(len(episodes) == model_report['episodes'], 'Training episode count')
        for record in episodes:
            episode_count += 1
            entry = records[record['map_id']]
            check(entry['split'] == 'train', 'No validation/test training episode')
            check(record['coordinate_sha256'] == entry['coordinate_sha256'], 'Episode actual training map hash')
            job = folder / 'training' / f'episode_{record["episode"]:04d}'
            native_episode = json.loads((job / 'lkh' / 'episode.json').read_text(encoding='ascii'))
            cost = independent_tour_cost(job / 'lkh' / 'best.tour', coordinates[entry['id']])
            check(cost == native_episode['length'] == record['policy_length'], 'Actual native training return')
            check(record['baseline_length'] == baseline[entry['id']], 'Same-map baseline identity')
            check(native_episode['solver_seed'] == record['solver_seed'] == entry['solver_seed'], 'Paired solver seed')
            check(native_episode['max_trials'] == report['config']['max_trials'], 'Policy trial cap')
            reward = report['config']['reward_scale'] * (baseline[entry['id']] - cost) / baseline[entry['id']]
            check(abs(record['terminal_reward'] - reward) < 1e-10, 'True terminal reward recomputation')
            expected_advantage = float(np.float32(record['terminal_reward']) - np.float32(record['critic_value']))
            check(abs(record['advantage'] - expected_advantage) < 1e-12,
                  'Critic baseline advantage')
            check(record['return_is_actual_native_lkh_final_tour'] is True, 'Actual-return marker')
            check(sha(job / 'actions.csv') == record['actions_sha256'], 'Sampled action hash')
            with (job / 'actions.csv').open(encoding='ascii', newline='') as stream:
                rows = list(csv.DictReader(stream))
            actual_actions = Counter((int(r['node']), int(r['target'])) for r in rows)
            graph = graphs[entry['id']]
            expected_actions = Counter(zip(graph.nodes.tolist(), graph.targets.tolist()))
            check(actual_actions == expected_actions, 'Sampled actions exact candidate union permutation')
            check(record['decision_count'] == len(rows), 'Exact PL decision count')
            native_stats = native_episode['native_stats']
            check(native_episode['mode'] == 'actions' and native_stats['active'] is True,
                  'Sampled policy applied inside native engine')
            check(native_stats['scored_candidates'] == len(rows), 'Native/action union count identity')
            check(native_stats['selected_candidates'] == 5 * entry['n'] and
                  native_stats['cost_checks'] == 5 * entry['n'], 'Five valid internal candidates per node')
            check(native_stats['best5opt_calls'] > 0, 'Selected candidates used by internal LK move search')
            check(native_stats['rerank_calls'] >= 1, 'Learned order restored after native candidate reset')
            check(abs(record['mean_joint_log_probability'] * len(rows) - record['joint_log_probability']) < 1e-7,
                  'Joint PL gradient scaling')
            check(all(math.isfinite(record[key]) for key in (
                'terminal_reward', 'critic_value', 'advantage', 'mean_joint_log_probability',
                'loss', 'actor_loss', 'critic_loss', 'gradient_norm_before_clip')), 'Finite actual-return update')
            check(sha(job / 'lkh' / 'best.tour') == record['tour_file_sha256'], 'Training tour hash')
            reward_signs['positive' if reward > 0 else 'negative' if reward < 0 else 'zero'] += 1
            action_edges += len(rows)
        for checkpoint in scores:
            root = folder / f'checkpoint_{checkpoint["episode"]:04d}' / 'validation'
            summary = json.loads((root / 'validation_summary.json').read_text(encoding='ascii'))
            improvements = []
            for record in summary['records']:
                validation_count += 1
                entry = records[record['id']]
                check(entry['split'] == 'validation', 'Held-out validation map')
                cost = independent_tour_cost(root / entry['id'] / 'best.tour', coordinates[entry['id']])
                check(cost == record['policy_length'], 'Validation cost independently recomputed')
                native_episode = json.loads((root / entry['id'] / 'episode.json').read_text(encoding='ascii'))
                check(native_episode['mode'] == 'model' and native_episode['native_stats']['has_model'] is True,
                      'Validation used actual native deep-network forward')
                improvements.append((baseline[entry['id']] - cost) / baseline[entry['id']])
            check(abs(float(np.mean(improvements)) - checkpoint['mean_improvement_fraction']) < 1e-14,
                  'Validation score recomputed')
    result = {'passed': not errors, 'errors': errors, 'checks': checks,
              'models': len(report['models']), 'training_maps': report['train_map_count'],
              'validation_maps': report['validation_map_count'],
              'training_episodes_recomputed': episode_count, 'validation_episodes_recomputed': validation_count,
              'sampled_candidate_edge_decisions_audited': action_edges,
              'training_terminal_reward_signs': dict(reward_signs),
              'return_audit': 'All saved LKH tours independently parsed and scalar integer EUC_2D costs recomputed',
              'audit_scope': 'Artifact/gradient bookkeeping; source inspection and native hook tests additionally required'}
    (directory / 'training_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='ascii')
    print(json.dumps(result), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
