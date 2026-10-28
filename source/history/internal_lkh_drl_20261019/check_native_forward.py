"""Compare real native C edge scores and selected arrays to independent forward."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from edge_model import load_native_actor, load_policy, native_reference_logits, read_native_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--training-directory', required=True, type=Path)
    parser.add_argument('--training-seed', type=int, default=3101)
    parser.add_argument('--maps', type=int, default=8)
    args = parser.parse_args()
    root = args.training_directory.resolve()
    binary = args.binary.resolve()
    native_path = root / f'seed_{args.training_seed}' / 'model_D.txt'
    model_path = root / f'seed_{args.training_seed}' / 'model_D.json'
    native = load_native_actor(native_path)
    model = load_policy(model_path)
    manifest = json.loads((root / 'train_manifest.json').read_text(encoding='ascii'))[:args.maps]
    records = []
    for entry in manifest:
        folder = root / 'real_native_parity' / entry['id']
        folder.mkdir(parents=True, exist_ok=False)
        parameter = folder / 'run.par'
        parameter.write_text('\n'.join([
            'PROBLEM_FILE = ' + (root / 'tsp' / (entry['id'] + '.tsp')).as_posix(),
            'OUTPUT_TOUR_FILE = ' + (folder / 'best.tour').as_posix(),
            'CANDIDATE_SET_TYPE = ALPHA', 'MAX_CANDIDATES = 5',
            'RUNS = 1', 'MAX_TRIALS = 1', 'STOP_AT_OPTIMUM = NO',
            'SEED = ' + str(entry['solver_seed']), 'TRACE_LEVEL = 1',
        ]) + '\n', encoding='ascii')
        env = {k: v for k, v in os.environ.items() if not k.startswith('INTERNAL_RL_')}
        env.update({'INTERNAL_RL_MODE': 'model', 'INTERNAL_RL_MODEL': str(native_path),
                    'INTERNAL_RL_DUMP': str(folder / 'features.csv'),
                    'INTERNAL_RL_SCORES': str(folder / 'scores.csv'),
                    'INTERNAL_RL_SELECTED': str(folder / 'selected.csv'),
                    'INTERNAL_RL_STATS': str(folder / 'stats.json')})
        with (folder / 'stdout.log').open('wb') as out, (folder / 'stderr.log').open('wb') as err:
            result = subprocess.run([str(binary), str(parameter)], cwd=folder, env=env,
                                    stdout=out, stderr=err, timeout=30,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode != 0:
            raise AssertionError('Native parity run failed: ' + str(folder))
        with (folder / 'features.csv').open(encoding='ascii', newline='') as stream:
            feature_rows = list(csv.DictReader(stream))
        with (folder / 'scores.csv').open(encoding='ascii', newline='') as stream:
            score_rows = list(csv.DictReader(stream))
        with (folder / 'selected.csv').open(encoding='ascii', newline='') as stream:
            selected_rows = list(csv.DictReader(stream))
        features = np.asarray([[float(row['f' + str(i)]) for i in range(12)] for row in feature_rows])
        reference = native_reference_logits(features, native)
        actual = np.asarray([float(row['score']) for row in score_rows])
        max_error = float(np.max(np.abs(reference - actual)))
        float_error = float(np.max(np.abs(model.logits(features) - actual)))
        if max_error > 2e-10 or float_error > 2e-5:
            raise AssertionError('Native deep forward disagreement')
        feature_ids = [(int(r['node']), int(r['target'])) for r in feature_rows]
        score_ids = [(int(r['node']), int(r['target'])) for r in score_rows]
        if feature_ids != score_ids:
            raise AssertionError('Native score and feature row mismatch')
        chosen = {}
        for row in selected_rows:
            chosen.setdefault(int(row['node']), []).append(int(row['target']))
        for node in range(1, entry['n'] + 1):
            indices = [i for i, pair in enumerate(feature_ids) if pair[0] == node]
            order = sorted(indices, key=lambda i: (-reference[i], feature_ids[i][1]))
            optional, expected = 0, []
            for i in order:
                protected = int(feature_rows[i]['protected'])
                if protected or optional < 3:
                    expected.append(feature_ids[i][1])
                    optional += not protected
            if chosen[node] != expected or len(expected) != 5:
                raise AssertionError('Native selected candidate array differs from decoder')
        stats = read_native_stats(folder / 'stats.json')
        if stats['cost_checks'] != 5 * entry['n'] or stats['selected_candidates'] != 5 * entry['n']:
            raise AssertionError('Native candidate-count/cost invariant')
        if stats['best5opt_calls'] < 1:
            raise AssertionError('Native ranked candidates never reached internal LK move search')
        records.append({'map_id': entry['id'], 'n': entry['n'],
                        'native_double_forward_max_error': max_error,
                        'float32_training_forward_max_error': float_error,
                        'all_internal_selected_candidates_match': True,
                        'native_stats': stats})
    report = {'passed': True, 'maps': len(records), 'records': records,
              'all_online_native_scores_match_exported_network': True,
              'all_candidate_arrays_match_policy_decoder': True,
              'internal_lk_move_search_observed': True}
    (root / 'real_native_forward_audit.json').write_text(json.dumps(report, indent=2), encoding='ascii')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
