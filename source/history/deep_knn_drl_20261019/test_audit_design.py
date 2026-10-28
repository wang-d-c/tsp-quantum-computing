"""Independent examples ensure prospective audit rejects altered rewards/orders."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import audit_design as design


class AuditFixtures(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parent / 'test_cache'
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='deep_audit_', dir=cache)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_training_reward_rebuild_and_tamper_rejection(self):
        entry = dict(coordinate_sha256='maphash', initial_length=1000, final_length=990,
                     reward_identity_passed=True, improvement_fraction=.01,
                     total_exact_gain=10, reward_scale_initial_mean_edge=10.,
                     selection_seed=123 ^ 0x5E1EC7, proposal_seed=123,
                     steps=[dict(best_gain=10, parent_gain=10, reward=1., value=.2,
                                 chosen_probability=.5)],
                     updates=[dict(loss=.1, gradient_norm_before_clip=.5,
                                   encoder_gradient_norm_after_clip=.1)])
        path = self.root / 'training_D.jsonl'
        path.write_text('\n'.join(json.dumps(entry) for _ in range(2)), encoding='utf-8')
        audit = design.Audit()
        result = design.episode_checks(audit, path, {'maphash'}, True, 2, 128)
        self.assertFalse(audit.errors)
        self.assertEqual(result['steps'], 2)
        corrupt = copy.deepcopy(entry)
        corrupt['steps'][0]['reward'] = 0.
        path.write_text(json.dumps(corrupt) + '\n' + json.dumps(entry), encoding='utf-8')
        audit = design.Audit()
        design.episode_checks(audit, path, {'maphash'}, True, 2, 128)
        self.assertIn('episode_reward_uses_global_incumbent', {x['code'] for x in audit.errors})

    def test_balanced_schedule_and_tamper_rejection(self):
        records = [dict(id='map' + str(i)) for i in range(12)]
        protocol = dict(development_smoke=False, methods=list(design.METHODS),
                        seeds=list(design.TEST_SEEDS), total_budget_seconds=30,
                        planned_jobs=180, quantum_execution=False, bootstrap_draws=20000,
                        uncertainty='95% and97.5%', primary='D-A', secondary='D-U',
                        settings=dict(preprocessing_fraction_target=.25, local_max_steps=128,
                                      pool_size=3, d_candidate_count=192, d_exact_quota=1, knn=16))
        schedule = []
        orders = list(itertools.permutations(design.METHODS))
        block = 0
        for record in records:
            for seed in design.TEST_SEEDS:
                order = orders[block % 6]
                block += 1
                for position, method in enumerate(order, 1):
                    schedule.append(dict(job=len(schedule) + 1, map_id=record['id'], seed=seed,
                                         method=method, block=block, position=position,
                                         block_order=''.join(order)))
        for name, value in [('frozen_protocol.json', protocol), ('schedule.json', schedule)]:
            (self.root / name).write_text(json.dumps(value), encoding='utf-8')
        audit = design.Audit()
        design.schedule_checks(audit, self.root, records)
        self.assertFalse(audit.errors)
        schedule[0]['position'] = 3
        (self.root / 'schedule.json').write_text(json.dumps(schedule), encoding='utf-8')
        audit = design.Audit()
        design.schedule_checks(audit, self.root, records)
        self.assertIn('block_schedule_metadata_consistent', {x['code'] for x in audit.errors})


if __name__ == '__main__':
    unittest.main()
