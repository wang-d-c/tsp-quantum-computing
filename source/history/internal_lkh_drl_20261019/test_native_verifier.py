"""Meaningful corruption/gate checks for the independent native verifier."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_native_study as verifier


class NativeVerifierTests(unittest.TestCase):
    def fixture(self,root):
        coords=[[i*113+(i%3)*7,(i*i*71)%997] for i in range(20)]
        knn=verifier.independent_knn(coords)
        directory=root/'j001'
        folder=directory/'lkh'
        folder.mkdir(parents=True)
        model=root/'zero_actor.txt'
        model.write_text('IRL_MLP_V1 12 64 32 1\n'+' '.join(['0']*2945)+'\n',encoding='ascii')
        scores,chosen=[],[]
        for node in range(1,21):
            targets=knn['targets'][node-1]
            protected=set(targets[:2])
            optional=sorted(set(targets)-protected)[:3]
            selected=sorted(protected|set(optional))
            for rank,target in enumerate(targets):
                selected_rank=selected.index(target) if target in selected else -1
                scores.append(dict(node=node,target=target,score=0.,selected=int(selected_rank>=0),
                                   selected_rank=selected_rank,alpha=rank if rank<5 else 2147483646,
                                   alpha_rank=rank if rank<5 else -1,knn_rank=rank,protected=int(rank<2)))
            for rank,target in enumerate(selected):
                union=next(r for r in scores if r['node']==node and r['target']==target)
                x,y=coords[node-1]
                u,v=coords[target-1]
                square=(x-u)**2+(y-v)**2
                lower=__import__('math').isqrt(square)
                distance=lower+int(square-lower*lower>lower)
                chosen.append(dict(node=node,rank=rank,target=target,cost=distance*100+node*3+target*3,
                                   alpha=union['alpha'],protected=union['protected'],score=0.,pi_from=node*3,pi_to=target*3))
        def write_csv(path,rows):
            with path.open('w',encoding='ascii',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_csv(folder/'scored_candidates.csv',scores)
        write_csv(folder/'selected_candidates.csv',chosen)
        stats=dict(schema=1,active=True,has_model=True,dimension=20,features=12,hidden=[64,32],
                   scored_candidates=len(scores),selected_candidates=100,rerank_calls=0,best5opt_calls=1,cost_checks=100,
                   model_load_seconds=.001,feature_build_seconds=.002,inference_seconds=.003,rank_seconds=.004)
        (folder/'internal_stats.json').write_text(json.dumps(stats)+'\n{"partial":',encoding='ascii')
        task=dict(method='D',seed=101,settings={'policy_by_test_seed':{'101':{'D':str(model)}}})
        worker=dict(lkh={'internal_policy_statistics':stats,'elapsed_seconds':1.,'observed_tour_files':1})
        return directory,coords,knn,task,worker,scores,chosen,write_csv

    def test_known_zero_policy_and_partial_final_snapshot_pass(self):
        cache=Path(__file__).resolve().parent/'test_cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache,prefix='native_verify_') as temporary:
            root=Path(temporary).resolve()
            directory,coords,knn,task,worker,*_=self.fixture(root)
            audit=verifier.Auditor(root)
            result=verifier.audit_native(audit,directory,task,worker,coords,knn,'fixture')
            self.assertFalse(audit.errors)
            self.assertTrue(result['hook_reached'])
            self.assertEqual(result['max_score_absolute_error'],0.)
            self.assertEqual(audit.counts['native_best5opt_consumption_proved'],1)

    def test_score_cost_and_union_corruption_are_rejected(self):
        cache=Path(__file__).resolve().parent/'test_cache'
        cache.mkdir(exist_ok=True)
        for corruption in ('score','cost','union'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory(dir=cache,prefix='native_tamper_') as temporary:
                root=Path(temporary).resolve()
                directory,coords,knn,task,worker,scores,chosen,write_csv=self.fixture(root)
                if corruption=='score':scores[0]['score']=.1
                elif corruption=='cost':chosen[0]['cost']+=1
                else:scores[7]['alpha_rank']=-1;scores[7]['knn_rank']=-1
                write_csv(directory/'lkh/scored_candidates.csv',scores)
                write_csv(directory/'lkh/selected_candidates.csv',chosen)
                audit=verifier.Auditor(root)
                verifier.audit_native(audit,directory,task,worker,coords,knn,'fixture')
                self.assertTrue(audit.errors)

    def test_incomplete_study_gate_precedes_all_outcome_reads(self):
        cache=Path(__file__).resolve().parent/'test_cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache,prefix='incomplete_') as temporary:
            with patch.object(verifier.sys,'argv',['verify_native_study.py',temporary]), \
                 patch.object(verifier.Auditor,'payload',side_effect=AssertionError('premature outcome read')), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):verifier.main()


if __name__=='__main__':unittest.main()
