"""Routing must remain development-only and must never mix feature matrices."""
import importlib
import unittest

policy=importlib.import_module('case studies.cardiomyocytes.ctnt_flowcytometry.runners.prediction.plate_ssl_routing')


class RoutingTests(unittest.TestCase):
    def fixture(self):
        rows=[];zero=[];fine=[];labels={}
        for plate,nval in [('large',2),('small',1)]:
            for usage,n in [('development_train',2),('development_val',nval),('evaluation_only',2)]:
                for i in range(n):
                    well=f'{plate}_{usage}_{i}'
                    rows.append(dict(well=well,plate_id=plate,usage=usage))
                    pred=dict(well=well,plate_id=plate,usage=usage,image_ids=[well],predicted_ctnt_positive_percent=40.)
                    zero.append(pred);fine.append({**pred,'predicted_ctnt_positive_percent':49.})
                    if usage!='evaluation_only':labels[well]=50.
        return rows,zero,fine,labels

    def test_eligibility_counts_wells_not_images_and_does_not_move_splits(self):
        rows,zero,fine,labels=self.fixture()
        eligibility=policy.eligible_plates(rows+rows)
        self.assertTrue(eligibility['large']['eligible'])
        self.assertFalse(eligibility['small']['eligible'])
        self.assertEqual(eligibility['large']['counts']['development_val'],2)

    def test_route_uses_validation_only_and_applies_consistently(self):
        rows,zero,fine,labels=self.fixture();eligibility=policy.eligible_plates(rows)
        routes=policy.choose_routes(zero,fine,labels,eligibility)
        self.assertEqual(routes['large']['branch'],'finetuned')
        self.assertEqual(routes['small']['branch'],'zero_shot')
        altered=[{**r,'predicted_ctnt_positive_percent':0.} if r['usage']=='evaluation_only' else r for r in fine]
        self.assertEqual(routes,policy.choose_routes(zero,altered,labels,eligibility))
        merged=policy.merge_predictions(zero,fine,routes)
        self.assertEqual(len(merged),len(zero))
        self.assertTrue(all(r['prediction_branch']==('finetuned' if r['plate_id']=='large' else 'zero_shot') for r in merged))

    def test_ties_no_checkpoint_and_worse_candidates_keep_zero_shot(self):
        rows,zero,fine,labels=self.fixture();eligibility=policy.eligible_plates(rows)
        for candidate in [None,zero,[{**r,'predicted_ctnt_positive_percent':0.} for r in fine]]:
            routes=policy.choose_routes(zero,candidate,labels,eligibility)
            self.assertTrue(all(r['branch']=='zero_shot' for r in routes.values()))

    def test_rejects_heldout_targets_and_mismatched_branch_rows(self):
        rows,zero,fine,labels=self.fixture();eligibility=policy.eligible_plates(rows)
        with self.assertRaises(ValueError):
            policy.choose_routes(zero,fine,{**labels,next(r['well'] for r in zero if r['usage']=='evaluation_only'):50.},eligibility)
        with self.assertRaises(ValueError):policy.choose_routes(zero,fine[:-1],labels,eligibility)


if __name__=='__main__':unittest.main()
