import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('full_analysis',Path(__file__).resolve().parents[1]/'evaluation/historical/six_arm_analysis.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RegressionReadoutTests(unittest.TestCase):
    def test_tolerance_is_inclusive_and_uses_percentage_points(self):
        r=mod.regression_readouts([10,20,30,40],[15,10,45,60])
        self.assertEqual([r['within_pp'][str(t)]['count'] for t in [5,10,15,20]],[1,2,3,4])
        self.assertEqual(r['mae_pp'],12.5)
        self.assertEqual(r['bias_pp'],7.5)

    def test_constant_predictions_have_no_correlation(self):
        r=mod.regression_readouts([0,50,100],[50,50,50])
        self.assertIsNone(r['pearson_r'])
        self.assertEqual(r['ccc'],0)

    def test_perfect_predictions(self):
        r=mod.regression_readouts([0,25,80],[0,25,80])
        self.assertEqual(r['ccc'],1)
        self.assertEqual(r['r2'],1)
        self.assertEqual(r['within_pp']['5']['fraction'],1)

    def test_rejects_missing_or_unpaired_values(self):
        for y,p in [([],[]),([1],[1,2]),([1,float('nan')],[1,2])]:
            with self.assertRaises(ValueError): mod.regression_readouts(y,p)


if __name__=='__main__': unittest.main()
