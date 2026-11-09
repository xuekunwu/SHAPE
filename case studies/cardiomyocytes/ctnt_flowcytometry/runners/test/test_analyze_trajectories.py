import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('trajectory_analysis',Path(__file__).resolve().parents[1]/'evaluation/historical/six_arm_trajectories.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class UsageScopeTests(unittest.TestCase):
    def test_counts_only_usage_once_not_prompt_reasoning_or_llm_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image.jsonl'
            events=[{'kind':kind,'label':'round','text':json.dumps({'total_tokens':100,'reasoning_tokens':30})} for kind in ['prompt','usage','reasoning','llm_call']]
            path.write_text('\n'.join(json.dumps(e) for e in events))
            s=mod.log_summary(path)
            self.assertEqual(s['labels']['round']['calls'],1)
            self.assertEqual(s['labels']['round']['total_tokens'],100)
            self.assertEqual(s['labels']['round']['reasoning_tokens'],30)

    def test_missing_log_is_explicit(self):
        s=mod.log_summary(Path('/nonexistent/ctnt_log.jsonl'))
        self.assertFalse(s['log_exists'])
        self.assertEqual(s['labels'],{})


if __name__=='__main__': unittest.main()
