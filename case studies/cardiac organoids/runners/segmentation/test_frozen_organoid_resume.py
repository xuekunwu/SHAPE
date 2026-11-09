"""Cost-sensitive launcher checks; these tests never load an inference environment."""
import hashlib,importlib.util,json,os,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('frozen_shard',Path(__file__).with_name('run_frozen_organoid_shard.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class NoRepeatTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'manifests').mkdir()
        self.write(self.root/'plan.json',dict(tasks=dict(shape=119),full_manifest='full'))
        self.full=self.root/'manifests/full.json';self.write(self.full,dict(pairs=[dict(image_id='already_done')]))

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def completed(self,slot,manifest,n,arm='shape'):
        self.write(self.root/arm/f'shard_{slot:03d}'/'stage_receipt.json',dict(technical_complete=True,n_expected=n,n_recorded=n,source_commit=runner.SOURCE,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))

    def invoke(self,slot,arm='shape'):
        argv=['runner','--arm',arm,'--study-root',str(self.root),'--source-root',str(self.root),'--shard',str(slot)]
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,SLURM_JOB_ID='local-test',GROUP_HOME=str(self.root/'group'),GROUP_SCRATCH=str(self.root/'group_scratch')))
            stack.enter_context(patch.object(sys,'argv',argv))
            stack.enter_context(patch.object(runner.subprocess,'check_output',side_effect=[runner.SOURCE+'\n','']))
            call=stack.enter_context(patch.object(runner.subprocess,'run',side_effect=AssertionError('A completed/empty slot must not call inference')))
            runner.main();self.assertEqual(call.call_count,0)

    def continuation(self,pairs):
        path=self.root/'manifests/full_remaining.json';self.write(path,dict(pairs=pairs))
        self.write(self.root/'continuation.json',dict(first_continuation_slot=9,manifest='manifests/full_remaining.json',manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return path

    def test_completed_stage_makes_zero_model_calls(self):
        self.completed(0,self.full,1);self.invoke(0)

    def test_changed_manifest_cannot_reuse_completed_receipt(self):
        self.completed(0,self.full,1);self.write(self.full,dict(pairs=[dict(image_id='different')]))
        with self.assertRaisesRegex(ValueError,'Completed receipt'):self.invoke(0)

    def test_continuation_uses_remaining_manifest_relative_index(self):
        path=self.continuation([dict(image_id=str(i)) for i in range(65)])
        self.completed(10,path,1);self.invoke(10)

    def test_empty_continuation_records_zero_call_receipt(self):
        self.continuation([]);self.invoke(9)
        receipt=json.loads((self.root/'shape/shard_009/stage_receipt.json').read_text())
        self.assertTrue(receipt['technical_complete']);self.assertTrue(receipt['empty_continuation_slot'])
        self.assertEqual(receipt['new_model_calls'],0)

    def test_react_full_manifest_reuses_completed_stage(self):
        self.write(self.root/'plan.json',dict(tasks=dict(react=2),agent_manifest_pattern='full_{shard:03d}.json'))
        path=self.root/'manifests/full_001.json';self.write(path,dict(pairs=[dict(image_id='react_done')]))
        self.completed(1,path,1,arm='react');self.invoke(1,arm='react')

if __name__=='__main__':unittest.main()
