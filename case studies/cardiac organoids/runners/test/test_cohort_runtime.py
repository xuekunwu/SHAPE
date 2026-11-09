"""Persistence, corruption rejection and real subprocess timeout/restart contracts."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'segmentation'))
from cohort_runtime import RunStore,IsolatedWorker,atomic_json


def fake_worker(conn,config):
    conn.send({'status':'READY'})
    while True:
        try:request=conn.recv()
        except EOFError:return
        if request.get('sleep'):time.sleep(30)
        conn.send({'status':'DONE','value':request.get('value')})

class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run'
    def test_completed_failure_is_replayed_instead_of_retried(self):
        with RunStore(self.root,{'source':'a'}) as store:
            store.save('passes/a',{'status':'TIMEOUT','error':'budget exhausted'})
        with RunStore(self.root,{'source':'a'},resume=True) as store:
            self.assertEqual(store.load('passes/a')['status'],'TIMEOUT')
    def test_resume_requires_same_source_inputs_environment_and_budget(self):
        with RunStore(self.root,{'source':'a','budget':5}):pass
        with self.assertRaisesRegex(ValueError,'identity'):
            RunStore(self.root,{'source':'a','budget':6},resume=True)
    def test_artifact_corruption_is_not_silently_recomputed(self):
        with RunStore(self.root,{'source':'a'}) as store:
            p=self.root/'mask';p.write_bytes(b'original')
            store.save('passes/a',{'ok':True},[p]);p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact'):store.load('passes/a')
    def test_metadata_corruption_is_rejected(self):
        with RunStore(self.root,{'source':'a'}) as store:
            store.save('passes/a',{'status':'DONE'})
            p=self.root/'passes/a/receipt.json';d=json.loads(p.read_text());d['payload']['status']='success';atomic_json(p,d)
            with self.assertRaisesRegex(ValueError,'Corrupt'):store.load('passes/a')
    def test_second_writer_is_rejected(self):
        with RunStore(self.root,{'source':'a'}):
            with self.assertRaisesRegex(ValueError,'active writer'):RunStore(self.root,{'source':'a'},resume=True)
    def test_timeout_stops_worker_and_next_request_uses_fresh_process(self):
        worker=IsolatedWorker(fake_worker,{},startup_seconds=20);self.addCleanup(worker.stop)
        worker.start()
        self.assertEqual(worker.request({'sleep':True},seconds=.1)['status'],'TIMEOUT')
        self.assertIsNone(worker.process)
        self.assertEqual(worker.request({'value':42},seconds=2)['value'],42)
        self.assertEqual(worker.starts,2)

if __name__=='__main__':unittest.main()
