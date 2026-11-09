"""Fixed-cohort and image-identity guards for the aggregate report."""
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import compare_cohort as c

class ComparisonContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.ids=[f"fixture-{i}" for i in range(5)]
        source=dict(commit=c.SOURCE,dirty=False,files={"shape/example.py":"3"*64})
        source['source_sha256']=c.old.sha256(json.dumps(source['files'],sort_keys=True).encode())
        self.meta=dict(inference_source=source,selection=dict(hf_revision=c.old.HF_REVISION,selection_sha256=c.SELECTION_SHA),
            manifest_fingerprint=c.FINGERPRINTS[5],model="gpt-5.6-luna",hardware=dict(cuda="12.1"),checkpoint_sha256="f"*64,
            summary=dict(n_planned=5,n_recorded=5,n_delivered=4,n_failed=1,ap50=.8,ap75=.8,map_50_95=.8,
                         tool_calls=5,native_calls=5,native_cache_hits=0,parameter_sources={"per_image_vrr":5}))
        self.rows=[]
        for n,i in enumerate(self.ids):
            success=n<4
            row=dict(image_id=i,arm="shape_cohort",image_sha256="1"*64,ground_truth_sha256="2"*64,
                     image_height_px="10",image_width_px="10",checkpoint_sha256="f"*64,source_commit=c.SOURCE,
                     source_sha256=source['source_sha256'],manifest_fingerprint=c.FINGERPRINTS[5],
                     evaluation_status="success" if success else "failed",delivered=str(success).lower(),error="" if success else "conceded",
                     tool_calls="1",native_calls="1",native_cache_hits="0",parameter_source="per_image_vrr")
            row.update({k:str(float(success)) for k in c.old.SCORES})
            self.rows.append(row)
        self.historical={a:{r['image_id']:copy.deepcopy(r) for r in self.rows} for a in ('shape','shape_seg_co')}
        self.save()

    def save(self):
        raw=c.old.csv_bytes(self.rows)
        (self.root/'shape_cohort.csv').write_bytes(raw)
        self.meta['file']=dict(sha256=c.old.sha256(raw),bytes=len(raw),rows=len(self.rows))
        (self.root/'manifest.json').write_text(json.dumps(self.meta),encoding='utf-8')

    def load(self):
        return c.load_export(self.root,self.ids,self.historical)

    def test_recorded_failure_is_retained(self):
        meta,rows,_=self.load()
        self.assertEqual(len(rows),5)
        self.assertEqual(c.old.mean(rows,'ap50'),.8)
        self.assertEqual(meta['summary']['n_failed'],1)

    def test_rehashed_reordered_csv_is_rejected(self):
        self.rows.reverse();self.save()
        with self.assertRaisesRegex(ValueError,'membership or ordering'):self.load()

    def test_rehashed_wrong_gt_is_rejected(self):
        self.rows[0]['ground_truth_sha256']='9'*64;self.save()
        with self.assertRaisesRegex(ValueError,'Image/GT'):self.load()

    def test_csv_tampering_is_rejected(self):
        with (self.root/'shape_cohort.csv').open('ab') as f:f.write(b'x')
        with self.assertRaisesRegex(ValueError,'hash or size'):self.load()

    def test_summary_denominator_change_is_rejected(self):
        self.meta['summary']['n_planned']=4;self.save()
        with self.assertRaisesRegex(ValueError,'Coverage'):self.load()

if __name__=='__main__':unittest.main()
