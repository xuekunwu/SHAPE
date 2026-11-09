"""Coverage, audit attribution and failure-inclusive export contracts."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import export_cohort as export

COMMIT="a"*40


class ExportContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sha="f"*64
        pairs=[dict(image_id=i,shape=[10,10],image_sha256="1"*64,label_sha256="2"*64) for i in ("a","b")]
        manifest=dict(schema=1,pairs=pairs)
        digest=lambda o: hashlib.sha256(json.dumps(o,sort_keys=True).encode()).hexdigest()
        manifest["fingerprint"]=digest(manifest)
        files={"shape/tool.py":"3"*64}
        source=dict(commit=COMMIT,dirty=False,files=files,source_sha256=digest(files))
        summary=dict(n_planned=2,n_recorded=2,n_delivered=1,n_failed=1,native_calls=2,tool_calls=2,llm_calls=1,tokens=17,native_cache_hits=0,parameter_sources={"per_image_vrr":2},ap50=.5,ap75=.5,map_50_95=.5)
        meta=dict(status="COMPLETED_WITH_FAILURES",source=source,manifest=manifest,checkpoint_sha256=self.sha,summary=summary,model="gpt-5.6-luna",reasoning_effort="high",packages={},hardware={"gpu":"fixture"},job_id="1",cost_scope="fixture")
        self.write("run_meta.json",meta);self.write("summary.json",summary);self.write("cohort.json",{})
        records=[]
        for i,delivered in (("a",True),("b",False)):
            value=1. if delivered else 0.
            row=dict(image_id=i,delivered=delivered,error=None if delivered else "native candidate failed",seconds=3.,n_managed_passes=1,parameter_source="per_image_vrr",mask_sha256="4"*64 if delivered else None,_thresholds={"0.50":{"true_positive":int(delivered)}})
            row.update({k:value for k in (*export.SCORES,*export.COUNTS)})
            records.append(row)
        self.lines("records.jsonl",records)
        self.lines("TOOL_CALLS.jsonl",[dict(parameters={"image_id":i},native_requests=1,result={}) for i in ("a","b")])
        self.lines("NATIVE_CALLS.jsonl",[dict(device="cuda",weights_sha256=self.sha) for _ in range(2)])
        self.lines("LLM_CALLS.jsonl",[dict(model=meta["model"],reasoning_effort="high",tokens=17)])

    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value),encoding="utf-8")

    def lines(self,name,rows):
        (self.root/name).write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")

    def read_lines(self,name):
        return [json.loads(s) for s in (self.root/name).read_text().splitlines()]

    def test_failure_stays_in_denominator_and_shared_usage_is_not_allocated(self):
        data,receipt=export.build(self.root,COMMIT)
        rows=list(csv.DictReader(io.StringIO(data.decode())))
        self.assertEqual([r["evaluation_status"] for r in rows],["success","failed"])
        self.assertEqual(sum(float(r["ap50"]) for r in rows)/len(rows),.5)
        self.assertNotIn("tokens",rows[0])
        self.assertEqual(json.loads(receipt)["summary"]["tokens"],17)

    def test_missing_prediction_is_rejected(self):
        self.lines("records.jsonl",self.read_lines("records.jsonl")[:1])
        with self.assertRaisesRegex(ValueError,"Missing"):
            export.build(self.root,COMMIT)

    def test_unattributed_tool_call_is_rejected(self):
        rows=self.read_lines("TOOL_CALLS.jsonl");rows[0]["parameters"]["image_id"]="unknown"
        self.lines("TOOL_CALLS.jsonl",rows)
        with self.assertRaisesRegex(ValueError,"Unattributed"):
            export.build(self.root,COMMIT)

    def test_llm_without_usage_is_rejected(self):
        rows=self.read_lines("LLM_CALLS.jsonl");rows[0]["tokens"]=None
        self.lines("LLM_CALLS.jsonl",rows)
        with self.assertRaisesRegex(ValueError,"positive token"):
            export.build(self.root,COMMIT)

    def test_cpu_call_is_rejected(self):
        rows=self.read_lines("NATIVE_CALLS.jsonl");rows[0]["device"]="cpu"
        self.lines("NATIVE_CALLS.jsonl",rows)
        with self.assertRaisesRegex(ValueError,"CUDA"):
            export.build(self.root,COMMIT)

    def test_mask_hash_change_is_rejected(self):
        p=self.root/"images/a/mask.tif";p.parent.mkdir(parents=True);p.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"mask hash"):
            export.build(self.root,COMMIT,verify_masks=True)


if __name__=="__main__":
    unittest.main()
