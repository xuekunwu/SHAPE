#!/usr/bin/env python3
"""Materialize an explicit challenge/pilot selection from a frozen metadata manifest; no image decoding."""
import argparse
import json
from pathlib import Path
from common import read_manifest,digest,protect_output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',required=True,type=Path)
    p.add_argument('--selection',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    source=read_manifest(a.source_manifest,verify=False)
    selection=json.loads(a.selection.read_text())
    if source['fingerprint']!=selection['source_manifest_fingerprint']:
        raise ValueError('Validation source manifest changed')
    by_id={pair['image_id']:pair for pair in source['pairs']}
    seen=set();manifests={}
    for name,cohort in selection['cohorts'].items():
        if name not in ('challenge','pilot'):raise ValueError('Unsupported validation cohort name')
        ids=cohort['image_ids']
        if not ids or len(ids)!=len(set(ids)) or seen.intersection(ids) or not set(ids)<=set(by_id):
            raise ValueError('Invalid/overlapping challenge and pilot selection')
        seen.update(ids)
        manifest=dict(schema=1,split=cohort['split'],pairs=[by_id[i] for i in ids],
            selection=dict(policy=cohort['policy'],selection_sha256=digest(selection),source_manifest_fingerprint=source['fingerprint']))
        manifest['fingerprint']=digest(manifest);manifests[name]=manifest
    protect_output(a.output_dir,source['pairs'])
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,manifest in manifests.items():
        (a.output_dir/(name+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({name:dict(n=len(m['pairs']),fingerprint=m['fingerprint']) for name,m in manifests.items()}))

if __name__=='__main__':main()
