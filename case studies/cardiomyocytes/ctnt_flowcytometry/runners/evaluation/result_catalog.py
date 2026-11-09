"""Locate published results and replay frozen exports without image inference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT.parent


def result_sets():
    return json.loads((ROOT/'docs/RESULTS.json').read_text(encoding='utf-8'))['results']


def replay(result_id,out):
    records={r['id']:r for r in result_sets()}
    if result_id not in records:raise ValueError('unknown result ID; use results')
    entry=records[result_id]
    if not entry.get('replay_script'):raise ValueError('this is an audit record, not a replayable model result')
    source=(CASE/entry['path']).resolve();out=Path(out).resolve()
    if not source.is_relative_to((CASE/'results').resolve()):raise ValueError('result path outside results')
    if out.exists() or out.is_relative_to((CASE/'results').resolve()):raise ValueError('use a fresh output outside published results')
    script=source/entry['replay_script']
    if not script.is_file() or not script.resolve().is_relative_to(source):raise ValueError('missing or invalid embedded replay script')
    # Historical evidence is copied before a replay script is allowed to regenerate reports.
    shutil.copytree(source,out)
    subprocess.run([sys.executable,str(out/entry['replay_script'])],cwd=out,check=True)
    print('OFFLINE_REPLAY_COMPLETE',out)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('results',help='map published results to their verified replay and runtime requirements')
    p.add_argument('id',nargs='?');p.add_argument('--json',action='store_true')
    p=sub.add_parser('replay',help='copy an existing public export and run its own offline verifier/reporter; no inference')
    p.add_argument('id');p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='results':
        records=[r for r in result_sets() if not args.id or r['id']==args.id]
        if not records:parser.error('unknown result ID')
        if args.json or args.id:print(json.dumps(records,indent=2,ensure_ascii=False))
        else:
            for r in records:print(f"{r['id']:22} {r['path']}\n  {r['summary']}")
    else:replay(args.id,args.out)


if __name__=='__main__':main()
