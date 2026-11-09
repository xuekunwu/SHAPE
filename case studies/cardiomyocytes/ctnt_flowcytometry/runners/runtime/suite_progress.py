"""Read-only suite progress from configuration and proofs, independent of GPU type."""
import argparse
from pathlib import Path
import re
from datetime import datetime, timezone
from .artifacts import read_json

STAGES=('preflight','segment','crop','ssl_search','embed','batch','predict','readouts','score')


def optional(path):
    try:return read_json(path)
    except (OSError,ValueError):return {}


def progress_bar(count,total,width=12):
    if not total:return '[.....>......] --%'
    size=min(width,max(0,int(width*count/total)))
    return '['+'#'*size+'.'*(width-size)+f'] {count}/{total}'


def arm_status(folder,stage):
    meta=optional(folder/'RUN_META.json');config=optional(folder/'CONFIG.json')
    proof=folder/({'preflight':'PREFLIGHT.json','readouts':'STAGE_READOUTS.json','score':'RESULT.json'}.get(stage,stage+'_COMPLETE.json'))
    if proof.exists():return '[############] DONE'
    if meta.get('current_stage')!=stage:return '[............] WAIT'
    if meta.get('status')=='failed':return '[!!!!!!!!!!!!] FAILED'
    total=len(config.get('images',[]));wait=optional(folder/'GPU_WAIT.json')
    if wait.get('stage')==stage and wait.get('status')=='waiting':return '[............] GPU QUEUE'
    if stage=='segment':
        output=optional(folder/'segmentation.json');count=len(output) if isinstance(output,list) else 0
        if not count:count=len(list((folder/'segment').glob('*/defaults.json')))
        if not count:count=len({p.parent.parent.name for p in (folder/'agent/segment/waves').glob('*/i*/c*/result.json')})
        return progress_bar(count,total)+(' masks; VERIFY' if total and count==total else ' RUN')
    if stage=='crop':return progress_bar(len(list((folder/'crop').glob('*/result.json'))),total)+' RUN'
    if stage=='readouts':
        row=optional(folder/'READOUTS_PROGRESS.json')
        return progress_bar(row.get('completed',0),row.get('total'))+' '+row.get('phase','RUN')
    if stage=='embed':
        path=folder/'logs/embed.log'
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-32768));tail=stream.read().decode(errors='replace')
            found=re.findall(r'EMBED (\d+)/(\d+)',tail)
            if found:return progress_bar(*map(int,found[-1]))+' infer'
    return '[.....>......] RUNNING'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('suite',type=Path)
    args=parser.parse_args(argv);suite=args.suite.resolve()
    info=optional(suite/'SUITE_META.json') or optional(suite/'SUITE.json')
    arms=list(info.get('arms') or sorted(p.name for p in suite.iterdir() if p.is_dir() and (p/'RUN_META.json').exists()))
    if not arms:parser.error('no configured arms or arm run metadata found')
    print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),info.get('status','UNKNOWN'))
    print('stage'.ljust(12)+' | '+' | '.join(arm.upper().ljust(36) for arm in arms));print('-'*(15+39*len(arms)))
    for stage in STAGES:
        if stage=='ssl_search' and not any((suite/a/'ELIGIBILITY.json').exists() for a in arms):continue
        print(stage.ljust(12)+' | '+' | '.join(arm_status(suite/a,stage).ljust(36) for a in arms))
    for arm in arms:
        folder=suite/arm;individual=list((folder/'segment').glob('*/SEARCH.json'));cohort=list((folder/'segment').glob('*/COHORT_SEARCH.json'))
        if individual or cohort:print(arm,'segmentation verification records:',len({p.parent.name for p in individual+cohort}),'| individual',len(individual),'| cohort',len(cohort))
        meta=optional(folder/'RUN_META.json')
        if meta.get('error'):print(arm,'ERROR',meta['error'])


if __name__=='__main__':main()
