"""Bounded I/O concurrency for exact crop integrity checks; no digest caching."""
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
import time

from .data import digest


def verify_crop_inputs(cells, workers=1, progress=None):
    """Hash every input, preserving serial error order and bounded queued work.

    Every invocation reads every byte again. Shared files and repeated cell paths
    are deliberately not deduplicated, so input records retain their own checks.
    """
    if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
        raise ValueError('positive integer integrity workers required')
    started=time.monotonic()
    def check(cell):
        if digest(Path(cell['crop_path']))!=cell['sha256']:
            raise ValueError(f"raw crop changed before embedding: {cell['cell_id']}")
    verified=0
    if workers==1:
        for cell in cells:
            check(cell);verified+=1
            if progress is not None:progress(verified)
    else:
        iterator=iter(cells)
        # Python 3.12 Executor.map eagerly submits its iterable. Batch explicitly
        # so hundreds of thousands of crops cannot allocate that many futures.
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='crop-integrity') as pool:
            while batch:=list(islice(iterator,workers*8)):
                for _ in pool.map(check,batch):verified+=1
                if progress is not None:progress(verified)
    return dict(verified_files=verified,workers=workers,seconds=time.monotonic()-started,
                algorithm='sha256',cached_digests=False,queued_files_at_most=workers*8 if workers>1 else 1)
