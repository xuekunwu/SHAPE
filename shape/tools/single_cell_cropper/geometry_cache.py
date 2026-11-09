"""Bounded, run-local reuse of unchanged objects' pure native geometry.

The full label mask hash binds neighboring geometry. Only the latest setting per
object is retained, so changing other objects does not evict frozen objects.
Neither source image pixels nor model judgements are cached here.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import threading

import numpy as np


def configure(tool, max_objects=200000):
    if max_objects<1:raise ValueError('positive geometry-cache capacity required')
    if hasattr(tool,'geometry_cache_stats'):return tool.geometry_cache_stats
    original=tool.cell_crop_box;entries=OrderedDict();lock=threading.RLock();local=threading.local()
    stats=dict(hits=0,misses=0,max_objects=max_objects)
    def cached(region,labeled_mask,cell_mask,image_shape,cell_override,**kwargs):
        with lock:
            arrays=getattr(local,'masks',None)
            known=arrays.get(id(labeled_mask)) if arrays is not None else None
            if known is None or known[0] is not labeled_mask:
                contiguous=np.ascontiguousarray(labeled_mask)
                digest=hashlib.sha256(memoryview(contiguous).cast('B')).hexdigest()
                if arrays is not None:arrays[id(labeled_mask)]=(labeled_mask,digest)
            else:digest=known[1]
            key=(digest,str(labeled_mask.dtype),tuple(labeled_mask.shape),int(region.label))
            setting=json.dumps(dict(shape=list(image_shape),override=cell_override,
                kwargs={k:v for k,v in kwargs.items() if k!='full_field_areas'}),sort_keys=True,default=str)
            previous=entries.get(key)
            if previous is not None and previous[0]==setting:
                entries.move_to_end(key);stats['hits']+=1;return deepcopy(previous[1])
        result=original(region,labeled_mask,cell_mask,image_shape,cell_override,**kwargs)
        with lock:
            entries[key]=(setting,deepcopy(result));entries.move_to_end(key);stats['misses']+=1
            while len(entries)>max_objects:entries.popitem(last=False)
        return result
    # The native generator treats its label mask as immutable during one call.
    # Hash once in that call, then discard array identities. A later call must
    # rehash even if the caller mutated and reused the same ndarray object.
    if hasattr(tool,'_generate_cell_crops'):
        generate=tool._generate_cell_crops
        def generation(*args,**kwargs):
            local.masks={}
            try:return generate(*args,**kwargs)
            finally:local.masks=None
        tool._generate_cell_crops=generation
    tool.cell_crop_box=cached;tool.geometry_cache_stats=stats
    return stats
