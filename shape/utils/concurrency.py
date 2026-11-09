"""Ordered results with prompt error propagation and no new work after failure."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def ordered_map(fn, items, workers=1, on_result=None):
    items = list(items)
    if workers <= 1:
        results=[]
        for item in items:
            value=fn(item)
            if on_result:on_result(value)
            results.append(value)
        return results
    cancelled = threading.Event()
    skipped = object()
    def call(item):
        if cancelled.is_set():
            return skipped
        try:
            return fn(item)
        except BaseException:
            cancelled.set()
            raise
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(call, item): i for i, item in enumerate(items)}
    results = [None] * len(items)
    try:
        for future in as_completed(futures):
            value=future.result()
            if value is skipped:
                continue
            results[futures[future]] = value
            if on_result:on_result(value)
    except BaseException:
        cancelled.set()
        for future in futures:
            future.cancel()
        # Python cannot interrupt a running thread. Callers must use bounded I/O;
        # do not hide the exception while other calls finish their own timeouts.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results
