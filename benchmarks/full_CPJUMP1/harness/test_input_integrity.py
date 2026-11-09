from pathlib import Path
import threading
import pytest
from . import input_integrity as integrity


def records(tmp_path,n=25):
    result=[]
    for i in range(n):
        p=tmp_path/f'{i}.tif';p.write_bytes(bytes([i%256])*1024)
        result.append(dict(cell_id=str(i),crop_path=str(p),sha256=integrity.digest(p)))
    return result


@pytest.mark.parametrize('workers',[1,4])
def test_checks_all_bytes_on_every_invocation_and_rejects_corruption(tmp_path,workers):
    cells=records(tmp_path)
    report=integrity.verify_crop_inputs(cells,workers)
    assert report['verified_files']==25 and not report['cached_digests']
    Path(cells[17]['crop_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw crop changed before embedding: 17'):
        integrity.verify_crop_inputs(cells,workers)


def test_first_bad_record_is_deterministic_with_multiple_threads(tmp_path):
    cells=records(tmp_path)
    for i in (2,5):Path(cells[i]['crop_path']).write_bytes(b'changed')
    for _ in range(3):
        with pytest.raises(ValueError,match='embedding: 2$'):integrity.verify_crop_inputs(cells,4)


def test_queue_and_progress_are_bounded_and_each_record_is_checked(tmp_path,monkeypatch):
    cells=records(tmp_path,83);seen=[];lock=threading.Lock();original=integrity.digest
    def counted(path):
        with lock:seen.append(path)
        return original(path)
    monkeypatch.setattr(integrity,'digest',counted)
    progress=[]
    result=integrity.verify_crop_inputs(iter(cells),3,progress.append)
    assert result['queued_files_at_most']==24 and len(seen)==83
    assert progress==[24,48,72,83]


@pytest.mark.parametrize('workers',[0,-1,True,1.5])
def test_rejects_invalid_worker_count(workers):
    with pytest.raises(ValueError):integrity.verify_crop_inputs([],workers)
