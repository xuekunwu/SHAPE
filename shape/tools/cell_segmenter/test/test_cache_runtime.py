import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shape.tools.cell_segmenter.cache_runtime import configure, native_rng


@pytest.mark.parametrize('raises', [False, True])
def test_native_rng_is_repeatable_and_restores_all_cpu_streams(raises):
    random.seed(17); np.random.seed(18); torch.manual_seed(19)
    states = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    values = []
    for _ in range(2):
        try:
            with native_rng(20261015, 'cpu'):
                values.append((random.random(), float(np.random.random()), float(torch.rand(()))))
                if raises:
                    raise RuntimeError('interrupted native attempt')
        except RuntimeError:
            assert raises
    assert values[0] == values[1]
    assert random.getstate() == states[0]
    after = np.random.get_state()
    assert after[0] == states[1][0] and np.array_equal(after[1], states[1][1]) and after[2:] == states[1][2:]
    assert torch.equal(torch.random.get_rng_state(), states[2])


def test_configuration_hashes_actual_loaded_weights_and_runtime_without_loading_a_model(tmp_path):
    checkpoint=tmp_path/'loaded.weights'
    checkpoint.write_bytes(b'already loaded checkpoint')
    tool=SimpleNamespace(model=SimpleNamespace(pretrained_model=str(checkpoint), net=torch.nn.Linear(2, 2)),
                         device=torch.device('cpu'))
    tool.set_trial_cache=lambda cache:setattr(tool, '_native_trial_cache', cache)
    identity=configure(tool,tmp_path/'cache',seed=20261015)
    assert identity['runtime']['seed']==20261015 and identity['runtime']['device']=='cpu'
    assert identity['runtime']['parameter_dtypes']==['torch.float32']
    assert identity['runtime']['packages']['torch']
    assert tool._trial_cache_seed==20261015
    assert len(identity['source_sha256'])==64
    checkpoint.write_bytes(b'different already loaded checkpoint')
    assert configure(tool,tmp_path/'other-cache',seed=20261015)['weights_sha256']!=identity['weights_sha256']


def test_unattested_server_or_unresolved_alias_cannot_enable_trial_reuse(tmp_path):
    tool=SimpleNamespace(model=None,device=torch.device('cpu'),_sam_server_url='http://localhost')
    with pytest.raises(ValueError,match='local model'):
        configure(tool,tmp_path/'cache',seed=1)


def test_actual_execute_wrapper_reuses_files_across_vrr_passes_and_retains_origin_cost(tmp_path, monkeypatch):
    from functools import wraps
    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
    from shape.tools.cell_segmenter.trial_cache import SegmentationTrialCache
    original=Cell_Segmenter_Tool._execute_native
    calls=[]
    @wraps(original)
    def native(self,**kwargs):
        calls.append(kwargs)
        folder=__import__('pathlib').Path(kwargs['query_cache_dir'])
        np.save(folder/'mask.npy',np.ones((4,4),dtype='uint32'))
        (folder/'overlay.png').write_bytes(b'complete native overlay')
        return dict(status='success',cell_count=1,mask_path=str(folder/'mask.npy'),
                    overlay_path=str(folder/'overlay.png'),seconds=12.5)
    monkeypatch.setattr(Cell_Segmenter_Tool,'_execute_native',native)
    tool=Cell_Segmenter_Tool.__new__(Cell_Segmenter_Tool)
    tool.device=torch.device('cpu')
    identity=dict(source_sha256='a'*64,weights_sha256='b'*64,runtime=dict(device='cpu',seed=1))
    tool.set_trial_cache(SegmentationTrialCache(tmp_path/'cache',identity))
    image=tmp_path/'raw.tif';image.write_bytes(b'unchanged pixels')
    first=tool.execute(str(image),image_id='i1',query_cache_dir=str(tmp_path/'cohort'))
    second=tool.execute(str(image),image_id='i1',query_cache_dir=str(tmp_path/'full-vrr'))
    assert len(calls)==1 and first['mask_path']==second['mask_path']
    assert second['native_cache']['hit']
    assert second['native_cache']['original_native_seconds']==first['native_cache']['original_native_seconds']
    assert second['native_cache']['original_reported_seconds']==12.5
    tool.execute(str(image),image_id='i1',bsize=512)
    assert len(calls)==2 and calls[-1]['bsize']==512
