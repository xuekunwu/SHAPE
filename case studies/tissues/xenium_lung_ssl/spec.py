"""The Dataset declaration the shared end-to-end harness uses for the Xenium lung cancer crops (embed stage only)."""
from benchmarks.full_CPJUMP1.harness.pipeline import Dataset

from .data import SPEC

MODULE = 'case studies.tissues.xenium_lung_ssl'
CHANNELS = tuple(SPEC['channels'])
XENIUM_LUNG = Dataset(
    name='xenium_lung_ssl', cli=MODULE + '.e2e', data=MODULE + '.data', score=MODULE + '.e2e',
    n_images=round(SPEC['sample']['fraction'] * SPEC['expected_cells']), marker='XENIUM_LUNG_SSL_COMPLETE',
    image_shape=(0, 0), channels=CHANNELS, dtype=SPEC['dtype'], plate_type='Xenium tissue section', mask_label=False,
    allowed_sites=range(1, SPEC['expected_cells'] + 1),   # a well holds any number of crops (its sites)
    tasks={'embed': 'Fit SSL on four-channel single-cell crops of one Xenium human lung cancer FFPE section from the '
                    'training tiles; the validation tiles select the SSL settings. Every crop, held-out tiles '
                    'included, is embedded after the fit. Each crop is one site; a well is the crops of one 250 um '
                    'neighborhood of the section. No cell type, gene expression or position is visible.'},
    inputs={'embed': 'Inputs: {n} single-cell crops, each a four-channel {dtype} TIFF (DAPI, ATP1A1/CD45/E-Cadherin, '
                     '18S, alphaSMA/Vimentin) cut around one Xenium-segmented cell, 8 to 187 px square; each crop is '
                     'one site, a well is the crops of one 250 um neighborhood of the section. Calls fit '
                     'and validate on development crops only: development_train crops train, development_val crops '
                     '(spatial tiles disjoint from training) validate; held-out crops never enter a call. input is '
                     'the pretrained backbone (zero_shot) on those crops, not a call. After finish, a fresh worker '
                     'embeds every crop, held-out included, with the delivered call\'s frozen checkpoint (the '
                     'pretrained backbone for input).'},
    input_channels={}, rules={},
    withheld='Cell types, gene expression, cell positions and the tile layout are withheld from agent and tool '
             'decisions.')


def for_config(config):
    return XENIUM_LUNG
