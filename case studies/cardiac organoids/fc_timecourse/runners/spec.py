"""The Dataset declaration the shared end-to-end harness uses for the FC time course: phase contrast at every
stage (the original study), or the `ssl_two_channel` variant, where segmentation and crop decisions still read the
phase-contrast image alone and the SSL stage receives two-channel (phase contrast, GFP) crops. The cross-validation
protocol (`co-fc-timecourse-ssl19-cvcol4-v1`) has no held-out wells in the SSL stage and its own agent wording."""
from benchmarks.full_CPJUMP1.harness.pipeline import Dataset
from . import data

MODULE = 'case studies.cardiac organoids.fc_timecourse.runners'
CO_FC = Dataset(
    name='co_fc_timecourse', cli=MODULE + '.e2e', data=MODULE + '.data', score=MODULE + '.score',
    n_images=184, marker='CO_FC_TIMECOURSE_COMPLETE', image_shape=(904, 1224), channels=('PhaseContrast',),
    dtype='uint16', plate_type='cardiac organoid time course', mask_label=True, allowed_sites=(8,),
    allow_empty_masks=True, segment_tool='organoid_segmenter', usages=('development_train', 'development_val', 'evaluation_only'),
    tasks={
        'segment': 'Segment every whole cardiac organoid in the phase-contrast image, one instance per organoid '
                   'over its visible extent. Constituent cells, internal texture, halos and debris are not organoids.',
        'crop': 'Choose a complete square crop box per organoid from the phase-contrast image and its predicted '
                'mask; the box must contain the whole organoid. No other channel is available.',
        'embed': 'Fit SSL on phase-contrast organoid crops from training wells only; validation wells select, '
                 'held-out wells are inference-only. No condition, time or fluorescence label is visible.',
        'batch': 'Not run: one plate, no technical batch key.'},
    inputs={
        'segment': 'Inputs: {n} {dtype} phase-contrast images (904 x 1224) of cardiac organoids, one well per image, '
                   'usually one organoid per well. A valid all-zero mask records no detection, not verified absence; '
                   'do not fabricate an organoid to satisfy a count.',
        'crop': 'Inputs: {n} {dtype} phase-contrast images and this arm\'s organoid masks. Return crop geometry from '
                'the phase-contrast image alone. An all-zero input mask requires an explicit zero-object crop record.',
        'embed': 'Inputs: phase-contrast organoid crops, at most {cells} development crops per image. Train/val wells '
                 'only during SSL selection.',
        'batch': 'Not run.'},
    input_channels={'segment': 'PhaseContrast', 'crop': 'PhaseContrast'}, rules={},
    withheld='Experimental condition, culture day, well position, GFP pixels and GFP-derived values, source file '
             'names and the split labels of held-out wells are withheld from agent and tool decisions.')

# The two-channel SSL variant (user request 2026-10-25): the same segment and crop stages; the crop boxes chosen on
# the phase-contrast image cut a two-channel (PhaseContrast, GFP) stack at identical coordinates, which is the SSL input.
TWO_CHANNEL_SSL = dict(
    channels=('PhaseContrast', 'GFP'),
    tasks={**CO_FC.tasks,
           'crop': 'Choose a complete square crop box per organoid from the phase-contrast image and its predicted '
                   'mask; the box must contain the whole organoid. The box is applied to a two-channel (phase contrast, '
                   'GFP) stack at identical coordinates, but only the phase-contrast image is available here.',
           'embed': 'Fit SSL on two-channel (phase contrast, GFP) organoid crops from training wells only; validation '
                    'wells select, held-out wells are inference-only. The GFP channel is a reporter fluorescence image '
                    'of the same organoid; no condition, time or fluorescence-derived label is visible.'},
    inputs={**CO_FC.inputs,
            'embed': 'Inputs: two-channel (PhaseContrast, GFP) {dtype} organoid crops, at most {cells} development crops '
                     'per image. Train/val wells only during SSL selection.'},
    withheld='Experimental condition, culture day, well position, GFP-derived values, source file names and the split '
             'labels of held-out wells are withheld from agent and tool decisions. The GFP pixels enter only the SSL '
             'crops of the embed stage; the segment and crop stages see the phase-contrast image alone.')


# the cross-validation protocol: the SSL stage has no held-out wells, the folds belong to the read-out
CV_EMBED = ('Fit SSL on phase-contrast organoid crops from the training wells; the validation wells select the SSL '
            'settings. Every crop is embedded after the fit. No condition, time or fluorescence label is visible.')
CV_WITHHELD = ('Experimental condition, culture day, well position, GFP pixels and GFP-derived values, source file names '
               'and the cross-validation folds of the read-out are withheld from agent and tool decisions.')


def for_config(config):
    """The run's Dataset: a technical smoke uses incomplete wells, a study run needs all eight time points; the
    `variant` of CONFIG.json (from the prepared data) selects the SSL channel set and its `protocol` the agent wording."""
    from dataclasses import replace
    from benchmarks.full_CPJUMP1.harness.pipeline import CPJUMP1
    sites = tuple(range(1, 9)) if config.get('smoke') else CO_FC.allowed_sites
    spec = replace(CO_FC, allowed_sites=sites, rules=CPJUMP1.rules)
    if config.get('variant', 'phase_only') == 'ssl_two_channel':
        spec = replace(spec, **TWO_CHANNEL_SSL)
    if config.get('protocol') == data.CV_PROTOCOL:
        spec = replace(spec, tasks={**spec.tasks, 'embed': CV_EMBED}, withheld=CV_WITHHELD)   # `usages` stays the 3-tuple: the cross-validation rows simply have no held-out usage
    return spec
