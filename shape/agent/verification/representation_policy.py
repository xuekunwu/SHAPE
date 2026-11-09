"""Opt-in SSL search policy; legacy experiments retain their original behavior."""
import math

POLICY = 'evidence_cost_v1'
REQUIRED = ('view_pairing', 'contrastive_mask', 'batch_sampling', 'intensity_norm',
            'augmentation', 'crop_scale_min', 'trainable_pct', 'learning_rate')


def validate_policy(value):
    if value not in ('legacy', POLICY):
        raise ValueError('unknown SSL search policy')
    return value


def expert_start():
    """Two search opportunities, not unconditional training jobs; VRR may stop."""
    return [dict(view_pairing='same_well', contrastive_mask='same_well',
                 batch_sampling='per_plate', intensity_norm='per_plate', augmentation='light',
                 crop_scale_min=.6, trainable_pct=50, learning_rate=3e-5),
            dict(init_from='parent', trainable_pct=100, learning_rate=1e-5)]


def continuation_evidence(paired):
    """Use the same well-paired 2-SE evidence as delivery, without weakening it."""
    delta, se = paired.get('delta'), paired.get('se')
    measurable = (isinstance(delta, (int, float)) and isinstance(se, (int, float))
                  and math.isfinite(delta) and math.isfinite(se) and se >= 0
                  and paired.get('n_wells', 0) >= 2)
    convincing = bool(measurable and delta > 2 * se)
    return dict(policy=POLICY, delta=delta, se=se, n_wells=paired.get('n_wells', 0),
                threshold=2 * se if measurable else None, continue_required=convincing,
                reason='beyond_2se' if convincing else 'within_2se' if measurable else 'not_measurable')


def execution_audit(params, result=None):
    """Fail before training on incomplete inherited recipes; record actual tool facts."""
    frozen = params.get('adaptation_depth') == 'zero_shot' or params.get('embed_only') is True
    if not frozen:
        missing = [key for key in REQUIRED if key not in params or params[key] is None]
        if missing:
            raise ValueError('SSL training requires complete effective parameters: ' + ', '.join(missing))
    record = dict(policy=POLICY, requested_effective_kwargs=dict(params),
                  parent_checkpoint=params.get('init_checkpoint'), execution='frozen' if frozen else 'training',
                  explicit_training_recipe=not frozen)
    if result is not None:
        training = result.get('training') or {}
        record['observed'] = dict(trainable_blocks=training.get('trainable_blocks'),
            learning_rate=(result.get('analyzer_kwargs') or {}).get('learning_rate'),
            intensity_norm=result.get('intensity_norm_used'), best_epoch=training.get('best_epoch'),
            epochs_trained=result.get('epochs_trained'), train_seconds=training.get('train_seconds'),
            native_seconds=result.get('seconds'))
    return record
