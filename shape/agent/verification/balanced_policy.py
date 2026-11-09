"""Opt-in VRR policy for development experiments; legacy remains the default."""
import json
import math
import os

POLICY = 'balanced_v1'


def selected_policy():
    value=os.environ.get('SHAPE_VRR_POLICY','legacy')
    if value not in ('legacy',POLICY):raise ValueError('unknown SHAPE_VRR_POLICY')
    return value


def balanced_override(call, baseline, metrics, baseline_metrics, assessments, comparison):
    """A visual pairwise decision, not an object-count or foreground maximizer.

    The caller enforces native artifact guards. Proxy loss is a warning requiring
    a visible explanation; proxy improvement alone cannot authorize delivery.
    """
    if not comparison or comparison.get('preferred_call')!=call or comparison.get('baseline_call')!=baseline:
        return False,'explicit pairwise comparison against the original baseline is required'
    if comparison.get('confidence')!='high' or not str(comparison.get('reason','')).strip():
        return False,'high-confidence visual reasoning is required'
    if len(comparison.get('cited_readouts') or [])<2:
        return False,'cite at least two observations; foreground alone is not ground truth'
    ranks={'none':0,'minor':1,'major':2}
    a=assessments.get(call,{});b=assessments.get(baseline,{})
    if a.get('severity') not in ranks or b.get('severity') not in ranks:
        return False,'both candidates require recorded visual assessments'
    if ranks[a['severity']]>ranks[b['severity']]:
        return False,'candidate has worse assessed whole-cell defects than baseline'
    losses=[]
    for key in ('cell_count','foreground_coverage'):
        before=baseline_metrics.get(key);after=metrics.get(key)
        if before is not None and after is not None and float(before)>0:
            if not math.isfinite(float(after)):return False,'nonfinite candidate readout'
            if float(after)<.75*float(before):losses.append(key)
    if losses and (not comparison.get('loss_explained') or not str(comparison.get('loss_explanation','')).strip()):
        return False,'large loss of '+', '.join(losses)+' needs an explicit visual explanation'
    return True,'balanced pairwise visual preference; native guards preserved; proxy losses explained' if losses else 'balanced pairwise visual preference; native guards preserved'


def repair_packet(verifier, plan, accepted, rejected, room):
    """Repair only the rejected structure; do not resend images or all trial history."""
    payload=dict(task=verifier.task_text(),invalid_plan=plan.model_dump(),
                 valid_levers=list(verifier.levers),rejected=rejected,
                 accepted=[dict(parent=r['parent'],change=r['param_delta']) for r in accepted],
                 replacement_slots=room,pending_focus=list(verifier._pending_focus),
                 narrowing=verifier.narrowing,search_strategy=verifier.search_strategy)
    raw=json.dumps(payload,default=str)
    # Do not silently truncate a requested per-cell action: use the full packet.
    return ['Structural repair only. Preserve the scientific hypothesis and the valid search phase. '
            'Correct only rejected moves using the supplied errors and schema. Do not repeat accepted moves. '
            'Do not invent unseen image evidence; return no replacement if the errors cannot be resolved. '
            'All original parameter, duplicate, focus and search-policy validators still apply.\n'+raw]
