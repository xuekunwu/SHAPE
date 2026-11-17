"""Development-only routing of two independently fitted percentage predictors."""

from collections import Counter, defaultdict
import math


def eligible_plates(images, minimum_validation_wells=2):
    wells = {}
    for row in images:
        value = (row['plate_id'], row['usage'])
        if row['well'] in wells and wells[row['well']] != value:
            raise ValueError('inconsistent well split or plate')
        wells[row['well']] = value
    counts = defaultdict(Counter)
    for plate, usage in wells.values():
        counts[plate][usage] += 1
    return {plate: dict(counts=dict(values), eligible=values['development_val'] >= minimum_validation_wells
                        and values['development_train'] > 0)
            for plate, values in sorted(counts.items())}


def choose_routes(zero, finetuned, development_labels, eligibility):
    """Neither held-out targets nor held-out prediction errors enter routing."""
    zero_by = {r['well']:r for r in zero}
    if len(zero_by) != len(zero):
        raise ValueError('duplicate wells')
    wanted = {r['well'] for r in zero if r['usage'] != 'evaluation_only'}
    if set(development_labels) != wanted:
        raise ValueError('routing accepts development labels only, with exact well coverage')
    fine_by = {r['well']:r for r in finetuned} if finetuned is not None else None
    if fine_by is not None:
        if len(fine_by) != len(finetuned) or set(fine_by) != set(zero_by):
            raise ValueError('branch well sets differ')
        for well, original in zero_by.items():
            if any(original[key] != fine_by[well][key] for key in ('usage','plate_id','image_ids')):
                raise ValueError('branch metadata differs')
    routes = {}
    for plate, info in eligibility.items():
        val = [r for r in zero if r['plate_id'] == plate and r['usage'] == 'development_val']
        zerr = [abs(r['predicted_ctnt_positive_percent']-development_labels[r['well']]) for r in val]
        ferr = [abs(fine_by[r['well']]['predicted_ctnt_positive_percent']-development_labels[r['well']])
                for r in val] if fine_by is not None else []
        zmae = sum(zerr)/len(zerr) if zerr else None
        fmae = sum(ferr)/len(ferr) if ferr else None
        if any(not math.isfinite(value) for value in [*zerr,*ferr]):
            raise ValueError('nonfinite development errors')
        use = bool(info['eligible'] and fine_by is not None and len(val) >= 2 and fmae < zmae)
        reason = ('validation_mae_improved' if use else 'insufficient_validation_wells' if not info['eligible']
                  else 'ssl_candidate_not_adopted' if fine_by is None else 'no_validation_mae_improvement')
        routes[plate] = dict(branch='finetuned' if use else 'zero_shot',reason=reason,
                             validation_wells=len(val),zero_shot_validation_mae=zmae,
                             finetuned_validation_mae=fmae,minimum_improvement_percentage_points=0.)
    return routes


def merge_predictions(zero, finetuned, routes):
    fine_by = {r['well']:r for r in finetuned} if finetuned is not None else {}
    merged = []
    for row in zero:
        branch = routes[row['plate_id']]['branch']
        chosen = fine_by[row['well']] if branch == 'finetuned' else row
        merged.append({**chosen,'prediction_branch':branch})
    return merged
