"""Best-observed crop histories for diagnostics and explicit delivery policies.

Scores are measured crop proxies, not biological truth. Recombining a fixed
historical search is a diagnostic, not a replay of an adaptive agent run.
"""
from copy import deepcopy
import math

EPS = 1e-12
POLICIES = {'quality_only': None, 'retention_preserved': 0.0,
            'retention_loss_5pp': .05}
NATIVE_METRICS = ('target_retention', 'neighbor_contamination', 'centroid_offset')


def observation(candidate, number, row, pixels, parameters, score='crop_quality_score'):
    box = row.get('crop_bbox', [])
    q = row.get('qc', {})
    if (len(box) != 4 or any(type(v) is not int for v in box) or
            box[0] < 0 or box[1] < 0 or box[2] <= box[0] or
            box[2] - box[0] != box[3] - box[1] or row.get('centroid_inside') is False):
        raise ValueError('Invalid native crop geometry')
    if (any(type(q.get(k)) not in (int, float) or not math.isfinite(q[k]) for k in NATIVE_METRICS) or
            not .90 <= q['target_retention'] <= 1 or not 0 <= q['neighbor_contamination'] <= 1 or
            q['centroid_offset'] < 0):
        raise ValueError('Invalid native crop measurements')
    if (score not in pixels or not pixels or
            any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in pixels.values())):
        raise ValueError('Invalid pixel measurements')
    return dict(candidate=candidate, observed_round=number, row=deepcopy(row),
                pixels=deepcopy(pixels), parameters=deepcopy(parameters))


class CropTrajectory:
    """Keep every measured option, including rejected whole-policy candidates.

    Each policy selects its highest *strictly* improved `score` (a key of each
    observation's measurements, by default the pixel crop_quality_score). Exact ties
    retain the earliest observation, avoiding gratuitous geometry changes. The
    initial observation always participates; routing-warning thresholds never
    filter the option pool. Retention budgets refer to the original target.
    """
    def __init__(self, rows, pixels, parameters, score='crop_quality_score'):
        if not rows or not (rows.keys() == pixels.keys() == parameters.keys()):
            raise ValueError('Initial evidence must be nonempty and paired')
        self.score = score
        self.history = {k: [observation('initial', 0, r, pixels[k], parameters[k], score)]
                        for k, r in rows.items()}
        self.best = {p: {k: values[0] for k, values in self.history.items()} for p in POLICIES}
        self.names = set();self.last_round = 0

    def add_candidate(self, candidate, number, rows, pixels, parameters):
        if (candidate in self.names or candidate == 'initial' or type(number) is not int or
                number < 1 or number < self.last_round):
            raise ValueError('Candidate identity/round is duplicated or unordered')
        if (not rows or not rows.keys() <= self.history.keys() or
                not (rows.keys() == pixels.keys() == parameters.keys())):
            raise ValueError('Candidate evidence must match known crop identities')
        # Fail before publishing ANY option if an observation is corrupt.
        options = {k: observation(candidate, number, r, pixels[k], parameters[k], self.score)
                   for k, r in rows.items()}
        updates = {p: 0 for p in POLICIES}
        for k, option in options.items():
            self.history[k].append(option)
            original = self.history[k][0]['row']['qc']['target_retention']
            for policy, budget in POLICIES.items():
                retention = option['row']['qc']['target_retention']
                if budget is not None and retention < original - budget - EPS:
                    continue
                if option['pixels'][self.score] > self.best[policy][k]['pixels'][self.score] + EPS:
                    self.best[policy][k] = option;updates[policy] += 1
        self.names.add(candidate);self.last_round = number
        return updates
