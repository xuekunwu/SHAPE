"""One conditional VRR session over a cohort; native passing images are frozen.

No dataset labels or channel assumptions enter this controller. The caller owns
native execution/measurement and evidence rendering. At most eight model replies
are consumed, including malformed replies. Every image retains its own history.
"""
from copy import deepcopy
import hashlib
import json
import math

from shape.utils.concurrency import ordered_map

POLICY = 'conditional_cohort_v2'


def identity(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, allow_nan=False).encode()).hexdigest()


def quality(observation):
    return (-len(observation['problems']), observation['score'])


def valid(observation):
    return (observation.get('guard_ok') is True
            and isinstance(observation.get('score'), (int, float))
            and math.isfinite(observation['score']))


def _large_loss(candidate, reference):
    return any(reference['metrics'].get(k, 0) > 0 and
               candidate['metrics'].get(k, 0) < .75 * reference['metrics'][k]
               for k in ('cell_count', 'foreground_coverage'))


def _matches(rule, item, current):
    if rule.get('items') and item['image_id'] not in rule['items']:
        return False
    issues = rule.get('issues') or []
    if issues and not set(issues) & {p.split()[0] for p in current['problems']}:
        return False
    for condition in rule.get('when') or []:
        value = current['metrics'].get(condition['metric'])
        if value is None or not condition.get('min', -math.inf) <= value <= condition.get('max', math.inf):
            return False
    return True


def run(items, defaults, measure, propose, coerce, *, workers=1, max_rounds=8,
        checkpoint=None, resume=None, fatal=lambda exc: False):
    """measure(item, effective_params) returns result/metrics/score/problems/guard_ok.

    propose(packet) returns reviews plus conditional rules. Reviews bind both the
    candidate and incumbent IDs; a stale or ungrounded review cannot change state.
    Native improvements can be retained below pass thresholds. Proxy tradeoffs
    require a high-confidence pairwise visual review in a subsequent shared round.
    """
    if not 1 <= max_rounds <= 8 or not items:
        raise ValueError('nonempty cohort and global budget in [1,8] required')
    by_id = {i['image_id']: i for i in items}
    if len(by_id) != len(items):
        raise ValueError('duplicate image IDs')
    history = {i: {} for i in by_id}; best = {}; frozen = set(); flags = set(); errors = []
    rounds = []; rule_log = []; stop = 'budget_exhausted'
    baseline_complete=False

    def execute(job):
        ident, params, number = job
        try:
            result = deepcopy(measure(by_id[ident], params))
            result.update(params=deepcopy(params), candidate=identity(params), round=number)
            if not isinstance(result.get('problems'), list) or not isinstance(result.get('metrics'), dict):
                raise ValueError('native measurement lacks problems/metrics')
            if not valid(result):
                result['guard_ok'] = False
            return ident, result, None
        except Exception as exc:
            if fatal(exc):
                raise
            return ident, None, dict(image_id=ident, round=number, params=params,
                                    type=type(exc).__name__, message=str(exc))

    def save():
        if checkpoint:
            checkpoint(dict(policy=POLICY, budget_scope='cohort_model_replies', max_rounds=max_rounds,
                baseline_complete=baseline_complete,
                rounds=deepcopy(rounds), history=deepcopy(history), selected=deepcopy(best),
                frozen=sorted(frozen), accepted_with_flags=sorted(flags), errors=deepcopy(errors),
                rules=deepcopy(rule_log), stop_reason=stop))

    if resume is not None:
        if (resume.get('policy')!=POLICY or resume.get('max_rounds')!=max_rounds
                or set(resume['history'])!=set(by_id) or len(resume['rounds'])>max_rounds):
            raise ValueError('checkpoint scope/budget differs')
        history=deepcopy(resume['history']);best=deepcopy(resume['selected'])
        rounds=deepcopy(resume['rounds']);rule_log=deepcopy(resume['rules'])
        baseline_complete=resume.get('baseline_complete',True)
        frozen=set(resume['frozen']);flags=set(resume['accepted_with_flags']);errors=deepcopy(resume['errors'])
        if not frozen<=set(best) or not flags<=frozen:
            raise ValueError('invalid checkpoint frozen state')
        for i,values in history.items():
            for key,o in values.items():
                if key!=identity(o['params']):raise ValueError('checkpoint parameter identity changed')
            if values and identity(defaults) not in values:
                raise ValueError('checkpoint baseline parameters changed')
            if i in best and best[i]['candidate'] not in values:
                raise ValueError('checkpoint selected unknown candidate')
            if i in best and best[i]!=values[best[i]['candidate']]:
                raise ValueError('checkpoint selected record differs from measured history')
        if any(not valid(best[i]) or (best[i]['problems'] and i not in flags) for i in frozen):
            raise ValueError('checkpoint froze an unresolved native result')
    if not baseline_complete:
        def baseline_result(value):
            ident,observation,error=value
            if error:
                errors.append(error);save();return
            history[ident][observation['candidate']] = observation
            if (observation.get('result') is not None and isinstance(observation.get('score'),(int,float))
                    and math.isfinite(observation['score'])):
                best[ident] = observation
                if valid(observation) and not observation['problems']:
                    frozen.add(ident)
            save()
        attempted={e['image_id'] for e in errors if e['round']==0}
        pending=[(i,dict(defaults),0) for i in by_id if not history[i] and i not in attempted]
        save()
        ordered_map(execute,pending,workers,on_result=baseline_result)
        baseline_complete=True
    save()
    for number in range(len(rounds)+1, max_rounds + 1):
        active = sorted(set(best) - frozen)
        if not active:
            stop = 'all_deliverable_items_resolved' if len(best) == len(items) else 'native_errors_remaining'
            break
        packet = dict(round=number, rounds_remaining=max_rounds-number, unresolved=active,
            errors=[e for e in errors if e['image_id'] in active],recent_decisions=deepcopy(rounds[-2:]),
            items=[dict(image_id=i, current=best[i], history=list(history[i].values())) for i in active])
        record = dict(round=number, problems_before=len(active), improved=0, newly_passing=0,
                      reverted=0, native_calls=0, rule_coverage=0)
        rounds.append(record)
        # Reserve the shared decision slot before network I/O. An interrupted
        # request cannot reset the eight-reply budget on resume.
        record['phase']='awaiting_decision';save()
        try:
            decision = propose(packet)
            if not isinstance(decision, dict) or set(decision) - {'reviews','rules','stop','reason'}:
                raise ValueError('unknown global decision fields')
            reviews, rules = decision.get('reviews', []), decision.get('rules', [])
            if not isinstance(reviews,list) or not isinstance(rules,list) or not isinstance(decision.get('stop',False),bool):
                raise ValueError('reviews/rules must be lists and stop must be boolean')
            if decision.get('stop') and rules:
                raise ValueError('stop cannot propose new native work')
            # Validate the entire plan before publishing any decision.
            prepared = []
            for rule in rules:
                if not str(rule.get('reason','')).strip() or not rule.get('params'):
                    raise ValueError('rules require effective changes and a reason')
                if set(rule.get('items') or []) - set(active):
                    raise ValueError('a rule targets unknown or frozen items')
                effect=rule.get('expected_effect')
                if not isinstance(effect,dict) or not effect or any(v not in ('up','down','preserve') for v in effect.values()):
                    raise ValueError('rules require explicit expected native effects')
                if any(not any(k in best[i]['metrics'] for i in active) for k in effect):
                    raise ValueError('unknown expected-effect readout')
                for c in rule.get('when') or []:
                    if set(c)-{'metric','min','max'} or not c.get('metric'):
                        raise ValueError('invalid conditional readout')
                    if not any(c['metric'] in best[i]['metrics'] for i in active):
                        raise ValueError('unknown conditional readout')
                    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v)
                           for k,v in c.items() if k!='metric'):
                        raise ValueError('nonfinite conditional bounds')
                    if c.get('min',-math.inf)>c.get('max',math.inf):
                        raise ValueError('inverted conditional bounds')
                prepared.append((rule, {k: coerce(k,v) for k,v in rule['params'].items()}))
            review_updates = []; seen = set()
            for review in reviews:
                i = review['image_id']
                if i not in active or i in seen:
                    raise ValueError('duplicate, unknown or frozen review ID')
                seen.add(i)
                candidate = history[i].get(review['candidate'])
                if (candidate is None or review.get('baseline') != best[i]['candidate']
                        or review.get('confidence') != 'high' or not review.get('reason')
                        or review.get('image_quality_sufficient') is not True):
                    raise ValueError('review lacks a bound high-confidence comparison')
                observations = review.get('observations') or []
                if len(set(observations)) < 2:
                    raise ValueError('pairwise review needs two distinct observations')
                if not valid(candidate) or (_large_loss(candidate,best[i]) and not review.get('loss_explanation')):
                    raise ValueError('native guards or unexplained large loss')
                if review.get('accept_with_flags'):
                    views={'raw_overview','mask_overlay','size_extreme_zoom'}
                    if (not review.get('normal_heterogeneity') or not review.get('image_quality_sufficient')
                            or review.get('visible_faults') or not views<=set(review.get('evidence_views') or [])):
                        raise ValueError('flag acceptance requires complete, sufficient visual evidence without faults')
                    if any(p.split()[0] not in ('tiny_mask_fraction','large_mask_fraction') for p in candidate['problems']):
                        raise ValueError('only statistical size alerts may be accepted with flags')
                review_updates.append((i,candidate,review))
        except Exception as exc:
            if fatal(exc): raise
            record.update(invalid_decision=dict(type=type(exc).__name__,message=str(exc)),
                          problems_after=len(active),phase='invalid_decision')
            save(); continue
        for i,candidate,review in review_updates:
            if candidate['candidate'] != best[i]['candidate']:
                best[i] = candidate; record['improved'] += 1
            candidate['visual_review'] = deepcopy(review)
            history[i][candidate['candidate']]['visual_review']=deepcopy(review)
            if review.get('accept_with_flags') or not candidate['problems']:
                frozen.add(i); record['newly_passing'] += 1
                if candidate['problems']: flags.add(i)
        jobs = [];job_effects={}
        failed_keys={(e['image_id'],identity(e['params'])) for e in errors}
        for i in active:
            if i in frozen: continue
            params = dict(best[i]['params']); matched = [];effects={}
            for rule,changes in prepared:
                if _matches(rule,by_id[i],best[i]):
                    params.update(changes); matched.append(rule['reason']);effects.update(rule['expected_effect'])
            if matched:
                record['rule_coverage'] += 1
                if identity(params) not in history[i] and (i,identity(params)) not in failed_keys:
                    jobs.append((i,params,number));job_effects[i]=(deepcopy(best[i]['metrics']),effects)
        rule_log.append(dict(round=number,rules=deepcopy(rules),reviews=deepcopy(reviews)))
        record['phase']='native_measurement';record['planned_native_calls']=len(jobs);save()
        def candidate_result(value):
            i,candidate,error=value
            record['native_calls'] += 1
            if error:
                errors.append(error);save();return
            before,effects=job_effects[i];checks={}
            for key,direction in effects.items():
                x,y=before.get(key),candidate['metrics'].get(key)
                supported=(x is not None and y is not None and
                    (y>x+1e-9 if direction=='up' else y<x-1e-9 if direction=='down' else abs(y-x)<=1e-9))
                checks[key]=dict(expected=direction,before=x,after=y,supported=supported)
            candidate['hypothesis_checks']=checks
            candidate['hypothesis_supported']=all(c['supported'] for c in checks.values())
            history[i][candidate['candidate']] = candidate
            # Preserve the best measured improvement even below pass thresholds.
            if (valid(candidate) and valid(best[i]) and not _large_loss(candidate,best[i])
                    and quality(candidate) > quality(best[i])):
                best[i] = candidate;record['improved'] += 1
                if not candidate['problems']:
                    frozen.add(i);record['newly_passing'] += 1
            else:
                record['reverted'] += 1
            record['problems_after']=len(set(best)-frozen)
            save()
        ordered_map(execute,jobs,workers,on_result=candidate_result)
        record['problems_after'] = len(set(best)-frozen)
        record['phase']='complete'
        save()
        if decision.get('stop'):
            stop='agent_stop';break
    if len(frozen)==len(items): stop='all_deliverable_items_resolved'
    delivery=dict(total=len(items),delivered=len(best),qc_pass=len(frozen-flags),
                  accepted_with_flags=len(flags),unresolved=len(set(best)-frozen),
                  failed=len(set(by_id)-set(best)))
    save()
    return best,dict(policy=POLICY,max_rounds=max_rounds,rounds=rounds,history=history,
        rules=rule_log,stop_reason=stop,budget_exhausted=stop=='budget_exhausted',
        delivery_summary=delivery,errors=errors)
