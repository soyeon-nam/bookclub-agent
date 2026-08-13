"""Deterministic speaker fairness rules; model fit remains a separate signal."""
from collections import Counter
import re

POLICY = {'window':8,'ban_count':3,'fit_floor':5,'third_turn_penalty':2,
          'starve_turns':6,'starve_bonus':2}


def snapshot(messages, participants):
    counts=Counter(m.split(':',1)[0].strip() for m in messages[-POLICY['window']:])
    last={m.split(':',1)[0].strip():i for i,m in enumerate(messages)}
    return [{'speaker':name,'recent_count':counts[name],
             'silence':len(messages)-last[name]-1 if name in last else None,
             'raw_fit':None,'adjusted_score':None,'adjustments':[],'excluded':None}
            for name in participants]


def select(signals, messages, participants, quality):
    rows=snapshot(messages,participants)
    by_name={s['speaker']:s for s in signals}
    last_speaker=messages[-1].split(':',1)[0].strip() if messages else ''
    for row in rows:
        raw=by_name.get(row['speaker'])
        if raw is None:
            row['excluded']='last_speaker' if row['speaker']==last_speaker else 'unscored'
            continue
        row.update(raw_fit=raw['score'],want_to_speak=raw['want_to_speak'],reason=raw.get('reason',''),target=raw.get('target','none'))
        changes=[]
        if row['recent_count']==POLICY['ban_count']-1:changes.append(('third_turn',-POLICY['third_turn_penalty']))
        if row['silence'] is None or row['silence']>POLICY['starve_turns']:changes.append(('starvation',POLICY['starve_bonus']))
        if quality.get('question_target')==row['speaker']:changes.append(('question_target',3))
        if quality.get('agreement_dominated') and re.search(r'\b(disagree|challenge|counter|wrong)\b|push back',raw.get('reason',''),re.I):changes.append(('counterpoint',2))
        row['adjustments']=[{'reason':reason,'delta':delta} for reason,delta in changes]
        row['adjusted_score']=max(0,row['raw_fit']+sum(delta for _,delta in changes))
    passes=[('strict',False,False,False),('relaxed_fit',True,False,False),
            ('relaxed_fit_and_frequency',True,True,False),('single_speaker_repeat',True,True,True)]
    stages=[]
    for label,relax_fit,relax_count,relax_last in passes:
        eligible=[];excluded={}
        for row in rows:
            name=row['speaker'];reason=None
            if name==last_speaker and not relax_last: reason='last_speaker'
            elif row['raw_fit'] is None: reason='unscored'
            elif row['recent_count']>=POLICY['ban_count'] and not relax_count: reason='recent_frequency'
            elif row['raw_fit']<POLICY['fit_floor'] and not relax_fit: reason='low_fit'
            if reason: excluded[name]=reason
            else: eligible.append(row)
        stages.append({'policy':label,'excluded':excluded})
        if eligible:
            chosen=max(eligible,key=lambda r:(r['adjusted_score'],len(messages)+1 if r['silence'] is None else r['silence'],-participants.index(r['speaker'])))
            for row in rows:row['excluded']=excluded.get(row['speaker'])
            return chosen['speaker'],{'policy':dict(POLICY),'fallback':None if label=='strict' else label,'candidates':rows,'filter_passes':stages}
        if label=='strict' and quality.get('president_allowed') and not quality.get('has_unanswered_question'):
            for row in rows:row['excluded']=excluded.get(row['speaker'])
            return 'President',{'policy':dict(POLICY),'fallback':'no_eligible_after_cooldown','candidates':rows,'filter_passes':stages}
    raise ValueError('No scored participant available')
