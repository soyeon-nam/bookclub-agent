"""Evaluate active participant turns; archived/user/moderator turns are excluded."""
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
from . import metrics


def validate_session(session):
    if not isinstance(session, dict) or session.get('schema_version') != 1:
        raise ValueError('Expected a schema_version=1 session log')
    if not isinstance(session.get('utterances'), list) or not isinstance(session.get('participants'), list):
        raise ValueError('Session must contain utterances and participants lists')
    if not all(isinstance(name,str) and name for name in session['participants']):
        raise ValueError('Participant names must be nonempty text')
    metadata = session.get('metadata',{})
    if not isinstance(metadata,dict) or not isinstance(metadata.get('personas',{}),dict):
        raise ValueError('Metadata and personas must be objects')
    if not isinstance(session.get('discarded',[]),list):
        raise ValueError('Discarded turns must be a list')
    seen = set()
    for turn in session['utterances']:
        if not isinstance(turn, dict) or not isinstance(turn.get('turn_id'), str) or not turn['turn_id']:
            raise ValueError('Every turn must have a nonempty turn_id')
        if turn['turn_id'] in seen:
            raise ValueError('Duplicate active turn_id')
        seen.add(turn['turn_id'])
        for key in ('speaker','text','topic'):
            if not isinstance(turn.get(key), str):
                raise ValueError(f'Turn {key} must be text')
        if turn.get('kind') not in ('user','moderator','participant'):
            raise ValueError('Unknown turn kind')
        evidence = turn.get('evidence', {})
        if not isinstance(evidence, dict):
            raise ValueError('Turn evidence must be an object')
        for key in ('query','rag_context','move','response_prompt','stance_before','stance_after'):
            if key in evidence and not isinstance(evidence[key], str):
                raise ValueError(f'Evidence {key} must be text')
    return session


def participant_turns(session):
    validate_session(session)
    return [turn for turn in session['utterances'] if turn['kind'] == 'participant']


def automatic_metrics(session):
    turns = participant_turns(session)
    texts = [turn['text'] for turn in turns]
    speakers = [turn['speaker'] for turn in turns]
    moves = [turn.get('evidence',{}).get('move','') for turn in turns]
    contexts = [turn.get('evidence',{}).get('rag_context','') for turn in turns]
    configured = set(session.get('metadata',{}).get('personas',{})) or {
        name for name in session['participants'] if name != session.get('user_name') and name.lower() != 'president'
    }
    configured |= set(speakers)
    distribution = {name: speakers.count(name) for name in sorted(configured)}
    entropy = metrics.turn_entropy(speakers)
    eligible = [i for i, turn in enumerate(session['utterances'][:-1])
                if turn['kind']=='participant' and turn.get('evidence',{}).get('move')=='ASK_QUESTION']
    answer_moves = {'BUILD_ON','AGREE_AND_EXTEND','DISAGREE_SPECIFIC','SHARE_ANECDOTE','REACT_EMOTIONALLY'}
    answered = sum(session['utterances'][i+1].get('evidence',{}).get('move') in answer_moves for i in eligible)
    return {
        'counts': {'active_turns':len(session['utterances']), 'participant_turns':len(turns),
                   'discarded_turns':len(session.get('discarded',[])),
                   'with_retrieval':sum(bool(ctx.strip()) for ctx in contexts),
                   'invalid_or_missing_move':sum(move not in metrics.VALID_MOVES for move in moves)},
        'speaker_distribution':distribution,
        'distinct_1':metrics.distinct_n(texts,1), 'distinct_2':metrics.distinct_n(texts,2),
        'turn_entropy':entropy,
        'normalized_turn_entropy_selected':entropy/math.log2(len(configured)) if len(configured)>1 else 0.0,
        'planned_move_diversity':metrics.move_diversity(moves),
        'name_mention_rate':metrics.direct_address_rate(texts,sorted(configured)),
        'planned_rebuttal_rate':metrics.rebuttal_rate(moves),
        'planned_question_followup':{'eligible_questions':len(eligible),'answer_move_after':answered,
                                     'rate':answered/len(eligible) if eligible else None},
        'rolling_lexical_similarity':metrics.topic_drift_score(texts),
        'grounding_proxy':metrics.grounding_score(texts,contexts),
        'lexical_diversity_ttr':metrics.lexical_diversity(texts),
        'mattr_50':metrics.moving_average_ttr(texts),
        'response_length_words':metrics.avg_response_length(texts),
    }


def build_report(session, source_bytes=None):
    canonical = json.dumps(session,sort_keys=True,ensure_ascii=False).encode()
    return {'schema_version':1,'session_id':session.get('session_id'),
            'source_sha256':hashlib.sha256(source_bytes if source_bytes is not None else canonical).hexdigest(),
            'automatic':automatic_metrics(session),
            'limitations':[
                'Active AI participant turns only; moderator, user and discarded turns are excluded from lexical and turn-distribution metrics.',
                'Move labels are generation plans, not independent annotations of realized speech acts.',
                'Question follow-up measures the immediate next planned AI move, not semantic answer correctness; user answers are not classified.',
                'English lexical overlap and scene patterns are proxies, not proof of faithfulness or factual accuracy.',
                'Rolling windows overlap; lexical similarity is not an independent measure of topic coherence.',
                'No-question and insufficient-data results are null; this tool does not establish scientific validity or system improvement.',
            ]}
