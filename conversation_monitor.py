"""Stateless, reproducible quality signals from the current active conversation."""
from collections import Counter
import re
from dialogue_moves import planned_move

STOPWORDS = set('about their which would could should being there where other every those these after before still rather doesn think story scene character'.split())


def mentioned_participant(content, participants):
    """Match whole names, or an unambiguous first name; never shared titles."""
    for name in participants:
        pattern = r'(?<!\w)' + r'[ _]+'.join(map(re.escape, name.split('_'))) + r'(?!\w)'
        if re.search(pattern, content, re.I):
            return name
    first_names = Counter(name.split('_')[0].lower() for name in participants)
    for name in participants:
        first = name.split('_')[0]
        if first.lower() not in {'dr', 'prof', 'chef'} and first_names[first.lower()] == 1:
            if re.search(r'(?<!\w)' + re.escape(first) + r'(?!\w)', content, re.I):
                return name
    return ''


def analyze(messages, participants, moves=(), book=''):
    last_content = messages[-1].split(':', 1)[-1].strip() if messages else ''
    has_question = '?' in last_content
    frequency = Counter()
    ignored = STOPWORDS | set(re.findall(r'\w+', book.lower()))
    for message in messages[-8:]:
        words = set(re.findall(r'[^\W_]+', message.split(':', 1)[-1].lower()))
        frequency.update(word for word in sorted(words) if len(word) >= 5 and word not in ignored)
    repeated = [word for word, count in frequency.most_common() if count >= 4][:5]
    recent_speakers = [m.split(':', 1)[0].strip() for m in messages[-10:]]
    recent_moves = [planned_move(m) for m in list(moves)[-6:]]
    agreements = {'EXTEND'}
    consecutive = 0
    for move in reversed(recent_moves):
        if move not in agreements:
            break
        consecutive += 1
    since_president = 0
    for message in reversed(messages):
        if message.split(':', 1)[0].strip().lower() == 'president':
            break
        since_president += 1
    recent_participant = [m for m in messages if m.split(':',1)[0].strip() in participants][-8:]
    drought = len(recent_participant) == 8 and not any('?' in m.split(':',1)[-1] for m in recent_participant)
    return {
        'question_drought': drought, 'question_drought_window': 8,
        'has_unanswered_question': has_question,
        'question_target': mentioned_participant(last_content, participants) if has_question else '',
        'repeated_keywords': repeated, 'is_repetitive': bool(repeated),
        'consecutive_agreements': consecutive,
        'agreement_dominated': len(recent_moves) >= 3 and sum(m in agreements for m in recent_moves)/len(recent_moves) > .7,
        'silent_speakers': [name for name in participants if name not in recent_speakers],
        'turns_since_president': since_president,
        'president_allowed': since_president >= 7,
    }
