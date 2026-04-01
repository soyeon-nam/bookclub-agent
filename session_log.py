"""Session-scoped evidence, atomic JSON saves, and explicit rollback archives."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from uuid import uuid4


def _save(session):
    path = Path(session['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def start_session(state, metadata):
    session_id = uuid4().hex
    state['_session'] = {
        'schema_version': 1, 'session_id': session_id,
        'path': str(Path('conversation_log') / ('session-' + session_id) / 'session.json'),
        'book': state['book'], 'participants': list(state['participants_name']),
        'user_name': state['user_name'], 'metadata': metadata,
        'utterances': [], 'discarded': [], 'events': [],
    }


def record_turn(state, speaker, audio='', evidence=None):
    if '_session' not in state:
        return  # External callers can still use the lower-level state functions.
    session = deepcopy(state['_session'])
    turn_id = uuid4().hex
    record = {'turn_id': turn_id, 'turn_idx': len(session['utterances']),
              'speaker': speaker, 'kind': 'user' if speaker == state['user_name'] else
              ('moderator' if speaker.lower() == 'president' else 'participant'),
              'text': state['messages'][-1].split(':', 1)[-1].strip(),
              'topic': state['topics_history'][-1] if state['topics_history'] else '',
              'topics_history': list(state['topics_history']),
              'evidence': deepcopy(evidence or {}), 'audio': None}
    target = None
    try:
        if audio and Path(audio).is_file():
            relative = Path('audio') / (turn_id + '.wav')
            target = Path(session['path']).parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio, target)
            record['audio'] = str(relative)
        session['utterances'].append(record)
        session['events'].append({'type': 'utterance', 'turn_id': turn_id})
        _save(session)
    except Exception:
        if target is not None:
            target.unlink(missing_ok=True)
        raise
    state['_session'] = session


def restore_session(restored, current):
    if '_session' not in current:
        return
    session = deepcopy(restored['_session'])
    retained = {turn['turn_id'] for turn in session['utterances']}
    removed = [turn for turn in current['_session']['utterances'] if turn['turn_id'] not in retained]
    session['discarded'] = deepcopy(current['_session']['discarded'] + removed)
    session['events'] = deepcopy(current['_session']['events'])
    session['events'].append({'type': 'rollback', 'removed_turn_ids': [turn['turn_id'] for turn in removed]})
    _save(session)
    restored['_session'] = session


def publish_turn(state, speaker, audio='', evidence=None):
    """If durable logging fails, undo the just-generated turn and plain transcript."""
    try:
        record_turn(state, speaker, audio, evidence)
    except Exception:
        from participant import rollback_agents
        message = state['messages'].pop()
        if speaker.lower() == 'president':
            state['topics_history'].pop()
        elif speaker != state['user_name']:
            rollback_agents([message])
        Path('book_club_conversation.txt').write_text(
            '\n'.join(m.rstrip() for m in state['messages']) + '\n', encoding='utf-8')
        raise
