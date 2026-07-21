"""Best-effort, append-only operational events with unique identifiers."""
import argparse
import json
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4

_LOCK = Lock()
_RUN_ID = uuid4().hex


def log_path(session_id=None):
    if session_id is None:
        return Path('conversation_log') / ('ops-' + _RUN_ID + '.jsonl')
    if not isinstance(session_id, str) or not re.fullmatch(r'[0-9a-f]{32}', session_id):
        raise ValueError('Invalid session identifier')
    return Path('conversation_log') / ('session-' + session_id) / 'ops.jsonl'


def ops_event(name, *, session_id=None, **fields):
    """Operational logging must never invalidate a generated or committed turn.

    Callers supply identifiers explicitly so background work or another session
    cannot silently inherit the wrong session.
    """
    try:
        record = dict(fields, event=name, event_id=uuid4().hex, session_id=session_id)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
        path = log_path(session_id)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(line)
        return record
    except Exception:
        return None


def summarize(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    counts = {}
    for path in paths:
        try:
            with Path(path).open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                        if not isinstance(value, dict) or not isinstance(value.get('event'), str):
                            raise ValueError('Invalid event')
                        event = value['event']
                    except (ValueError, TypeError):
                        event = '<malformed>'
                    counts[event] = counts.get(event, 0) + 1
        except FileNotFoundError:
            continue
    unmapped = counts.get('action_unmapped', 0)
    failed = counts.get('action_failed', 0)
    total = counts.get('action_chosen', 0) + unmapped + failed
    return {'counts': counts, 'action_decisions': total,
            'unmapped_actions': unmapped, 'failed_action_calls': failed,
            'action_fallback_rate': (unmapped + failed) / total if total else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count operational attempts, including rolled-back turns.')
    parser.add_argument('paths', nargs='*', help='JSONL paths; defaults to all local operational logs')
    args = parser.parse_args()
    paths = args.paths or sorted(Path('conversation_log').glob('**/ops*.jsonl'))
    print(json.dumps(summarize(paths), ensure_ascii=False, indent=2))
