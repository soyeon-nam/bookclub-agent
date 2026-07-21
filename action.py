"""Normalize model animation labels without changing the existing Unity IDs."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ops_log import ops_event

load_dotenv()
MODEL_NAME = os.getenv('ACTION_MODEL', 'gpt-4o')
llm = ChatOpenAI(model=MODEL_NAME, temperature=0, request_timeout=30, max_retries=1)

# Questionning is a historical Unity label. Preserve its ID and public spelling.
action_list = ['Normal 1', 'Normal 2', 'Questionning', 'Thinking', 'Angry', 'Happy', 'Sad']
action_map = {name: index + 1 for index, name in enumerate(action_list)}
_LABELS = {name.casefold(): name for name in action_list}
_LABELS['questioning'] = 'Questionning'
_FALLBACK = 'Normal 1'


def _normalize(action):
    if not isinstance(action, str):
        return None
    key = ' '.join(action.strip().strip('\"\'“”‘’').split()).casefold()
    return _LABELS.get(key)


def choose_action(speach: str, *, session_id=None, turn_id=None) -> str:
    context = {'session_id': session_id, 'turn_id': turn_id, 'model': MODEL_NAME,
               'source': 'choose_action'}
    try:
        response = llm.invoke([HumanMessage(content=(
            'Choose the animation for this speech. Allowed labels: ' + ', '.join(action_list) + '. '
            'Use Normal 1 or Normal 2 for ordinary speech and emotional actions only for strongly emotional speech. '
            'Return only one label. Speech: ' + speach))])
        raw = response.content
    except Exception as error:
        ops_event('action_failed', **context, error_type=type(error).__name__,
                  label=_FALLBACK, animation_id=action_map[_FALLBACK])
        return _FALLBACK
    label = _normalize(raw)
    ops_event('action_chosen' if label else 'action_unmapped', **context,
              raw=raw, label=label or _FALLBACK, animation_id=action_map[label or _FALLBACK])
    return label or _FALLBACK


def action_to_id(action: str) -> int:
    label = _normalize(action)
    if label is None:
        ops_event('action_mapping_fallback', source='action_to_id', raw=action,
                  label=_FALLBACK, animation_id=action_map[_FALLBACK])
    # Mapping choose_action's canonical result does not double-count a decision.
    return action_map[label or _FALLBACK]
