"""Validate a single model result before publishing any participant state."""
from langchain_core.messages import HumanMessage
from rag_search import parse_json


def validate_turn(content, allowed_moves):
    if not isinstance(content, str):
        raise ValueError("Turn must contain text")
    value = parse_json(content)
    if not isinstance(value, dict):
        raise ValueError('Turn must be a JSON object')
    if set(value) != {'stance', 'move', 'evidence_quote', 'response'}:
        raise ValueError('Turn has missing or unknown fields')
    for key in ('stance', 'move', 'evidence_quote', 'response'):
        if not isinstance(value.get(key), str):
            raise ValueError(f'{key} must be a string')
        value[key] = value[key].strip()
    if not value['stance'] or not value['response']:
        raise ValueError('stance and response must be nonempty')
    if value['move'] not in allowed_moves:
        raise ValueError('move must be one of the allowed moves')
    if len(value['stance']) > 1000 or len(value['evidence_quote']) > 1000 or len(value['response']) > 3000:
        raise ValueError('Turn fields exceed size limits')
    return {key: value[key] for key in ('stance', 'move', 'evidence_quote', 'response')}


def generate_turn(llm, messages, allowed_moves):
    """One successful structured call; at most one schema repair, no partial state."""
    errors = []
    for attempt in range(2):
        request = messages
        if errors:
            request = messages + [HumanMessage(content=(
                'The previous result was invalid: '+errors[-1]+'. Return the complete JSON object. '
                'Allowed moves: '+', '.join(allowed_moves)))]
        content = llm.invoke(request).content
        try:
            value = validate_turn(content, allowed_moves)
            return value, {'format':'json', 'calls':attempt+1, 'validation_errors':errors}
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
    raise ValueError('Invalid structured turn after one repair: '+errors[-1])


def unchanged_count(prior, current, previous_count):
    stopwords = set('the a an is it in of and to that this i my but not be'.split())
    old = set(prior.lower().split()) - stopwords
    new = set(current.lower().split()) - stopwords
    if old and new and len(old & new) / max(len(old), len(new)) > .6:
        return previous_count+1
    return 0
