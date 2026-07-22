"""Validate a single model result before publishing any participant state."""
from langchain_core.messages import HumanMessage, AIMessage
from rag_search import parse_json
import re


def validate_turn(content, allowed_moves):
    if not isinstance(content, str):
        raise ValueError("Turn must contain text")
    value = parse_json(content)
    if not isinstance(value, dict):
        raise ValueError('Turn must be a JSON object')
    if set(value) != {'stance', 'move', 'evidence_quote', 'warrant', 'revision_reason', 'response'}:
        raise ValueError('Turn has missing or unknown fields')
    for key in ('stance', 'move', 'evidence_quote', 'warrant', 'revision_reason', 'response'):
        if not isinstance(value.get(key), str):
            raise ValueError(f'{key} must be a string')
        value[key] = value[key].strip()
    if not value['stance'] or not value['response'] or not value['warrant']:
        raise ValueError('stance, warrant and response must be nonempty')
    if value['move'] not in allowed_moves:
        raise ValueError('move must be one of the allowed moves')
    if any(len(value[key]) > (3000 if key == 'response' else 1000) for key in value):
        raise ValueError('Turn fields exceed size limits')
    return {key: value[key] for key in ('stance', 'move', 'evidence_quote', 'warrant', 'revision_reason', 'response')}


def normalize_quote(text):
    text = text.translate(str.maketrans({'“':'"', '”':'"', '‘':"'", '’':"'"}))
    return re.sub(r"\s+", " ", text).strip().casefold().strip(" .,!?:;\"'")


def validate_grounding(value, passages, prior_stance):
    quote = normalize_quote(value['evidence_quote'])
    matched = [i for i, passage in enumerate(passages.split('\n---\n'))
               if quote and re.search(r'(?<!\w)' + re.escape(quote) + r'(?!\w)', normalize_quote(passage))]
    errors = []
    if value['evidence_quote'] and not matched:
        errors.append('evidence_quote must be copied from a provided passage without word changes')
    revising = value['move'] == 'CONCEDE_AND_REVISE'
    if revising and (not prior_stance or not value['revision_reason']
                     or normalize_quote(prior_stance).casefold() == normalize_quote(value['stance']).casefold()):
        errors.append('CONCEDE_AND_REVISE requires a prior stance, a changed stance and a nonempty concrete revision_reason')
    if not revising and value['revision_reason']:
        errors.append('revision_reason must be empty unless the move is CONCEDE_AND_REVISE')
    if errors:
        raise ValueError('; '.join(errors))
    return {'status':'matched' if matched else 'not_provided', 'passage_indices':matched,
            'scope':'Excerpt presence after whitespace/case normalization only; logical support is not automatically proven.'}


class TurnGenerationError(ValueError):
    def __init__(self, errors, attempts):
        super().__init__('Invalid structured turn after one repair: '+errors[-1])
        self.attempts = attempts


def generate_turn(llm, messages, allowed_moves, passages='', prior_stance=''):
    """One successful structured call; at most one schema repair, no partial state."""
    errors = []
    attempts = []
    for attempt in range(2):
        request = messages
        if errors:
            request = messages + [AIMessage(content=attempts[-1]["content"]), HumanMessage(content=(
                'The previous result was invalid: '+errors[-1]+'. Return the complete JSON object. '
                'For evidence_quote, copy a contiguous excerpt exactly from the supplied passages; '
                'never paraphrase. Use an empty quote if no excerpt supports the claim. ' +
                ('No book passages are available: evidence_quote MUST be an empty string. ' if not passages.strip() else '') +
                'Allowed moves: '+', '.join(allowed_moves)))]
        content = llm.invoke(request).content
        try:
            value = validate_turn(content, allowed_moves)
            grounding = validate_grounding(value, passages, prior_stance)
            return value, {'format':'json', 'calls':attempt+1, 'validation_errors':errors,
                           'grounding':grounding}
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
            attempts.append({"content":content, "error":str(error)})
    raise TurnGenerationError(errors, attempts)


def unchanged_count(prior, current, previous_count):
    stopwords = set('the a an is it in of and to that this i my but not be'.split())
    old = set(prior.lower().split()) - stopwords
    new = set(current.lower().split()) - stopwords
    if old and new and len(old & new) / max(len(old), len(new)) > .6:
        return previous_count+1
    return 0
