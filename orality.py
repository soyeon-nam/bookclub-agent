"""English-oriented spoken length controls and descriptive surface measurements."""
import math
import os
import re

MAX_SPOKEN_WORDS = 70
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def word_count(text):
    return len(_WORD.findall(text))


def spoken_style_enabled():
    return os.getenv('SPOKEN_REGISTER','1').strip().lower() not in ('0','false','off','')


def style_instructions(enabled):
    return ('This will be heard aloud. Use conversational rhythm, short sentences and one idea per sentence. '
            'Keep your persona and domain vocabulary, but explain them as you would in live conversation. '
            'Avoid essay connectors such as moreover/furthermore/indeed/thus, stacked clauses and formulaic antithesis. '
            'A friendly opener does not make an essay body conversational.\n') if enabled else ''


def percentile(values,p):
    if not values:return None
    ordered=sorted(values);position=(len(ordered)-1)*p
    lower=math.floor(position);upper=math.ceil(position)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def statistics(texts):
    from tts import split_sentences
    lengths=[word_count(t) for t in texts]
    sentence_lengths=[word_count(s) for t in texts for s in split_sentences(t)]
    p95=percentile(lengths,.95)
    return {'n':len(texts),'words_per_turn':lengths,'words_mean':sum(lengths)/len(lengths) if lengths else None,
            'words_p95':p95,'design_p95_at_most_75':p95<=75 if p95 is not None else None,
            'sentences':len(sentence_lengths),'words_per_sentence_mean':sum(sentence_lengths)/len(sentence_lengths) if sentence_lengths else None,
            'words_per_sentence_p95':percentile(sentence_lengths,.95),
            'essay_connector_turns':sum(bool(re.search(r'\b(moreover|furthermore|indeed|thus)\b',t,re.I)) for t in texts),
            'scope':'English-oriented tokenization and linear-interpolated empirical p95. Surface description, not listening quality or a validated threshold.'}
