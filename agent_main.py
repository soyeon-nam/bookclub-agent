"""Shared LangGraph nodes for automatic and single-turn discussion execution."""
from copy import deepcopy
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agentstate import BookAgentState
from tools import make_speak, president, choose_next, user_call, append_line_to_file
from participant import participant_creation, reset_agents, AGENTS, PARTICIPANT_DESCRIPTIONS, MODEL_NAME as PARTICIPANT_MODEL, TURN_SIGNAL_STYLE
from session_log import start_session, record_turn, publish_turn
from rag import EMBEDDING_MODEL, init_rag
from tts import tts_generate

MODEL_NAME = os.getenv("GPT_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL_NAME)
DEFAULT_MAX_MESSAGES = 100


def session_participants(state, include_user=False):
    names = list(participant_creation(state.get("participants_name")))
    user = state.get("user_name", "")
    if not isinstance(user, str) or not user.strip() or ":" in user or "\n" in user or "\r" in user:
        raise ValueError("User name must be nonempty and contain no colon or newline")
    if user.casefold() in {name.casefold() for name in names + ["President"]}:
        raise ValueError("User name must differ from the selected participants and President")
    return names + ([user] if include_user else [])


def _initialize_session(state: BookAgentState, include_user=False):
    names = session_participants(state, include_user=include_user)
    reset_agents()
    init_rag(state["book"])

    def generate(prompt):
        text = llm.invoke([HumanMessage(content=prompt)]).content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("The model returned an empty session introduction")
        return text.strip()

    book_summary = generate("Write a short summary of 100 words about " + state["book"])
    opening = generate(
        f"You are the president of a book club about '{state['book']}'.\n"
        f"Book summary:\n{book_summary}\n\n"
        "Write a warm, welcoming opening: briefly welcome everyone and ask ONE easy "
        "warm-up question about a first impression, memorable moment, or character. "
        "Help everyone feel comfortable talking. Two to three conversational sentences."
    )
    if "?" in opening:
        opening = opening.split("?", 1)[0] + "?"
    topic = generate(
        "In one short sentence, what is the core discussion topic in this opening statement?\n"
        + opening
    )
    message = "President: " + opening
    tts_generate(opening, "president", os.path.join("dialogue_audio", "President_Introduction.wav"))
    append_line_to_file("book_club_conversation.txt", message)
    state.update(participants_name=names, book_summary=book_summary,
                 topics_history=[topic], messages=[message], _routing={},
                 turns_since_president=0, force_question=False)
    start_session(state, {"chat_model": PARTICIPANT_MODEL, "embedding_model": EMBEDDING_MODEL,
                          "turn_signal_style": TURN_SIGNAL_STYLE,
                          "personas": {name: PARTICIPANT_DESCRIPTIONS[name] for name in names if name in AGENTS}})
    record_turn(state, "President", os.path.join("dialogue_audio", "President_Introduction.wav"),
                {"book_summary": book_summary})
    return state


def init(state: BookAgentState):
    return _initialize_session(state, include_user=True)


def controlled_init(state: BookAgentState):
    return _initialize_session(state)


def select_speaker(state):
    speaker = choose_next(state, state.get('turns_since_president', 0))
    if speaker not in state.get('participants_name', list(AGENTS)) + ['President', state.get('user_name', '')]:
        raise ValueError('Router selected an unknown speaker')
    return {'next_speaker': speaker, '_routing': deepcopy(state.get('_routing', {}))}


def _speak(state, kind):
    # Publish a copied message/topic list, so failed work cannot leak into caller state.
    current = dict(state, messages=list(state['messages']), topics_history=list(state['topics_history']))
    speaker = state['next_speaker']
    evidence = {}
    audio = ''
    if kind == 'president':
        topic, message, audio = president(current['messages'], current['book'], current['topics_history'], evidence=evidence)
        current['topics_history'].append(topic)
        count = 0
    elif kind == 'user':
        message = user_call(current)
        count = current.get('turns_since_president', 0) + 1
    else:
        quality = current.get('_routing', {}).get('quality', {})
        drought = quality.get('question_drought', False) and not quality.get('has_unanswered_question', False)
        current['force_question'] = current.get('force_question', False) or drought
        message, audio = make_speak(current['messages'], current['book'], current['topics_history'],
                                    speaker, current.get('force_question', False), current.get('user_name', ''))
        count = current.get('turns_since_president', 0) + 1
        if speaker in AGENTS and AGENTS[speaker].turn_evidence:
            evidence = AGENTS[speaker].turn_evidence[-1]
    current['messages'].append(message)
    publish_turn(current, speaker, audio, dict(evidence, routing=deepcopy(current.get('_routing', {}))))
    current.update(turns_since_president=count, force_question=False, last_audio_path=audio)
    return current


def president_speak(state): return _speak(state, 'president')
def participant_speak(state): return _speak(state, 'participant')
def user_speak(state): return _speak(state, 'user')


def route_speaker(state):
    if state['next_speaker'] == 'President': return 'president_speak'
    if state['next_speaker'] == state.get('user_name'): return 'user_speak'
    return 'participant_speak'


def should_continue(state):
    return 'select_speaker' if len(state.get('messages', [])) < state.get('max_messages', DEFAULT_MAX_MESSAGES) else END


def build_graph(mode):
    if mode not in ('full', 'loop', 'turn'): raise ValueError('Unknown graph mode')
    builder = StateGraph(BookAgentState)
    builder.add_node('select_speaker', select_speaker)
    nodes = {'president_speak': president_speak, 'participant_speak': participant_speak, 'user_speak': user_speak}
    for name, node in nodes.items():
        builder.add_node(name, node)
        if mode == 'turn': builder.add_edge(name, END)
        else: builder.add_conditional_edges(name, should_continue, {'select_speaker':'select_speaker', END:END})
    builder.add_conditional_edges('select_speaker', route_speaker, {n:n for n in nodes})
    if mode == 'full':
        builder.add_node('init', init)
        builder.add_edge(START, 'init')
        builder.add_conditional_edges('init', should_continue, {'select_speaker':'select_speaker', END:END})
    elif mode == 'loop':
        builder.add_conditional_edges(START, should_continue, {'select_speaker':'select_speaker', END:END})
    else: builder.add_edge(START, 'select_speaker')
    return builder.compile()


full_graph = build_graph('full')
loop_graph = build_graph('loop')
turn_graph = build_graph('turn')


def _invoke(graph, state, **overrides):
    limit = state.get('max_messages', DEFAULT_MAX_MESSAGES)
    if type(limit) is not int or limit < 1: raise ValueError('max_messages must be a positive integer')
    working = dict(state, **overrides)
    # Only committed nodes reach callers. Earlier durable turns survive a later failure.
    for chunk in graph.stream(working, config={'recursion_limit':limit*3+20}, stream_mode='updates'):
        for node, update in chunk.items():
            working.update(update)
            if node != 'select_speaker': state.update(working)
    return state


def run_discussion(state):
    return _invoke(full_graph, state)


def assistant(state):
    return _invoke(loop_graph, state)


def controlled_assistant(state, last: int, force_question: Optional[bool] = False):
    result = _invoke(turn_graph, state, turns_since_president=last, force_question=bool(force_question))
    return result, result['next_speaker'], result.get('last_audio_path', '')
