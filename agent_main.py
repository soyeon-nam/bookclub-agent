from copy import deepcopy
from typing import TypedDict, Annotated, Optional, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from tools import make_speak, president, choose_next, user_call
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
import os 
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from participant import participant_creation, reset_agents, AGENTS, PARTICIPANT_DESCRIPTIONS, MODEL_NAME as PARTICIPANT_MODEL, TURN_SIGNAL_STYLE
from session_log import start_session, record_turn, publish_turn
from rag import EMBEDDING_MODEL
from tts import tts_generate
from tools import append_line_to_file
from rag import init_rag
    # ---------- LLM Setup ----------

from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

# Use a lightweight GPT model by default, can be overridden with env var GPT_MODEL
MODEL_NAME = os.getenv("GPT_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(model=MODEL_NAME)

    # ---------- State Management ----------

from agentstate import BookAgentState

    # ---------- List of Tools ----------

# Tools list using @tool decorators
tools = [make_speak, president]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    # ---------- Nodes ----------


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
                 topics_history=[topic], messages=[message], _routing={})
    start_session(state, {"chat_model": PARTICIPANT_MODEL, "embedding_model": EMBEDDING_MODEL,
                          "turn_signal_style": TURN_SIGNAL_STYLE,
                          "personas": {name: PARTICIPANT_DESCRIPTIONS[name] for name in names if name in AGENTS}})
    record_turn(state, "President", os.path.join("dialogue_audio", "President_Introduction.wav"),
                {"book_summary": book_summary})
    return state


def init(state: BookAgentState):
    return _initialize_session(state, include_user=True)


def assistant(state: BookAgentState):
    # make the discussion run until 100 messages are reached
    last=0 #Number of messages since the last president intervention
    while len(state["messages"])<state.get("max_messages", 100):
        next_speaker = choose_next(state,last)
        evidence = {}
        filepath = ""
        
        if next_speaker == "President": #Make the president speak and fine a new topic
            last=0 
            new_topic,new_message,filepath = president(
                state["messages"],
                state["book"],
                state["topics_history"],
                evidence=evidence,
            )
            state["topics_history"].append(new_topic)
        
        elif next_speaker==state["user_name"]: #Make the user speak
            last+=1
            new_message=user_call(state)
            
        
        else: #Make a participant speak
            last+=1
            new_message, filepath = make_speak(
                state["messages"],
                state["book"],
                state["topics_history"],
                next_speaker,
                user=state.get("user_name", ""),
            )

        state["messages"].append(new_message)
        if next_speaker in AGENTS and AGENTS[next_speaker].turn_evidence:
            evidence = AGENTS[next_speaker].turn_evidence[-1]
        evidence = dict(evidence, routing=deepcopy(state.get("_routing", {})))
        publish_turn(state, next_speaker, filepath, evidence)


    
    
    return state


def controlled_init(state: BookAgentState):
    return _initialize_session(state)


def controlled_assistant(state: BookAgentState, last:int, force_question:Optional[bool]=False):
    # make 1 message in the discussion
    
    
    next_speaker = choose_next(state,last)
    evidence = {}
    
    if next_speaker == "President": #Make the president speak and fine a new topic
        last=0 
        new_topic,new_message,filepath = president(
            state["messages"],
            state["book"],
            state["topics_history"],
            evidence=evidence,
        )
        state["topics_history"].append(new_topic)
    
        
    
    else: #Make a participant speak
        last+=1

        
        new_message,filepath = make_speak(
            state["messages"],
            state["book"],
            state["topics_history"],
            next_speaker, force_question, state["user_name"]
        )

    state["messages"].append(new_message)
    if next_speaker in AGENTS and AGENTS[next_speaker].turn_evidence:
        evidence = AGENTS[next_speaker].turn_evidence[-1]
    evidence = dict(evidence, routing=deepcopy(state.get("_routing", {})))
    publish_turn(state, next_speaker, filepath, evidence)

    
    
    return state , next_speaker, filepath

