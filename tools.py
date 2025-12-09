import ast
import json
import shutil
from langchain_core.tools import tool
from typing import List, Tuple
from agentstate import BookAgentState
from participant import participant_get_speak, participant_turn_signal
from typing import Union, Tuple
import requests
import os
from typing import TypedDict, List, Dict, Any, Optional
import io
import base64
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import HumanMessage
from agentstate import BookAgentState
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from tts import tts_generate
import datetime


from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()
# Use a lightweight GPT model via langchain_openai
MODEL_NAME = os.getenv("GPT_MODEL", "gpt-4o-mini")


from langchain_core.messages import AIMessage

def choose_next(state,last) -> str:
    """
    Choose the next participant to speak in the book club.

    Args:
        state (BookAgentState): The current state of the book club discussion.
        last (int): Number of messages since the last president intervention
        

    Returns:
        new_message (str): The name of the next participant to speak.
    """
    recent_messages = state["messages"][-6:]
    topic = state.get("topics_history", [])
    current_topic = topic[-1] if topic else "General impression of the book"

    candidate_names = [
        name for name in state["participants_name"]
        if name != state.get("user_name")
    ]

    signals = []
    for name in candidate_names:
        try:
            signal = participant_turn_signal(
                name=name,
                book_description=state["book"],
                recent_historic=recent_messages,
                topic_discussion=current_topic,
                user=state.get("user_name", ""),
            )
        except Exception as exc:
            signal = {
                "speaker": name,
                "want_to_speak": False,
                "score": 0,
                "reason": f"Signal error: {exc}",
                "target": "none",
            }
        signals.append(signal)

    president_allowed = last > 10
    prompt = (
        f"You are the central discussion router for a book club about '{state['book']}'.\n"
        f"Current topic: {current_topic}\n"
        f"Recent conversation history: {recent_messages}\n"
        f"Participant turn signals: {json.dumps(signals, ensure_ascii=True)}\n"
        f"President eligible this turn: {president_allowed}\n\n"
        "Choose exactly one next speaker.\n"
        "Priority rules:\n"
        "- Prefer participants with higher scores when their reason clearly connects to the latest message.\n"
        "- Vary the flow naturally and avoid repetitive back-and-forth patterns.\n"
        "- Choose President only if president eligible is true and the discussion seems stale, blocked, or needs a new angle.\n"
        "- Never choose the user.\n"
        f"Return only one name from this list: {', '.join(candidate_names)}, President."
    )
    llm = ChatOpenAI(model=MODEL_NAME)
    response = llm.invoke([HumanMessage(content=prompt)])

    text = (getattr(response, "text", None) or getattr(response, "content", None) or str(response)).strip()
    cleaned = text.splitlines()[0].strip().strip(".:,;!?'\"")

    valid_choices = set(candidate_names + ["President"])
    if cleaned in valid_choices:
        return cleaned

    for choice in valid_choices:
        if choice in text:
            return choice

    ranked_signals = sorted(
        signals,
        key=lambda signal: (signal["want_to_speak"], signal["score"]),
        reverse=True,
    )
    if ranked_signals and ranked_signals[0]["want_to_speak"]:
        return ranked_signals[0]["speaker"]

    return "President" if president_allowed else candidate_names[0]



def make_speak(messages:list[str] ,book_name: str, topic_discussion: str, name: str, force: bool = False, user: str = "") -> str:
    """
    Make a participant of the book club speak in the book club.

    Args:
        messages (list[str]): list of the 5 more Recents messages (if they exists)in the book club.
        book_name (str): The name of the book being discussed.
        topic_discussion (str): The current topic of discussion.
        name (str): The name of the participant who will speak.

    Returns:
        new_message (str): The new message added to the book club discussion.
        filepath: path to the audio file generated
    """
    print(f"Making participant {name} speak...")
    # Generate message
    if len(messages)>4:
        recent=messages[-4:]
    else:
        recent=messages
    
    participant_message = participant_get_speak(
        name=name,
        book_description=book_name,
        recent_historic=recent,
        topic_discussion=topic_discussion[-1],
        force=force,
        user=user
    )
    output_dir = "dialogue_audio"

# inside your loop / function
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{name}_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)

    tts_generate(participant_message, name, filepath)


    new_message = f"{name}: {participant_message}\n"
    append_line_to_file("book_club_conversation.txt", new_message)

    # Return a partial update, not full state
    
    return (new_message,filepath)


def president(messages: list[str], book_name: str, topic_discussion: str) -> str:
    """
    Make the president of the book club speak about a topic. 

    Args:
        messages (list[str]): recent messages (used for context)
        book_name (str): the book title
        topic_discussion (str): the topic to introduce

    Returns:
        new_topic: the new topic introduced by the president (string)
        new_message: the new message produced by the president (string)
        filepath: path to the audio file generated (string)
    """

    # Keep recent history small for prompt clarity
    recent = messages[-4:] if len(messages) > 4 else messages

    # Find the new subject 

    prompt = (
        f"A disccusion is taking place in a book club about the book '{book_name}'. "
        f"The previous topic of discussion are '{topic_discussion}'. "
        f"Based on the recent conversation history:\n{chr(10).join(recent)}\n"
        f"Find a new consice, engaging topic that encourages discussion among participants."
        f"Only say the new topic."
    )

    llm = ChatOpenAI(model=MODEL_NAME)
    response = llm.invoke([HumanMessage(content=prompt)])
    text = getattr(response, "text", None) or str(response)
    new_topic = text.strip()
    topic_discussion.append(new_topic)


    # Generate message
    prompt = (
        f"You are the president of a book club discussing '{book_name}'. "
        f"Introduce a concise, engaging question or topic about '{topic_discussion[-1]}' that encourages discussion among participants. "
        f"Use one or two sentences and speak like a real human. Do not add extra explanation. "
        f"Recent conversation history:\n{chr(10).join(recent)}"
    )

    llm = ChatOpenAI(model=MODEL_NAME)
    response = llm.invoke([HumanMessage(content=prompt)])
    text = getattr(response, "text", None) or str(response)
    text = text.strip()

    output_dir = "dialogue_audio"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"president_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    tts_generate(text, "president", filepath)

    new_message = f"president: {text}\n"
    append_line_to_file("book_club_conversation.txt", new_message)

    # Return the produced message string (tools are expected to update state)
    return new_topic,new_message,filepath

def append_line_to_file(filename: str, text: str):
    """Append one line of text to a .txt file."""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def user_call(state: BookAgentState) -> str:
    """Take a text input from the user.
    returns:
        str: The message entered by the user.
        
    """

    user_text = input("Enter your text: ")

    new_message = f"{state['user_name']}: {user_text}\n"
    append_line_to_file("book_club_conversation.txt", new_message)

    # Return the user message
    return new_message
