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
from participant import participant_creation, reset_agents
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


def init(state: BookAgentState):
    reset_agents()
    init_rag(state["book"])
    participants=participant_creation()
    participants_name=participants.keys()
    state["participants_name"]=list(participants_name)
    state["participants_name"].append(state["user_name"])
    # Generate a short summary using the configured lightweight GPT model
    summary_prompt = "write a short summary of 100 words about "+state["book"] #+"by"+state["author"]
    summary_resp = llm.invoke([HumanMessage(content=summary_prompt)]) # SystemMessage 수정
    
    book_summary =  summary_resp.content.strip()
    state["book_summary"]=book_summary

    mes="Welcome everyone! Lets talk about "+ state["book"] +", to start, did you guys liked the book?"
    presmes="President: "+mes   
    messages= [presmes]

    state["topics_history"]=["General impression of the book"]
    tts_generate(mes, "president", os.path.join("dialogue_audio", "President_Introduction.wav"))
    append_line_to_file("book_club_conversation.txt", presmes)
    state["messages"]=messages
    return state


def assistant(state: BookAgentState):
    # make the discussion run until 100 messages are reached
    last=0 #Number of messages since the last president intervention
    while len(state["messages"])<state.get("max_messages", 100):
        next_speaker = choose_next(state,last)
        
        if next_speaker == "President": #Make the president speak and fine a new topic
            last=0 
            new_topic,new_message,_ = president(
                state["messages"],
                state["book"],
                state["topics_history"],
            )
            state["topics_history"].append(new_topic)
        
        elif next_speaker==state["user_name"]: #Make the user speak
            last+=1
            new_message=user_call(state)
            
        
        else: #Make a participant speak
            last+=1
            new_message, _ = make_speak(
                state["messages"],
                state["book"],
                state["topics_history"],
                next_speaker,
            )

        state["messages"].append(new_message)


    
    
    return state


def controlled_init(state: BookAgentState):
    reset_agents()
    init_rag(state["book"])
    #Does'nt activate the fact that the user is part of the participants
    participants=participant_creation()
    participants_name=participants.keys()
    state["participants_name"]=list(participants_name)
    # Generate a short summary using the configured lightweight GPT model
    summary_prompt = "write a short summary of 100 words about "+state["book"]
    summary_resp = llm.invoke([SystemMessage(content=summary_prompt)])
    
    book_summary =  summary_resp.content.strip()
    state["book_summary"]=book_summary

    mes="Welcome everyone ! Lets talk about "+ state["book"] +", to start, did you guys liked the book?"
    presmes="President: "+mes   
    messages= [presmes]

    state["topics_history"]=["General impression of the book"]
    tts_generate(mes, "president", os.path.join("dialogue_audio", "President_Introduction.wav"))
    append_line_to_file("book_club_conversation.txt", presmes)
    state["messages"]=messages
    return state

def controlled_assistant(state: BookAgentState, last:int, force_question:Optional[bool]=False):
    # make 1 message in the discussion
    
    
    next_speaker = choose_next(state,last)
    
    if next_speaker == "President": #Make the president speak and fine a new topic
        last=0 
        new_topic,new_message,filepath = president(
            state["messages"],
            state["book"],
            state["topics_history"],
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


    
    
    return state , next_speaker, filepath

