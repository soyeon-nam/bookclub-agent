from agent_main import controlled_init,controlled_assistant
from agentstate import BookAgentState
import os
from agent_main import init, assistant
from tools import append_line_to_file
from action import choose_action
from participant import rollback_agents
#This is a simple controlled discussion loop script
# It allows to control when the discussion continues, and to add user messages or rollback the discussion by a number of messages
# Must be use for a controlled discussion when a real play can interact




# Build an empty BookAgentState (all fields empty / None as requested)
state: BookAgentState = {
    "messages": [],
    "book": "pinocchio",
    "book_summary": "",
    "participants_name": [],
    "user_name": "Soyeon",
    "topics_hystory": [],
}
audio_action=[] #List of [Character_name, audio_file_path,action]
state = controlled_init(state)
audio_action.append(["President", os.path.join("dialogue_audio", "President_Introduction.wav"), "Normal 1"])

last=0
last_user=0
while True:
    user_input = input("Continuation of the discussion? (type 'end' to stop, 'c' to continue, a sentence for a user message): ")
    # Check if the user wants to exit
    if user_input.lower() == "end":
        print("End of the discussion.")
    elif user_input.lower() == "c":
        force=False
        if last_user>10:
            last_user=0
            force=True

        
        state,speaker,filepath=controlled_assistant(state,last,force)
        print(state["messages"][-1])
        action = choose_action(state["messages"][-1] if state["messages"] else "")
        audio_action.append([speaker,filepath, action])
        print(f"{speaker} spoke. Audio file at: {filepath}, action chosen: {action}")   
        if force:
            print(f"Wait for the participant {state['user_name']} to answer the question.")
        
    elif user_input.strip() == "replace":
        user_input = input("Number of roleback sentences wanted?: ")
        num_rollback = int(user_input)
        if num_rollback >= len(state["messages"]):
            print("Cannot rollback that many messages.")
        else:
            removed = state["messages"][-num_rollback:]
            state["messages"] = state["messages"][:-num_rollback]
            rollback_agents(removed)
    else:
        # User message
        speaker=state["user_name"]
        user_message = f"{state['user_name']}: {user_input}\n"
        state["messages"].append(user_message)
        append_line_to_file("book_club_conversation.txt", user_message)
        print(f"{speaker} spoke. No audio file generated for user messages.")
    if speaker!="President":
        last+=1
    if speaker=="President":
        last=0
    if speaker!=state["user_name"]:
        last_user+=1
    if speaker==state["user_name"]:
        last_user=0
       


