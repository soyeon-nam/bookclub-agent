"""Interactive book-club discussion with synchronized memory rollback."""
from copy import deepcopy
import os

from agent_main import controlled_init, controlled_assistant
from action import choose_action
from participant import rollback_agents
from tools import append_line_to_file


def turns_since(messages, speaker):
    for count, message in enumerate(reversed(messages)):
        if message.split(":", 1)[0].strip().lower() == speaker.lower():
            return count
    return len(messages)


def run_cli():
    state = {
        "messages": [], "book": "pinocchio", "book_summary": "",
        "participants_name": [], "user_name": "Soyeon", "topics_history": [],
    }
    # Each invocation is a new session; rollback also updates this transcript.
    with open("book_club_conversation.txt", "w", encoding="utf-8"):
        pass
    state = controlled_init(state)
    audio_action = [["President", os.path.join("dialogue_audio", "President_Introduction.wav"), "Normal 1"]]
    snapshots = []
    while True:
        try:
            command = input("Continue: c | end | replace | your message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("End of the discussion.")
            return state
        if command.lower() == "end":
            print("End of the discussion.")
            return state
        if not command:
            continue
        if command.lower() == "replace":
            try:
                count = int(input("Number of messages to roll back: "))
            except ValueError:
                print("Enter a positive whole number.")
                continue
            except (EOFError, KeyboardInterrupt):
                print("End of the discussion.")
                return state
            if count <= 0 or count > len(snapshots):
                print("Cannot roll back that many messages; keep the opening message.")
                continue
            removed = state["messages"][-count:]
            rollback_agents(removed)
            state, audio_count = snapshots[-count]
            del snapshots[-count:]
            del audio_action[audio_count:]
            with open("book_club_conversation.txt", "w", encoding="utf-8") as transcript:
                transcript.write("\n".join(m.rstrip() for m in state["messages"]) + "\n")
            print(f"Rolled back {count} message(s).")
            continue

        before = (deepcopy(state), len(audio_action))
        if command.lower() == "c":
            last = turns_since(state["messages"], "President")
            force = turns_since(state["messages"], state["user_name"]) > 10
            state, speaker, filepath = controlled_assistant(state, last, force)
            print(state["messages"][-1])
            try:
                action = choose_action(state["messages"][-1])
            except Exception as error:
                print(f"Animation classification failed ({type(error).__name__}); using Normal 1.")
                action = "Normal 1"
            audio_action.append([speaker, filepath, action])
            print(f"{speaker} spoke. Audio file at: {filepath}, action: {action}")
            if force:
                print(f"Waiting for {state['user_name']}'s response.")
        else:
            message = f"{state['user_name']}: {command}"
            state["messages"].append(message)
            append_line_to_file("book_club_conversation.txt", message)
        snapshots.append(before)


if __name__ == "__main__":
    try:
        run_cli()
    except Exception as error:
        print(f"Discussion stopped ({type(error).__name__}). Check API access and the audio setup.")
        raise SystemExit(1)
