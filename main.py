from agent_main import init,assistant
from agentstate import BookAgentState

from agent_main import init, assistant
# Build an empty BookAgentState (all fields empty / None as requested)
state: BookAgentState = {
    "messages": [],
    "book": "Les misérables",
    "book_summary": "",
    "participants_name": [],
    "user_name": "Léonard",
    "topics_hystory": [],
}

state = init(state)

state=assistant(state)
