from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o")

action_list = ["Normal 1","Normal 2", "Questionning", "Thinking", "Angry", "Happy", "Sad" ]
action_map = {name: i + 1 for i, name in enumerate(action_list)}

def choose_action(speach: str) -> str:
    
    message = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You are an assistant that help to choose the best action for someone character during his speach to do based on his speach. "
                        "You have the choice between the following actions: " + ", ".join(action_list) + ". "
                        "Based on the following speach: " + speach + " "
                        "Choose the most appropriate action that fit the most with the speach. Choose normal 1 or 2 actions for normal speach, and more emotional actions for very emotional speach. "
                        "Only return the name of the action without any additional explanation."
         ) }])]
            

    # Call the llm model
    response = llm.invoke(message)
    action =response.content.strip()


    return action

def action_to_id(action: str) -> int:
    return action_map.get(action, 0)

# print(choose_action("Hello everyone , welcome to this book club."))
# print(choose_action("Yes, i think that this book is really interesting and i loved the characters development."))
# print(choose_action("Oui clairement , mais bon je trouve que valjean pourrais etre plus developpé sur le point de la mélancolie"))
# print(choose_action("I don't like this book at all, it's boring and the plot is weak."))
# print(choose_action("The themes explored in this book are profound and thought-provoking."))
# print(choose_action("I am so happy to have read this book, it brought me joy and inspiration."))
# print(choose_action("This book made me sad, it touched on some really emotional topics."))
# print(choose_action("I am angry about the way the story ended, it felt unfair to the characters."))
# print(choose_action("I need to think more about the implications of the book's message."))
# print(choose_action("Can someone explain the symbolism used in the book?"))
# print(choose_action("What do you think about the character's motivations?"))
# print(choose_action("How does the setting influence the story?"))
# print(choose_action("What are your thoughts on the author's writing style?"))
