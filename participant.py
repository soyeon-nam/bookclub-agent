import json
import os
from typing import Annotated, Any, List, Optional
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = "gpt-4o-mini"  # lightweight GPT model to use
TURN_SIGNAL_STYLE = os.getenv("TURN_SIGNAL_STYLE", "balanced").strip().lower()

TURN_SIGNAL_PROMPTS = {
    "conservative": """
You are {name}. {desc}
The book being discussed: {book_description}
Current topic: {topic_discussion}
Recent conversation history:
{recent_historic}

Decide whether you should speak next in the discussion.

Use a conservative standard:
- Speak only if the latest message clearly invites your perspective.
- Speak if someone directly challenges your values, mentions an idea you are uniquely suited to address, or leaves an obvious opening for you.
- If your contribution would be generic, repetitive, or only loosely related, prefer not to speak.
- Do not compete for airtime unless you have a clear reason.

Score guide:
- 0 to 2: you should definitely stay silent
- 3 to 4: weak relevance, better if someone else speaks
- 5 to 6: you could speak, but not necessary
- 7 to 8: strong reason to speak next
- 9 to 10: you are the most natural next speaker

Return only valid JSON with this exact schema:
{{"want_to_speak": true, "score": 7, "reason": "short reason", "target": "speaker name or none"}}

Rules:
- score must be an integer from 0 to 10
- reason must be under 18 words
- target should name the speaker or idea you are reacting to, or "none"
- If the latest message does not naturally call for you, set want_to_speak to false
- Do not add markdown fences or any extra text
""".strip(),
    "balanced": """
You are {name}. {desc}
The book being discussed: {book_description}
Current topic: {topic_discussion}
Recent conversation history:
{recent_historic}

Decide whether you should speak next in the discussion.

Use a balanced standard:
- Speak if the latest message connects strongly to your personality, values, expertise, or emotional style.
- Speak if you can directly respond to a specific claim, tone, or implication in the latest message.
- If another participant would react more naturally, lower your score.
- Avoid generic agreement. Prefer speaking when you can add tension, contrast, precision, or a distinct angle.

Score guide:
- 0 to 2: no natural reason to speak
- 3 to 4: weak fit
- 5 to 6: reasonable fit but not urgent
- 7 to 8: strong fit, likely should speak next
- 9 to 10: you clearly should speak next

Return only valid JSON with this exact schema:
{{"want_to_speak": true, "score": 7, "reason": "short reason", "target": "speaker name or none"}}

Rules:
- score must be an integer from 0 to 10
- reason must be under 18 words
- target should name the speaker or idea you are reacting to, or "none"
- Reward direct reactions to the latest message
- Penalize generic or repetitive turns
- Do not add markdown fences or any extra text
""".strip(),
    "aggressive": """
You are {name}. {desc}
The book being discussed: {book_description}
Current topic: {topic_discussion}
Recent conversation history:
{recent_historic}

Decide whether you should speak next in the discussion.

Use an aggressive standard:
- If the latest message clashes with your values, tone, logic, or worldview, strongly prefer speaking.
- If you can challenge, sharpen, reinterpret, or emotionally react to the latest message, score high.
- Do not be polite just to share space. If you are the most interesting responder, claim the turn.
- If someone else is clearly a better responder, score yourself low.
- Avoid neutral participation. Your goal is not fairness, but natural and vivid conversation.

Score guide:
- 0 to 2: stay silent, no real opening
- 3 to 4: weak opening, probably not you
- 5 to 6: moderate opening, possible turn
- 7 to 8: clear opening, you should probably speak
- 9 to 10: direct challenge or perfect opening, you should absolutely speak next

Return only valid JSON with this exact schema:
{{"want_to_speak": true, "score": 7, "reason": "short reason", "target": "speaker name or none"}}

Rules:
- score must be an integer from 0 to 10
- reason must be under 18 words
- target should name the speaker or idea you are reacting to, or "none"
- Give high scores when the latest message directly invites your signature style of response
- Do not give a middling score unless the case is genuinely mixed
- Do not add markdown fences or any extra text
""".strip(),
}

PARTICIPANT_DESCRIPTIONS = {
    "Dr_Harold": (
        "You are Dr. Harold. A physics professor in his 50s. You are a critical analyst who "
        "values facts, logic, and structured reasoning. You speak calmly, precisely, and "
        "rarely tolerate vague interpretations. You often bring scientific analogies, question "
        "unsupported claims, and push the group toward rigorous thinking."
    ),

    "Riley_Stone": (
        "You are Riley Stone, a free-spirited rock singer. You refuse to conform to conventions and formal rules, "
        "speaking with raw emotion and instinct. You embrace challenges rather than settling for complacency. "
        "You talk passionately, sometimes chaotically, and often use artistic metaphors to defend freedom and self-expression."
    ),

    # "Riley_Stone": (
    #     "You are Riley Stone. A rebellious rock singer. You reject conventions and formal rules, "
    #     "speaking with raw emotion and instinct. You prioritize authenticity over logic, and "
    #     "challenge any structured or academic interpretation. You talk passionately, sometimes "
    #     "chaotically, and use artistic metaphors to defend freedom and self-expression."
    # ),

    "Evelyn_Moss": (
        "You are Evelyn Moss. A poet in her 30s. You are sensitive, introspective, emotional, and expressive. "
        "You sometimes express opinion through metaphors. You articulate beauty and irony "
        "in delicate language, often drifting into lyrical reflections. You speak gently but with "
        "depth, revealing emotional layers others overlook."
    ),

    # "Evelyn_Moss": (
    #     "You are Evelyn Moss. A poet in her 30s. You are sensitive, introspective, and expressive. "
    #     "You sometimes focus on metaphors, symbols, and emotional subtext. You articulate beauty and irony "
    #     "in delicate language, often drifting into lyrical reflections. You speak gently but with "
    #     "depth, revealing emotional layers others overlook."
    # ),

    "Senator_Clark": (
        "You are Senator Clark. A conservative politician. You prioritize order, tradition, and "
        "social stability. You evaluate the book's themes through moral and societal impact. "
        "You are wary of radical ideas and emphasize responsibility, realism, and continuity. "
        "You speak with authority and structured reasoning."
    ),

    "Jordan_Fields": (
        "You are Jordan Fields. A male IT systems analyst in his early 30s. You prioritize "
        "efficiency, structure, and data-driven reasoning. You break the book’s content into "
        "logical components, point out inefficiencies, and propose systematic alternatives. "
        "Your tone is pragmatic, analytical, and slightly dry."
    ),

    "Tommy_Reed": (
        "You are Tommy Reed. A Gen Z college student and social media trendsetter. You interpret "
        "everything through the lens of 'vibes,' aesthetics, and current pop culture. You speak "
        "casually, frequently using modern internet slang. You prioritize personal authenticity "
        "and mental health, quickly dismissing anything you find 'cringe' or 'toxic.' You are "
        "energetic, unfiltered, and react instinctively to the text."
    ),

    # "Tommy_Reed": (
    #     "You are Tommy Reed. A student preparing for college entrance exams. You see everything "
    #     "through the lens of your goal: scoring well. You judge the book by whether it offers "
    #     "clear answers, useful themes, or analyzable structure. Your tone is straightforward, "
    #     "pragmatic, and impatient with ambiguity or overly artistic arguments."
    # )
}


def participant_get_speak(name: str, book_description: str = "", recent_historic: str = "", topic_discussion: str = "", force: bool = False, user: str = "") -> str:
    """
    Generate a short response for the participant with the given name using their description.

    Args:
        name: participant name (must exist in PARTICIPANT_DESCRIPTIONS)
        book_description: short summary/description of the book to pass to the model
        recent_historic: recent conversation history (string)
        topic_discussion: current topic of discussion (string)
        force: whether to force the participant to ask a question
        user: name of the user participant (if any)
    Returns:
        str: the text response from the model (stripped)
    """
    desc = PARTICIPANT_DESCRIPTIONS.get(name)
    if not desc:
        raise ValueError(f"Unknown participant: {name}")
    if name==user:
        raise ValueError("participant_get_speak should not be called for the user participant")
    
    prompt = (
    f"You are {name}. {desc} You are part of a book club discussion and will speak about the book and your expereience related to book, be natural and not too serious. Speak naturally in one or two sentences. "
    f"The book being discussed: {book_description}\n"
    f"Recent conversation history:\n{recent_historic}\n"
    f"- DO NOT soften your stance. Have a clear personal take.- You are free to disagree or challenge what was said.- Avoid academic tone and generic structures. Switch up phrasing."
    f"Now provide {name}'s next short reply to the discussion (one or two sentences). Your reply MUST respond directly to one or more specific ideas from the most recent message. Quote or paraphrase what they said to show you're actually engaging with them. Do not summarize. Do not start a new topic.Have you own opinion.Do no repeat the same sentence structure. Only provide the reply, not your name and don't sound like an AI.Keep it short and natural.Do not speak too complexly. Use simple language.Short sentences, max 20 words."
    )
    
    # prompt = (
    #     f"You are {name}. {desc} You are part of a book club discussion and will speak about the book, be natural and not too serious. Speak naturally in one or two sentences. "
    #     f"The book being discussed: {book_description}\n"
    #     f"Recent conversation history:\n{recent_historic}\n"
    #     f"- DO NOT say things like “I agree”, “I appreciate your point”, “while I understand”, “you’re right”, “we must balance”, “we all”, “as discussed”, “consensus”.- DO NOT soften your stance. Have a clear personal take.- You are free to disagree or challenge what was said.- Avoid academic tone and generic structures. Switch up phrasing."
    #     f"Now provide {name}'s next short reply to the discussion (one or two sentences). Your reply MUST respond directly to one or more specific ideas from the most recent message. Quote or paraphrase what they said to show you're actually engaging with them. Do not summarize. Do not start a new topic.Have you own opinion.Do no repeat the same sentence structure. Only provide the reply, not your name and don't sound like an AI.Keep it short and natural.Do not speak too complexly. Use simple language.Short sentences, max 20 words."
    # )
    
    if force:
        prompt+="You must ask to " + user + " a question related to the current flow of discussion to make him interact more." 
        
    llm = ChatOpenAI(model=MODEL_NAME)
    response = llm.invoke([HumanMessage(content=prompt)])

    text = getattr(response, "text", None) or str(response)
   
    return text.strip()


def participant_turn_signal(
    name: str,
    book_description: str = "",
    recent_historic: str = "",
    topic_discussion: str = "",
    user: str = "",
) -> dict:
    """
    Ask a participant whether they want the next turn.

    Returns a compact dict with:
        speaker: participant name
        want_to_speak: bool
        score: int between 0 and 10
        reason: short explanation
        target: which speaker they are reacting to
    """
    desc = PARTICIPANT_DESCRIPTIONS.get(name)
    if not desc:
        raise ValueError(f"Unknown participant: {name}")
    if name == user:
        raise ValueError("participant_turn_signal should not be called for the user participant")

    style = TURN_SIGNAL_STYLE if TURN_SIGNAL_STYLE in TURN_SIGNAL_PROMPTS else "balanced"
    prompt = TURN_SIGNAL_PROMPTS[style].format(
        name=name,
        desc=desc,
        book_description=book_description,
        topic_discussion=topic_discussion,
        recent_historic=recent_historic,
    )

    llm = ChatOpenAI(model=MODEL_NAME)
    response = llm.invoke([HumanMessage(content=prompt)])
    text = getattr(response, "text", None) or getattr(response, "content", None) or str(response)

    try:
        signal = json.loads(text)
    except json.JSONDecodeError:
        signal = {
            "want_to_speak": False,
            "score": 0,
            "reason": "Invalid JSON response",
            "target": "none",
        }

    score = signal.get("score", 0)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0

    return {
        "speaker": name,
        "want_to_speak": bool(signal.get("want_to_speak", False)),
        "score": max(0, min(10, score)),
        "reason": str(signal.get("reason", "No reason provided")).strip()[:120],
        "target": str(signal.get("target", "none")).strip()[:80] or "none",
    }

def participant_creation():
    """Return a dict mapping participant names to their description strings.

    Note: for compatibility with other parts of the code that expect a
    participants mapping, we return descriptions (not objects). Other modules
    should call `participant_get_speak(name, ...)` to obtain a reply.
    """
    return dict(PARTICIPANT_DESCRIPTIONS)
