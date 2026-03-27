"""Run an automated discussion (message limit includes the opening)."""
import argparse
from agent_main import init, assistant
from participant import get_all_participant_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default="Les misérables")
    parser.add_argument("--user", default="Léonard")
    parser.add_argument("--max-messages", type=int, default=100)
    parser.add_argument("--participants", nargs="+", choices=get_all_participant_names())
    args = parser.parse_args()
    if args.max_messages < 1:
        parser.error("--max-messages must be positive")
    state = {
        "messages": [], "book": args.book, "book_summary": "",
        "participants_name": args.participants or [], "user_name": args.user,
        "topics_history": [], "max_messages": args.max_messages,
    }
    with open("book_club_conversation.txt", "w", encoding="utf-8"):
        pass
    state = assistant(init(state))
    print(f"Discussion complete: {len(state['messages'])} messages.")


if __name__ == "__main__":
    main()
