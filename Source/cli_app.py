"""
cli_app.py – Lightweight CLI for MediBot.
Usage: python Source/cli_app.py [--show-sources]
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached


def main(show_sources: bool = False):
    print("\n💊 MediBot – Medical FAQ CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    try:
        print("Loading pipeline…")
        vectorstore = load_vector_store()
        qa_chain = initialize_rag_chain(vectorstore)
        print("Ready!\n")
    except (FileNotFoundError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! Stay healthy 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! Stay healthy 👋")
            break

        answer, sources = ask_question_cached(qa_chain, user_input, user_id="cli_user")
        print(f"\nMediBot: {answer}\n")

        if show_sources:
            print("Sources:")
            for doc in sources:
                q = doc.metadata.get("question", "N/A")
                src = doc.metadata.get("source", "N/A")
                print(f"  - {q}  [{src}]")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediBot CLI")
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()
    main(show_sources=args.show_sources)