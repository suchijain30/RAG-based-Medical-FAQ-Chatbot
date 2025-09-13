import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached

def main(show_sources: bool = False):
    print("\n💊 MediBot – Medical FAQ CLI Chatbot")
    print("Type your medical question and press Enter.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    # Load RAG pipeline
    print("Initializing MediBot, please wait...")
    qa_chain = load_vector_store()  # FAISS + embeddings
    qa_chain = initialize_rag_chain(qa_chain)
    print("MediBot is ready!\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting MediBot. Stay healthy! 👋")
            break

        if not user_input:
            continue

        answer, sources = ask_question_cached(qa_chain, user_input)
        print(f"MediBot: {answer}\n")

        if show_sources:
            print("Retrieved Source Chunks:")
            for doc in sources:
                q = doc.metadata.get("question", "N/A")
                src = doc.metadata.get("source", "N/A")
                print(f"- Original Question: {q} | Source ID: {src}")
            print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediBot CLI")
    parser.add_argument("--show-sources", action="store_true",
                        help="Display retrieved source chunks with answers")
    args = parser.parse_args()
    main(show_sources=args.show_sources)