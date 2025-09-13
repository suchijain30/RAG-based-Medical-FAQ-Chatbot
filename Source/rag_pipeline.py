import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#paths of vector database and model name
FAISS_INDEX_FILE = r"D:\data_sci_code\RAG_ChatBot\VectorStore\medical_faq_index"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10


#prompt template for medical safety
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "input"],
    template="""
            You are a responsible and concise medical assistant AI. 
                - Use ONLY the context provided to answer the patient's question. 
                - Do NOT provide personal medical advice or unrelated information. 
            Rules:
                1. If the question is about medical topics and the answer is in the context, answer clearly and concisely.
                2. If the question is about medical topics but the answer is NOT in the context, reply: 
                      "I'm sorry, I don't have enough information in my knowledge base to answer that accurately. Please consult a healthcare professional for personalized advice."
                3. If the question is NOT about medical topics, politely reply:
                      "I'm here to answer medical-related questions only. Please ask a question about health, symptoms, treatments, or medical conditions."

Context:
{context}

Question:
{input}

Answer (max 4-5 sentences):
"""
)

#load vector store
def load_vector_store(faiss_index_file: str = FAISS_INDEX_FILE):
    print(f"Loading FAISS vector store from '{faiss_index_file}'...")
    vectordb = FAISS.load_local(
        faiss_index_file,
        embeddings=HuggingFaceEmbeddings(model_name=HF_MODEL_NAME),
        allow_dangerous_deserialization=True
    )
    print("FAISS vector store loaded successfully!")
    return vectordb

#
#initialize rag chain
def initialize_rag_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=MEDICAL_PROMPT
    )

    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return qa_chain

# manual chaching for repeated queries.
_cache = {}

def ask_question_cached(qa_chain, question: str):
    """
    Query RAG chain and cache results for repeated queries.
    """
    if question in _cache:
        return _cache[question]
    
    result = qa_chain.invoke({"input": question})
    answer = result["answer"]
    source_docs = result["context"]
    
    _cache[question] = (answer, source_docs)
    return answer, source_docs

#main execution
if __name__ == "__main__":
    # Load FAISS vector store
    vectorstore = load_vector_store()

    # Initialize RAG chain
    qa_chain = initialize_rag_chain(vectorstore)

    # Example questions
    example_questions = [
        "What are the early symptoms of diabetes?",
        "Can children take paracetamol?",
        "What foods are good for heart health?"
        "What is the birth date of Virat Kohli"
    ]

    # Ask questions and display results
    for question in example_questions:
        answer, sources = ask_question_cached(qa_chain, question)
        print("\n")
        print("Question:", question)
        print("Answer:", answer)
        print("Retrieved Source Chunks:")
        for doc in sources:
            q = doc.metadata.get("question", "N/A")
            src = doc.metadata.get("source", "N/A")
            print(f"- Original Question: {q} | Source ID: {src}")





