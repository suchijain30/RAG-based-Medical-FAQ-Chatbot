# # import streamlit as st
# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# # from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached

# # # Streamlit Page Configuration
# # st.set_page_config(
# #     page_title="MediBot – Medical FAQ Chatbot",
# #     page_icon="💊",
# #     layout="centered"
# # )

# # # Load RAG Pipeline (Cached)
# # @st.cache_resource
# # def load_pipeline():
# #     vectorstore = load_vector_store()
# #     qa_chain = initialize_rag_chain(vectorstore)
# #     return qa_chain

# # with st.spinner("Initializing MediBot, please wait..."):
# #     qa_chain = load_pipeline()


# # # Header & Introduction
# # st.markdown(
# #     """
# #     <div style='text-align:center;'>
# #         <h1>🩺 MediBot – Health Assistant</h1>
# #         <p style='font-size:16px; color:#555; line-height:1.6; margin-top:-10px;'>
# #             Ask any medical-related question based on our FAQ database.<br>
# #             ⚠️ MediBot provides informational answers only and does not offer personal medical advice.
# #         </p>
# #     </div>
# #     """, unsafe_allow_html=True
# # )
# # st.markdown("---")

# # # Session State Initialization
# # if "history" not in st.session_state:
# #     st.session_state.history = []

# # # Clear Chat Button
# # if st.button("Clear Chat"):
# #     st.session_state.history = []

# # # User Input
# # user_question = st.text_input("Type your question here:")

# # if st.button("Ask") and user_question.strip():
# #     with st.spinner("Generating answer..."):
# #         answer, sources = ask_question_cached(qa_chain, user_question)
# #         st.session_state.history.append({
# #             "question": user_question,
# #             "answer": answer,
# #             "sources": sources
# #         })

# # # Display Chat History (Newest First)
# # for chat in reversed(st.session_state.history):
# #     # User message
# #     st.markdown(f"<b style='color:#1f77b4'>You:</b> {chat['question']}", unsafe_allow_html=True)
# #     # Bot answer
# #     st.markdown(f"<b style='color:#2ca02c'>MediBot:</b> {chat['answer']}", unsafe_allow_html=True)
    
# #     # Collapsible section for retrieved sources
# #     with st.expander("View retrieved source chunks"):
# #         for doc in chat.get('sources', []):
# #             q = doc.metadata.get("question", "N/A")
# #             src = doc.metadata.get("source", "N/A")
# #             st.markdown(f"- Original Question: {q} | Source ID: {src}")
    
# #     st.markdown("---")

# import streamlit as st
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached
# from embeddings import create_vector_store  # import your embedding creation function

# # Streamlit Page Configuration
# st.set_page_config(
#     page_title="MediBot – Medical FAQ Chatbot",
#     page_icon="💊",
#     layout="centered"
# )

# # Load RAG Pipeline (Cached)
# @st.cache_resource
# def load_pipeline():
#     # Automatically create FAISS index if missing
#     faiss_index_file = r"D:\data_sci_code\RAG_ChatBot\VectorStore\medical_faq_index"
#     index_path = os.path.join(faiss_index_file, "index.faiss")
    
#     if not os.path.exists(index_path):
#         st.info("FAISS index not found. Creating a new one, this may take a few minutes...")
#         create_vector_store()  # This will create and save the FAISS index
    
#     vectorstore = load_vector_store()
#     qa_chain = initialize_rag_chain(vectorstore)
#     return qa_chain

# with st.spinner("Initializing MediBot, please wait..."):
#     qa_chain = load_pipeline()

# # Header & Introduction
# st.markdown(
#     """
#     <div style='text-align:center;'>
#         <h1>🩺 MediBot – Health Assistant</h1>
#         <p style='font-size:16px; color:#555; line-height:1.6; margin-top:-10px;'>
#             Ask any medical-related question based on our FAQ database.<br>
#             ⚠️ MediBot provides informational answers only and does not offer personal medical advice.
#         </p>
#     </div>
#     """, unsafe_allow_html=True
# )
# st.markdown("---")

# # Session State Initialization
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Clear Chat Button
# if st.button("Clear Chat"):
#     st.session_state.history = []

# # User Input
# user_question = st.text_input("Type your question here:")

# if st.button("Ask") and user_question.strip():
#     with st.spinner("Generating answer..."):
#         answer, sources = ask_question_cached(qa_chain, user_question)
#         st.session_state.history.append({
#             "question": user_question,
#             "answer": answer,
#             "sources": sources
#         })

# # Display Chat History (Newest First)
# for chat in reversed(st.session_state.history):
#     # User message
#     st.markdown(f"<b style='color:#1f77b4'>You:</b> {chat['question']}", unsafe_allow_html=True)
#     # Bot answer
#     st.markdown(f"<b style='color:#2ca02c'>MediBot:</b> {chat['answer']}", unsafe_allow_html=True)
    
#     # Collapsible section for retrieved sources
#     with st.expander("View retrieved source chunks"):
#         for doc in chat.get('sources', []):
#             q = doc.metadata.get("question", "N/A")
#             src = doc.metadata.get("source", "N/A")
#             st.markdown(f"- Original Question: {q} | Source ID: {src}")
    
#     st.markdown("---")


import streamlit as st
import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached
from embeddings import create_vector_store  # import your embedding creation function

# Streamlit Page Configuration
st.set_page_config(
    page_title="MediBot – Medical FAQ Chatbot",
    page_icon="💊",
    layout="centered"
)

# Load RAG Pipeline (Cached)
@st.cache_resource
def load_pipeline():
    # Automatically create FAISS index if missing
    faiss_index_file = r"D:\data_sci_code\RAG_ChatBot\VectorStore\medical_faq_index"
    index_path = os.path.join(faiss_index_file, "index.faiss")
    
    if not os.path.exists(index_path):
        st.info("FAISS index not found. Creating a new one, this may take a few minutes...")
        create_vector_store()  # This will create and save the FAISS index
    
    vectorstore = load_vector_store()
    qa_chain = initialize_rag_chain(vectorstore)
    return qa_chain

with st.spinner("Initializing MediBot, please wait..."):
    qa_chain = load_pipeline()

# Header & Introduction
st.markdown(
    """
    <div style='text-align:center;'>
        <h1>🩺 MediBot – Health Assistant</h1>
        <p style='font-size:16px; color:#555; line-height:1.6; margin-top:-10px;'>
            Ask any medical-related question based on our FAQ database.<br>
            ⚠️ MediBot provides informational answers only and does not offer personal medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("---")

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []

# Clear Chat Button
if st.button("Clear Chat"):
    st.session_state.history = []

# User Input
user_question = st.text_input("Type your question here:")

if st.button("Ask") and user_question.strip():
    with st.spinner("Generating answer..."):
        answer, sources = ask_question_cached(qa_chain, user_question)
        st.session_state.history.append({
            "question": user_question,
            "answer": answer,
            "sources": sources
        })

# Display Chat History (Newest First)
for chat in reversed(st.session_state.history):
    # User message
    st.markdown(f"<b style='color:#1f77b4'>You:</b> {chat['question']}", unsafe_allow_html=True)
    # Bot answer
    st.markdown(f"<b style='color:#2ca02c'>MediBot:</b> {chat['answer']}", unsafe_allow_html=True)
    
    # Collapsible section for retrieved sources
    with st.expander("View retrieved source chunks"):
        for doc in chat.get('sources', []):
            q = doc.metadata.get("question", "N/A")
            src = doc.metadata.get("source", "N/A")
            st.markdown(f"- Original Question: {q} | Source ID: {src}")
    
    st.markdown("---")