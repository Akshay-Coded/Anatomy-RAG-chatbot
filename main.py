import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# -----------------------------
# 🌱 Load environment variables
# -----------------------------
load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not groq_api_key:
    st.error("❌ GROQ API key not found in .env file.")
    st.stop()

# -----------------------------
# 📦 Load FAISS Index & Embedder
# -----------------------------
st.title("🧠 Anatomical-GPT")
st.markdown("All questions related to human anatomy are answered with respect to famous anatomical texts")

embedding_model = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

faiss_path = "vectorstore/faiss_index"
if not os.path.exists(faiss_path):
    st.error(f"⚠️ FAISS index not found at {faiss_path}.")
    st.stop()

vectorstore = FAISS.load_local(
    faiss_path,
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 🤖 Groq LLM
# -----------------------------
llm = ChatGroq(
    temperature=0.2,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# -----------------------------
# 🧠 RetrievalQA Chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# 💬 Chat UI
# -----------------------------
query = st.text_input("🔍 Ask a medical or anatomy question:")

# Add your custom instruction here
base_instruction = (
    "You are an expert medical tutor specialized in human anatomy. "
    "Use the provided context from authoritative anatomy texts to answer and your knowledge to build a great explanation. "
    "I want you to strictly answer the question without any extra header or footer words\n\n"
)

def is_irrelevant(sources, threshold=100):
    return all(len(doc.page_content.strip()) < threshold for doc in sources)

if query:
    # Prepend instruction to the question
    query_with_instruction = base_instruction + query

    with st.spinner("💭 Thinking..."):
        result = qa_chain.invoke({"query": query_with_instruction})
        sources = result.get("source_documents", [])

        if not sources or is_irrelevant(sources):
            st.warning("🤷 Sorry, I couldn’t find enough information in the documents to answer that.")
        else:
            st.success(result["result"])
            with st.expander("📚 View sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(doc.page_content[:500] + "...")
