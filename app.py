import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get backend URL from environment
BACKEND_URL = os.getenv("BACKEND_LOCAL_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Code Documentation RAG", layout="centered")
st.title("Intelligent chunking methods for code documentation RAG")
st.caption("RAG pipeline that processes the text corpus into chunks, generates embeddings, and computes the chosen retrieval quality metrics on your evaluation dataset with questions and golden excerpts")

# Sidebar for options
st.sidebar.header("Configuration")

# Dataset options
dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=["chatlogs", "state_of_the_union", "wikitext"]
)

# Chunker options
chunker = st.sidebar.selectbox("Select Chunker", ["FixedTokenChunker"])

# Embedding function
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    options=["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"]
)

# Chunk size and number of retrieved chunks
chunk_size = st.sidebar.slider("Chunk Size (tokens)", 100, 500, 200, step=50)
top_k = st.sidebar.slider("Number of Retrieved Chunks", 1, 10, 5)

if st.button("Run Retrieval Pipeline"):
    with st.spinner("Running pipeline..."):
        payload = {
            "dataset": dataset,
            "chunker": chunker,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "top_k": top_k
        }
        print(payload)
        try:
            response = requests.post(f"{BACKEND_URL}/query-retriever-multiple", json=payload)
            response.raise_for_status()
            results = response.json()
            print(results)
            st.json(results)
            st.success("Results ready!")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.info("Hit 'Run Retrieval Pipeline' to start")
