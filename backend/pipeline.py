from typing import Dict, List, Tuple, Set
from pathlib import Path
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import re

def load_data(dataset: str) -> Tuple[List[str], List[Dict]]:
    """
    Load both corpus and evaluation data
    :param dataset: dataset name
    :return: corpus documents and evaluation data
    """
    # Load corpus
    corpus_path = Path("data") / dataset / "corpus"
    loader = DirectoryLoader(
        str(corpus_path),
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    corpus_documents = loader.load()
    
    # Load evaluation data
    eval_path = Path("data") / dataset / "evaluation.json"
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    
    return corpus_documents, eval_data

def chunk_documents(documents: List[str], chunk_size: int) -> List[str]:
    """
    Split documents into chunks
    :param documents: list of documents
    :param chunk_size: size of each chunk
    :return: list of chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def create_embeddings(chunks: List[str], embedding_model: str) -> FAISS:
    """
    Create embeddings for chunks using specified model
    :param chunks: list of text chunks
    :param embedding_model: name of the embedding model
    :return: FAISS vector store
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def tokenize(text: str) -> Set[str]:
    """
    Tokenize text into a set of tokens
    :param text: input text
    :return: set of tokens
    """
    # Simple tokenization by splitting on whitespace and removing punctuation
    tokens = re.findall(r'\w+', text.lower())
    return set(tokens)

def calculate_token_metrics(retrieved_chunks: List[str], golden_excerpt: str) -> Dict[str, float]:
    """
    Calculate token-wise precision, recall, and IoU
    :param retrieved_chunks: list of retrieved text chunks
    :param golden_excerpt: golden excerpt containing relevant tokens
    :return: dictionary of metrics
    """
    # Get sets of tokens
    te = tokenize(golden_excerpt)  # tokens in golden excerpt
    tr = set()  # tokens in retrieved chunks
    
    # Combine tokens from all retrieved chunks
    for chunk in retrieved_chunks:
        tr.update(tokenize(chunk))
    
    # Calculate intersection
    intersection = te.intersection(tr)
    
    # Calculate metrics
    if len(tr) == 0:
        precision = 0.0
    else:
        precision = len(intersection) / len(tr)
    
    if len(te) == 0:
        recall = 0.0
    else:
        recall = len(intersection) / len(te)
    
    # Calculate IoU
    union = len(te) + len(tr) - len(intersection)
    if union == 0:
        iou = 0.0
    else:
        iou = len(intersection) / union
    
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "num_relevant_tokens": len(te),
        "num_retrieved_tokens": len(tr),
        "num_intersection_tokens": len(intersection)
    }

def evaluate_retrieval(vectorstore: FAISS, eval_data: List[Dict], top_k: int) -> Dict:
    """
    Evaluate retrieval performance using token-wise metrics
    :param vectorstore: FAISS vector store
    :param eval_data: evaluation data with questions and golden excerpts
    :param top_k: number of retrieved chunks
    :return: evaluation metrics
    """
    all_precisions = []
    all_recalls = []
    all_ious = []
    all_relevant_tokens = []
    all_retrieved_tokens = []
    all_intersection_tokens = []
    
    for item in eval_data:
        question = item["question"]
        golden_excerpt = item["golden_excerpt"]
        
        # Retrieve chunks
        retrieved_chunks = vectorstore.similarity_search(question, k=top_k)
        retrieved_texts = [doc.page_content for doc in retrieved_chunks]
        
        # Calculate token-wise metrics
        metrics = calculate_token_metrics(retrieved_texts, golden_excerpt)
        
        all_precisions.append(metrics["precision"])
        all_recalls.append(metrics["recall"])
        all_ious.append(metrics["iou"])
        all_relevant_tokens.append(metrics["num_relevant_tokens"])
        all_retrieved_tokens.append(metrics["num_retrieved_tokens"])
        all_intersection_tokens.append(metrics["num_intersection_tokens"])
    
    return {
        "average_precision": np.mean(all_precisions),
        "average_recall": np.mean(all_recalls),
        "average_iou": np.mean(all_ious),
        "total_queries": len(eval_data),
        "token_statistics": {
            "average_relevant_tokens": np.mean(all_relevant_tokens),
            "average_retrieved_tokens": np.mean(all_retrieved_tokens),
            "average_intersection_tokens": np.mean(all_intersection_tokens)
        }
    }

def run_query_retriever_multiple(config: dict) -> dict:
    """
    Run the complete retrieval evaluation pipeline
    :param config: configuration dictionary containing:
        - dataset: name of the dataset
        - chunk_size: size of chunks
        - top_k: number of retrieved chunks
        - embedding_model: name of the embedding model
    :return: evaluation results and metrics
    """
    # Extract configuration
    dataset = config.get("dataset", "chatlogs")
    chunk_size = config.get("chunk_size", 200)
    top_k = config.get("top_k", 5)
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")

    # 1. Load data
    corpus_documents, eval_data = load_data(dataset)
    
    # 2. Chunk documents
    chunks = chunk_documents(corpus_documents, chunk_size)
    
    # 3. Create embeddings
    vectorstore = create_embeddings(chunks, embedding_model)
    
    # 4. Evaluate retrieval
    metrics = evaluate_retrieval(vectorstore, eval_data, top_k)
    
    # 5. Return results
    return {
        "metrics": metrics,
        "config": {
            "dataset": dataset,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "embedding_model": embedding_model
        },
        "summary": {
            "total_documents": len(corpus_documents),
            "total_chunks": len(chunks),
            "total_queries": len(eval_data)
        }
    }
    
