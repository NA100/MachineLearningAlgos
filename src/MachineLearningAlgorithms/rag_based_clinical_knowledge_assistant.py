# Project 2: RAG-Based Clinical Knowledge Assistant

"""
This project builds a Retrieval-Augmented Generation (RAG) system for answering clinician queries.
It uses a local vector store of medical documents + an LLM to provide grounded answers.
"""

# --------------------------
# Step 1: Load Knowledge Base
# --------------------------

sample_docs = [
    "CBT (Cognitive Behavioral Therapy) is an evidence-based approach for treating anxiety and depression.",
    "SSRIs (Selective Serotonin Reuptake Inhibitors) are a common first-line pharmacological treatment for depression.",
    "Exposure therapy is effective for treating phobias and PTSD by gradually confronting feared stimuli.",
    "Dialectical Behavior Therapy (DBT) is used particularly for borderline personality disorder and emotional regulation."
]

# --------------------------
# Step 2: Embed + Store (FAISS)
# --------------------------

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # small and fast
embeddings = model.encode(sample_docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --------------------------
# Step 3: Retrieve Context
# --------------------------

def retrieve_context(query, k=2):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k)
    return [sample_docs[i] for i in I[0]]

# --------------------------
# Step 4: Prompt LLM with Context
# --------------------------

import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_llm_with_context(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"""
    You are a clinical assistant. Use the following context to answer the clinician's query accurately and concisely:

    Context:
    {context}

    Query:
    {query}

    Answer:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    return response["choices"][0]["message"]["content"].strip()

# --------------------------
# Step 5: Run Full Pipeline
# --------------------------

if __name__ == "__main__":
    query = "What therapy is effective for PTSD?"
    context = retrieve_context(query)
    answer = query_llm_with_context(query, context)

    print("Clinician Query:", query)
    print("\nRetrieved Context:", "\n- ".join(context))
    print("\nLLM Answer:", answer)

# --------------------------
# Step 6 (Optional): Future Work
# --------------------------
"""
- Replace sample_docs with actual documents from PubMed, NICE guidelines, DSM-5 summaries, etc.
- Use LangChain or LlamaIndex for more advanced RAG architecture
- Add Streamlit/Gradio UI for doctors to interact
- Add source citations and highlight retrieval matches
- Log each interaction for feedback and improvement
"""


"""
In a RAG (Retrieval-Augmented Generation) system like the one in Project 2: Clinical Knowledge Assistant, context comes from a retrieval system, typically a vector database or in-memory vector store.

Let me break it down technically:

🔁 RAG Architecture (High-Level)
User Input
The user asks a question:
"What are the symptoms of generalized anxiety disorder?"
Retrieval Phase
The input question is converted into a vector using an embedding model.
This vector is used to search a vector store (like FAISS) that contains embedded chunks of clinical documents.
The top k most relevant chunks (e.g., text paragraphs) are returned — this is your context.
Augmented Generation Phase
These retrieved chunks are appended as context to the original user query in a prompt.
The LLM (e.g., gpt-4) uses this prompt to generate an informed, grounded answer.
🧠 In Our Code (Key Components)
1. Context Source: sample_docs
sample_docs = [
    "Generalized Anxiety Disorder (GAD) is characterized by excessive anxiety and worry...",
    "Cognitive Behavioral Therapy (CBT) is an evidence-based treatment for anxiety disorders...",
    "SSRIs such as sertraline or escitalopram are first-line pharmacological treatments for GAD...",
]
These are the source documents (contextual knowledge base). In a real system, these would come from:
medical textbooks
clinical research articles
guidelines (e.g., DSM-5)
EHR notes (after de-identification)
2. Vector Embeddings
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(sample_docs, embedding=embeddings)
Each sample_doc is turned into a vector using OpenAIEmbeddings.
All vectors are stored in a FAISS vector store.
3. Retriever (Context Fetching)
retriever = db.as_retriever()
retrieved_docs = retriever.get_relevant_documents(user_query)
The retriever searches for semantically similar documents based on the user’s query.
These documents become the context.
4. Context Passed to LLM
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
response = qa.run(user_query)
LangChain’s RetrievalQA handles:
combining retrieved context + user query into a prompt
sending it to the LLM
returning the final grounded answer
"""