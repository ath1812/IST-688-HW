import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')
collection = chroma_client.get_or_create_collection('HW4Collection')

st.title("HW4: Syracuse Student Org Chatbot")

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text = text + page.extract_text()
    return text

def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    text = soup.get_text()
    return text

def chunk_text(text):
    midpoint = len(text) // 2
    first_half = text[:midpoint]
    second_half = text[midpoint:]
    return [first_half, second_half]

def add_to_collection(collection, text, chunk_id):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[chunk_id],
        embeddings=[embedding]
    )

if collection.count() == 0:
    html_files = Path("./html_data/su_orgs").glob("*.html")
    for html_path in html_files:
        text = extract_text_from_html(html_path)
        chunks = chunk_text(text)
        file_name = html_path.name
        for i in range(len(chunks)):
            chunk_id = file_name + "_chunk_" + str(i + 1)
            add_to_collection(collection, chunks[i], chunk_id)



system_prompt = {
    "role": "system",
    "content": "You are a helpful assistant that answers questions about Syracuse University student organizations, using the retrieved organization information provided to you."
}

if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_input = st.chat_input("Ask me something")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    client = st.session_state.openai_client

    query_response = client.embeddings.create(
        input=user_input,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_text = ""
    for i in range(len(search_results["documents"][0])):
        doc_name = search_results["ids"][0][i]
        doc_text = search_results["documents"][0][i]
        retrieved_text = retrieved_text + f"From {doc_name}:\n{doc_text}\n\n"

        history = st.session_state.messages[1:]

    max_interactions = 5
    max_messages = max_interactions * 2

    buffer = history[-max_messages:]

    context_message = {
        "role": "system",
        "content": "The following is student organization information retrieved for this specific question. When you use this material in your answer, start your response with 'Based on the organization information:' so it's clear you're using retrieved knowledge. If the material doesn't help answer the question, say so and answer from general knowledge instead.\n\n" + retrieved_text
    }

    messages_to_send = [system_prompt] + buffer + [context_message]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages_to_send,
                stream=True,
            )
            response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})