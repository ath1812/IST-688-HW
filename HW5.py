import streamlit as st
from openai import OpenAI
import tiktoken
import sys
import json

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection('Lab4Collection')

st.title("HW5: Smart Syllabus Chatbot")

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text = text + page.extract_text()
    return text


def add_to_collection(collection, text, file_name):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding]
    )


if collection.count() == 0:
    pdf_files = Path("./pdf_data").glob("*.pdf")
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        file_name = pdf_path.name
        add_to_collection(collection, text, file_name)


def relevant_course_info(query):
    client = st.session_state.openai_client

    query_response = client.embeddings.create(
        input=query,
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

    return retrieved_text


tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_course_info",
            "description": "Search the course syllabus database for information relevant to a student's question about courses, topics, or content covered in class.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A search query describing what course information is needed."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


max_tokens = 1000

encoding = tiktoken.encoding_for_model("gpt-4o")

system_prompt = {
    "role": "system",
    "content": "You are a helpful assistant that answers student questions about course syllabi. Use the relevant_course_info tool whenever a question needs specific course details. Clearly say when your answer is based on retrieved course material."
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

    history = st.session_state.messages[1:]

    buffer = []
    total_tokens = 0

    for message in reversed(history):
        message_content = message["content"] if isinstance(message["content"], str) else ""
        message_tokens = len(encoding.encode(message_content))
        if total_tokens + message_tokens > max_tokens:
            break
        buffer.insert(0, message)
        total_tokens = total_tokens + message_tokens

    messages_to_send = [system_prompt] + buffer

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            first_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages_to_send,
                tools=tools,
                tool_choice="auto"
            )

            response_message = first_response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages_to_send.append(response_message)

                for tool_call in tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    query = function_args.get("query", user_input)

                    retrieved_text = relevant_course_info(query)

                    messages_to_send.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "relevant_course_info",
                        "content": retrieved_text
                    })

                second_response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages_to_send,
                    stream=True
                )
                response = st.write_stream(second_response)
            else:
                st.write(response_message.content)
                response = response_message.content

    st.session_state.messages.append({"role": "assistant", "content": response})