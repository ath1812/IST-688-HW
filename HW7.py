import streamlit as st
from openai import OpenAI
import sys
import json

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW7')
collection = chroma_client.get_or_create_collection('HW7NewsCollection')

st.title("News Information Bot")

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def search_news(query, n_results=5):
    client = st.session_state.openai_client

    query_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    articles = []
    for i in range(len(results["documents"][0])):
        article_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        articles.append({
            "company": metadata["company_name"],
            "date": metadata["date"],
            "url": metadata["url"],
            "content": article_text
        })

    return articles


def find_interesting_news(n_candidates=10):
    client = st.session_state.openai_client

    interestingness_query = "major, high-impact, significant business news with notable consequences"

    query_response = client.embeddings.create(
        input=interestingness_query,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates
    )

    articles = []
    for i in range(len(results["documents"][0])):
        article_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        articles.append({
            "company": metadata["company_name"],
            "date": metadata["date"],
            "url": metadata["url"],
            "content": article_text
        })

    return articles


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search for news articles about a specific topic or company. Use this when the user asks about a particular subject, company, or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic, company name, or keyword to search for in the news articles."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_interesting_news",
            "description": "Find and rank the most interesting, high-impact news stories across all companies. Use this when the user asks for the most interesting, important, or notable news without naming a specific topic.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

system_prompt = {
    "role": "system",
    "content": "You are a news reporting bot for a law firm, monitoring news about its clients. Use search_news to answer questions about a specific topic or company. Use find_interesting_news when asked for the most interesting or important news. When listing articles, include the company, date, and a brief reason why each one matters, and cite the source URL."
}

if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_input = st.chat_input("Ask about the news (e.g. 'find the most interesting news' or 'news about Coca-Cola')")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    client = st.session_state.openai_client

    messages_to_send = list(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Searching the news..."):

            first_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_to_send,
                tools=tools,
                tool_choice="auto"
            )

            response_message = first_response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages_to_send.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "search_news":
                        query = function_args.get("query")
                        articles = search_news(query)

                    if function_name == "find_interesting_news":
                        articles = find_interesting_news()

                    messages_to_send.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(articles)
                    })

                second_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_to_send,
                    stream=True
                )
                response = st.write_stream(second_response)
            else:
                st.write(response_message.content)
                response = response_message.content

    st.session_state.messages.append({"role": "assistant", "content": response})