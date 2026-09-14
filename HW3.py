import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
import tiktoken

st.title("HW3 - Chat About a URL")

st.write(
    "This chatbot lets you talk about the content of up to two web pages. "
    "Paste one or two URLs in the sidebar, pick which LLM you want to chat with, "
    "and ask questions. The page content is pulled once and placed into a system "
    "prompt that stays attached to every message, so the bot always has that "
    "context even as the conversation grows. To keep the conversation from "
    "growing forever, this app uses a token-based memory buffer: it counts the "
    "tokens in the chat history using tiktoken and only sends the most recent "
    "messages that fit under a 2,000 token budget, along with the system "
    "prompt, to the model."
)

encoding = tiktoken.encoding_for_model("gpt-4o")
max_tokens = 2000

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def read_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()


def gemini_text_generator(gemini_stream):
    for chunk in gemini_stream:
        yield chunk.text


st.sidebar.header("Setup")

url_1 = st.sidebar.text_input("URL 1")
url_2 = st.sidebar.text_input("URL 2 (optional)")

llm_choice = st.sidebar.selectbox(
    "Choose an LLM",
    ("OpenAI - GPT-5.6 Sol", "Google - Gemini 3.1 Pro"),
)

url_context = ""

if url_1:
    url_context = url_context + "Content from URL 1 (" + url_1 + "):\n"
    url_context = url_context + read_url_content(url_1) + "\n\n"

if url_2:
    url_context = url_context + "Content from URL 2 (" + url_2 + "):\n"
    url_context = url_context + read_url_content(url_2) + "\n\n"

system_prompt_text = (
    "You are a helpful assistant that answers questions using the reference "
    "material provided below. Base your answers on this material whenever it "
    "is relevant, and say so if the material does not cover something.\n\n"
    + url_context
)

system_prompt = {"role": "system", "content": system_prompt_text}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_urls" not in st.session_state:
    st.session_state.last_urls = ""

current_urls = url_1 + "|" + url_2

if current_urls != st.session_state.last_urls:
    st.session_state.messages = []
    st.session_state.last_urls = current_urls

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask something about the page(s) above")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    buffer = []
    total_tokens = 0

    for message in reversed(st.session_state.messages):
        message_tokens = len(encoding.encode(message["content"]))
        if total_tokens + message_tokens > max_tokens:
            break
        buffer.insert(0, message)
        total_tokens = total_tokens + message_tokens

    messages_to_send = [system_prompt] + buffer

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if llm_choice == "OpenAI - GPT-5.6 Sol":
                stream = openai_client.chat.completions.create(
                    model="gpt-5.6-sol",
                    messages=messages_to_send,
                    stream=True,
                )
                response = st.write_stream(stream)

            if llm_choice == "Google - Gemini 3.1 Pro":
                gemini_model = genai.GenerativeModel(
                    "gemini-3.1-pro",
                    system_instruction=system_prompt_text,
                )
                gemini_history = []
                for message in buffer:
                    if message["role"] == "user":
                        gemini_history.append(
                            {"role": "user", "parts": [message["content"]]}
                        )
                    if message["role"] == "assistant":
                        gemini_history.append(
                            {"role": "model", "parts": [message["content"]]}
                        )

                gemini_stream = gemini_model.generate_content(
                    gemini_history, stream=True
                )

                response = st.write_stream(gemini_text_generator(gemini_stream))

    st.session_state.messages.append({"role": "assistant", "content": response})