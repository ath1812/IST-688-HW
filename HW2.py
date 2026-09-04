import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai

st.title("HW 2: URL Summarizer")

url = st.text_input("Enter a web page URL")

summary_type = st.sidebar.selectbox("Summary type", ["100 words", "2 paragraphs", "5 bullet points"])
language = st.sidebar.selectbox("Output language", ["English", "Spanish", "French"])
llm_choice = st.sidebar.selectbox("Choose LLM", ["OpenAI", "Gemini"])
use_advanced = st.sidebar.checkbox("Use advanced model")

generate = st.button("Generate Summary")

if generate and url:

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    page_text = soup.get_text()

    if summary_type == "100 words":
        instruction = "Summarize this in 100 words."
    elif summary_type == "2 paragraphs":
        instruction = "Summarize this in 2 paragraphs."
    else:
        instruction = "Summarize this in 5 bullet points."

    prompt = instruction + " Write it in " + language + ".\n\n" + page_text

    if llm_choice == "OpenAI":
        if use_advanced:
            model_name = "gpt-4.1"
        else:
            model_name = "gpt-5-nano"

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response.choices[0].message.content

    else:
        if use_advanced:
            model_name = "gemini-3.5-flash"
        else:
            model_name = "gemini-3.5-flash-lite"

        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        summary = response.text

    st.subheader("Summary")
    st.write(summary)
    st.write("Model used:", model_name)