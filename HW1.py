import streamlit as st
import fitz
from openai import OpenAI

MODEL_OPTIONS = [
    "gpt-3.5",
    "gpt-4.1",
    "gpt-5-chat-latest",
    "gpt-5-nano",
]


def run_model(client, model_name, messages):
    """Run a model call and surface a user-friendly error if the model is unavailable."""
    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )
    except Exception as exc:
        st.warning(
            f"{model_name} is not available on this OpenAI account or API key. "
            "This is a model-access issue, not a document problem. "
            f"Error: {exc}"
        )
        return None


def read_pdf(uploaded_file):
    """Read text from an uploaded PDF file."""
    try:
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_pages = [page.get_text() for page in pdf_document]
        pdf_document.close()
        return "\n".join(text_pages)
    except Exception as exc:
        st.error(f"Unable to read PDF file: {exc}")
        return ""


def read_document(uploaded_file):
    """Read either a TXT or PDF document."""
    uploaded_file.seek(0)
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "txt":
        return uploaded_file.read().decode("utf-8", errors="replace")
    if file_extension == "pdf":
        return read_pdf(uploaded_file)

    st.error("Unsupported file type. Please upload a .txt or .pdf file.")
    return ""


# Show title and description.
st.title("📄 Document QA with model comparison")
st.write(
    "Upload a .txt or .pdf document and ask a question about it. "
    "This version lets you compare four OpenAI models on the same document and question."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)",
        type=("txt", "pdf"),
    )

    question = st.text_area(
        "Question about the document",
        value="",
        placeholder="Is this course hard?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        document = read_document(uploaded_file)
        if not document:
            st.stop()

        messages = [
            {
                "role": "user",
                "content": (
                    "Use the uploaded document to answer the question accurately. "
                    f"Document:\n{document}\n\nQuestion:\n{question}"
                ),
            }
        ]

        selected_model = st.selectbox("Choose a model", MODEL_OPTIONS, index=0)

        if st.button("Run answer"):
            with st.spinner(f"Running {selected_model}..."):
                stream = client.chat.completions.create(
                   model="gpt-5",
                   messages=messages,
                   stream=True,
                   temperature=1,
            )
                
                st.subheader(f"Answer using {selected_model}")
                st.write_stream(stream)

     
         
