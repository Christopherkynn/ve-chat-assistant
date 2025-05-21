import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
import os

# Set page config
st.set_page_config(page_title="VE Chat App", layout="wide")

# Title
st.title("📘 Value Engineering Chat Assistant")
st.markdown("Upload your VE reports, FAST diagrams, risk matrices, and more — and start asking questions!")

# Load API key from secrets.toml
openai_api_key = st.secrets["openai"]["api_key"]

# Setup OpenAI LLM
llm = OpenAI(api_key=openai_api_key, model="gpt-4")

# Load and index documents from the /docs folder
@st.cache_resource(show_spinner="Indexing documents...")
def load_index():
    reader = SimpleDirectoryReader(input_dir="docs", recursive=True)
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_index()
chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm)

# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything about your documents...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = chat_engine.chat(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": str(response)})

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    else:
        st.chat_message("assistant").write(chat["content"])
