import streamlit as stl
import os
from dotenv import load_dotenv  # for accessing the .env file which contains the OPENAI_API_KEY
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI
# Load environment variables
load_dotenv()
storage_path = "./vectorstore"
documents_path = "./data"

Settings.llm = OpenAI(model="gpt-3.5-turbo", max_tokens=512)

# Initialize the index
@stl.cache_resource(show_spinner=False)
def initialize(): 
    if not os.path.exists(storage_path):
        documents = SimpleDirectoryReader(documents_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
    return index

index = initialize()

# Streamlit UI setup
stl.title("RAG CHATBOT")
if "messages" not in stl.session_state:
    stl.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything from your file! I am here to help you."}
    ]

# Create a chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Instruction to the model
instruction = "Please answer only questions relevant to the provided documents. Do not answer out-of-context questions. If out of context is asked avoid hallucinating the answer. Provide a short detailed answer for contextual answers only."

# Input prompt
if prompt := stl.chat_input("Your question"):
    stl.session_state.messages.append({"role": "user", "content": prompt})

# Display previous messages
for messages in stl.session_state.messages:
    with stl.chat_message(messages["role"]):
        stl.write(messages["content"])

casual_responses = {
    "thank you": "You're welcome!",
    "thanks": "You're welcome!",
    "hello": "Hi there! How can I assist you today?",
    "hi": "Hello! What can I do for you?",
    "bye": "Goodbye! Have a great day!"
}

def get_casual_response(prompt):
    return casual_responses.get(prompt.lower(), None)

# Process new user input
if stl.session_state.messages[-1]["role"] != "assistant":
    with stl.chat_message("assistant"):
        with stl.spinner("Thinking...."):
            casual_response = get_casual_response(prompt)
            if casual_response:
                response_content = casual_response
            else:
                # Retrieve relevant documents
                retriever = index.as_retriever()
                retrieved_docs = retriever.retrieve(prompt)
                
                augmented_prompt = f"{instruction}\n\n{prompt}\n\nRelevant Documents:\n"
                for node_with_score in retrieved_docs:
                    document_content = node_with_score.node.text  
                    augmented_prompt += f"{document_content}\n\n"

                # Generate response
                response = chat_engine.chat(augmented_prompt)
                response_content = response.response

            stl.write(response_content)
            message = {"role": "assistant", "content": response_content}
            stl.session_state.messages.append(message)

# use 'streamlit run run_chatbot.py'in the terminal for running the application