import streamlit as st
from langchain.llms import HuggingFaceEndpoint

# To Access the HuggingFace API key securely from Streamlit secrets
api_key = st.secrets["HUGGINGFACE_API_KEY"]

# Define the endpoint URL 
endpoint_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

# Initializing the HuggingFace model using LangChain and the API key
llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url, 
    api_key=api_key, 
    timeout=300 
)

# Function to return the response from HuggingFace Inference API
def load_answer(question):
    return llm.invoke(question)

# Streamlit UI
st.set_page_config(page_title="LangChain HuggingFace Demo", page_icon=":robot:")
st.header("LangChain HuggingFace Demo")

# Get user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

# If the user submits a question
if st.button('Generate'):
    if user_input:
        response = load_answer(user_input)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.error("Please enter a question.")
