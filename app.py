import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.llms import HuggingFaceEndpoint

# Load the environment variables from the .env-sample file
load_dotenv('.env')

# Function to return the response from Hugging Face Inference API
def load_answer(question):
    # Get the API key from the environment variable
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please ensure it's set in the .env-sample file.")

    # Replace with your Hugging Face Inference API endpoint URL
    endpoint_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    
    # Use the HuggingFaceEndpoint instead of OpenAI
    llm = HuggingFaceEndpoint(endpoint_url=endpoint_url, api_key=api_key)
    
    # Invoke the model with the user's question
    answer = llm.invoke(question)
    
    return answer

# App UI starts here
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

# Gets the user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

# If the user provides input, load the answer
if user_input:
    response = load_answer(user_input)

submit = st.button('Generate')  

# If the generate button is clicked, display the response
if submit:
    st.subheader("Answer:")
    st.write(response)
