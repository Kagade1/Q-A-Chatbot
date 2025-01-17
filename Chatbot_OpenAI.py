import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OpenAI"

## Prompt templet
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Yoy are a helpful assistant. Please responce to the user query"),
        ("user","Question:{question}")
    ]
)

def generate_responce(question, api_key, llm, temperature, 
max_token):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

## title of the app
st.title("Enhanced Q&A Chatbot with OpenAI")

## sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

## drop down to select various API model
llm = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4o","gpt-4-turbo","gpt-4"])

## adjust responce parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.7)
max_token = st.sidebar.slider("Max_token", min_value=50, max_value=300, value=150)

## main interface for user input
st.write("Go ahead and ask any question")
user_input  = st.text_input("You:")

if user_input:
    response = generate_responce(user_input,api_key,llm,temperature,max_token)
    st.write(response)
else:
    st.write("Please provide the query")