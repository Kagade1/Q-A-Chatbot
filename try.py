import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#from langchain.chat_models import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
import time
import os

# Initialize session state variables
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None

# Set up OpenAI API Key
st.title('Groq')
st.sidebar.title('Configuration')
os.environ['OLLAMA_API_KEY'] = st.sidebar.text_input("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", type="password")

def create_stuff_documents_chain(llm, prompt):
    stuff_prompt = PromptTemplate(template=prompt, input_variables=["context", "input"])
    return StuffDocumentsChain(llm_chain=LLMChain(llm=llm, prompt=stuff_prompt), document_variable_name="context")

def create_retrieval_chain(retriever, document_chain):
    return RetrievalQA(retriever=retriever, combine_documents_chain=document_chain)

def create_vector_embedding():
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.session_state.embeddings = OllamaEmbeddings(api_key=os.getenv("OLLAMA_API_KEY"))
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        if uploaded_file:
            st.session_state.loader = PyPDFDirectoryLoader(uploaded_file.name)
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Document embedding created successfully.")
        else:
            st.error("Please upload a PDF document to create embeddings.")

# Inputs
llm_model = st.sidebar.selectbox('LLM Model', ['gpt-3.5-turbo', 'gpt-4'])
prompt = st.text_area('Prompt', 'Answer questions from the provided research paper.')
user_prompt = st.text_area('Enter your Query')

# LLM Initialization
llm = Ollama(model=llm_model, temperature=0.7)

# Document Embedding Button
if st.button('Documents Embedding'):
    create_vector_embedding()

# Query Handling
if user_prompt:
    if st.session_state.vectors is None:
        st.error("Please click 'Documents Embedding' to initialize the vector database before querying.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrival_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrival_chain.invoke({"input": user_prompt})
        print(f"Response time: {time.process_time() - start}")

        if response and 'context' in response:
            st.write(response['answer'])

            # Expand results
            with st.expander("Document similarity search"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write("--------------------")
        else:
            st.write("No relevant documents found.")
