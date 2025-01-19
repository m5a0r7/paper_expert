import streamlit as st
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv(override=True)


# Cache the function to load and process PDF documents
@st.cache(allow_output_mutation=True)
def load_and_process_pdfs(pdf_folder_path):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, file)
            try:
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Failed to load and process {pdf_path}: {e}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits


# Cache the function to initialize the vector store with documents
@st.cache(allow_output_mutation=True)
def initialize_vectorstore(_splits):
    return FAISS.from_documents(
        documents=_splits,
        embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
    )


pdf_folder_path = "./pdf_files"
splits = load_and_process_pdfs(pdf_folder_path)
vectorstore = initialize_vectorstore(splits)
prompt_template = """You are a paper scholar expert. You need to answer the question related to hydrology papers. 
Given below is the context and question of the user. Don't answer question outside the context provided.
context = {context}
question = {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": vectorstore.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
# Streamlit app
st.title("Paper Expert")
user_input = st.text_input("Enter your question about hydrology:", "")
if st.button("Submit"):
    try:
        response = rag_chain.invoke(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")
