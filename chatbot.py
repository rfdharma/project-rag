import streamlit as st
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv
import os
from groq import Groq
from langchain_groq import ChatGroq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


db = FAISS.load_local("faiss_index", embedding_function,allow_dangerous_deserialization=True)

# Load environment variables
load_dotenv()


retriever  = db.as_retriever(search_kwargs={"k": 10})


# Initialize Chat model
model = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Define prompt template
template = """You are an expert assistant with access to a reliable knowledge base. Answer the question using the provided context and do not include information outside this context. If the answer cannot be determined from the context, respond with "The context does not provide sufficient information."

Context: 
{context}

Question: 
{question}

Preferred answer language: 
{language}

Guidelines:
1. Base your answer solely on the provided context.
2. Ensure your answer is concise, accurate, and directly addresses the question.
3. Avoid assumptions or elaborations not supported by the context.
4. Use the same tone and formality level as the context.
"""
prompt = ChatPromptTemplate.from_template(template)

# Define chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Streamlit app
st.title("Question Answering")

# Input fields
question = st.text_input("Enter your question:")

# Process question and display answer
if st.button("Get Answer"):
    if question.strip():
        try:
            result = chain.invoke({"question": question, "language": "indonesia"})
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")