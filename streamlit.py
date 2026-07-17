import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
#--------------------------------- PAGE CONFIG -----------------------------------------

st.title("CHAT with Hooked Design Expert")

#--------------------------------- CONSTANTS --------------------------------------------
PDF_PATH="docs/Hooked.epub"
LLM_EMBEDDING_MODEL="text-embedding-3-large"
DB_DIR="./CHROMA-DB"

@st.cache_resource(show_spinner=False)
def initialise_rag():
    """loads new chroma DB if not there else loads old one"""
    embeddings=OpenAIEmbeddings(model=LLM_EMBEDDING_MODEL,api_key=os.environ.get("OPENAI_API_KEY"))
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR))>0:
        return Chroma(persist_directory=DB_DIR,embedding_function=embeddings)
    
    if not os.path.exists(PDF_PATH):
        return f"error pdf path doesn't exist create one: {PDF_PATH} "
    loader=UnstructuredEPubLoader(file_path=PDF_PATH,mode="elements")
    docs=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ) 
    splits=text_splitter.split_documents(docs)
    vector_store=Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR,
        
    )
    # print(vector_store)
    return vector_store

with st.spinner("Loading Resources, Please wait ... "):
     try:    
        vector=initialise_rag()
        retriever=vector.as_retriever(search_kwargs={"k":3})
        llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview",api_key=os.environ.get("GOOGLE_API_KEY"),verbose=True)
     except Exception as e:
         st.error("AI and Resources failed to load :(")
         st.stop()
         
system_prompt=(
 "You are an expert AI assistant specialized exclusively in Nir Eyal's book \"Hooked: How to Build Habit-Forming Products\". "
    "Your primary task is to answer user questions using only the provided context.\n\n"
    "CRITICAL RULES:\n"
    "1. If the answer cannot be found in the context, or the input is insufficient, say exactly: "
    "\"Sorry, that information is not in the context of the document.\"\n"
    "2. Immediately after, list the core principles of the book to guide the user.\n\n"
    "Context:\n{context}\n\n"
    "Core Principles to list if context is missing:\n"
    "• Trigger (Internal & External): What prompts the user to action?\n"
    "• Action: What is the simplest behavior done in anticipation of a reward?\n"
    "• Variable Reward: How do you satisfy the user's need while leaving them wanting more?\n"
    "• Investment: What bit of 'work' does the user do to increase the likelihood of returning?\n"
    "• Habit Loops & Ethics: The Manipulation Matrix (Facilitator, Peddler, Entertainer, Dealer).\n\n"
    "If the context does not contain the answer, output the exact phrase followed by the bullet points above."
   
)
prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{input}")
])

question_answer_chain= create_stuff_documents_chain(llm=llm,prompt=prompt,verbose=True)
rag_chain= create_retrieval_chain(
    retriever, question_answer_chain
)

if "messages" not in st.session_state:
    st.session_state.messages=[]
    
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
#------------------------ USER INPUT FIELD ------------------------------------------

if user_query := st.chat_input("Ask about product design and we will answer about it..."):
    
    #Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
        
    st.session_state.messages.append(
        {"role":"user", "content": user_query}
    )    
    
    #Display AI message
    with st.chat_message("assistant"):
         response_placeholder=st.empty()
         with st.spinner("Searching Resources and Thinking to Answer ..."):
             response=rag_chain.invoke(
                 {"input": user_query}
                 )     
             answer=response["answer"]
             response_placeholder.markdown(answer)
    #save assistant message to history
    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )
    