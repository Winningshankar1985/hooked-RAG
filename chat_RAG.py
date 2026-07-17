import os

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.prompts import MessagesPlaceholder
import streamlit as st
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.globals import set_debug,set_verbose
from langchain.tools import tool
from tavily import TavilyClient
#--------------------------------- PAGE CONFIG -----------------------------------------
set_debug(True)
set_verbose(True)


tavily_client=TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


st.title("CHAT with Hooked Design Expert")

#--------------------------------- CONSTANTS --------------------------------------------
PDF_PATH="docs/Hooked.epub"
LLM_EMBEDDING_MODEL="text-embedding-3-large"
DB_DIR="./CHROMA-DB"


@tool
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
    retriever=vector_store.as_retriever(search_kwargs={"k":3})
    return retriever

@tool
def web_search(query:str)->str:
    """use this tool to search for answers when you can't find the answer directly from vectorDB, 
    use this tool as much time as possible to understand user questions 
    and to search the vectorDB resources correctly 
    and continue untill you get a definitive answer for the user's question"""
    
    return tavily_client.search(query)
    
system_prompt=(
 """
 
 You are an advanced, multi-step research assistant. Your goal is to provide the most accurate, comprehensive, and up-to-date answers by intelligently combining private knowledge and live web data.

You have access to the following tools:
1. `web_search`: Searches the live internet for recent developments, news, and external verification.

CRITICAL OPERATIONAL PIPELINE:
For every user query, you must follow this strict execution loop:

1. INTERNAL FIRST: Always search the internal knowledge base first. Evaluate the retrieved documents deeply.
2. GAP ANALYSIS: If the vector database yields no results, incomplete answers, or outdated information, identify exactly what is missing.
3. EXTERNAL EXPANSION: Use `web_search` to fill in those specific informational gaps.
4. SYNTHESIS & RE-CHECK: If the web search reveals new keywords, terms, or historical context, pivot back and query `vector_db_search` with these refined terms to check if related internal data was missed.
5. ITERATE: Repeat this loop until you have synthesized the absolute best, most comprehensive answer possible.

RESPONSE FORMAT RULES:
- If an answer is found exclusively in the internal database, flag it as [Internal Source].
- If an answer required web research, clearly cite your external findings alongside internal policies.
- If a conflict arises between internal documents and web data, highlight the contradiction clearly for the user.
- Remain objective, concise, and professional. 
 """
)

with st.spinner("Loading Resources, Please wait ... "):
     try:    
       
        # llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview",api_key=os.environ.get("GOOGLE_API_KEY"))
        llm=ChatOpenAI(model="gpt-4o",api_key=os.environ.get("OPENAI_API_KEY"),temperature=0)
       
        prompt=ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        tools=[web_search,initialise_rag]
        agent=create_tool_calling_agent(
            llm,tools,prompt
            
        )
      
        agent_executor=AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )
       
        
     except Exception as e:
         st.error("AI and Resources failed to load :(")
         st.stop()
         




if "messages" not in st.session_state:
    st.session_state.messages=[]
    
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
#------------------------ USER INPUT FIELD ------------------------------------------
def main():
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
                response=agent_executor.invoke(
                    {"input": user_query, "chat_history": st.session_state.messages}
                    ) 
                # print(response)    
                answer=response["output"]
                response_placeholder.markdown(answer)
        #save assistant message to history
        st.session_state.messages.append(
            {"role":"assistant","content":answer}
        )
    
if __name__ == "__main__":
    main()