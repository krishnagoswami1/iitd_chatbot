from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from qdrant_client import QdrantClient
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="IIT Delhi Chatbot", page_icon="🎓")

st.markdown("""
<style>
            .title{
            position: fixed;
            top:50px;
            left: 0;
            width: 100%;
            padding: 10px;
            z-index: 1000;
            text-align: center;
            background-color: #fff;
            border-bottom: 1px solid #ccc;
            }
""", unsafe_allow_html=True)

st.markdown("<h1 class = 'title'>🎓 IIT Delhi Chatbot</h1", unsafe_allow_html=True)

# load_dotenv()  # Load environment variables from .env file

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets['GOOGLE_API_KEY'])


@st.cache_resource
def init_qdrant():
    # qdrant_api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=st.secrets['QDRANT_URL'], api_key=st.secrets['QDRANT_API_KEY'])
    with st.chat_message("assistant"):
        st.write(client.get_collections())
    collection_name = "iitd_chatbot"
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store

vector_store = init_qdrant()

rag_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant of IIT Delhi. Use the following context to answer student queries.If the context does not contain the answer say so.
    Context: {context}
    
    Question: {question}
        
    Answer:"""
)





# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("You: "):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check for exit command
    if prompt.lower() == "exit":
        with st.chat_message("assistant"):
            st.write("Exiting chat...")
    else:
        # Generate response
        try:
            #Using RAG for intelligent context aware responses
            with st.spinner("Searching for relevant information...."):
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            formatted_prompt = rag_prompt.format(context = context, question = prompt)
            

            response = model.invoke(formatted_prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response.content)
                
        except Exception as e:
            with st.chat_message("assistant"):
                st.write(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
