import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import cassio
from PyPDF2 import PdfReader
import os
import time
import gdown

load_dotenv(Path(".env"))

st.set_page_config(page_title="Chat with Shrirang", layout="wide")
st.title("Chat with Shrirang..!!")

pdfreader = PdfReader("All About Shrirang.pdf")


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "astra_vector_index" not in st.session_state:
    st.session_state.astra_vector_index = None

with st.spinner("Loading"):
    if pdfreader:
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        if not st.session_state.pdf_processed:
            cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"), database_id=os.getenv("ASTRA_DB_ID"))
            llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
            embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            astra_vector_store = Cassandra(
                    embedding=embedding,
                    table_name="resume_chatbot_finalv1",
                    session=None,  
                    keyspace=None,
                )

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)
            astra_vector_store.add_texts(texts[:50])

            st.session_state.astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
            st.session_state.pdf_processed = True

st.sidebar.markdown("## **Welcome to Chat with Shrirang**")
st.sidebar.markdown('##### Shrirang is currently a student at IIT Madras')
st.sidebar.markdown('##### This chatbot is specifically build for recruiters or someone who wants to know about Shrirang')
st.sidebar.markdown('##### Additionally this also shows shrirang\'s skill in generative AI.')
st.sidebar.markdown(' If anything goes wrong do hard refresh by using **Shift** + **F5** key')

def typing_animation(text, speed):
            for char in text:
                yield char
                time.sleep(speed)

if "intro_displayed" not in st.session_state:
    st.session_state.intro_displayed = True
    intro = "Hello, I am Shrirang, currently a student at IIT Madras"
    intro2= "You can chat with Shrirang"
    st.write_stream(typing_animation(intro,0.02))
    st.write_stream(typing_animation(intro2,0.02))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#initialised prePrompt_selected
if "prePrompt_selected" not in st.session_state:
    st.session_state.prePrompt_selected = False

if "btn_selected" not in st.session_state:
    st.session_state.btn_selected = True

#defined callback fn
def btn_callback():
    st.session_state.prePrompted_selected = False
    st.session_state.btn_selected=False

prePrompt = None
if st.session_state.btn_selected:
    
    with st.expander("What can you ask?"):
        col1, col2,col3=st.columns(3, gap="small")
        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4, gap="small")
        row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4, gap="small")
        row4_col1, row4_col2, row4_col3, row4_col4 = st.columns(4, gap="small")
        with row2_col1:
            button_a = st.button('# Tell me in detail about Shrirang')
        with row2_col2:    
            button_b = st.button('# What education Shrirang have completed?')
        with row2_col3:  
            button_c = st.button('# Why should I hire Shrirang for AI role?')
        with row2_col4:  
            button_d = st.button('# List down Genrative AI projects shrirang have done')
        with row3_col1:  
            button_h = st.button('# List down all the skills Shrirang has')
        with row3_col2:  
            button_e = st.button('# What is Shrirang\'s GitHub id?')
        with row3_col3:  
            button_f = st.button('# What is Shrirang\'s LinkedIn id?')
        with row3_col4:  
            button_g = st.button('# List down Machine Learning projects shrirang have done')
        with row4_col2:  
            button_i = st.button('# What all position of responsibility Shrirang have took?')
        with row4_col3:  
            button_j = st.button('# What all hobbies Shrirang have?')
        with col1:
            button_x= st.button('# x',on_click=btn_callback ,  type='primary', key='close_btn')
           

    if button_a:
        st.session_state.prePrompt_selected = True
        prePrompt = 'Tell me in detail about Shrirang'   

    if button_b:
        st.session_state.prePrompt_selected = True
        prePrompt = 'What education Shrirang have completed? answer in points' 

    if button_c:
        st.session_state.prePrompt_selected = True
        prePrompt = 'Analyse and answer Why a recruiter should hire Shrirang for AI role? answer in points'  
    
    if button_d:
        st.session_state.prePrompt_selected = True
        prePrompt = 'List down Machine Learning projects shrirang have done? answer in points'  
    
    if button_e:
        st.session_state.prePrompt_selected = True
        prePrompt = 'What is Shrirang\'s GitHub id?' 
        
    if button_f:
        st.session_state.prePrompt_selected = True
        prePrompt = 'What is Shrirang\'s LinkedIn id?'  
    
    if button_g:
        st.session_state.prePrompt_selected = True
        prePrompt = 'List down Machine Learning projects shrirang have done? answer in points'
    
    if button_h:
        st.session_state.prePrompt_selected = True
        prePrompt = 'List down all the skills Shrirang has? answer in points'

    if button_i:
        st.session_state.prePrompt_selected = True
        prePrompt = 'What all position of responsibility Shrirang have took?'

    if button_j:
        st.session_state.prePrompt_selected = True
        prePrompt = 'What all hobbies Shrirang have? answer in points'

if st.session_state.prePrompt_selected and prePrompt is not None:
    
    query_text = prePrompt.strip() 
    if st.session_state.astra_vector_index is not None:
        answer = st.session_state.astra_vector_index.query(query_text, llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))).strip()
        typing_speed = 0.02
        if "context" or "no" in answer:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer, typing_speed))
        else:        
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer,typing_speed))
                
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # button_x= st.button('#### X',on_click=btn_callback)

# if "prePrompt_buttons_container" not in st.session_state:
#     st.session_state.prePrompt_buttons_container = st.container()

# with st.session_state.prePrompt_buttons_container:
#     prePrompt = None
#     col1, col2,col3, col4=st.columns(4)
#     with col1:
#         if st.button('Who is Shrirang',type='secondary'):
#             st.session_state.prePrompt_buttons_container.empty()
#             prePrompt = 'Who is Shrirang'
#             st.session_state.prePrompt_selected =True
#     with col2:
#         if st.button('What education Shrirang have completed?'):
#             st.session_state.prePrompt_buttons_container.empty()
#             prePrompt='What education Shrirang have completed?'
#             st.session_state.prePrompt_selected =True
#     with col3:
#         if st.button('Why should I hire Shrirang for AI role?'):
#             st.session_state.prePrompt_buttons_container.empty()
#             prePrompt='Why should I hire Shrirang for AI role?'
#             st.session_state.prePrompt_selected =True
#     with col4:
#         if st.button('Why should I hire Shrirang for AI role???'):
#             st.session_state.prePrompt_buttons_container.empty()
#             prePrompt='Why should I hire Shrirang for AI role?'
#             st.session_state.prePrompt_selected =True

# print(prePrompt)



# if st.session_state.prePrompt_selected: 
#     st.session_state.prePrompt_selected = False   
#     query_text = prePrompt.strip() 
#     if st.session_state.astra_vector_index is not None:
#         answer = st.session_state.astra_vector_index.query(query_text, llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))).strip()
#         typing_speed = 0.02
#         if "context" or "no" in answer:
#             with st.chat_message("assistant"):
#                 st.write_stream(typing_animation(answer, typing_speed))
#         else:        
#             with st.chat_message("assistant"):
#                 st.write_stream(typing_animation(answer,typing_speed))
                
#         st.session_state.messages.append({"role": "assistant", "content": answer})
#     else:
#         st.error("Database not initialized. Kindly reload and upload the PDF first.")   
    
        
# else:
#     # st.write("You have already selected a Prompt")   
#     pass

prompt = st.chat_input("Chat with Shrirang...")
 
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    query_text = prompt.strip()

    if st.session_state.astra_vector_index is not None:
        
        answer = st.session_state.astra_vector_index.query(query_text, llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))).strip()
        
        typing_speed = 0.02
        if "context" or "no" in answer:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer, typing_speed))
        else:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer,typing_speed))
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Database not initialized. Kindly reload and upload the PDF first.")