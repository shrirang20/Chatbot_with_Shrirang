import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
from PyPDF2 import PdfReader
from huggingface_hub import login
import os
import time
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from pydantic import Field, BaseModel
from typing import Any, List, Optional, Dict
# from flask import Flask
# from database import db
# from models import Prompt

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///queries_and_answers.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db.init_app(app)

class GeminiLLM(LLM, BaseModel):
    model_name: str = Field(..., description="model/gemini-1.5-flash")
    model: Any = Field(None, description="The GenerativeModel instance")
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        try:
            self.model = genai.GenerativeModel(model_name=self.model_name)
        except AttributeError as e:
            st.error(f"Model initialization failed: {e}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text
    
    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
load_dotenv(Path(".env"))

def generate_rag_prompt(query, context):
    prompt=("""
    You are an AI-powered virtual assistant representing Shrirang Sapate, a BS Data Science and Programming student at IIT Madras.\
    You have access to detailed information about Shrirang's academic background, professional experiences, technical skills,\
    projects, and personal interests. Your primary role is to answer any questions about Shrirang in a clear, concise, and professional\
    manner. Ensure that your responses are up-to-date and accurately reflect Shrirang's achievements and capabilities.\
    Respond to queries as if you are Shrirang speaking in the first person.\
    Key points to include in your responses:

        - Academic background and coursework
        - Technical skills and proficiencies
        - Professional experiences and internships
        - Projects and notable achievements
        - Extracurricular activities and hobbies
        - Personal interests and goals
        - Always maintain a polite and professional tone, and provide comprehensive and accurate information.

    Key guidelines:

        1. Maintain a professional and friendly tone at all times.
        2. Provide concise, relevant answers to questions about Shrirang's professional background.
        3. If asked about personal information beyond what's typically included in a resume, politely decline to answer\
           and redirect the conversation to professional topics or ask them to conact Shrirang at official.shriraang@gmail.com .
        4. Be prepared to elaborate on any aspect of Shrirang's professional experience when asked.
        5. If you're uncertain about a specific detail, say so rather than providing potentially inaccurate information.
        6. Offer to provide more details or clarification if your initial response doesn't fully address the query.
        7. Be able to summarize Shrirang's key qualifications and experiences succinctly when asked.
        8. If asked about skills or experiences not listed in your knowledge base, politely state that you don't have that information.
        9. Maintain consistency in all your responses about Shrirang's background.
        10. Be able to tailor responses to highlight relevant experiences for specific job roles or industries when asked.

    Remember, your purpose is to represent Shrirang's professional qualifications effectively and help potential employers or connections learn more about Shrirang's career and capabilities.
            
            QUESTION: '{query}'
            CONTEXT: '{context}'

            Response: 
""").format(query=query,context=context)
    return prompt

def generate_answer(prompt):
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    login(token="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")
    llm = genai.GenerativeModel(model_name='model/gemini-1.5-flash')
    answer = llm.generate_content(prompt)
    return answer.text

# with app.app_context():
#     db.create_all()

st.set_page_config(page_title="Chat with Shrirang", layout="wide", )
st.title("Chat with Shrirang..!!")

pdfreader = PdfReader("All About Shrirang.pdf")


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "faiss_vector_index" not in st.session_state:
    st.session_state.faiss_vector_index = None

with st.spinner("Loading"):
    if pdfreader:
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        if not st.session_state.pdf_processed:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            login(token="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")
            llm=genai.GenerativeModel(model_name='model/gemini-1.5-flash')

            embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            faiss_vector_store = FAISS.from_texts([raw_text], embedding_function)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
            )

            texts = text_splitter.split_text(raw_text)
            faiss_vector_store.add_texts(texts[:50])

            st.session_state.faiss_vector_index = VectorStoreIndexWrapper(vectorstore=faiss_vector_store)

            st.session_state.pdf_processed =True

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
    gemini_llm = GeminiLLM(model_name='model/gemini-1.5-flash')
    if st.session_state.faiss_vector_index is not None:
        context = raw_text
        prompt = generate_rag_prompt(query=query_text,context=context)
        answer = generate_answer(prompt)

        # with app.app_context():
        #     new_prompt = Prompt(prompt=query_text, answer=answer)
        #     db.session.add(new_prompt)
        #     db.session.commit()
            
        typing_speed = 0.02
        if "context" or "no" in answer:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer, typing_speed))
        else:        
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer,typing_speed))
                
        st.session_state.messages.append({"role": "assistant", "content": answer})


prompt = st.chat_input("Chat with Shrirang...")
 
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    query_text = prompt.strip()
    gemini_llm = GeminiLLM(model_name='model/gemini-1.5-flash')
    if st.session_state.faiss_vector_index is not None:
        context = raw_text
        print("Query:",query_text)
        prompt = generate_rag_prompt(query=query_text,context=context)
        answer = generate_answer(prompt)
        # print("prompt:",prompt)
        print("answer:",answer)

        # Save the interaction to the database
        # with app.app_context():
        #     new_prompt = Prompt(prompt=query_text, answer=answer)
        #     db.session.add(new_prompt)
        #     db.session.commit()
        
        typing_speed = 0.02
        if "context" or "no" in answer:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer, typing_speed))
        else:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer,typing_speed))
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # st.experimental_rerun()

    else:
        st.error("Database not initialized. Kindly reload and upload the PDF first.")
 
