import streamlit as st
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="ChatBot",page_icon="🤖")

st.title("ChatBot")



schema = {
    "properties": {
        "location": {"type": "string", "example": "Where is the restaurant located?"},
        "dish": {"type": "string", "example": ["What dishes are recommended at the restaurant?", "Can you suggest a vegetarian dish?"]},
        "cuisine": {"type": "string", "example": "What type of cuisine does the restaurant serve?"},
        "type_of_restaurant": {"type": "string", "example": "Is it a fine dining place or a casual dining restaurant?"},
        "ambience": {"type": "string", "example": "What's the ambiance like at the restaurant?"},
        "average_cost": {"type": "string", "example": "What's the average cost for a meal for two?"},
        "user_reviews": {"type": "string", "example": ["Are there any recent reviews about the restaurant?", "What do people usually say about their service?"]},
        "rating": {"type": "string", "example": "What is the restaurant's overall rating?"},
    },
    "required": ["location", "cuisine"]
}



#get response

def get_response(query):
    template = template = """
As an AI knowledgeable about various restaurants, your task is to provide detailed information based on user inquiries. Ensure your responses are precise, utilizing the data available. If certain information is requested but not available, kindly inform the user with a polite message indicating the absence of that specific detail. Always maintain a courteous and supportive tone.



User's current question: "{user_question}"


Your responses should directly address the user's question. If, the necessary information is not detailed in our records, ensure to use the polite placeholder responses as indicated.
"""

        
    prompt = ChatPromptTemplate.from_template(template=template)

    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

    chain_e = create_extraction_chain(schema, llm)

    chain =  prompt | chain_e

    return chain.stream({
        
        "user_question" : query,
        })




#user input
user_query = st.chat_input("Your message")
if user_query is not None and user_query !="":
    

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query))

        st.markdown(AIMessage(ai_response))