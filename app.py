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

def get_response(query, chat_history):
    template = template = """
As an AI knowledgeable about various restaurants, your task is to provide detailed information based on user inquiries. Ensure your responses are precise, utilizing the data available. If certain information is requested but not available, kindly inform the user with a polite message indicating the absence of that specific detail. Always maintain a courteous and supportive tone.

Given the user's questions and the conversation's context:

Chat history: {chat_history}

User's current question: "{user_question}"


Your responses should directly address the user's question, leveraging the provided chat history for context where necessary. If the necessary information is not detailed in our records, ensure to use the polite placeholder responses as indicated.
"""

# This template should be dynamically filled with the actual conversation history (`chat_history`), the user's current question (`user_question`), and data for each property. If data for a certain property is not available, the specified default message should be used instead.


# This is a Python template string. To use it, you'd fill in `chat_history` and `user_question` with the actual conversation history and the current user's question, respectively.

    
        
    prompt = ChatPromptTemplate.from_template(template=template)

    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

    chain_e = create_extraction_chain(schema, llm)

    chain =  prompt | chain_e

    return chain.stream({
        "chat_history": chat_history,
        "user_question" : query,
        })



#conversation
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)
#user input
user_query = st.chat_input("Your message")
if user_query is not None and user_query !="":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(ai_response))