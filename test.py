import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="ChatBot", page_icon="🤖")

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

def get_response(query, chat_history):
    template = """You are a helpful assistant. Answer the following questions considering the history of the conversation.
    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)

    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

    chain_e = create_extraction_chain(schema, llm)

    chain = prompt | chain_e 

    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })

# Display conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.info(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        st.success(f"AI: {message.content}")
    # Example for SystemMessage, if needed in the future
    elif isinstance(message, SystemMessage):
        st.warning(f"System: {message.content}")

# User input
user_query = st.text_input("Your message", key="user_query")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Simulate AI response for demonstration. Replace the next line with actual AI response logic.
    ai_response = "This is a simulated response based on your query."
    st.session_state.chat_history.append(AIMessage(content=ai_response))

    # Clear the input field after the response.
    st.experimental_rerun()

