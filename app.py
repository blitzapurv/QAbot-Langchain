import streamlit as st
import os
import time
import logging
from src.chat_engine import ChatBotModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

chat_bot = ChatBotModel(logger, model_name="TheBloke/Llama-2-7b-Chat-GPTQ", embedding_model_name="all-MiniLM-L6-v2")
qa = chat_bot.create_qa_instance(file_path="./sample_pdf.pdf")

def run():
    st.title('Conversation Bot Demo')
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask something.."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        model_output = qa(prompt)
        response = f"AI: {model_output['answer']}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == '__main__':

    run()
