import streamlit as st
import os

os.environ["PANDASAI_API_KEY"] = "$2a$10$Psgcj0HiVxCmEscv1W5Dc.BorjRmFuaQppP4iXmtExYi0Ljyum2em"
import pandas as pd
import plotly.express as px
from pandasai import SmartDataframe


def get_bot_response(user_input):
    # Load Excel file into pandas DataFrame
    excel_file = "UMH24 - FinTech Dataset.xlsx"  # Replace with your file path
    df = pd.read_excel(excel_file)
    agent = SmartDataframe(df)
    
    # Get response from the agent
    response = agent.chat(user_input)
    
    return response

# Function to generate graph
def generate_graph(data):
    # Use Plotly Express to create a simple bar chart
    fig = px.bar(data, x='Category', y='Amount', title='Expenses by Category')
    return fig

st.title("Personal Finance Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

     # Accept user input
prompt= st.chat_input("What is up?")
if prompt :
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

          # Get response from the bot
    with st.spinner('Thinking...'):
        bot_response = get_bot_response(prompt)
        
  # Display assistant response in chat message container
    with st.chat_message("assistant"):
         response = get_bot_response(prompt)
         st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})   
          
           # Generate graph based on bot response
    if 'data' in response:
        data = response['data']
        fig = generate_graph(data)
        # Display graph in Streamlit app
        st.plotly_chart(fig)