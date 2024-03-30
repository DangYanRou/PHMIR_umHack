import streamlit as st
import os
import openai

from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
import pandasai.pandas as pd

os.environ["PANDASAI_API_KEY"] = "$2a$10$Psgcj0HiVxCmEscv1W5Dc.BorjRmFuaQppP4iXmtExYi0Ljyum2em"
os.environ["OPENAI_API_KEY"] = "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"

import pandas as pd
import plotly.express as px
from pandasai import SmartDataframe

llm = OpenAI()
 # Load Excel file into pandas DataFrame
excel_file = "UMH24 - FinTech Dataset.xlsx"  # Replace with your file path
df = pd.read_excel(excel_file)
# conversational=False is supposed to display lower usage and cost
df = SmartDataframe(df, config={"llm": llm, "conversational": False})

def get_bot_response(user_input):
   
    agent = SmartDataframe(df)
    # Get response from the agent
    response = agent.chat(user_input)
    
    return response

#prompt engineering
def chat_with_openai(user_prompt, response):
    if isinstance(response, pd.DataFrame):
        response = response.to_string(index=False)
    try:
        prompt = "The user prompt: " + user_prompt + "The response to user: " + response + """
        You are to act like a financial advisor.
        So if the user prompt ask more
        about future or trends stuff and the response to user is not that precise or good, you can respond more details
        if not applicable such that the user ask about his current data, then just say if got more question can ask more."""
        
        completion = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        
        return completion.choices[0].text.strip()

    except Exception as e:
        print("An error occurred:", e)
        return "If you have more questions, feel free to ask."


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
         generated_response = chat_with_openai(prompt, response)
         st.markdown(response+"\n"+generated_response)
    st.session_state.messages.append({"role": "assistant", "content": response+"\n"+generated_response})   
          
           # Generate graph based on bot response
    #if 'data' in response:
       #data = response['data']
    #fig = generate_graph(data)
  #Display graph in Streamlit app
       