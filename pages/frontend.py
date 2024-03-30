import requests
import streamlit as st
import os
import openai
import sys

# recommendation 
from tavily import TavilyClient
import json
import time

from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
import pandasai.pandas as pd

os.environ["PANDASAI_API_KEY"] = "$2a$10$YWEW2uYAvqkP/n3sucIwY.cTG4OhViG72IxL4WWdjvaaM6Vs9yLtq"
os.environ["OPENAI_API_KEY"] = "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"
tavily_client = TavilyClient(api_key="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")

import pandas as pd
import plotly.express as px
from pandasai import SmartDataframe

#pdf library
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

import matplotlib.pyplot as plt

# for time series analysis
from statsmodels.tsa.arima.model import ARIMA





llm = OpenAI()
 # Load Excel file into pandas DataFrame
excel_file = "../UMH24 - FinTech Dataset.xlsx"  # Replace with your file path
sheet_name = "Ahmad"
#print(sys.argv[0])
#file_path = sys.argv[2]

df = pd.read_excel(excel_file, sheet_name)

# drop the unnecessary transaction ID
df.drop(columns="TRANSACTION ID", inplace=True)

# fill the NaN with 0
df[['WITHDRAWAL AMT', 'DEPOSIT AMT']] = df[['WITHDRAWAL AMT', 'DEPOSIT AMT']].fillna(0)

# Set 'DATE' column as the index
df.set_index('DATE', inplace=True)

# conversational=False is supposed to display lower usage and cost
df = SmartDataframe(df, config={"llm": llm, "conversational": False})



client = openai.OpenAI(api_key="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")
# Create an assistant
assistant = client.beta.assistants.create(
    instructions="""You are a finance expert. 
    Your goal is to provide answers based on information from the internet. 
    You must use the provided Tavily search API function to find relevant online information. 
    You should never use your own knowledge to answer questions.
    Please include relevant url sources in the end of your answers.
    """,
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Get information on recent events from the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to use. For example: 'Latest news on Nvidia stock performance'"},
                },
                "required": ["query"]
            }
        }
    }]
)
assistant_id = assistant.id
print(f"Assistant ID: {assistant_id}")

# Create a thread
thread = client.beta.threads.create()
print(f"Thread: {thread}")

# Function to perform a Tavily search
def tavily_search(query):
    search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    return search_result

# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "tavily_search":
            output = tavily_search(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )

# Function to print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages:
        st.session_state.messages.append({"role": "assistant", "content": msg})
        print(f"{msg.role}: {msg.content[0].text.value}")
        
        
def chatbot_recommendation(user_input):
    # Create a message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    # Create a run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    print(f"Run ID: {run.id}")

    # Wait for run to complete
    run = wait_for_run_completion(thread.id, run.id)

    if run.status == 'failed':
        print(run.error)
        return
    
    elif run.status == 'requires_action':
        run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
        run = wait_for_run_completion(thread.id, run.id)

    # Print messages from the thread
    print_messages_from_thread(thread.id)


def getTrend(category=None):
    dfTemp = pd.read_excel(excel_file, sheet_name)

    # drop the unnecessary transaction ID
    dfTemp.drop(columns="TRANSACTION ID", inplace=True)

    # fill the NaN with 0
    dfTemp[['WITHDRAWAL AMT', 'DEPOSIT AMT']] = dfTemp[['WITHDRAWAL AMT', 'DEPOSIT AMT']].fillna(0)

    # Set 'DATE' column as the index
    dfTemp.set_index('DATE', inplace=True)
    
    if category == None:
        withdrawals = dfTemp['WITHDRAWAL AMT']
    else:
        withdrawals = dfTemp.loc[dfTemp['CATEGORY'] == category, 'WITHDRAWAL AMT']
    
    # Fit an ARIMA model
    model = ARIMA(withdrawals, order=(5,1,1))  # Example ARIMA model, you may need to tune parameters
    model_fit = model.fit()
    
    # Make forecasts for the next 7 days
    forecast_values = model_fit.forecast(steps=7)
    
    
    # Plot original time series data
    plt.plot(withdrawals.index, withdrawals, label='Original Data')
    
    # Plot forecasted values
    forecast_index = pd.date_range(withdrawals.index[-1], periods=8)[1:]  # Forecasted index for the next 7 days
    plt.plot(forecast_index, forecast_values, color='red', linestyle='--', label='Forecast')
    
    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Net Amount')
    plt.title('Forecast')
    plt.legend()
    
    # Show plot
    #plt.show()
    
    fig = plt.gcf()
    st.pyplot(fig)
    st.session_state.messages.append({"role": "assistant", "content": fig})
    

def create_pdf(text_and_image_paths, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    margin = 20  # Margin from all sides
    max_width = width - 2 * margin  # Maximum width for text/image

    y_coordinate = height - margin  # Initial y-coordinate for drawing elements
    remaining_text = ""

    for item in text_and_image_paths:
        content = item["content"]
        
        if content.endswith('.jpg') or content.endswith('.jpeg') or content.endswith('.png') or content.endswith('.gif'):
            # If content is an image, draw the image
            try:
                img = Image.open(content)
                img_width, img_height = img.size
                aspect = img_height / img_width

                # Scale image to fit within the page width
                scaled_width = min(img_width, max_width)
                scaled_height = scaled_width * aspect
                
                if y_coordinate - scaled_height < margin:
                    c.showPage()  # New page
                    y_coordinate = height - margin  # Reset y-coordinate
                
                c.drawImage(content, margin, y_coordinate - scaled_height, width=scaled_width, height=scaled_height)
                y_coordinate -= scaled_height + margin  # Adjust y-coordinate for the next element
            except Exception as e:
                print(f"Error while processing image {content}: {e}")
        else:
            # If content is not an image, draw text
            lines = (remaining_text + content).split("\n")
            remaining_text = ""
            for line in lines:
                while c.stringWidth(line) > max_width:
                    # Find the maximum index that fits within the max_width
                    index = 0
                    while c.stringWidth(line[:index]) <= max_width:
                        index += 1
                    index -= 1

                    # Draw the portion of the line that fits within the max_width
                    text_line = line[:index]
                    c.drawString(margin, y_coordinate, text_line)
                    y_coordinate -= margin  # Adjust y-coordinate for the next line

                    # Update line to the remaining text
                    line = line[index:]

                    # Check if new line exceeds the page, if yes, move to the next page
                    if y_coordinate < margin:
                        c.showPage()  # New page
                        y_coordinate = height - margin  # Reset y-coordinate
                else:
                    # Draw the remaining part of the line
                    c.drawString(margin, y_coordinate, line)
                    y_coordinate -= margin  # Adjust y-coordinate for the next line
            
    c.save()

def chatbot_future(prompt):
    flag = False
    categories = ['Income/Salary', 'Utilities', 'Other Expenses', 'Government Services', 
                  'Groceries', 'Insurance', 'Dining', 'Health & Fitness', 'Entertainment', 
                  'Transportation', 'Debts/Overpayments', 'Education', 'Shopping', 
                  'Savings', 'Investment', 'Travel']
    
    for category in categories:
        keywords = {'Income/Salary': ['income', 'salary'],
                    'Utilities': ['utilities', 'utility'],
                    'Other Expenses': ['other expenses','other expense'],
                    'Government Services': ['government services','government'],
                    'Groceries': ['groceries','grocery'],
                    'Insurance': ['insurance'],
                    'Dining': ['dining','eat'],
                    'Health & Fitness': ['health', 'fitness'],
                    'Entertainment': ['entertainment','relaxation','enjoy'],
                    'Transportation': ['transportation','transport'],
                    'Debts/Overpayments': ['debts', 'overpayments'],
                    'Education': ['education'],
                    'Shopping': ['shopping'],
                    'Savings': ['savings','saving'],
                    'Investment': ['investment','invest'],
                    'Travel': ['travel']}
        
        for keyword in keywords[category]:
            if keyword in prompt.lower():
                getTrend(category)
                flag = True
                break;
        
    if(not flag):
        getTrend();



def chatbot_current(user_input):
   
    agent = SmartDataframe(df)
    # Get response from the agent
    response = agent.chat(user_input)
    
    return response

#prompt engineering
def chat_with_openai(user_prompt, response):
    if isinstance(response, pd.DataFrame):
        response = response.to_string(index=False)
    try:
        
        prompt = """
        you can respond more details
        else if nothing to comment can just say If you have more questions, feel free to ask.
        """
        
        messages = [{"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response},
                    {"role": "assistant", "content": prompt}]
        
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        print("An error occurred:", e)
        return "If you have more questions, feel free to ask."


# Function to generate graph
def generate_graph(data):
    # Use Plotly Express to create a simple bar chart
    fig, ax = plt.subplots(figsize = (1200, 600))
    fig = px.bar(data, x='Category', y='Amount', title='Expenses by Category')
    plt.setp(ax.get_xticklabels(), rotation = 90)
    return fig

st.title("Personal Finance Chatbot")
if st.button("Save Chat as PDF"):
    create_pdf(st.session_state.messages, "chat.pdf")

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
    with st.spinner('Loading...'):
        #bot_response = chatbot_current(prompt)
        pass
    
  # Display assistant response in chat message container
    with st.chat_message("assistant"):
        generated_response = ""
        
        # if the users ask about trends
        if any(keyword in prompt.lower() for keyword in ["trend", "future", "predict", "prediction", "forecast"]):
            chatbot_future(prompt)
            
        # if the users ask about recommendation
        elif any(keyword in prompt.lower() for keyword in ["recommendation", "suggestion", "recommend", "suggest"]):
            chatbot_recommendation(prompt)
        
        # send the users current data analysis
        else:
            response = chatbot_current(prompt)
            # to conver the response to string in case it is a data frame
            if isinstance(response, pd.DataFrame):
                response = response.to_string(index=False)
                
            # if the pandasai fail to analyse the prompt then don't output it
            if not ("Unfortunately" in response or "error" in response or "was not able to answer" in response):
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
                
            generated_response = chat_with_openai(prompt, response)
            st.markdown("\n"+generated_response)
        st.session_state.messages.append({"role": "assistant", "content": generated_response})   
          
    # Generate graph based on bot response
    if 'generate graph' in prompt.lower():
        data = df  
        fig = generate_graph(data)
        st.plotly_chart(fig, use_container_width=True)
