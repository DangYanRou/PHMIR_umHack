# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:44:54 2024

@author: szeyu
"""


#%%

import os
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from IPython.display import Markdown
import textwrap


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
#%%
# Set the environment variable directly in the script
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

#%%

gemini_api_key = os.environ.get("GEMINI_API_KEY")
print(gemini_api_key)
openai_api_key = os.environ.get("OPENAI_API_KEY")
print(openai_api_key)

#%%

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')


#%%

response = model.generate_content("What is the meaning of life?")

#%%

print(response.text)


#%%

import pandas as pd

# Load Excel file into pandas DataFrame
excel_file = "UMH24 - FinTech Dataset.xlsx"  # Replace with your file path
df = pd.read_excel(excel_file)

df[['WITHDRAWAL AMT', 'DEPOSIT AMT']] = df[['WITHDRAWAL AMT', 'DEPOSIT AMT']].fillna(0)
# Set 'DATE' column as the index
df.set_index('DATE', inplace=True)

#%%

# Convert DataFrame to string
df_string = df.to_string(index=False)

# Print the string representation
print(df_string)

#%%

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
texts = text_splitter.split_text(df_string)
print(texts)

#%%

# Initialize GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)

vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})


#%%


# Verify initialization of model and vector_index
print(type(model))
print(type(vector_index))

# Check parameters being passed to RetrievalQA.from_chain_type
print(model)
print(vector_index)


#%%

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True

)

#%%


# Example query
query = "What is the amount of the fleet services fees?"

# Prompt query and get answer
answer = qa_chain.ask(query)

# Print the answer
print("Answer:", answer)



#%%
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=gemini_api_key,
                             temperature=0.2,convert_system_message_to_human=True)

query = "What is UMHackathon"
#qa = RetrievalQA.from_chain_type(llm=model, retriever=vectordb.as_retriever())

#response = qa.run(query)




#%%


import os
import pandas as pd
from pandasai import SmartDataframe

# Sample DataFrame
sales_by_country = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "sales": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})
#%%

# Get your FREE API key signing up at https://pandabi.ai.
# You can also configure it in your .env file.
os.environ["PANDASAI_API_KEY"] = ""

#%%
agent = SmartDataframe(sales_by_country)
agent.chat('Which are the top 5 countries by sales?')


#%%


df_pandasai = SmartDataframe(df)


#%%

df_pandasai.chat("Can you tell me what is my saving trends?")

#%%
from pandasai import Agent

agent = Agent(df);


# Train the model
query = "What is the total sales for the current fiscal year?"
response = """
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df[['WITHDRAWAL AMT', 'DEPOSIT AMT']] = df[['WITHDRAWAL AMT', 'DEPOSIT AMT']].fillna(0)
# Set 'DATE' column as the index
df.set_index('DATE', inplace=True)

withdrawals = df['WITHDRAWAL AMT']

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
plt.show()

"""
agent.train(queries=[query], codes=[response])



response = agent.chat("Can you tell me what is my saving trends?")
print(response)



#%%

from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
import pandasai.pandas as pd

llm = OpenAI()


# conversational=False is supposed to display lower usage and cost
df = SmartDataframe(df, config={"llm": llm, "conversational": False})

user_prompt = "Based on my expenses,what expenses can I cut down to save more money"

with get_openai_callback() as cb:
    response = df.chat("""The user prompt:""" + user_prompt + """
                       
                       The money format must be 2 decimal places
                       If user prompt about future or trends, then
                       State out what it is look like for the past spending data (Summary)
                       Given your past spending data, let's predict your expenses for the upcoming month. Please provide the following information:

                        1. Your total income for the month.
                        2. Any known fixed expenses (rent/mortgage, utilities, insurance, etc.).
                        3. Average monthly spending on variable expenses (groceries, dining out, entertainment, etc.).
                        4. Any upcoming one-time expenses (travel, major purchases, etc.).
                        5. Savings goals or investment contributions for the month.
                        
                        Based on this data, the model will generate a more accurate prediction of your spending for the next month. Additionally, it will offer suggestions or insights to help you manage your finances effectively.
                        """)

    print(response)
    # print(cb)
    
    
    


#%%

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI()

print(user_prompt)
print(response)

try:

    prompt = "The user prompt: " + user_prompt + "The response to user: " + response + """So if the user prompt ask more
    about future or trends stuff and the response to user is not that precise or good, you can respond more details
    if not applicable such that the user ask about his current data, then just say if got more question can ask more."""
    
    completion = client.completions.create(
        model="gpt-4",
        prompt=prompt,
    )
    print(completion.choices[0].message.content)

except:
    print("If you got more question feel free to ask more")

#%%

import openai

def chat_with_openai(user_prompt, response):
    if isinstance(response, pd.DataFrame):
        response = response.to_string(index=False)
    try:
        
        prompt = """
        
        The output format will be the response from your last chat history and also your enhancement to the answer.
        
        if not the user ask about his current data,
        then just say if got more question can ask more.
        
        else
        You are to act like a financial advisor.
        So if the user prompt ask more
        about future or trends stuff and the response to user is not that precise or good,
        you can respond more details
        
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

#%%

print("User prompt:", user_prompt)
print("Response:", response)

#%%

generated_response = chat_with_openai(user_prompt, response)
print(generated_response)



#%%


from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
import pandasai.pandas as pd

llm = OpenAI()


# conversational=False is supposed to display lower usage and cost
df = SmartDataframe(df, config={"llm": llm, "conversational": False})

user_prompt = "Can you tell me what is my saving trends?"

with get_openai_callback() as cb:
    response = df.chat("""The user prompt:""" + user_prompt + """

                       The money format must be 2 decimal places
                       If user prompt about future or trends, then
                       State out what it is look like for the past spending data (Summary)
                       Given your past spending data, let's predict your expenses for the upcoming month. Please provide the following information:

                        1. Your total income for the month.
                        2. Any known fixed expenses (rent/mortgage, utilities, insurance, etc.).
                        3. Average monthly spending on variable expenses (groceries, dining out, entertainment, etc.).
                        4. Any upcoming one-time expenses (travel, major purchases, etc.).
                        5. Savings goals or investment contributions for the month.
                        
                        Based on this data, the model will generate a more accurate prediction of your spending for the next month. Additionally, it will offer suggestions or insights to help you manage your finances effectively.
                        """)
    if not ("Unfortunately" in response or "error" in response or "was not able to answer" in response):
        print(response)
    generated_response = chat_with_openai(user_prompt, response)
    
    print(generated_response)
    # print(cb)
    
    


#%%

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

def create_pdf(text_and_image_paths, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    y_coordinate = height - 50  # Initial y-coordinate for drawing elements

    for item in text_and_image_paths:
        if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png') or item.endswith('.gif'):
            # If item is an image, draw the image
            try:
                img = Image.open(item)
                img_width, img_height = img.size
                aspect = img_height / img_width
                max_width = 400  # Adjust according to your needs
                max_height = int(max_width * aspect)
                
                if y_coordinate - max_height < 50:
                    c.showPage()  # New page
                    y_coordinate = height - 50  # Reset y-coordinate
                c.drawImage(item, 100, y_coordinate - max_height, width=max_width, height=max_height)
                c.drawString(100, y_coordinate - max_height - 20, item)  # Display the image path below the image
                y_coordinate -= max_height + 40  # Adjust y-coordinate for the next element
            except Exception as e:
                print(f"Error while processing image {item}: {e}")
        else:
            # If item is not an image, draw text
            if y_coordinate < 50:
                c.showPage()  # New page
                y_coordinate = height - 50  # Reset y-coordinate
            c.drawString(100, y_coordinate, item)
            y_coordinate -= 20  # Adjust y-coordinate for the next element

    c.save()

# Example usage:
text_and_image_paths = [
    "This is some text.",
    "C:\\Users\\szeyu\\Downloads\\20240320_144825.jpg",
    "Another text.",
    "C:\\Users\\szeyu\\Downloads\\20240320_150045.jpg",
    "Last text.",
    "text again",
    "another text again"
]

output_file = "output.pdf"
create_pdf(text_and_image_paths, output_file)







#%%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def getTrend(df, category=None):
    
    if category == None:
        withdrawals = df['WITHDRAWAL AMT']
    else:
        withdrawals = df.loc[df['CATEGORY'] == category, 'WITHDRAWAL AMT']
    
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
    plt.show()

#print(forecast_values)

#%%

getTrend(df, 'Investment')







#%%