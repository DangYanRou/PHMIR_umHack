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
os.environ["GEMINI_API_KEY"] = "AIzaSyCt0auU8CC-s8BuiNM4tWlhK3MFK181dZ0"
os.environ["OPENAI_API_KEY"] = "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"

#%%

gemini_api_key = os.environ.get("GEMINI_API_KEY")
print(gemini_api_key)


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

# Get your FREE API key signing up at https://pandabi.ai.
# You can also configure it in your .env file.
os.environ["PANDASAI_API_KEY"] = "$2a$10$YWEW2uYAvqkP/n3sucIwY.cTG4OhViG72IxL4WWdjvaaM6Vs9yLtq"

agent = SmartDataframe(sales_by_country)
agent.chat('Which are the top 5 countries by sales?')


#%%


df_pandasai = SmartDataframe(df)


#%%

df_pandasai.chat("How much did I spend on 2024-01-01?")





#%%

from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
import pandasai.pandas as pd

llm = OpenAI()


# conversational=False is supposed to display lower usage and cost
df = SmartDataframe(df, config={"llm": llm, "conversational": False})

user_prompt = "Can you show my last 5 transactions record?"

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
    
    
    
    print(cb)
    


#%%




#%%








#%%











#%%








#%%










#%%