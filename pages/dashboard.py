import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

st.set_page_config(layout="wide")

# Define a function to load the dashboard page

excel_file="C:/Users/leeyu/Downloads/UMH24 - FinTech Dataset.xlsx"
df=pd.read_excel(excel_file)

#def load_page(selected_page):
#     if selected_page == "Home":
#          st.switch_page("Home.py")
#     elif selected_page == "Dashboard":
#          st.set_page_config(layout="wide")
          


#----- side bar
st.sidebar.header("Please Filter Here:")
category=st.sidebar.multiselect("Select the Category:",options=df["CATEGORY"].unique(),default=df["CATEGORY"].unique())
start_date = st.sidebar.date_input("Select Start Date:")
end_date = st.sidebar.date_input("Select End Date:")

# Filtering DataFrame based on selected categories
df_selection = df[df["CATEGORY"].isin(category)]

if start_date and end_date:
     start_date = pd.to_datetime(start_date)
     end_date = pd.to_datetime(end_date)
     df_selection = df_selection[(df_selection["DATE"] >= start_date) & (df_selection["DATE"] <= end_date)]


#mainpage
st.title(":bar_chart: Financial Dashboard")
st.markdown("##")



#Finacial Health KPI
# Filter the DataFrame for transactions in the current month
# Convert the 'DATE' column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Get the current date
current_date = datetime.now()
current_month_transactions = df[df['DATE'].dt.month == current_date.month]


# Filter the DataFrame for income and debt/overpayment transactions
income_and_debt = current_month_transactions[(current_month_transactions['CATEGORY'] == 'Income/Salary') | (current_month_transactions['CATEGORY'] == 'Debts/Overpayments')]

total_income= income_and_debt[income_and_debt['CATEGORY']=='Income/Salary']['DEPOSIT AMT'].sum()
total_debt_overpayment= income_and_debt[income_and_debt['CATEGORY']=='Debts/Overpayments']['WITHDRAWAL AMT'].sum()

# Calculate the debt-to-income ratio
debt_to_income_ratio = total_debt_overpayment / total_income

#calculate spending
total_spent=df_selection["WITHDRAWAL AMT"].sum()

#calculate balane
current_balance=df["DEPOSIT AMT"].sum()-df["WITHDRAWAL AMT"].sum()

# Determine color based on debt-to-income ratio value
color = "green" if debt_to_income_ratio <= 0.35 else "red"

left_column,middle_column,right_column=st.columns(3)
with left_column:
     st.subheader("Debt-to-Income Ratio:")
     st.markdown(f'<span style="color:{color};font-size:30px;font-weight:bold;">{debt_to_income_ratio*100:.2f} % :smile:</span>', unsafe_allow_html=True)
with middle_column:
     st.subheader("Total Spent:")
     st.subheader(f"RM {total_spent}")
with right_column:
     st.subheader("Current Balance:")
     st.subheader(f"RM {current_balance}")





df_selection
st.markdown("---")    
#total spent by category
total_spent_category=(df_selection.groupby(by=["CATEGORY"]).sum()[["WITHDRAWAL AMT"]])
total_spent_category_sorted = total_spent_category.sort_values(by="WITHDRAWAL AMT", ascending=True)
     
fig_category=px.bar(total_spent_category,x=total_spent_category.index,y="WITHDRAWAL AMT", orientation="v",title="<b>Total Spent by Category</b>",color_discrete_sequence=["#0083B8"]*len(total_spent_category_sorted),template="plotly_white",)
fig_category.update_layout(yaxis_title="Total Withdrawal Amount",width=1200,height=600,)
st.plotly_chart(fig_category)

# Filter the DataFrame to include only rows where the Category is "Other Expenses"
other_expenses_df=df_selection[df_selection["CATEGORY"]=="Other Expenses"]

# Sort the DataFrame by the "Transaction Details" column in ascending order
other_expenses_df_sorted = other_expenses_df.sort_values(by="TRANSACTION DETAILS",ascending=True)
grouped_df=other_expenses_df_sorted.groupby("TRANSACTION DETAILS")["WITHDRAWAL AMT"].sum().reset_index()

#Create a plot using Plotly Express
fig=px.bar(grouped_df,x="TRANSACTION DETAILS",y="WITHDRAWAL AMT",title="Other Expenses Breakdown",labels={"Transaction Details":"Subcategory","Withdrawal Amount":"Total Withdrawal Amount"})

# Customize the layout if needed
fig.update_layout(xaxis_title="Subcategory", yaxis_title="Total Withdrawal Amount",width=1200,height=600,)

# Display the plot
st.plotly_chart(fig)