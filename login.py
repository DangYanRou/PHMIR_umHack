import streamlit as st
import sys
import subprocess
# cd .\pages
# python -m streamlit run login.py

# Define page functions
def page_1():
    st.write("Welcome Ahamad!")
    str_parse = "Ahmad"

def page_2():
    st.write("Welcome Bryan!")
    str_parse = "Bryan"

def page_3():
    st.write("Welcome Charles!")
    str_parse = "Charles"

def page_4():
    st.write("Welcome Danish")
    str_parse = "Danish"

def page_5():
    st.write("Welcome Emily!")
    str_parse = "Emily"

# Dictionary to store user credentials and corresponding page names
users = {
    "ahmad": {"password": "password1", "page": Ahmad},
    "bryan": {"password": "password2", "page": Bryan},
    "charles": {"password": "password3", "page": Charles},
    "danish": {"password": "password4", "page": Danish},
    "emily": {"password": "password5", "page": Emily},
}


st.title("Login Page")

# Get user input
username = st.text_input("Username")
password = st.text_input("Password", type="password")

str_parse = ""

# Check if the user has entered correct credentials
if st.button("Login"):
    if username in users:
        if password == users[username]["password"]:
            st.success("Logged in successfully!")
            # Show page links in the sidebar based on the user
            st.sidebar.title("Navigation")
            to_page = users[username]["page"]
            if to_page == "Page 1":
                page_1()
            elif to_page == "Page 2":
                page_2()
            elif to_page == "Page 3":
                page_3()
            elif to_page == "Page 4":
                page_4()
            elif to_page == "Page 5":
                page_5()
            st.switch_page("pages/frontend.py")
        else:
            st.error("Invalid password. Please try again.")
    else:
        st.error("Invalid username. Please try again.")
