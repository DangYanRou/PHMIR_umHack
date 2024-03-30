import streamlit as st

# Define page functions
def page_1():
    st.write("Welcome to Page 1")

def page_2():
    st.write("Welcome to Page 2")

def page_3():
    st.write("Welcome to Page 3")

def page_4():
    st.write("Welcome to Page 4")

def page_5():
    st.write("Welcome to Page 5")

# Dictionary to store user credentials and corresponding page names
users = {
    "user1": {"password": "password1", "page": "Page 1"},
    "user2": {"password": "password2", "page": "Page 2"},
    "user3": {"password": "password3", "page": "Page 3"},
    "user4": {"password": "password4", "page": "Page 4"},
    "user5": {"password": "password5", "page": "Page 5"},
}


st.title("Login Page")

# Get user input
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# Check if the user has entered correct credentials
if st.button("Login"):
    if username in users:
        if password == users[username]["password"]:
            st.success("Logged in successfully!")
            # Show page links in the sidebar based on the user
            st.sidebar.title("Navigation")
            page = users[username]["page"]
            if page == "Page 1":
                page_1()
            elif page == "Page 2":
                page_2()
            elif page == "Page 3":
                page_3()
            elif page == "Page 4":
                page_4()
            elif page == "Page 5":
                page_5()
        else:
            st.error("Invalid password. Please try again.")
    else:
        st.error("Invalid username. Please try again.")




