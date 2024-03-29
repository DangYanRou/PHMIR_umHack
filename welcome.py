import streamlit as st
import subprocess
# python -m streamlit run welcome.py
primary = '#6fb3b8'
background = '#f6f6f2'

# set page configuration
st.set_page_config(
    page_title="Welcome to PHMIR Chat",
)

# CSS define for header
header_style = f"""
    <style>
        [data-testid = "stHeader"] {{
            background-color: {primary};
            position: relative;
            text-align: center;
        }}
    </style>
"""
st.markdown(header_style, unsafe_allow_html=True)

st.markdown('''<h1 style="font-size: 72px">Welcome</h1>''', unsafe_allow_html=True)
st.write("")

st.markdown('''<h3 style="font-size: 24px">PHMIR is an AI chat-bot that provides personal finance assitance.</h3>''', unsafe_allow_html=True)
st.write("")

st.page_link("pages/login.py", label="Get Started →")