import streamlit as st
import subprocess
# streamlit run welcome.py --theme.base="light" --theme.primaryColor="#6fb3b8" --theme.backgroundColor="#f6f6f2" --theme.secondaryBackgroundColor="#F0F2F6" --theme.textColor="#3B6467" --theme.font="sans serif"

primary = '#6fb3b8'
background = '#f6f6f2'

# set page configuration
st.set_page_config(
    page_title="Welcome to PHMIR Chat",
)

# CSS define for background color
background_style = f"""
    <style>
        [data-testid = "stAppViewContainer"] {{
            background-color: {background};
        }}
    </style>
"""

# Display background colour with HTML
st.markdown(background_style, unsafe_allow_html=True)

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

st.markdown('''<h1 style="font-size: 64px">Welcome</h1>''', unsafe_allow_html=True)
st.markdown('''<h3 style="font-size: 24px">PHMIR is an AI chat-bot that provides personal finance assitance.</h3>''', unsafe_allow_html=True)

getStarted_btn = st.button("Get Started →")
if getStarted_btn:
    st.page_link("frontend.py")