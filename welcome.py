import streamlit as st

primary = '#6fb3b8'
background = '#f6f6f2'

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