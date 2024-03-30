import streamlit as st
import json
from streamlit_lottie import st_lottie
import requests
st.set_page_config(layout="centered")
st.title("Pricing")

html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pricing Page</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }

        .plan {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 10px;
            margin: 10px;
            padding: 20px;
            max-width: 300px;
            width: 100%;
            box-sizing: border-box;
        }

        .plan h2 {
            color: #333;
            font-size: 24px;
            margin-top: 0;
        }

        .plan ul {
            list-style: none;
            padding: 0;
        }

        .plan ul li {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>

<div class="plan">
    <h2>Plan A: Free</h2>
    <ul>
        <li>Max file input size: 25MB</li>
        <li>Prompts per day: 15</li>
        <li>Access to prediction model: No</li>
        <li>Supported file format: CSV</li>
    </ul>
</div>

<div class="plan">
    <h2>Plan B: $15/month</h2>
    <ul>
        <li>Max file input size: 100MB</li>
        <li>Prompts per day: 50</li>
        <li>Access to prediction model: Yes</li>
        <li>Supported file formats: CSV, XLSX</li>
    </ul>
</div>

<div class="plan">
    <h2>Plan C: $30/month</h2>
    <ul>
        <li>Max file input size: 300MB</li>
        <li>Prompts per day: 100</li>
        <li>Access to prediction model: Yes</li>
        <li>Supported file formats: CSV, PDF, DOCX, XLSX</li>
    </ul>
</div>

<div class="plan">
    <h2>Plan D: $40/month</h2>
    <ul>
        <li>Max file input size: 300MB</li>
        <li>Prompts per day: 500</li>
        <li>Access to prediction model: Yes</li>
        <li>Supported file formats: CSV, PDF, DOCX, XLSX</li>
    </ul>
</div>

</body>
</html>

"""



st.markdown(html_code, unsafe_allow_html=True)
