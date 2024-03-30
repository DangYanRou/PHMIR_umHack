import streamlit as st

st.set_page_config(layout="centered")
st.title("Subscriptions")

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
            max-width: 500px;
            width: 100%;
            box-sizing: border-box;
            animation-name: gradient;
            animation-duration: 5s;
        }

        @keyframes gradient {
            from {background-color: #4f4f4f;}
            to {background-color: #f4f4f4;}
        }

        .plan h2 {
            color: #000000;
            font-size: 24px;
            margin-top: 0;
        }

        .plan ul {
            list-style: none;
            padding: 0;
        }

        .plan ul li {
            color: #000000; /* Changed text color to black */
            margin-bottom: 10px;
        }
    </style>
</head>
<body>

<div class="plan">
    <h2>Plan A: Free</h2>
    <ul>
        <li>Prompts per day: 15</li>
        <li>Access to prediction model: No</li>
    </ul>
</div>

<div class="plan">
    <h2>Plan B: RM15/month</h2>
    <ul>
        <li>Prompts per day: 50</li>
        <li>Access to prediction model: Yes</li>
    </ul>
</div>

<div class="plan">
    <h2>Plan C: RM30/month</h2>
    <ul>
        <li>Prompts per day: 100</li>
        <li>Access to prediction model: Yes</li>
    </ul>
</div>

<div class="plan">
    <h2>Plan D: RM40/month</h2>
    <ul>
        <li>Prompts per day: 500</li>
        <li>Access to prediction model: Yes</li>
    </ul>
</div>

</body>
</html>
"""

st.markdown(html_code, unsafe_allow_html=True)
