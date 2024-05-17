import streamlit as st
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configure the Google Gemini API
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_GEMINI_API")
genai.configure(api_key=gemini_api_key)

# Create the Streamlit app
st.title('Product Background Recommendations')

product = st.text_input('Enter your product:', value='', key="input")

if "cache" not in st.session_state:
    st.session_state.cache = []

if "chosen" not in st.session_state:
    st.session_state.chosen = False

if "product" not in st.session_state:
    st.session_state.product = ''

if product != '' and product != st.session_state.product:

    st.session_state.product = product
    st.session_state.chosen = False

    # Generative model setup
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    if product:
        # Generate recommendations
        prompt = f'What are some great background places for a {product}? Give me the answer as a list of places. There should be 5 relevant places. The answer\'s format should be as follows: \'["on a beach", "in a park", "near a flowing river"]\' You need to follow this answer\'s format strictly. Don\'t ask more questions.'
        response = model.generate_content(prompt)
        st.session_state.recs = json.loads(response.text)

if "recs" in st.session_state:
    if not st.session_state["chosen"]:
        st.write("### Here's your list of places reccommendations:")

    for rec in st.session_state.recs:
        button_key = f"{product}_{rec.replace(' ', '_')}"
        if not st.session_state["chosen"]:
            st.session_state[button_key] = False
            if st.button(rec):
                st.session_state[button_key] = True
                st.session_state["chosen"] = True
                st.rerun()
        
        elif st.session_state[button_key]:
            st.write(f"### You selected {rec}!")
        
            

            
    