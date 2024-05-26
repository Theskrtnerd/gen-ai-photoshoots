import google.generativeai as genai
import streamlit as st
import json


def validate_gemini_api_key(my_api_key):
    """
    Validate the provided Gemini API key by configuring the genai client and
    testing a content generation request.

    Args:
        my_api_key (str): The Gemini API key to be validated.

    Returns:
        bool: True if the API key is valid and the content generation request
              is successful, False otherwise.

    This function performs the following steps:
        1. Configures the genai client with the provided API key.
        2. Attempts to create a GenerativeModel instance and generate content
           to validate the API key.
        3. Returns True if successful; otherwise, displays the error message
           and returns False.
    """
    try:
        genai.configure(api_key=my_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        model.generate_content("What's my name?")
        return True

    except Exception as e:
        st.error(e)
        return False


def generate_reccommendations():
    """
    Generate recommended background places for the user's product using the
    Gemini API.

    Returns:
        list: A list of recommended background places for the product.

    This function performs the following steps:
        1. Retrieves the Gemini API key and product name from the session state.
        2. Configures the genai client with the API key.
        3. Creates a GenerativeModel instance and constructs a prompt to generate
           recommendations.
        4. Sends the prompt to the model and parses the JSON response to extract
           the list of recommended places.
    """
    gemini_api_key = st.session_state.api_key
    product_name = st.session_state.product_name
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f'What are some great recommended background places for a {product_name}?'
    prompt += 'Give me the answer as a list of places. There should be 5 relevant places.'
    prompt += 'The answer\'s format should be as follows: \'["on a beach", "in a park", "near a flowing river"]\'.'
    prompt += 'You need to follow this answer\'s format strictly. Don\'t ask more questions.'
    response = model.generate_content(prompt)
    return json.loads(response.text)
