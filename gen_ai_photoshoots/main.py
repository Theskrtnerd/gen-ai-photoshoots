import streamlit as st
import os
import google.generativeai as genai
from model_training import train_model
import shutil
import json
from images_generate import generate_image


def validate_gemini_api_key(my_api_key):
    try:
        genai.configure(api_key=my_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        model.generate_content("What's my name?")
        return True

    except Exception as e:
        st.error(e)
        return False


def generate_reccommendations():
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


def stage_0():
    st.write("Ready to showcase your product in its best light, but lacking professional photography tools?")
    st.write("Let our AI do the work for you.")
    st.write("Craft exquisite, professional-grade product images effortlessly, and at no cost.")
    st.write("Transform your brand's visual presentation with ease.")
    if st.button("Start Creating"):
        st.session_state.current_stage = 1
        st.rerun()


def stage_1():
    st.write("To start, go to [this link](https://aistudio.google.com/app/apikey)"
             + " to create your Gemini API key (Don't worry, it's free)")
    api_key = st.text_input("Enter your Google Gemini API Key here:")
    if api_key:
        if not validate_gemini_api_key(api_key):
            st.error("Not the correct API key")
        else:
            st.session_state.current_stage = 2
            st.session_state.api_key = api_key
            st.rerun()


def stage_2():
    st.write("Now upload 5-10 photos of your product, it can be photos taken of your product from different angles")
    files = st.file_uploader(
        "Upload images of your product:",
        type=["jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload here...")

    if files:
        if os.path.exists("./training_photos/"):
            shutil.rmtree("./training_photos/")
        os.makedirs("./training_photos/")
        for i, file in enumerate(files):
            with open(os.path.join("./training_photos/", f"image_{i}.jpg"), "wb") as f:
                f.write(file.getbuffer())
        st.session_state.current_stage = 3
        st.rerun()


def stage_3():
    st.write("What's your product's name?")
    st.write("E.g. corggi dog, champions league soccer ball, red scarf with patterns")
    product_name = st.text_input("Enter your product's name here:")
    if product_name:
        st.session_state.current_stage = 4
        st.session_state.product_name = product_name
        st.rerun()


def stage_4():
    st.write("Now it's time to train your model.")
    st.write("Please wait a while for us to understand your product.")
    product_name = st.session_state.product_name
    my_bar = st.progress(0, text="Training in progress...")
    try:
        train_model(product_name, my_bar)
        st.session_state.current_stage = 5
        st.rerun()
    except Exception as e:
        st.error(e)


def stage_5():
    st.write("Everything's finally done, Hooray!")
    st.write("Now's the time to create stunning photos with your product.")
    st.write("Enter a background you want or choose a reccommendation below")
    recs = generate_reccommendations()
    for rec in recs:
        if st.button(rec):
            st.session_state.current_stage = 6
            st.session_state.background = rec
            st.rerun()
    text_input = st.text_input("Enter a background: ")
    if text_input:
        st.session_state.current_stage = 6
        st.session_state.background = text_input
        st.rerun()


def stage_6():
    st.write("Nice Choice!")
    product_name = st.session_state.product_name
    background = st.session_state.background
    prompt = f"a photo of a {product_name} {background}"
    st.write(f"Finally, here's your {product_name} {background}")
    image = generate_image(prompt)
    st.image(image)


st.title("AI-Powered Product Photoshoot Wizard")

if "current_stage" not in st.session_state:
    st.session_state.current_stage = 0

curr_stage = st.session_state.current_stage

if curr_stage == 0:
    stage_0()

elif curr_stage == 1:
    stage_1()

elif curr_stage == 2:
    stage_2()

elif curr_stage == 3:
    stage_3()

elif curr_stage == 4:
    stage_4()

elif curr_stage == 5:
    stage_5()

elif curr_stage == 6:
    stage_6()
