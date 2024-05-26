import streamlit as st
import os
from gen_ai_photoshoots.train_model import train_model
import shutil
from gen_ai_photoshoots.generate_image import generate_image
from gen_ai_photoshoots.gemini_api import validate_gemini_api_key, generate_reccommendations

# Main Application with GUI


def stage_0():
    """
    Display the introductory page of the application.

    This page explains the purpose of the application and provides a button to
    start the creation process.
    """
    st.write("Ready to showcase your product in its best light, but lacking professional photography tools?")
    st.write("Let our AI do the work for you.")
    st.write("Craft exquisite, professional-grade product images effortlessly, and at no cost.")
    st.write("Transform your brand's visual presentation with ease.")
    if st.button("Start Creating"):
        st.session_state.current_stage = 1
        st.rerun()  # Move to next page


def stage_1():
    """
    Display the API key input page.

    This page instructs the user to create a Gemini API key and input it into
    the text box provided. The key is validated, and the user is moved to the
    next stage if the key is valid.
    """
    st.write("To start, go to [this link](https://aistudio.google.com/app/apikey)"
             + " to create your Gemini API key (Don't worry, it's free)")
    api_key = st.text_input("Enter your Google Gemini API Key here:")
    if api_key:
        if not validate_gemini_api_key(api_key):
            st.error("Not the correct API key")
        else:
            st.session_state.current_stage = 2
            st.session_state.api_key = api_key
            st.rerun()  # Move to next page


def stage_2():
    """
    Display the image upload page.

    This page allows the user to upload 5-10 photos of their product. The images
    are stored in a directory for later use in model training.
    """
    st.write("Now upload 5-10 photos of your product, it can be photos taken of your product from different angles")
    files = st.file_uploader(
        "Upload images of your product:",
        type=["jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload here...")

    if files:
        if os.path.exists("./training_photos/"):
            shutil.rmtree("./training_photos/")  # Create the folder for photos storage
        os.makedirs("./training_photos/")
        for i, file in enumerate(files):
            with open(os.path.join("./training_photos/", f"image_{i}.jpg"), "wb") as f:
                f.write(file.getbuffer())
        st.session_state.current_stage = 3
        st.rerun()  # Move to next page


def stage_3():
    """
    Display the product name input page.

    This page prompts the user to enter the name of their product, which will
    be used in model training and image generation.
    """
    st.write("What's your product's name?")
    st.write("E.g. corggi dog, champions league soccer ball, red scarf with patterns")
    product_name = st.text_input("Enter your product's name here:")
    if product_name:
        st.session_state.current_stage = 4
        st.session_state.product_name = product_name
        st.rerun()  # Move to next page


def stage_4():
    """
    Display the model training page.

    This page initiates the model training process using the uploaded images and
    product name. A progress bar is shown to indicate the training status.
    """
    st.write("Now it's time to train your model.")
    st.write("Please wait a while for us to understand your product.")
    product_name = st.session_state.product_name
    my_bar = st.progress(0, text="Training in progress...")  # Monitor the progress
    try:
        train_model(product_name, my_bar)  # Finetuning the model
        st.session_state.current_stage = 5
        st.rerun()  # Move to next page
    except Exception as e:
        st.error(e)


def stage_5():
    """
    Display the background selection page.

    This page allows the user to select or enter a background for the generated
    product images. Background recommendations are provided for convenience.
    """
    st.write("Everything's finally done, Hooray!")
    st.write("Now's the time to create stunning photos with your product.")
    st.write("Enter a background you want or choose a reccommendation below")
    if "recs" not in st.session_state:
        st.session_state.recs = generate_reccommendations()  # Create some background recommendations
    for rec in st.session_state.recs:
        if st.button(rec):
            st.session_state.current_stage = 6
            st.session_state.background = rec
            st.rerun()  # Move to next page
    text_input = st.text_input("Enter a background: ")
    if text_input:
        st.session_state.current_stage = 6
        st.session_state.background = text_input
        st.rerun()  # Move to next page


def stage_6():
    """
    Display the image generation and results page.

    This page generates and displays an image of the product with the selected
    background. Users can generate another image or choose a different background.
    """
    st.write("Nice Choice!")
    product_name = st.session_state.product_name
    background = st.session_state.background
    prompt = f"a photo of a {product_name} {background}"
    st.write(f"Finally, here's your {product_name} {background}")
    image = generate_image(prompt)
    st.image(image)
    if st.button("Generate another image"):
        st.rerun()  # Rerun stage 6
    if st.button("Choose another background"):
        st.session_state.current_stage = 5
        st.rerun()  # Return to stage 5


if __name__ == "__main__":  # Main page and page control system
    _ = """
    Main function to control the page flow of the application.

    This function checks the current stage stored in the session state and
    displays the appropriate page based on the stage.
    """
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
