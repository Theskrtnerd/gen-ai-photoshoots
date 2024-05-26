from streamlit.testing.v1 import AppTest


at = AppTest.from_file("gen_ai_photoshoots/main.py")
at.run()
