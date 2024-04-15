import streamlit as st
from PIL import Image
import numpy as np
import main

def app():
    st.title('Leaf Disease Classification')

    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = main.load_model_and_predict(np.array(image))
        st.write("Predicted class is: ", prediction)

if __name__ == "__main__":
    app()