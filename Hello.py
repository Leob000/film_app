import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to our application! 👋")

st.sidebar.success("Select a dataset above.")

st.markdown(
    """
    For this application, two datasets are available:
    - The first dataset is CLEVR, the one from the paper. The model used for this is a pretrained model, similar to the one used in the paper.
    - The other dataset is Faklevr, a custom dataset created by us. It uses simple, low resolution 2D images instead of 3D ones, and greatly limits the question types the model has seen during training. It also uses raw pixels for training, instead of having the images pre processed by a visual CNN like resnet. All this greatly reduce training time, making it possible to get decent results training less than an hour on CPU.
"""
)
