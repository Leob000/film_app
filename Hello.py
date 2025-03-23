import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to our application! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    ### CLEVR Dataset
    Select the CLEVR Dataset page to use a model on this dataset.

    There is the ability to train your own model, but the results will probably be poor unless trained on a GPU during a long time. The model will be trained using Cuda if available, else Mps (for M-series Macs), else on CPU.

    We have pretrained a model, see `README.md` to get the weights. Our pre-trained model uses the default hyperparameters from the FiLM paper, except that we use 3 FiLM layers, and resnet101 with 1024 features map as the vision CNN model. Our model was trained during around 15 hours, using a M2 Mac mini with 16 GPU cores.

    #### Help for questions
    Here are a few exemples of questions the model can answer:
    - How many brown objects are there?
    - What is the size of the object left of the red ball?
    - Is the yellow object metallic?

    The model is trained only on certain words, and will not understand words he has not seen during training.
    Here is the full list of words it has seen:

    "Are, Do, Does, How, Is, The, There, What, a, an, and, another, any, anything, are, as, ball, balls, behind, big, block, blocks, blue, both, brown, color, cube, cubes, cyan, cylinder, cylinders, does, either, else, equal, fewer, front, gray, greater, green, has, have, how, in, is, it, its, large, left, less, made, many, material, matte, metal, metallic, more, number, object, objects, of, on, or, other, purple, red, right, rubber, same, shape, shiny, side, size, small, sphere, spheres, than, that, the, there, thing, things, tiny, to, visible, what, yellow"

"""
)
