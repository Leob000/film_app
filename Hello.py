import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    ### Help for questions
    Here are a few exemples of questions the model can answer:
    - How many brown objects are there?
    - What is the size of the object left of the red ball?
    - Is the yellow object metallic?

    The model is trained only on certain words, and will not understand words he has not seen during training.
    Here is the full list of words it has seen:

    "Are, Do, Does, How, Is, The, There, What, a, an, and, another, any, anything, are, as, ball, balls, behind, big, block, blocks, blue, both, brown, color, cube, cubes, cyan, cylinder, cylinders, does, either, else, equal, fewer, front, gray, greater, green, has, have, how, in, is, it, its, large, left, less, made, many, material, matte, metal, metallic, more, number, object, objects, of, on, or, other, purple, red, right, rubber, same, shape, shiny, side, size, small, sphere, spheres, than, that, the, there, thing, things, tiny, to, visible, what, yellow"

"""
)
