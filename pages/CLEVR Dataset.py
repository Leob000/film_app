import streamlit as st
import subprocess
import platform
import os
import time
import string
import plotly_express as px
import numpy as np
import torch

# Chose the python interpreter path
current_os = platform.system()
if current_os == "Windows":
    python_executable = ".venv\Scripts\python.exe"
else:
    python_executable = "python"

st.title("CLEVR Dataset")

tab1, tab2 = st.tabs(["Help", "Visualizing"])

with tab1:
    # epoch = st.slider("Epoch", 1, 20, 1)
    # model_choice = st.selectbox("Model", ["resnet", "raw"])
    # if st.button("Train"):
    #     st.write(f"Training started with {model_choice} for {epoch} epochs")
    st.markdown(
        # There is the ability to train your own model, but the results will probably be poor unless trained on a GPU during a long time. The model will be trained using Cuda if available, else Mps (for M-series Macs), else on CPU.
        """
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

with tab2:
    # Display error message if not data/best.pt
    if not os.path.exists("data/best.pt"):
        st.error('No model found at "data/best.pt". Please train or download the model')

    # Select and display the image with default image 17
    img_number = st.selectbox(
        "Select an image number:", [str(i) for i in range(10, 20)], index=7
    )
    st.image(
        f"img/CLEVR_val_0000{img_number}.png",
        caption=f"CLEVR_val_0000{img_number}.png",
        # use_container_width=True,
        width=400,
    )

    # Checkbox to visualize attention
    attention_to_plot = st.selectbox(
        "Attention map from:",
        [
            "None",
            "conv-stem",
            "resnet101",
            "pool-feature-locations",
            "pre-pool",
            "grad-conv-stem",
        ]
        + [f"grad-resblock{i}" for i in range(3)]
        + [f"resblock{i}" for i in range(3)],
        index=3,
    )

    # Create a form so that hitting Enter submits the input
    with st.form(key="question_form"):
        user_input = st.text_input("Enter your question:")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        # Launch the process (adjust parameters as needed)
        process = subprocess.Popen(
            [
                python_executable,
                "scripts/run_model.py",
                "--image",
                f"img/CLEVR_val_0000{img_number}.png",
                "--streamlit",
                "True",
                "--visualize_attention",
                str(attention_to_plot != "None"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Preprocess the user input
        punctuation_removal = str.maketrans("", "", string.punctuation)
        user_input = user_input.translate(punctuation_removal)
        user_input = user_input.lower()

        # Send the user input to the process and capture the output
        output, error = process.communicate(input=user_input)
        output = output.strip()  # Remove leading/trailing whitespace

        # Display the output
        st.subheader("Model Response:")
        st.write(output)

        # Optionally display any error messages
        # if error:
        #     st.subheader("Errors:")
        #     st.write(error)

        # Display the image with attention, if requested
        if attention_to_plot != "None":
            attention_img_path = f"img/attention_visualizations/{user_input} {output}/{attention_to_plot}.png"
            # Wait for the image to be created
            while not os.path.exists(attention_img_path):
                time.sleep(1)
            st.image(
                attention_img_path,
                caption=f"Image with attention from {attention_to_plot}",
                width=400,
            )

        # importation and processing of the parameters values for the three resblocks
        parameters = torch.load(os.path.join("img", "params.pt"))
        beta = []
        gamma = []
        for i in range(3):
            beta.extend(parameters[0][i][0:128].tolist())
            gamma.extend(parameters[0][i][128:256].tolist())

        # ploting the histograms with Plotly
        hist_gammas = px.histogram(gamma, nbins=70, marginal="rug")
        hist_gammas.update_layout(
            title="Histogram of gammas values of the 3 resblocks",
            xaxis_title="Value",
            yaxis_title="Frequency",
        )
        st.plotly_chart(hist_gammas)
        hist_betas = px.histogram(beta, nbins=70, marginal="rug")
        hist_betas.update_layout(
            title="Histogram of beta values of the 3 resblocks",
            xaxis_title="Value",
            yaxis_title="Frequency",
        )
        st.plotly_chart(hist_betas)
