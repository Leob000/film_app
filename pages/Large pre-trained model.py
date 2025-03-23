import streamlit as st
import subprocess
import platform
import os
import time
import plotly_express as px
import numpy as np

# Chose the python interpreter path
current_os = platform.system()
if current_os == "Windows":
    python_executable = ".venv\Scripts\python.exe"
else:
    python_executable = ".venv/bin/python"

st.title("Feature-wise Linear Modulations")

tab1, tab2 = st.tabs(["Visualizing", "Training"])

with tab1:
    # Display error message if not data/best.pt
    if not os.path.exists("data/best.pt"):
        st.error("No model found at \"data/best.pt\". Please train or download the model")

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
    visualize = st.checkbox("Visualize attention")
    
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
                str(visualize),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Send the user input to the process and capture the output
        output, error = process.communicate(input = user_input)
        output = output.strip() # Remove leading/trailing whitespace

        # Display the output
        st.subheader("Model Response:")
        st.write(output)

        # Optionally display any error messages
        # if error:
        #     st.subheader("Errors:")
        #     st.write(error)

        # Display the image with attention, if requested
        if visualize:
            attention_img_path = f"img/attention_visualizations/{user_input} {output}/pool-feature-locations.png"
            # Wait for the image to be created
            while not os.path.exists(attention_img_path):
                time.sleep(1)
            st.image(attention_img_path, caption="Image with attention", width=400)
        
        # importation and processing of the parameters values for the three resblocks
        parameters=torch.load('D:\\projet FiLM deep learning\\img\\params.pt')
        beta=[]
        gamma=[]
        for i in range(3):
            beta.extend(parameters[0][i][0:128].tolist())
            gamma.extend(parameters[0][i][128:256].tolist())

        # ploting the histograms with Plotly
        hist_gammas = px.histogram(gamma, nbins=70, marginal='rug')
        hist_gammas.update_layout(title='Histogram of gammas values of the 3 resblocks', xaxis_title='Value', yaxis_title='Frequency')
        st.plotly_chart(hist_gammas)
        hist_betas = px.histogram(beta, nbins=70, marginal='rug')
        hist_betas.update_layout(title='Histogram of gammas values of the 3 resblocks', xaxis_title='Value', yaxis_title='Frequency')
        st.plotly_chart(hist_betas)

with tab2:
    epoch = st.slider("Epoch", 1, 20, 1)
    model_choice = st.selectbox("Model", ["resnet", "raw"])
    if st.button("Train"):
        st.write(f"Training started with {model_choice} for {epoch} epochs")
