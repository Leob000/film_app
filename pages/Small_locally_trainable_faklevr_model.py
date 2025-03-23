import streamlit as st
import subprocess
import platform
import os
import time
import string


def show():
    # Chose the python interpreter path
    current_os = platform.system()
    if current_os == "Windows":
        python_executable = ".venv\Scripts\python.exe"
    else:
        python_executable = "python"

    st.title("Feature-wise Linear Modulations")

    tab1, tab2 = st.tabs(["Visualizing", "Training"])

    with tab1:
        # Display error message if not data/best.pt
        if not os.path.exists("data/film_faklevr_raw.pt"):
            st.error(
                'No model found at "data/film_faklevr_raw.pt". Please train or download the model'
            )

        # Select and display the image with default image 17
        img_number = st.selectbox(
            "Select an image number:", [str(i) for i in range(10, 20)], index=7
        )
        st.image(
            f"data/faklevr/images/test/faklevr_test_0000{img_number}.png",
            caption=f"faklevr_test_0000{img_number}.png",
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
                    "--program_generator",
                    "data/film_faklevr_raw.pt",
                    "--execution_engine",
                    "data/film_faklevr_raw.pt",
                    "--image",
                    f"data/faklevr/images/test/faklevr_test_0000{img_number}.png",
                    "--streamlit",
                    "True",
                    "--visualize_attention",
                    str(visualize),
                    "--output_viz_dir",
                    "data/faklevr/images/attention_visualizations/",
                    "--image_width",
                    "56",
                    "--image_height",
                    "56",
                    "--cnn_model",
                    "none",
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
            if visualize:
                attention_img_path = f"data/faklevr/images/attention_visualizations/{user_input} {output}/pool-feature-locations.png"
                # Wait for the image to be created
                while not os.path.exists(attention_img_path):
                    time.sleep(1)
                st.image(attention_img_path, caption="Image with attention", width=400)

    with tab2:
        epoch = st.slider("Epoch", 1, 20, 1)
        model_choice = st.selectbox("Model", ["resnet", "raw"])
        if st.button("Train"):
            st.write(f"Training started with {model_choice} for {epoch} epochs")
    return ()
