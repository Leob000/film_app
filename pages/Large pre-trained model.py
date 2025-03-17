import streamlit as st
import subprocess
import platform
import os

# Chose the python interpreter path
current_os = platform.system()
if current_os == "Windows":
    python_executable = ".venv\Scripts\python.exe"
else:
    python_executable = ".venv/bin/python"

st.title("Feature-wise Linear Modulations")

tab1, tab2 = st.tabs(["Visualizing", "Training"])

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
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Send the user input to the process and capture the output
    output, error = process.communicate(input = user_input)

    # Display the output
    st.subheader("Model Response:")
    st.write(output)

    # Optionally display any error messages
    # if error:
    #     st.subheader("Errors:")
    #     st.write(error)

