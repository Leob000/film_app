import streamlit as st
#from scripts.run_model import visualize


st.title("Feature-wise Linear Modulations")

tab1, tab2 = st.tabs(["Training", "Visualizing"])

# Training tab
with tab1:
    st.header("Training")
    epoch = st.slider("Epoch", 1, 20, 1)
    model_choice = st.selectbox("Model", ["resnet", "raw"])
    if st.button("Train"):
        st.write(f"Training started with {model_choice} for {epoch} epochs")

# Visualizing tab
with tab2:
    st.header("Visualizing")
    query = st.text_input("Query")
    if query:
        #image = visualize(query)
        st.image(image, caption=f"FiLMed image for query : {query}")