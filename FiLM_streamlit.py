import streamlit as st

def main():
    st.set_page_config(page_title="Feature-Wise Linear Modulations", layout="wide")

    # Menu de navigation
    st.sidebar.title("Model Choice")
    page = st.sidebar.selectbox("Choose a model", ["Large CLEVR Model", "Small Faklevr Model"])
    
    if page == "Large CLEVR Model":
        import pages.Large_pre_trained_model as LargeModelPage
        LargeModelPage.show()
    elif page == "Small Faklevr Model":
        import pages.Small_locally_trainable_faklevr_model as SmallModelPage
        SmallModelPage.show()
    return()

if __name__ == "__main__":
    main()

