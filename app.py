import streamlit as st
from OCRmodel import load_models, extract_text
from PIL import Image
from utils import highlight_text

model, processor = load_models()

st.title("Smart OCR App")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    temp_image_path = "uploaded_image.png"
    image.save(temp_image_path)

    # Text extraction
    if st.button("Extract Text from Image"):
        #st.image(image, caption="Extracting text...", use_column_width=True)
        
        extracted_text = extract_text(model,processor,temp_image_path)

        # Store extracted text in Streamlit session state
        st.session_state.extracted_text = extracted_text

        st.subheader("Extracted Text:")
        st.text_area("Extracted Text", extracted_text, height=400)

# Display text query input and perform search if text is extracted
if 'extracted_text' in st.session_state:
    text_query = st.text_input("Enter your text query")

    if st.button("Search Text"):
        if text_query:
            # Highlight the queried text
            highlighted_output = highlight_text(st.session_state.extracted_text, text_query)

            st.subheader("Search Results (with query highlighted):")
            st.markdown(highlighted_output, unsafe_allow_html=True)
        else:
            st.warning("Please enter a query.")
else:
    st.info("Upload an image and extract text to get started.")