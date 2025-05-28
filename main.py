import streamlit as st
from utilities import extract_text_from_pdf, get_embedding, compute_similiarity
import io

st.title("Streamlit Resume Checker App")

uploaded_resume_file = st.file_uploader("upload you files")  # uploaded file

job_descriptions = st.text_area("Enter your Job descriptions here")

threshold = st.slider("Sets the Difficulty Bars Level", 0.0,1.0,0.7,0.1)


if st.button("Analyze Resumes"):
    if uploaded_resume_file is not None:
        pdf_bytes = uploaded_resume_file.getvalue()
        extracted_text_from_pdf = extract_text_from_pdf(io.BytesIO(pdf_bytes)) 
        similarity = compute_similiarity(extracted_text_from_pdf,job_descriptions)
        if similarity >= threshold:
            st.write("Resume matched,its a good resume for job , similarity score:",similarity)
        else:
            st.write("Resume did not match, similarity score is less:" ,similarity)
