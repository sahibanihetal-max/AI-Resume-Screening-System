import streamlit as st
import PyPDF2
import joblib


# Load trained model
model = joblib.load("resume_classifier.pkl")


# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)


# Custom CSS
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(135deg, #eef2ff, #f8fafc);
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #475569;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Main heading
st.markdown(
    '<div class="title">📄 AI Resume Screening System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your resume and let AI predict the suitable job category.'
    '</div>',
    unsafe_allow_html=True
)


# Upload PDF
uploaded_file = st.file_uploader(
    "📤 Upload Your Resume",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success("Resume uploaded successfully! ✅")

    # Read PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf_reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text

    # Show extracted text
    with st.expander("📖 View Extracted Resume Text"):
        st.write(resume_text)

    # Analyze button
    if st.button("🔍 Analyze Resume"):

        if resume_text.strip():

            prediction = model.predict([resume_text])[0]

            probabilities = model.predict_proba([resume_text])[0]

            confidence = max(probabilities) * 100

            st.subheader("🎯 Resume Analysis")

            st.success(
                f"Predicted Job Category: {prediction}"
            )

            st.info(
                f"Prediction Confidence: {confidence:.2f}%"
            )

        else:

            st.error(
                "Unable to extract text from the uploaded PDF."
            )