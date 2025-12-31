# SMS / Email Spam Classification App

import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK Resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# Load Model (Cached)-
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open('Code/vectorizer.pkl', 'rb'))
    model = pickle.load(open('Code/model.pkl', 'rb'))
    return vectorizer, model

vectorizer, model = load_model()

# UI
st.title("Email / SMS Spam Classifier")
st.caption("A Machine Learning based text classification system")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
        st.stop()

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display Result
    if result == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray; font-size:0.9rem;">
        Developed by <b>Dhruba Mondal</b> | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
