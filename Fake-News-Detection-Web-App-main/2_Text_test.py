import streamlit as st
import tensorflow
from transformers import pipeline

st.set_page_config(page_title="Fake News Detector", layout="wide")


@st.cache_resource
def fetch_model():
    MODEL = "jy46604790/Fake-News-Bert-Detect"
    return MODEL


MODEL = fetch_model()
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

st.title("Text test(Work in progress)")
text = st.text_input("Enter the news article's text")
if text:
    result = clf(text)
    if result[0]["label"] == "LABEL_1":
        st.write("The article is real")
        score = result[0]["score"]
        st.write(f"The score is: {score}")
    elif result[0]["label"] == "LABEL_0":
        st.write("The article is fake")
        score = result[0]["score"]
        st.write(f"The score is: {score}")
