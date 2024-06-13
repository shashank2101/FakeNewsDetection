import streamlit as st
import urllib.request
import PIL
import google.generativeai as ga
from newspaper import Article
from urllib.parse import urlparse
from transformers import pipeline


st.set_page_config(page_title="Fake News Detector", layout="wide")


@st.cache_resource
def fetch_model():
    MODEL = "jy46604790/Fake-News-Bert-Detect"
    return MODEL


MODEL = fetch_model()
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)


@st.cache_data
def configure_api_key():
    GOOGLE_API_KEY = "AIzaSyB2ml0oownht1RfN69k8O3msz8S547WgJ8"
    ga.configure(api_key=GOOGLE_API_KEY)


configure_api_key()

summarization_model = ga.GenerativeModel("gemini-pro")


st.title("URL test(Work in progress)")
URL = st.text_input("Enter the URL of the news article.")
trusted_sources = [
    "ndtv",
    "indiatoday",
    "thehindu",
    "news18",
    "deccanchronicle",
    "indiatimes",
    "hindustantimes",
    "livemint",
    "deccanherald",
    "thenewsminute",
    "nytimes",
    "cnn",
    "theguardian",
    "foxnews",
    "bbc",
    "dailymail",
    "washingtonpost",
    "wsj",
    "usatoday",
    "huffpost",
]

if URL:
    real = False
    parsed_url = urlparse(URL)
    domain_name_parts = parsed_url.netloc.split(".")
    if len(domain_name_parts) > 1:
        domain_name = domain_name_parts[1]
    else:
        domain_name = domain_name_parts[0]

    if domain_name in trusted_sources:
        real = True
    article = Article(URL)
    article.download()
    article.parse()
    # AUTHORS = article.authors
    PUBLISH_DATE = article.publish_date
    TITLE = article.title
    TEXT = article.text
    SUMMARY = summarization_model.generate_content(
        ["Generate a short and precise summary for the given text input", TEXT],
        stream=True,
    )
    SUMMARY.resolve()
    st.write(f"Title : {TITLE}")
    # st.write(f"Authors : {AUTHORS}")
    st.write(f"Publish date : {PUBLISH_DATE}")
    st.header("Summary")
    st.write(SUMMARY.text)

    TOP_IMAGE_URL = article.top_image
    urllib.request.urlretrieve(TOP_IMAGE_URL, "top_image.jpg")
    TOP_IMAGE = PIL.Image.open("top_image.jpg")
    captioning_model = ga.GenerativeModel("gemini-pro-vision")
    caption = captioning_model.generate_content(
        ["Generate a short and precise caption for the image", TOP_IMAGE], stream=True
    )
    caption.resolve()
    st.image(TOP_IMAGE)
    st.header("Caption:")
    st.write(caption.text)
    if TITLE:
        result = clf(TITLE)
        if result[0]["label"] == "LABEL_1":
            score = result[0]["score"]
            st.header(f"Text analysis: The article is real with a score of {score:.4f}")
        elif result[0]["label"] == "LABEL_0":
            score = result[0]["score"]
            st.header(f"Text analysis: The article is fake with a score of {score:.4f}")
    if real == True:
        st.header(
            f"Web Scraping and reverse search analysis: The article is likely real, it has been covered by {domain_name}"
        )
    else:
        st.header(
            f"Web Scraping and reverse search analysis: The article is likely fake, it has not been covered by {domain_name}"
        )
