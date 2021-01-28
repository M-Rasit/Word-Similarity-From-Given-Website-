import spacy
import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
import re

nlp = spacy.load("en_core_web_lg")
stopwords = nlp.Defaults.stop_words


def get_website(website):


    page = requests.get(website)
    soup = BeautifulSoup(page.content, 'html.parser')

    text = " ".join([i.get_text() for i in soup.find_all(['p', 'h'])])
    text_tokens = nlp(text.lower())
    text_lemma = " ".join([i.lemma_ for i in text_tokens if (i not in stopwords) and (not i.is_title) and (not i.is_punct) and (not i.like_num)])
    text_clean = nlp(text_lemma)
    return text_clean

def word_converter(key):

    word_token = nlp(key)
    if not word_token.has_vector:
        return "Error"
    return word_token





def main():

    html_temp = """
    <div style="background-color:tomato;padding:1.5px">
    <h1 style="color:white;text-align:center;">Word Similarity In Websites</h1>
    </div><br>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    website = st.text_input("Please enter your url:")
    key = st.text_input("Please enter your word:")

    if st.button("Analyze"):
        word_token = word_converter(key)
        if word_token == "Error":
            st.error("Please enter another word!!!")
        else:
            text_clean = get_website(website)
            dct = {"word":[], "similarity":[]}

            for token in text_clean:
                if token.text in dct["word"]:
                    continue
                else:
                    if token.has_vector:
                        x = word_token.similarity(token)
                        if  x > 0:
                            dct["word"].append(token.text)
                            dct["similarity"].append(x)

            df = pd.DataFrame.from_dict(dct)
            df = df.sort_values(by="similarity", ascending=False)

            if (key.lower() == df.loc[0,"word"]):
                for i in range(5):
                    st.success(df["word"].values[i])
            else:
                for i in range(1,6):
                    st.success(df["word"].values[i])

if __name__ == "__main__":
    main()
