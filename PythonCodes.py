import spacy
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

nlp = spacy.load("en_core_web_lg")
stopwords = nlp.Defaults.stop_words

page = requests.get("https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/")
soup = BeautifulSoup(page.content, 'html.parser')

text = " ".join([i.get_text() for i in soup.find_all(['p', 'h'])])
text_tokens = nlp(text.lower())
text_lemma = " ".join([i.lemma_ for i in text_tokens if (i not in stopwords) and (not i.is_title) and (not i.is_punct) and (not i.like_num)])
text_clean = nlp(text_lemma)


word = "data"
word_token = nlp(word)

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


if (word == df.loc[0,"word"]):
    print(df["word"][0:5].values)
else:
    print(df["word"][1:6].values)
