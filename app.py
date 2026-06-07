import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

#Load Dataset
df = pd.read_csv("spam.csv", encoding = "Latin-1")[["v1","v2"]]

#Renaming Columns
df.columns = ["label", "text"]

#Converting labels to numbers
df["label"] = df["label"].map({"ham":0, "spam":1})

#Create model
model = Pipeline([
 ("tfidf", TfidfVectorizer(stop_words = "english")),
 ("nb", MultinomialNB())
])

#Train Model on all data
model.fit(df["text"], df["label"])

#Streamlit UI
st.title("Spam detection app")
msg = st.text_input("Enter your message")

if st.button("Check"):
 prediction = model.predict([msg])[0]
 if prediction == 1:
  st.error("This is a spam")
 else:
  st.success("This is not a spam")