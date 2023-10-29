import streamlit as st
import pickle

# Load data
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Streamlit app
st.title('Sentiment Analysis')

text_input = st.text_input('Enter some text:')
if text_input:
    vectorized_text = tfidf_vectorizer.transform([text_input]) 
    prediction = model.predict(vectorized_text)[0]
    if prediction == 1:
        st.write('Positive sentiment')
    elif prediction == -1:
        st.write('Negative sentiment')
    else:
        st.write('Neutral sentiment')
