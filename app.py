import streamlit as st
import pickle

# Load model dan vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Deteksi Pesan Spam")

pesan = st.text_area("Masukkan Pesan")

if st.button("Prediksi"):
    data = vectorizer.transform([pesan])
    hasil = model.predict(data)

    if hasil[0] == 1:
        st.error("Spam")
    else:
        st.success("Bukan Spam")