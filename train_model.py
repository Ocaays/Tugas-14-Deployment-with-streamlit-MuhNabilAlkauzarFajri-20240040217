import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset sederhana
data = {
    "text": [
        "Selamat Anda memenangkan hadiah",
        "Transfer uang sekarang",
        "Meeting jam 10 pagi",
        "Tugas Machine Learning dikumpulkan besok",
        "Klik link ini untuk mendapatkan hadiah"
    ],
    "label": [1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["label"])

# Simpan model
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model berhasil disimpan")