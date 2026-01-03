import joblib

from model_training import preprocess_text 

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

texts = [
    "I hate twitter",
    "Weather is too awesome today"
]

cleaned = [preprocess_text(t) for t in texts]
vectors = tfidf.transform(cleaned)

predictions = model.predict(vectors)
print(predictions)
