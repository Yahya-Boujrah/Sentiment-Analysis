import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

# -----------------------------
# Fonction de prétraitement
# -----------------------------
def preprocess_text(text):
    text = re.sub(r"<.*?>|http\S+", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))  # on garde anglais pour le modèle entraîné
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# -----------------------------
# Chargement du modèle
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/sentiment_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model()

# -----------------------------
# Interface utilisateur
# -----------------------------
st.set_page_config(page_title="Analyse de Sentiment", page_icon="💬", layout="centered")

st.title("💬 Analyse de Sentiment")
st.write("Tapez un texte ci-dessous et découvrez si le sentiment est positif, négatif ou neutre.")

# ===== Prédiction texte unique =====
user_input = st.text_area("Entrez votre texte ici :")

if st.button("Analyser"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte !")
    else:
       # Prédiction du texte
        prediction = model.predict(tfidf.transform([preprocess_text(user_input)]))[0]

        # Affichage avec couleur de fond selon le sentiment
        if prediction == "negative":
            st.markdown(
                f"""
                <div style="padding: 10px; background-color: #f44336; color: white; border-radius: 5px;">
                    Sentiment prédit : <b>{prediction.capitalize()}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif prediction == "positive":
            st.success(f"Sentiment prédit : **{prediction.capitalize()}**")
        else:  # neutral
            st.info(f"Sentiment prédit : **{prediction.capitalize()}**")


# ===== Fonctionnalités optionnelles =====
st.markdown("---")
st.subheader("🔹 Fonctionnalités supplémentaires")

# 1️⃣ Prédiction par lot à partir d'un CSV
# uploaded_file = st.file_uploader(
#     "Téléversez un fichier CSV avec une colonne 'text' pour analyser plusieurs textes",
#     type=["csv"]
# )
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     if "text" not in df.columns:
#         st.error("Le CSV doit contenir une colonne 'text' !")
#     else:
#         df["cleaned"] = df["text"].apply(preprocess_text)
#         vectors = tfidf.transform(df["cleaned"])
#         df["sentiment"] = model.predict(vectors)
#         st.success("✅ Analyse terminée ! Voici les résultats :")
#         st.dataframe(df[["text", "sentiment"]])



@st.cache_data
def preprocess_batch(texts):
    return [preprocess_text(t) for t in texts]


# Couleurs plus douces pour chaque sentiment
SENTIMENT_STYLES = {
    "positive": {"color": "#2e7d32"}, 
    "negative": {"color": "#c62828"},  
    "neutral": {"color": "#1c83ffe6"} 
}

uploaded_file = st.file_uploader(
    "Téléversez un fichier CSV avec une colonne 'text' pour analyser plusieurs textes",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("Le CSV doit contenir une colonne 'text' !")
    else:
        st.info("Analyse du fichier en cours... ⏳")

        # Prétraitement & prédiction
        df["cleaned"] = preprocess_batch(df["text"])
        vectors = tfidf.transform(df["cleaned"])
        df["sentiment"] = model.predict(vectors)

        # Appliquer des couleurs douces pour l'affichage
        def style_sentiment(row):
            sentiment = row["sentiment"]
            color = SENTIMENT_STYLES.get(sentiment, {"color": "black"})["color"]
            return f"<span style='color: {color}; font-weight: bold'>{sentiment.capitalize()}</span>"

        df["Sentiment"] = df.apply(style_sentiment, axis=1)

        # Affichage du tableau final
        st.success("✅ Analyse terminée ! Voici les résultats :")
        st.write("")

        # Afficher la table avec HTML pour les couleurs
        st.write(
            df[["text", "Sentiment"]].to_html(index=False, escape=False),
            unsafe_allow_html=True
        )
