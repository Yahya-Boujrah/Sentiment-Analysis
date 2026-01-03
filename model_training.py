# # Install all necessary libraries and packages
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Preprocessing and text handling
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Warnings
# import warnings
# warnings.filterwarnings('ignore')

# # Load in the dataset
# df= pd.read_csv('dataset/Tweets.csv')
# df.head()

# # Check the shape
# print(df.shape)

# # Check for missing values and drop them
# print(df.isnull().sum())
# df = df.dropna()
# print(df.isnull().sum())

# # check the proportion of each sentiment
# x = df.sentiment.value_counts().sort_values()

# plt.figure(figsize=(7, 7))
# ax = plt.pie(x = x, labels=x.index, autopct = '%1.1f%%', explode = [0.03, 0.03, 0.08])
# plt.title('Proportion of sentiment')

# # plt.show()

# from wordcloud import WordCloud, STOPWORDS
# negative_df = df[df['sentiment'] == 'negative']
# positive_df = df[df['sentiment'] == 'positive']
# neutral_df = df[df['sentiment'] == 'neutral']

# # Define a function to generate and display a WordCloud
# def generate_wordcloud(data, title):
#     words = ' '.join(data['text'])
#     cleaned_word = " ".join([word for word in words.split()
#                             if 'http' not in word
#                                 and not word.startswith('@')
#                                 and word != 'RT' ])
#     wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black', 
#                           width=3000, height=800).generate(cleaned_word)
#     plt.figure(figsize=(15, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
# # Generate and display WordClouds for each sentiment category
# generate_wordcloud(negative_df, 'Negative Sentiments')
# generate_wordcloud(positive_df, 'Positive Sentiments ')
# generate_wordcloud(neutral_df, 'Neutral Sentiments ')

# # Define a function to clean and preprocess the text
# def preprocess_text(text):
#     # Remove HTML tags and URLs
#     text = re.sub(r'<.*?>|http\S+', '', text)
#     # Convert text to lower case
#     text = text.lower()
#     # Tokenize the text
#     tokens = word_tokenize(text)
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#     # Perform stemming
#     stemmer = PorterStemmer()
#     tokens = [stemmer.stem(word) for word in tokens]
#     # Join the tokens back into a single string
#     cleaned_text = ' '.join(tokens)
#     return cleaned_text

# # Apply preprocessing function to text column
# df['cleaned_text'] = df['text'].apply(preprocess_text)
# df.head()

# # TF-IDF Vectorization
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectors = tfidf_vectorizer.fit_transform(df['text'])

# # Split the data into training and test sets for the Logistic Regression model
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, df['sentiment'], test_size=0.2, random_state=42)


# # Importing & calling Machine learning models

# # Logistic Regression
# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression()

# # Support Vector Machine
# from sklearn.svm import SVC
# svc = SVC()

# # Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()

# # Gradient Boosting Classifier
# from sklearn.ensemble import GradientBoostingClassifier
# GB = GradientBoostingClassifier(random_state=42)

# from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
# models = [lr_model, svc, rfc, GB]
# accuracy_scores = []

# # training models
# for model in models:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     acc = accuracy_score(y_test,y_pred)
#     accuracy_scores.append(acc)
    
#     print(model)
#     print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
#     print()

#     models = ['Logistic Regression', 'Support Vector', 'Random Classifier', 'Gradient Boosting']
# plt.plot(models, accuracy_scores, label='Accuracy Score', marker='o', linestyle='-', color='b', linewidth=2)
# plt.fill_between(models, accuracy_scores, color='lightblue', alpha=1)
# plt.xlabel('Model')
# plt.ylabel('Accuracy Value')
# plt.title('Model Comparison')

# # Generate and plot the confusion matrix for Support Vector Classifier
# y_pred = svc.predict(X_test)
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=df['sentiment'].unique(), yticklabels=df['sentiment'].unique())
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Support Vector Classifier - Confusion Matrix')
# plt.show()

# print("Classification Report for Support Vector Classifier:\n", classification_report(y_test, y_pred, target_names=df['sentiment'].unique()))

# text = ["I hate twitter"]
# sentiment = lr_model.predict(tfidf_vectorizer.transform(text))
# print(sentiment)

# text = ["Weather is too awsome today"]
# sentiment = lr_model.predict(tfidf_vectorizer.transform(text))
# print(sentiment)

import pandas as pd
import re
import nltk
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ======================
# NLTK resources (run once)
# ======================
nltk.download("punkt")
nltk.download("punkt_tab") 
nltk.download("stopwords")

# ======================
# Load dataset
# ======================
df = pd.read_csv("dataset/Tweets.csv")
df = df.dropna()

# ======================
# Text preprocessing
# ======================
def preprocess_text(text):
    text = re.sub(r"<.*?>|http\S+", "", text)
    text = text.lower()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))

    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

# ======================
# TF-IDF Vectorization
# ======================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)   # 🔥 small improvement
)

X = tfidf.fit_transform(df["cleaned_text"])
y = df["sentiment"]

# ======================
# Train / Test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# Models
# ======================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Linear SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
}

# ======================
# Training, Evaluation & Saving
# ======================
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# ======================
# Save best model & vectorizer
# ======================
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\n✅ Best Model Saved")
print("Model:", best_model_name)
print("Accuracy:", best_accuracy)
