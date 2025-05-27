# main.py - Script principal para la categorización de reseñas de películas

import pandas as pd
import numpy as np
import nltk
import spacy
import torch
import logging
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from transformers import BertTokenizer, BertModel

# Configuración inicial
tqdm.pandas()
plt.style.use("seaborn")
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# Ruta del dataset
DATA_PATH = "data/imdb_reviews.tsv"

def cargar_datos():
    """Carga el dataset y realiza limpieza inicial."""
    df = pd.read_csv(DATA_PATH, sep="\t")
    columnas_relevantes = ["tconst", "review", "rating", "sp", "pos", "ds_part", "genres", "start_year"]
    df = df[columnas_relevantes].drop_duplicates()
    print(f"📊 Dataset cargado: {df.shape}")
    return df

def preprocesar_texto(text):
    """Lematización y limpieza del texto con spaCy."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
    doc = nlp(text.lower()[:1000])  # Limitar a 1000 caracteres
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def vectorizar_textos(df):
    """Vectorización con TF-IDF."""
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    df["final_review_text"] = df["review"].apply(preprocesar_texto)
    X_train, X_test, y_train, y_test = train_test_split(df["final_review_text"], df["pos"], test_size=0.2, random_state=42)
    train_features = tfidf_vectorizer.fit_transform(X_train)
    test_features = tfidf_vectorizer.transform(X_test)
    return train_features, test_features, y_train, y_test, tfidf_vectorizer

def entrenar_modelos(X_train, y_train, X_test, y_test):
    """Entrena modelos tradicionales y evalúa su rendimiento."""
    modelos = {
        "Dummy Classifier": DummyClassifier(strategy="constant", constant=1),
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "LGBM Classifier": LGBMClassifier(),
    }

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        score = modelo.score(X_test, y_test)
        print(f"🔹 {nombre}: Precisión = {score:.4f}")
    
    return modelos

def entrenar_bert(df):
    """Genera embeddings con BERT y entrena un modelo."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    def bert_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

    df["bert_features"] = df["final_review_text"].progress_apply(bert_embeddings)
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(df["bert_features"]), df["pos"], test_size=0.2, random_state=42)
    
    modelo_bert = LogisticRegression()
    modelo_bert.fit(X_train, y_train)
    score = modelo_bert.score(X_test, y_test)
    print(f"🔹 BERT + Regresión Logística: Precisión = {score:.4f}")

    return modelo_bert

def evaluar_reviews(modelos, tfidf_vectorizer):
    """Clasifica nuevas reseñas con los modelos entrenados."""
    my_reviews = pd.DataFrame([
        "I was really fascinated with the movie.",
        "What a rotten attempt at a comedy. Not a single joke lands.",
        "The movie had its upsides and downsides, but it's a decent flick.",
    ], columns=["review"])

    my_reviews["processed"] = my_reviews["review"].apply(preprocesar_texto)
    my_reviews_vectorized = tfidf_vectorizer.transform(my_reviews["processed"])

    for nombre, modelo in modelos.items():
        predicciones = modelo.predict(my_reviews_vectorized)
        my_reviews[f"{nombre}_sentiment"] = predicciones
    
    print(my_reviews[["review", "Dummy Classifier_sentiment", "Regresión Logística_sentiment", "LGBM Classifier_sentiment"]])

if __name__ == "__main__":
    print("🚀 Iniciando categorización de reseñas...")

    # Cargar y limpiar datos
    df_reviews = cargar_datos()

    # Vectorización y entrenamiento de modelos clásicos
    X_train, X_test, y_train, y_test, tfidf_vectorizer = vectorizar_textos(df_reviews)
    modelos = entrenar_modelos(X_train, y_train, X_test, y_test)

    # Generación de embeddings y entrenamiento con BERT
    entrenar_bert(df_reviews)

    # Evaluación de nuevas reseñas con modelos entrenados
    evaluar_reviews(modelos, tfidf_vectorizer)

    print("✅ Proceso completado.")