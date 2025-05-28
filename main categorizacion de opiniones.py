import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import torch
import transformers
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

# ==============================
# 1. Carga y limpieza de datos
# ==============================
def load_data(file_path):
    """Carga el dataset de reseñas IMDb."""
    df = pd.read_csv(file_path, sep='\t', dtype={'votes': 'Int64'})
    return df.drop_duplicates().dropna()

# ==============================
# 2. Exploratory Data Analysis (EDA)
# ==============================
def perform_eda(df):
    """Genera gráficos de análisis exploratorio."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating'], bins=10, kde=True, color='blue')
    plt.title('Distribución de Calificaciones de Reseñas')
    plt.xlabel('Calificación')
    plt.ylabel('Frecuencia')
    plt.show()

# ==============================
# 3. Preprocesamiento de Texto
# ==============================
def preprocess_text(text, nlp):
    """Preprocesamiento de texto: lematización y eliminación de stopwords."""
    doc = nlp(text.lower()[:1000])
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def apply_text_preprocessing(df, nlp):
    """Aplica la función de preprocesamiento de texto a todas las reseñas."""
    df['processed_review'] = df['review'].apply(lambda x: preprocess_text(x, nlp))
    return df

# ==============================
# 4. Vectorización de Texto
# ==============================
def vectorize_text(df_train, df_test):
    """Vectoriza los textos usando TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train = vectorizer.fit_transform(df_train['processed_review'])
    X_test = vectorizer.transform(df_test['processed_review'])
    return X_train, X_test, vectorizer

# ==============================
# 5. Entrenamiento de Modelos
# ==============================
def train_logistic_regression(X_train, y_train):
    """Entrena un modelo de Regresión Logística."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_lgbm(X_train, y_train):
    """Entrena un modelo LGBMClassifier."""
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    return model

# ==============================
# 6. Evaluación de Modelos
# ==============================
def evaluate_model(model, X_test, y_test, model_name):
    """Evalúa el modelo y muestra métricas clave."""
    y_pred = model.predict(X_test)
    print(f"\nEvaluación de {model_name}:")
    print("\nExactitud:", accuracy_score(y_test, y_pred))
    print("\nAUC-ROC:", roc_auc_score(y_test, y_pred))
    print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# ==============================
# 7. Ejecución del Pipeline
# ==============================
def main():
    """Ejecuta el pipeline completo."""
    file_path = "/datasets/imdb_reviews.tsv"
    df_reviews = load_data(file_path)
    
    perform_eda(df_reviews)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])
    df_reviews = apply_text_preprocessing(df_reviews, nlp)

    df_train, df_test = train_test_split(df_reviews, test_size=0.2, stratify=df_reviews['pos'], random_state=42)
    X_train, X_test, vectorizer = vectorize_text(df_train, df_test)
    y_train, y_test = df_train['pos'], df_test['pos']

    log_reg_model = train_logistic_regression(X_train, y_train)
    lgbm_model = train_lgbm(X_train, y_train)

    evaluate_model(log_reg_model, X_test, y_test, "Logistic Regression")
    evaluate_model(lgbm_model, X_test, y_test, "LGBM Classifier")

if __name__ == "__main__":
    main()