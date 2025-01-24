import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier
import json

# Charger les modèles et transformations sauvegardés
def load_pretrained_models():
    """Charge les modèles pré-entraînés et les encodeurs."""
    try:
        xgb_model = load("lgbm_model.joblib")
        encoder = load("onehot_encoder.joblib")
        scaler = load("scaler.joblib")  # Utiliser le scaler sauvegardé
        return xgb_model, encoder, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None, None, None

# Charger les données
def load_data(filepath):
    """Charge le fichier CSV dans un DataFrame."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Prétraitement des données
def preprocess_data(df, encoder, scaler, column_names):
    """Prétraite les données pour les prédictions."""
    try:
        # Supprimer les colonnes inutiles
        df = df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, errors='ignore')

        # Vérifier que les colonnes catégoriques nécessaires sont présentes
        required_categorical_columns = column_names['categorical_columns']
        missing_categorical = [col for col in required_categorical_columns if col not in df.columns]
        if missing_categorical:
            raise ValueError(f"Colonnes catégoriques manquantes : {missing_categorical}")

        # Vérifier que les colonnes numériques nécessaires sont présentes
        required_numerical_columns = column_names['numerical_columns']
        missing_numerical = [col for col in required_numerical_columns if col not in df.columns]
        if missing_numerical:
            raise ValueError(f"Colonnes numériques manquantes : {missing_numerical}")

        # Encodage des colonnes catégoriques
        X_cat = df[required_categorical_columns]
        X_cat_encoded = encoder.transform(X_cat)

        # Colonnes numériques
        X_numerical = df[required_numerical_columns]

        # Normalisation
        X_all = np.hstack([X_cat_encoded, scaler.transform(X_numerical)])

        return X_all
    except Exception as e:
        st.error(f"Erreur lors du prétraitement des données: {e}")
        return None

# Fonction pour effectuer les prédictions
def make_predictions(model, X):
    """Effectue des prédictions avec le modèle chargé."""
    return model.predict(X)

# Visualisation des données après prédiction
def visualize_post_prediction(df):
    """Affiche des visualisations et des statistiques descriptives après la prédiction."""
    st.write("### Analyse des Prédictions")

    # Distribution des prédictions
    st.write("#### Distribution des Prédictions")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Prédiction XGBoost', data=df, palette='viridis', ax=ax)
    ax.set_title("Distribution des Prédictions (Attrition)")
    st.pyplot(fig)

    # Comparaison des prédictions avec la réalité (si 'Attrition' est disponible)
    if 'Attrition' in df.columns:
        st.write("#### Comparaison des Prédictions avec la Réalité")
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        comparison = pd.crosstab(df['Attrition'], df['Prédiction XGBoost'], rownames=['Réalité'], colnames=['Prédiction'])
        st.write(comparison)

        # Matrice de confusion visuelle
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(comparison, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Matrice de Confusion")
        st.pyplot(fig)

    # Analyse des caractéristiques importantes
    st.write("#### Importance des Caractéristiques")
    feature_importance = xgb_model.feature_importances_
    feature_names = column_names['encoded_columns'] + column_names['numerical_columns']
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='magma', ax=ax)
    ax.set_title("Top 10 des Caractéristiques les Plus Importantes")
    st.pyplot(fig)

    # Statistiques descriptives
    st.write("#### Statistiques Descriptives")
    st.write(df.describe())

# Application Streamlit
st.title("Analyse et Prédiction de l'Attrition des Employés")
uploaded_file = st.file_uploader("Téléchargez un fichier CSV contenant les données des employés", type=["csv"])

if uploaded_file:
    # Charger les données
    df = load_data(uploaded_file)
    if df is not None:
        # Charger les modèles et les noms des colonnes
        xgb_model, encoder, scaler = load_pretrained_models()
        with open('column_names.json', 'r') as f:
            column_names = json.load(f)

        if xgb_model and encoder and scaler:
            # Prétraitement
            X = preprocess_data(df, encoder, scaler, column_names)
            if X is not None:
                # Prédictions
                df['Prédiction XGBoost'] = make_predictions(xgb_model, X)

                # Affichage des résultats
                st.write("### Résultats des Prédictions")
                st.dataframe(df[["Attrition", "Prédiction XGBoost"]].head(20))

                # Visualisations et statistiques après prédiction
                visualize_post_prediction(df)
