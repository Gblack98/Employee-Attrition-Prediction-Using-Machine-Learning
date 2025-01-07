import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Charger le modèle et l'encodeur
model_path = "model_logistique.joblib"  # Chemin vers le modèle sauvegardé
encoder_path = "onehot_encoder.joblib"  # Chemin vers l'encodeur sauvegardé
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# Titre de l'application
st.title("Prédiction d'Attrition des Employés")
st.write("Cette application prédit si un employé est susceptible de quitter l'entreprise.")

# Formulaire d'entrée des données utilisateur
st.header("Entrez les caractéristiques de l'employé :")

# Collecte des données utilisateur
age = st.slider("Âge", min_value=18, max_value=65, value=30)
travel = st.selectbox("Fréquence des déplacements professionnels", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
daily_rate = st.number_input("Taux journalier", min_value=100, max_value=1500, value=800)
department = st.selectbox("Département", ["Sales", "Research & Development", "Human Resources"])
distance_from_home = st.slider("Distance du domicile (km)", min_value=1, max_value=50, value=10)
education = st.selectbox("Niveau d'éducation", [1, 2, 3, 4, 5])
education_field = st.selectbox(
    "Domaine d'étude",
    ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
)
job_level = st.selectbox("Niveau du poste", [1, 2, 3, 4, 5])
job_role = st.selectbox(
    "Rôle de l'employé",
    [
        "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
    ]
)
overtime = st.selectbox("Heures supplémentaires", ["Yes", "No"])
monthly_income = st.number_input("Revenu mensuel", min_value=1000, max_value=20000, value=5000)
total_working_years = st.slider("Années totales de travail", min_value=0, max_value=40, value=10)
years_at_company = st.slider("Années dans l'entreprise", min_value=0, max_value=40, value=5)
years_in_current_role = st.slider("Années dans le rôle actuel", min_value=0, max_value=20, value=3)

# Ajout des colonnes manquantes avec des valeurs par défaut
gender = st.selectbox("Genre", ["Male", "Female"])
marital_status = st.selectbox("Statut matrimonial", ["Single", "Married", "Divorced"])

# Mapping des données d'entrée pour correspondre au dataset d'origine
input_data = pd.DataFrame({
    "Age": [age],
    "BusinessTravel": [travel],
    "DailyRate": [daily_rate],
    "Department": [department],
    "DistanceFromHome": [distance_from_home],
    "Education": [education],
    "EducationField": [education_field],
    "JobLevel": [job_level],
    "JobRole": [job_role],
    "OverTime": [overtime],  # Cette colonne n'était pas incluse lors de l'entraînement
    "MonthlyIncome": [monthly_income],
    "TotalWorkingYears": [total_working_years],
    "YearsAtCompany": [years_at_company],
    "YearsInCurrentRole": [years_in_current_role],
    "Gender": [gender],  # Colonne manquante ajoutée
    "MaritalStatus": [marital_status],  # Colonne manquante ajoutée
})

# Colonnes catégoriques utilisées lors de l'entraînement de l'encodeur
# Exclure "OverTime" si elle n'était pas incluse lors de l'entraînement
categorical_columns = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]

# Encodage des colonnes catégoriques
encoded_data = encoder.transform(input_data[categorical_columns]).toarray()

# Créer un DataFrame pour les données encodées
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Combiner les données numériques et les données encodées
numeric_columns = ["Age", "DailyRate", "DistanceFromHome", "Education", "JobLevel", "MonthlyIncome", 
                   "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole"]
final_input = pd.concat([input_data[numeric_columns].reset_index(drop=True), encoded_df], axis=1)

# Ajouter la colonne "OverTime" encodée manuellement
final_input["OverTime"] = input_data["OverTime"].map({"Yes": 1, "No": 0})

# Liste des colonnes attendues par le modèle (à adapter selon votre modèle)
expected_columns = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "JobLevel", "MonthlyIncome",
    "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole", "OverTime",
    "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
    "Department_Research & Development", "Department_Sales",
    "EducationField_Life Sciences", "EducationField_Medical", "EducationField_Other",
    "Gender_Female", "Gender_Male",
    "JobRole_Healthcare Representative", "JobRole_Laboratory Technician", "JobRole_Manager",
    "JobRole_Manufacturing Director", "JobRole_Research Director", "JobRole_Research Scientist",
    "JobRole_Sales Executive", "JobRole_Sales Representative",
    "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"
]

# Ajouter les colonnes manquantes avec des valeurs par défaut
missing_columns = set(expected_columns) - set(final_input.columns)
for col in missing_columns:
    final_input[col] = 0  # Valeur par défaut (vous pouvez ajuster selon vos besoins)

# Vérifier que l'ordre des colonnes correspond à celui utilisé lors de l'entraînement
final_input = final_input[expected_columns]

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(final_input)
    prob = model.predict_proba(final_input)[0][1]

    if prediction[0] == 1:
        st.error(f"L'employé est susceptible de quitter l'entreprise avec une probabilité de {prob:.2f}.")
    else:
        st.success(f"L'employé est peu susceptible de quitter l'entreprise avec une probabilité de {prob:.2f}.")

# Section pour les visualisations
st.header("Visualisations")

def plot_roc_curve():
    st.subheader("Courbe ROC")
    # Exemple de courbe ROC (remplacez par vos propres données de test)
    y_test = np.random.randint(0, 2, 100)  # Remplacez par vos étiquettes réelles
    y_proba = np.random.rand(100)          # Remplacez par vos probabilités prédites

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    st.pyplot(plt)

if st.button("Afficher la courbe ROC"):
    plot_roc_curve()