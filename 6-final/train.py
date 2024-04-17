import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import unidecode
import re

# Chargement des données depuis un fichier CSV
data_path = './4-modeling/elections_33_clean.csv'  # Mettre à jour ce chemin avec l'emplacement réel de votre fichier CSV
data = pd.read_csv(data_path)

# Définition de la fonction pour nettoyer et standardiser les noms de communes
def clean_commune_name(name):
    if pd.isnull(name):
        return name
    # Enlever les accents, convertir en majuscules et retirer les espaces superflus
    name = unidecode.unidecode(name.strip().upper())
    # Remplacer ' & ' par '-ET-', en tenant compte des espaces irréguliers
    name = re.sub(r'\s*&\s*', '-ET-', name)
    return name

# Application de la fonction de nettoyage sur la colonne des noms de commune
commune_name_column = 'Libellé de la commune'  # Nom de la colonne contenant les noms des communes
data[commune_name_column] = data[commune_name_column].apply(clean_commune_name)

# Suppression des colonnes avec un taux de valeurs manquantes trop élevé
threshold = 0.5 * len(data)
data_clean = data.dropna(thresh=threshold, axis=1)

# Remplacement des valeurs manquantes restantes dans les colonnes numériques par la médiane
for column in data_clean.select_dtypes(include=[np.number]).columns:
    median_value = data_clean[column].median()
    data_clean[column].fillna(median_value, inplace=True)

# Gestion de l'absence de la colonne 'Orientation'
if 'Orientation' not in data_clean.columns:
    raise ValueError("La colonne 'Orientation' est absente du jeu de données.")
data_clean = data_clean.dropna(subset=['Orientation'])  # Suppression des lignes où 'Orientation' est manquante

# Définition des caractéristiques pour le modèle en fonction des colonnes disponibles
features = ['Pourcentage_Blancs_et_nuls', 'Pourcentage_Abstentions', 'Pourcentage_Votants', 'Voix'] + \
           [col for col in data_clean.columns if 'Age' in col or 'Prof' in col]

# Colonnes pour le code géographique et le nom de la commune
geo_code_column = 'Code de la commune'
commune_name_column = 'Libellé de la commune'

# Préparation des données pour la modélisation
X = data_clean[features]
y = data_clean['Orientation']

# Standardisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialisation et entraînement du modèle RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction et calcul de l'erreur quadratique moyenne (MSE)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Prédiction des orientations pour toutes les données
all_predictions = model.predict(X_scaled)
current_year = 2023  # Année actuelle utilisée comme exemple
future_years = [current_year + i for i in range(1, 4)]  # Les trois prochaines années

# Création d'un DataFrame pour chaque année future et leur concaténation
future_predictions_dfs = []
for year in future_years:
    df = pd.DataFrame({
        geo_code_column: data_clean[geo_code_column],
        commune_name_column: data_clean[commune_name_column],
        'Year': year,
        'Orientation prédite': all_predictions
    })
    future_predictions_dfs.append(df)

final_predictions_df = pd.concat(future_predictions_dfs, ignore_index=True)

# Sauvegarde des prédictions dans un fichier CSV
output_path = './4-modeling/predicted_orientations_future.csv'  # Définir le chemin de sortie souhaité pour le fichier CSV
final_predictions_df.to_csv(output_path, index=False)
print(f"Les prédictions futures sont sauvegardées à {output_path}")

# Affichage d'un aperçu du DataFrame
print(final_predictions_df.head())
