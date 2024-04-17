import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('./4-modeling/elections_33_clean.csv')

# Fill missing values if necessary
df['Blancs_et_nuls'].fillna(0, inplace=True)
df = df.dropna(subset=['Orientation'])

# Specify the columns that need different preprocessing
categorical_features = ['Libellé de la commune', 'Sexe', 'Nom Complet']
numerical_features = ['Annee', 'Code de la commune', 'Inscrits', 'Abstentions', 'Votants', 'Exprimés',
                      'Blancs_et_nuls', 'Voix', 'Pourcentage_Blancs_et_nuls', 'Pourcentage_Abstentions', 'Pourcentage_Votants']

# Define the ColumnTransformer, including both scalers for numeric columns and one-hot encoding for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that processes the data and then runs the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Define the features and target variable
X = df.drop('Orientation', axis=1)

# Convert 'Orientation' to discrete classes
le = LabelEncoder()
y = le.fit_transform(df['Orientation'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Print the accuracy score
print("Accuracy: ", accuracy_score(y_test, predictions))

