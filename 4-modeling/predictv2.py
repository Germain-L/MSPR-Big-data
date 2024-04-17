import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def categorize_orientation(value):
    if value < -0.5:
        return 'Negative'  # class 0
    elif value > 0.5:
        return 'Positive'  # class 2
    else:
        return 'Neutral'   # class 1

def preprocess_data(data):
    # Convert categorical data to numeric
    label_encoders = {}
    for column in ['Libellé de la commune', 'Département', 'Sexe', 'Nom Complet']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Fill missing values if necessary
    data.fillna(data.mean(), inplace=True)
    return data, label_encoders

def add_time_features(data):
    # Create data frames for specified future years
    future_data = pd.DataFrame()
    specified_years = [2024, 2025, 2026]  # Specific years for prediction
    for year in specified_years:
        temp = data.copy()
        temp['year'] = year
        future_data = pd.concat([future_data, temp], axis=0)
    return future_data

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

def main(commune_name):
    # Load and preprocess the data
    data = load_data('./4-modeling/combined_all.csv')
    data, label_encoders = preprocess_data(data)

    # Ensure data includes at least some data up to year 2017 or any suitable cutoff
    if data['year'].max() < 2017:
        print("Insufficient data for predicting future years.")
        return

    # Filter data for the specified commune
    commune_data = data[data['Libellé de la commune'] == label_encoders['Libellé de la commune'].transform([commune_name])[0]]

    # Convert 'Orientation' from continuous to categorical
    commune_data['Orientation'] = commune_data['Orientation'].apply(categorize_orientation)
    label_encoder = LabelEncoder()
    commune_data['Orientation'] = label_encoder.fit_transform(commune_data['Orientation'])

    # Prepare data for modeling
    X = commune_data.drop(['Orientation'], axis=1)
    y = commune_data['Orientation']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = train_model(X_train, y_train)
    y_pred = make_predictions(model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')

    # Predict future orientations for specified years
    future_data = add_time_features(commune_data)
    X_future = future_data.drop(['Orientation'], axis=1)
    future_predictions = make_predictions(model, X_future)
    future_data['Predicted Orientation'] = label_encoder.inverse_transform(future_predictions)

    # Printing future data predictions year-wise
    for year in [2024, 2025, 2026]:
        print(f"Predictions for the year {year}:")
        print(future_data[future_data['year'] == year][['year', 'Predicted Orientation']].head())

if __name__ == "__main__":
    commune_name = input("Enter the name of the commune for prediction: ")
    main(commune_name)


if __name__ == "__main__":
    commune_name = input("Enter the name of the commune for prediction: ")
    main(commune_name)
