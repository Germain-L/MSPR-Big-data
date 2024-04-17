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

def add_time_features(data, base_year):
    # Create future data frames for prediction
    future_data = pd.DataFrame()
    for i in range(1, 4):  # Next 1, 2, 3 years
        temp = data.copy()
        temp['year'] = base_year + i
        future_data = pd.concat([future_data, temp], axis=0)
    return future_data

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

def main():
    # Load and preprocess the data
    data = load_data('./4-modeling/combined_all.csv')
    data, label_encoders = preprocess_data(data)

    # Convert 'Orientation' from continuous to categorical
    data['Orientation'] = data['Orientation'].apply(categorize_orientation)
    label_encoder = LabelEncoder()
    data['Orientation'] = label_encoder.fit_transform(data['Orientation'])

    # Prepare data for modeling
    X = data.drop(['Orientation'], axis=1)
    y = data['Orientation']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = train_model(X_train, y_train)
    y_pred = make_predictions(model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')

    # Assume future data preparation is handled similarly
    # Predict future orientations
    future_data = add_time_features(data[data['year'] == data['year'].max()], data['year'].max())
    X_future = future_data.drop(['Orientation'], axis=1)
    future_predictions = make_predictions(model, X_future)
    future_data['Predicted Orientation'] = label_encoder.inverse_transform(future_predictions)  # Converting back to original labels
    print(future_data[['year', 'Predicted Orientation']].head())

if __name__ == "__main__":
    main()
