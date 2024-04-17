from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the entire predictions data once to improve performance
predictions = pd.read_csv('predicted_orientations_future.csv')
predictions['Libellé de la commune'] = predictions['Libellé de la commune'].str.strip().str.lower()
# Store unique commune names sorted for dropdown
commune_names = predictions['Libellé de la commune'].drop_duplicates().sort_values()

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_commune = request.form.get('commune_name', '').strip().lower()  # Normalize input to match data cleaning
    predictions_html = ""

    if selected_commune:
        # Filter data
        filtered_predictions = predictions[predictions['Libellé de la commune'] == selected_commune]
        if not filtered_predictions.empty:
            predictions_html = filtered_predictions.to_html(index=False)
        else:
            predictions_html = "<p>No data available for the selected commune.</p>"
            print(f"No data found for {selected_commune}")

    return render_template('index.html', commune_names=commune_names.tolist(), predictions=predictions_html, selected_commune=selected_commune)

if __name__ == '__main__':
    app.run(debug=True)
