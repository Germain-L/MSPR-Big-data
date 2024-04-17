import pandas as pd
import os

def clean_election_data(year, num_candidates, input_path):
    # Specify dtype for columns known to cause DtypeWarning, or set low_memory=False
    df = pd.read_csv(input_path, low_memory=False)
    df["annee"] = year

    # Define common columns that exist across all files
    cols_communes = [
        'Code du département', 'Libellé du département', 'Code de la commune', 'Libellé de la commune',
        'Inscrits', 'Abstentions', '% Abs/Ins', 'Votants', '% Vot/Ins', 'Exprimés', '% Exp/Ins', '% Exp/Vot'
    ]

    # Add columns based on their existence in the DataFrame
    optional_cols = [
        ('Code de la circonscription', 'Libellé de la circonscription'),
        ('Code du b.vote',),
        ('Blancs', '% Blancs/Ins', '% Blancs/Vot'),
        ('Nuls', '% Nuls/Ins', '% Nuls/Vot'),
        ('Blancs et nuls',)  # For years like 2017 where 'Blancs' and 'Nuls' might be combined
    ]

    for cols in optional_cols:
        if all(col in df.columns for col in cols):
            cols_communes.extend(cols)

    cols_candidats_base = ['Sexe', 'Nom', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp']
    cols_candidats = []

    # Check for N°Panneau and adjust candidate columns accordingly
    if 'N°Panneau' in df.columns:
        cols_candidats.append('N°Panneau')
    cols_candidats.extend(cols_candidats_base)

    data = []

    for i in range(num_candidates):
        if i == 0:
            cols_selection = cols_communes + cols_candidats
        else:
            cols_selection = cols_communes + [f'{col}.{i}' if col not in cols_communes else col for col in cols_candidats]

        try:
            df_candidat = df[cols_selection].copy()
            if i != 0:  # Rename columns for candidates beyond the first
                df_candidat = df_candidat.rename(columns={f'{col}.{i}': col for col in cols_candidats if f'{col}.{i}' in df_candidat})
            df_candidat['Candidat_ID'] = f"{year}_{i}"
            data.append(df_candidat)
        except KeyError as e:
            print(f"KeyError for year {year} with candidate {i}: {e}")

    df_final = pd.concat(data, ignore_index=True)
    return df_final


# List to hold data from all years
all_years_data = []

election_years = {
    1995: 9,
    2002: 16,
    2007: 12,
    2012: 10,
    2017: 11,
    2022: 12,
}

for year, num_candidates in election_years.items():
    # Corrected input_path to point to the original CSV files location
    input_path = f"./1-extract/csv/{year}_tour_1.csv"
    df_year = clean_election_data(year, num_candidates, input_path)
    all_years_data.append(df_year)

# Combine all years into a single DataFrame
df_all_years = pd.concat(all_years_data, ignore_index=True)

# print all unique candidate names
print(df_all_years['Nom'].unique())

# Optionally, save the combined DataFrame to a CSV file
output_path = "./2-transform/elections/export/all_years_combined.csv"
df_all_years.to_csv(output_path, index=False)
  
print("Combined data saved to:", output_path)