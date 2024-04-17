import os
import pandas as pd

# Directories
source_dir = 'data/raw'
target_dir = 'data/csv'

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# List all files in the source directory
for file_name in os.listdir(source_dir):
    # Construct full file path
    file_path = os.path.join(source_dir, file_name)
    
    # Check if the file is an Excel file
    if file_name.endswith(('.xls', '.xlsx')):
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Construct CSV file name and path
            csv_file_name = os.path.splitext(file_name)[0] + '.csv'
            csv_file_path = os.path.join(target_dir, csv_file_name)
            
            # Write to a CSV file
            df.to_csv(csv_file_path, index=False)

            print(f"Converted '{file_name}' to CSV.")
        except Exception as e:
            print(f"Failed to convert '{file_name}': {e}")
