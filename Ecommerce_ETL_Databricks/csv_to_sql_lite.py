import os
import pandas as pd
import sqlite3

# Change this to your folder path
folder_path = r"C:\Users\Admin\Documents\ML_Projects\Databricks\databrick-mini-course-end-to-end-project\project_assets\project_assets\0_data\ecomm-raw-data\order_items"

# Loop through all CSV files in folder
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Create table name from file name (remove .csv)
        table_name = os.path.splitext(file)[0]
        
        # Write to SQLite
        sqlite_db_path =  os.path.join(folder_path, file.replace('.csv', '.db'))
        
        # Connect to SQLite database (creates if not exists)
        conn = sqlite3.connect(sqlite_db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"Loaded {file} into table '{table_name}'")

# Close connection
conn.close()

print("All CSV files converted to SQLite database successfully.")