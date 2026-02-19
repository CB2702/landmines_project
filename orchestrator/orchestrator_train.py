# add orchestration details here
import sys
import os
sys.path.insert(1, os.getcwd())
import pandas as pd
from packages.landmines.ml_ops.data_import.data_import import run_import_data
from packages.landmines.ml_ops.data_processing.data_processing import run_data_processing
from packages.landmines.ml_ops.training.train import run_model_training
import mlflow
import shutil

def main():
    project_name = "landmines"
    model_name = "dense_nn"
    path = 'data\Mine_Dataset.xls'
    print(f'Importing data from {path}.')
    df = run_import_data(path = path, sheet_name = 'Normalized_Data', name = 'landmines')
    print('Successfully imported data! See head sample below:')
    print(df.head())
    print(f"Number of rows in df: {len(df)}")
    print(f"Number of cols in df: {len(df.columns)}")

    print('Processing data for feature engineering.')
    df = run_data_processing(df)
    print('Successfully run feature engineering for dataset! See head sample below:')
    print(df.head())
    print(f"Number of rows in processed df: {len(df)}")
    print(f"Number of columns in processed df: {len(df.columns)}")

    print('Running model training. This may take some time.')
    model_bc, model_mc = run_model_training(df, 0.3, 0.5)
        
    print('Successfully completed model training.')
    mlflow.end_run()

    return None

if __name__ == '__main__':
    main()
