# add orchestration details here
import sys
import os
sys.path.insert(1, os.getcwd())
import pandas as pd
from packages.landmines.ml_ops.data_import.data_import import run_import_data
from packages.landmines.ml_ops.data_processing.data_processing import run_data_processing
from packages.landmines.ml_ops.training.train import run_model_training
import mlflow

def main():
    project_name = 'landmines'
    model_name = 'dense_nn'
    experiment_name = f"{project_name}_experiment_log"

    try:
        mlflow.set_experiment(experiment_name = experiment_name)
        mlflow.set_experiment_tag("version", "1.0")
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name = experiment_name)
        mlflow.set_experiment_tag("version", "1.0")
    mlflow.start_run()
    path = 'data\Mine_Dataset.xls'
    print(f'Importing data from {path}.')
    df = run_import_data(path = path, sheet_name = 'Normalized_Data', name = 'landmines')
    mlflow.log_metric('import_data_nrows', len(df))
    mlflow.log_metric('import_data_ncols', len(df.columns))
    print('Successfully imported data! See head sample below:')
    print(df.head())
    print(f'Number of rows in dataset: {len(df)}')
    print(f'Number of cols in dataset: {len(df.columns)}')

    print('Processing data for feature engineering.')
    df = run_data_processing(df)
    mlflow.log_metric("features_nrows", len(df))
    mlflow.log_metric("features_ncols", len(df.columns))
    print('Successfully run feature engineering for dataset! See head sample below:')
    print(df.head())
    print(f"Number of rows in processed df: {len(df)}")
    print(f"Number of columns in processed df: {len(df.columns)}")

    print('Running model training. This may take some time.')
    mlflow.tensorflow.autolog()
    bc_model, mc_model = run_model_training(df, 0.3, 0.5)
    mlflow.tensorflow.save_model(bc_model, f'{project_name}_{model_name}_binary_classifier')
    mlflow.tensorflow.save_model(mc_model, f'{project_name}_{model_name}_multiclass_classifier')
    print('Successfully completed model training.')
    mlflow.end_run()

if __name__ == '__main__':
    main()

