import sys
import os
sys.path.insert(1, os.getcwd())
from packages.landmines.ml_source.training.train import prepare_dataset, train_model, evaluate_model, build_binary_classifier_model, build_multiclass_classifier_model, build_conf_matrix

def run_model_training(df, test_val_size, test_size):
    print('Splitting binary data into training, validation and testing sets.')
    try:
        df_bc = df.copy()
        df_bc, X_train_bc, X_val_bc, y_train_bc, y_val_bc = prepare_dataset(df, ['is_M', 'M'], 'is_M', test_val_size, test_size, scale = True, encode=False)
    except:
        raise Exception('There was an issue with the splitting of the binary classification data.')
    try:
        df_mc = df.copy()
        df_mc, X_train_mc, X_val_mc, y_train_mc, y_val_mc = prepare_dataset(df, ['is_M', 'M'], 'M', test_val_size, test_size, scale = True, encode=True)
    except:
        raise Exception('There was an issue with the splitting of the multiclass classification data.')
    print('Done. See below for binary classifier model input:')
    print('Training models.')
    try:
        bc_model = build_binary_classifier_model()
        bc_model = train_model(bc_model, X_train_bc, y_train_bc)
    except:
        raise Exception('There was an issue with binary classifier model training.')
    
    try:
        mc_model = build_multiclass_classifier_model()
        mc_model = train_model(mc_model, X_train_mc, y_train_mc)
    except:
        raise Exception('There was an issue with model training.')
    print('Done')
    print('Validating models')
    try:
        bc_evaluation = evaluate_model(bc_model, X_val_bc, y_val_bc)
        mc_evaluation = evaluate_model(mc_model, X_val_mc, y_val_mc)
    except:
        raise Exception('There was an issue with validating the model.')
    print('Done')
    print(f'Binary classification scoring: {bc_evaluation}')
    print(f'Multiclass classification scoring: {mc_evaluation}')
    build_conf_matrix(bc_model, X_val_bc, y_val_bc, 'bc_conf_mat.png')
    build_conf_matrix(mc_model, X_val_mc, y_val_mc, 'mc_conf_mat.png')

    return bc_model, mc_model