import sys
import os
sys.path.insert(1, os.getcwd())
from packages.landmines.ml_source.data_processing.data_processing import set_var_as_category, enforce_category_values, get_dummies_for_col, get_binary_class_for_col

def run_data_processing(df):
    print('Running data processing on dataframe.')

    # Create categories
    print('Constructing categorical variables.')
    df = set_var_as_category(df, 'S')
    df = set_var_as_category(df, 'M')
    print('Done.')

    # Enforce cat values
    print('Enforcing categorical values.')
    df = enforce_category_values(df, 'S', use_current_values=True)
    df = enforce_category_values(df, 'M', [1, 2, 3, 4, 5])
    print('Done.')

    # One-hot-encode variables
    print('Encoding categorical features.')
    df = get_dummies_for_col(df, 'S')
    print('Done.')

    # Label encode variable
    print(f'Encoding binary class variable is_M')    
    df = get_binary_class_for_col(df, 'M', '1', inverse = True)
    print('Done.')
    
    return df