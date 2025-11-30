import pandas as pd
import numpy as np

def set_var_as_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype('category')
    
    return df

def enforce_category_values(df: pd.DataFrame, col: str, categories: list = [], use_current_values: bool = True) -> pd.DataFrame:
    if (use_current_values):
        categories = list(df[col].unique())
    else:
        categories = list(categories)
    df[col] = df[col].cat.set_categories(categories)

    return df

def get_dummies_for_col(df: pd.DataFrame, col: str, drop_first: bool = True):
    df = pd.get_dummies(df, columns=[col], dtype= int, drop_first=drop_first)

    return df

def get_binary_class_for_col(df: pd.DataFrame, col:str, positive_value: str, inverse: bool = False):
    if inverse:
        df[f'is_{col}'] = np.where(df[col].astype(str) == positive_value, 0, 1)
    else:
        df[f'is_{col}'] = np.where(df[col].astype(str) == positive_value, 1, 0)
    
    return df