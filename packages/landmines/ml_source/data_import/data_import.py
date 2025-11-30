### DATA IMPORT ML SOURCE ###
import pandas as pd

def import_data(path, sheet_name):
    full_path = path
    df = pd.read_excel(full_path, sheet_name = sheet_name)
    return df