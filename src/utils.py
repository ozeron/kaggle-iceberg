import pandas as pd

from os import path 


PROJECT_HOME = path.normpath(path.join(__file__, '../..'))
def load_dataframe(mode):
    file_path = path.join(PROJECT_HOME, 'data', mode, 'data.json')
    return pd.read_json(file_path)