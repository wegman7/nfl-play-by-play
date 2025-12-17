import pandas as pd
import numpy as np


def add_posteam_is_home(df: pd.DataFrame) -> pd.DataFrame:
    # add column indicating if posteam is home team
    df = df.copy()

    df['posteam_is_home'] = df['posteam'] == df['home_team']
    return df


def convert_posteam_to_home_pred(df: pd.DataFrame) -> pd.DataFrame:
    # make predictions home team centric
    
    return np.where(
        df['posteam_is_home'],
        df['prediction'],
        1 - df['prediction']
    )
