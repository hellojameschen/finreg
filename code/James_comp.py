import pandas as pd
import numpy as np

BASE_DIR = "/Users/jameschen/Team Name Dropbox/James Chen/FINREGRULEMAKE2/finreg/"

df = pd.read_csv(BASE_DIR + "test2_df.csv")

score_cols = list(filter(lambda x: "score" in x,df.columns))

for score_col in score_cols:
    threshold_fail = df[score_col]<0.95
    df.loc[threshold_fail, score_col] = np.NaN
    df.loc[threshold_fail, df.columns[df.columns.get_loc(score_col)-1]] = np.NaN

df['best_match_score'] = df[score_cols].max(axis=1)