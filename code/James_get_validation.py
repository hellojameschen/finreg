import pandas as pd
import numpy as np

BASE_DIR = '/Users/jameschen/Team Name Dropbox/James Chen/FINREGRULEMAKE2/finreg/data/match_data/'
in_file = "match_df_20230624.csv"
out_file = "validation_20230624.xlsx"


def get_best(df):
    score_cols = list(filter(lambda x: "score" in x,df.columns))
    df['best_match_score'] = df[score_cols].max(axis=1)
    df['best_match_name'] = df[score_cols].idxmax(axis=1).str.split(':').str[0] + ':best_match_name'
    df['best_match_name'] = df.apply(lambda row: row.loc[row['best_match_name']] if row['best_match_name']==row['best_match_name'] else None, axis=1)
    return df[['original_org_name', 'best_match_score', 'comment_org_name', 'best_match_name']]

df = get_best(pd.read_csv(BASE_DIR + in_file)).groupby('comment_org_name').first()
df = df.reset_index()
df = df.sort_values(by='best_match_score', ascending=False)
df = df[['original_org_name', 'best_match_score', 'comment_org_name', 'best_match_name']]
df['hand_match']=''



df.to_excel(BASE_DIR + out_file)

