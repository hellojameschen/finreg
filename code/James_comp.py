import pandas as pd
import numpy as np

BASE_DIR = '/Users/jameschen/Team Name Dropbox/James Chen/FINREGRULEMAKE2/finreg/data/match_data/'


def get_best(df):
    score_cols = list(filter(lambda x: "score" in x,df.columns))
    df['best_match_score'] = df[score_cols].max(axis=1)
    df['best_match_name'] = df[score_cols].idxmax(axis=1).str.split(':').str[0] + ':best_match_name'
    df['best_match_name'] = df.apply(lambda row: row.loc[row['best_match_name']] if row['best_match_name']==row['best_match_name'] else None, axis=1)
    return df[['comment_org_name', 'best_match_score', 'best_match_name']]

df1 = get_best(pd.read_csv(BASE_DIR + "test2_df.csv")).groupby('comment_org_name').first()
df2 = get_best(pd.read_csv(BASE_DIR + "match_df_20230615.csv")).groupby('comment_org_name').first()

result = df1.merge(df2, on ='comment_org_name', suffixes=('_old', '_new'))

result = result[result['best_match_name_old'] != result['best_match_name_new']]

result = result[result['best_match_name_old'] == result['best_match_name_old']]


