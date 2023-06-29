import pandas as pd
import numpy as np
from datetime import date
today_for_filenames = date.today()
curr_date = str(today_for_filenames.strftime("%Y%m%d"))

BASE_DIR = '/Users/jameschen/Team Name Dropbox/James Chen/FINREGRULEMAKE2/finreg/data/match_data/'
in_file = "match_all_covariates_df_20230627.csv"
out_file = f"validation_20230627.csv"


def get_best(df):
    score_cols = list(filter(lambda x: "score" in x,df.columns))
    df['best_match_score'] = df[score_cols].max(axis=1)
    df['dataset'] = df[score_cols].idxmax(axis=1).str.split(':').str[0]
    df['cleaned_best_match_name'] = df.apply(lambda row: row.loc[row['dataset'] + ':best_match_name'] if row['dataset']==row['dataset'] else None, axis=1)
    df['original_best_match_name'] = df.apply(lambda row: row.loc[row['dataset'] + ':original_match_name'] if row['dataset']==row['dataset'] else None, axis=1)

    return df[['original_org_name', 'best_match_score', 'comment_org_name', 'cleaned_best_match_name', 'original_best_match_name', 'dataset']]

temp = get_best(pd.read_csv(BASE_DIR + in_file)).groupby('original_org_name')
df = temp.first()
df['frequency']=temp.count()['best_match_score']
df = df.reset_index()
df = df.sort_values(by=['frequency','best_match_score'], ascending=False)
df = df[['frequency', 'best_match_score', 'original_org_name', 'comment_org_name', 'cleaned_best_match_name', 'original_best_match_name', 'dataset']]
df['hand_match']=''




df.to_csv(BASE_DIR + out_file, index=False)

