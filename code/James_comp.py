import pandas as pd
import numpy as np

BASE_DIR = '/Users/jameschen/Team Name Dropbox/James Chen/FINREGRULEMAKE2/finreg/data/match_data/'
old_file = "validation_20230627.csv"
new_file = "validation_df_20230705160725.csv"


def get_best(df):
    score_cols = list(filter(lambda x: "score" in x,df.columns))
    df['best_match_score'] = df[score_cols].max(axis=1)
    df['cleaned_best_match_name'] = df[score_cols].idxmax(axis=1).str.split(':').str[0] + ':cleaned_best_match_name'
    df['cleaned_best_match_name'] = df.apply(lambda row: row.loc[row['cleaned_best_match_name']] if row['cleaned_best_match_name']==row['best_match_name'] else None, axis=1)
    return df[['comment_org_name', 'best_match_score', 'cleaned_best_match_name']]

df1 = get_best(pd.read_csv(BASE_DIR + old_file)).groupby('comment_org_name').first()
df2 = get_best(pd.read_csv(BASE_DIR + new_file)).groupby('comment_org_name').first()

merged = df1.merge(df2, on ='comment_org_name', suffixes=('_' + old_file, '_' + new_file))
merged = merged[merged['cleaned_best_match_name_'+old_file] != merged['cleaned_best_match_name_'+new_file]]
merged = merged[merged['cleaned_best_match_name_'+old_file] == merged['cleaned_best_match_name_'+old_file]]

merged.to_csv(BASE_DIR + "comparison_" + old_file + "_" + new_file + ".csv")


