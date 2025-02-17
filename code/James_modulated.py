#!/usr/bin/env python
# coding: utf-8

import math
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import pandas as pd
from datetime import datetime
import pickle
import re
from fastDamerauLevenshtein import damerauLevenshtein
import apsw
import sys
import numpy as np
import corp_simplify_utils
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from difflib import SequenceMatcher

# nlp
import spacy
from spacy import displacy
from collections import Counter
# to install: $python3 -m spacy download en_core_web_lg
import en_core_web_lg

# analysis/regressions
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

# from statsmodels.graphics.gofplots import qqplot_2samples
from scipy import stats
from joypy import joyplot
from matplotlib import cm

from datetime import datetime
today_for_filenames = datetime.now()
curr_date = str(today_for_filenames.strftime("%Y%m%d%H%M%S"))

NUMBER_OF_MATCHES_TO_RECORD = 10
punc_remove_re = re.compile(r'\W+')
corp_re = re.compile('( (group|holding(s)?( co)?|inc(orporated)?|ltd|l ?l? ?[cp]|co(rp(oration)?|mpany)?|s[ae]|plc))+$')
and_re = re.compile(' & ')
punc1_re = re.compile(r'(?<=\S)[\'’´\.](?=\S)')
punc2_re = re.compile(r'[\s\.,:;/\'"`´‘’“”\(\)\[\]\{\}_—\-?$=!]+')

STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.remove("am")
STOPWORDS.remove("up")
STOPWORDS.remove("in")
STOPWORDS.remove("on")
STOPWORDS.remove("all")
STOPWORDS.remove("any")
STOPWORDS.remove("most")
STOPWORDS.remove("no")
STOPWORDS.remove("nor")
STOPWORDS.remove("own")
STOPWORDS.remove("same")
STOPWORDS.remove("so")
STOPWORDS.remove("very")
STOPWORDS.remove("s")
STOPWORDS.remove("t")
STOPWORDS.remove("d")
STOPWORDS.remove("ll")
STOPWORDS.remove("m")
STOPWORDS.remove("o")
STOPWORDS.remove("re")
STOPWORDS.remove("ve")
STOPWORDS.remove("y")

stopword_re_str = r""
for word in STOPWORDS:
	stopword_re_str += r'\b' + word + r'\b|'
stopword_re = re.compile(stopword_re_str[:-1]) # The negative 1 is for the fencepost |

BASE_DIR = "/Users/jameschen/Team Name Dropbox/James Chen/FINREGRULEMAKE2/finreg/"
# BASE_DIR = "/Users/jameschen/Team Name Dropbox/James Chen/JLW-FINREG-PARTICIPATION/"
# BASE_DIR = "/Users/jameschen/Documents/Code/JLW-FINREG-PARTICIPATION/"
# DB_PATH = BASE_DIR + "data/master.sqlite"
DB_PATH = BASE_DIR[:-7] + "Data/master.sqlite"
# LAST_SAVE_DATASET_DATE = "20210824"
LAST_SAVE_DATASET_DATE = "20220402" # Needs to be set to the last date the 'rebuild datasets' part of this code was run

sources = ["FDIC_Institutions", "FFIECInstitutions", "CreditUnions", "CIK", "compustat_resources", "nonprofits_resources", "opensecrets_resources_jwVersion", "SEC_Institutions"]


def get_longest_common_substring(string1, string2):
    match = SequenceMatcher(None, string1, string2, autojunk=False).find_longest_match()
    return string1[match.a:match.a + match.size]


# Function from Brad Hackinen's NAMA
def basicHash(s):
    '''
    A simple case and puctuation-insensitive hash
    '''
    s = s.lower()
    s = re.sub(and_re,' and ',s)
    s = re.sub(punc1_re,'',s)
    s = re.sub(punc2_re,' ',s)
    s = s.strip()

    return s

# Function from Brad Hackinen's NAMA
def corpHash(s):
    '''
    A hash function for corporate subsidiaries
    Insensitive to
        -case & punctation
        -'the' prefix
        -common corporation suffixes, including 'holding co'
    '''
    s = basicHash(s)
    if s.startswith('the '):
        s = s[4:]

    s = re.sub(corp_re,'',s,count=1)

    return s

def remove_junk(string):
    junk = ['inc', 'na', 'pllc', 'llc']
    lst = string.split()
    lst = list(filter(lambda x: not x in junk, lst))
    return ' '.join(lst)
    

# function to clean org names
def clean_fin_org_names(name):
    if name == None:
        return ''
    name = name.lower()
    name = name.strip()
    if name is None or not isinstance(name, str) or name == "NA":
        return ""
    else:
        # James strip metadata from name
        name = name.split(',\n(')[0]
        name = re.sub(" [0-9]* [k|m]b pdf","",name)

        name = name.translate(corp_simplify_utils.STR_TABLE)
        name = re.sub(stopword_re, '', name.lower())

        name = corpHash(name)
        name = remove_junk(name)
        
        return name
    

def clean_match_score(x):
    if x is None or x is np.nan or pd.isnull(x) or x == "":
        return np.nan
    elif isinstance(x, str) and not x.isnumeric():
        unit_multiplier = 1
        if "B" in x:
            x = x[:-1]
            unit_multiplier = 1000000000
        if "M" in x:
            x = x[:-1]
            unit_multiplier = 1000000
        if "K" in x:
            x = x[:-1]
            unit_multiplier = 1000
        x = x.replace(",", "")
        try:
            x = float(x) * unit_multiplier
            return x
        except:
            return np.nan
    else:
        return float(x)

def get_match_score(frequency_dict, org_name, candidate_name):
    org_tokens = org_name.split(' ')
    
    # tokenize the candidate match
    candidate_match_tokens = set(candidate_name.split(" "))

    # Calculate the match score
    total_inverse_frequency = 0
    total_matching_inverse_frequency = 0
    tokenized_name = org_tokens
    for token in tokenized_name:
        token_frequency = frequency_dict.get(token, 999999) # if token not found, give high frequency to ignore it
        total_inverse_frequency += 1.0/token_frequency
        if token in candidate_match_tokens:
            total_matching_inverse_frequency += 1.0/token_frequency
    match_score = total_matching_inverse_frequency / total_inverse_frequency

    # added by James
    weight = 0.1/len(org_name)
    longest_common_substring = get_longest_common_substring(org_name, candidate_name)
    match_score -= weight * len(candidate_name)/len(longest_common_substring) - weight

    # handle acronyms
    # candidate_acronym = ''.join([item[0] for item in candidate_name.split()])
    # for word in org_name.split():
    #     if word in candidate_acronym:
    #         match_score += 0.1 * (len(word)-1)/len(candidate_acronym)

    return match_score

# 3.1: Read the gathered datasets in as one dataframe
def get_covariate_dfs():
    covariate_dfs = {}
    financial_datasets = [("data/merged_resources/", "FDIC_Institutions"), 
                    ("data/merged_resources/", "FFIECInstitutions"),
                    ("data/", "CreditUnions"),
                    ("data/merged_resources/", "compustat_resources"),
                    ("data/merged_resources/", "nonprofits_resources"),
                    ("data/merged_resources/", "SEC_Institutions")
    ]
    for financial_dataset_tuple in financial_datasets:
        df = pd.read_csv(BASE_DIR + financial_dataset_tuple[0] + financial_dataset_tuple[1] + ".csv")
        covariate_dfs[financial_dataset_tuple[1]] = df
        
    # Read in opensecrets dataseparately to deal with quotechar
    df = pd.read_csv(BASE_DIR + "data/merged_resources/opensecrets_resources_jwVersion.csv", quotechar='"')
    covariate_dfs['opensecrets_resources_jwVersion'] = df
        
    # Merge compustat data to cik data on cik
    cik_df = pd.read_csv(BASE_DIR + "data/merged_resources/CIK.csv", dtype={"CIK":str})
    compustat_df = pd.read_csv(BASE_DIR + "data/merged_resources/compustat_resources.csv", dtype={"cik":str})
    compustat_df.sort_values(by=['year2', 'year1'], ascending=True, inplace=True)
    compustat_df = compustat_df.drop_duplicates(subset='cik', keep='last', ignore_index=True)
    compustat_df = compustat_df[['cik', 'marketcap']]

    # James: dtype convert
    cik_df['cik']= cik_df['cik'].astype('Int64')
    compustat_df['cik']= compustat_df['cik'].astype('Int64')

    cik_merged_df = cik_df.merge(compustat_df, how='left', left_on='cik', right_on='cik')
    del cik_merged_df['cik']
    covariate_dfs['CIK'] = cik_merged_df

    return covariate_dfs

def get_data_row(covariate_dfs, match_type, match_row_num, match_on_type):

    df = covariate_dfs[match_type]
    column_names = df.columns
    match_covariates = df.iloc[match_row_num]
    
    covariate_dict = {'row_id':match_row_num, 'row_type':match_type}
    for elem_idx, elem in enumerate(match_covariates):
        var_name = match_type + "-" + match_on_type + ":" + column_names[elem_idx]
        val = elem
        covariate_dict[var_name] = val

    return covariate_dict

def get_organization_dataset():

    ## PART 1: Match records from the gathered organization datasets (FDIC, FFEIC, Nonprofits, CIK, Compustat, etc.) to scraped comments

    if True:
        financial_datasets = []
        unique_ids = []
        all_org_names = []
        for financial_dataset in sources:
            intermediate_data_folder = "data/"
            col_name = ""
            read_from_file = False
            if financial_dataset == "FDIC_Institutions":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "NAME"
                read_from_file = True
            elif financial_dataset == "FFIECInstitutions":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "Financial Institution Name"
                read_from_file = True
            elif financial_dataset == "CreditUnions":
                col_name = "CU_NAME"
                read_from_file = True
            elif financial_dataset == "CIK":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "company_name"#"COMPANYNAME"
                read_from_file = True
            elif financial_dataset == "compustat_resources":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "conm"
                read_from_file = True
            elif financial_dataset == "nonprofits_resources":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "name"
                read_from_file = True
            elif financial_dataset == "opensecrets_resources_jwVersion":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "orgName"
                read_from_file = True
            elif financial_dataset == "SEC_Institutions":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "Name"
                read_from_file = True

            print(financial_dataset)
            if financial_dataset == "opensecrets_resources_jwVersion":
                df = pd.read_csv(BASE_DIR + intermediate_data_folder + financial_dataset + ".csv", quotechar='"')
            else:
                df = pd.read_csv(BASE_DIR + intermediate_data_folder + financial_dataset + ".csv")
            df['unique_id'] = financial_dataset + "-" + df.index.astype(str)
            df['financial_dataset'] = financial_dataset
            financial_datasets = financial_datasets + list(df['unique_id'])
            unique_ids = unique_ids + list(df['unique_id'])
            all_org_names = all_org_names + list(df[col_name])

        data = list(zip(unique_ids, all_org_names, financial_datasets))
        org_name_df = pd.DataFrame(data, columns=['unique_id', 'org_name', 'financial_dataset'])

        org_name_df['original_match_name'] = org_name_df['org_name']

        return org_name_df



def get_comment_dataset():
    # 1.2: Read and clean submitter and org names from scraped comments
    connection=apsw.Connection(DB_PATH)
    c=connection.cursor()

    c.execute("SELECT * FROM comments")
    key_names_df = c.fetchall()

    c.execute("PRAGMA table_info(comments);")
    column_names = [row[1] for row in c.fetchall()]

    print("Starting cleaning")
    df = pd.DataFrame(key_names_df, columns = column_names)
    cols = ['comment_url', 'submitter_name', 'organization', 'agency_acronym', 'docket_id', 'comment_title']

    df = df[cols]
    df['original_organization_name'] = df['organization']

    # FRS, take what is before first comma
    # df.loc[df['agency_acronym']=='FRS', "organization"] = df.loc[df['agency_acronym']=='FRS', "organization"].str.split(',').map(lambda x: x[0]).map(lambda x: '' if '(' in x else x)

    # # FDIC take before first comma, before with, and before -
    # df.loc[df['agency_acronym']=='FDIC', "organization"] = df.loc[df['agency_acronym']=='FDIC', "organization"].str.split(',').map(lambda x: x[0] if x else '')
    # df.loc[df['agency_acronym']=='FDIC', "organization"] = df.loc[df['agency_acronym']=='FDIC', "organization"].str.split(' with ').map(lambda x: x[1] if len(x)>1 else x[0])
    # df.loc[df['agency_acronym']=='FDIC', "organization"] = df.loc[df['agency_acronym']=='FDIC', "organization"].str.split(' - ').map(lambda x: x[0] if x else '')

    # SEC is difficult
    # df['submitter_name'] = df['submitter_name']
    # df['organization'] = df['organization']

    #replace none
    # df.loc[df['submitter_name'].isna(), "submitter_name"] = ''

    # key_names_df = df.iloc[:,:] # include how many to match

    return df

def get_candidate_match_dict(org_name_df):
    print('Preparing candidate match dictionary.')
    # 1.4: Create a dict mapping from tokens in the gathered org datasets to IDs and org_names that contain that token
    candidate_match_dict = {}
    for row_idx in tqdm(range(len(org_name_df))):
        row = org_name_df.iloc[row_idx]
        unique_id = row['unique_id']
        org_name = row['org_name']
        original_match_name = row['original_match_name']

        flags = org_name.split(" ")

        # include organization acronym
        org_acronym = ''.join([item[0] for item in org_name.split()])
        flags.append(org_acronym)

        for token in flags:
            if token in candidate_match_dict:
                candidate_match_dict[token].append((unique_id, org_name, original_match_name))
            else:
                candidate_match_dict[token] = [(unique_id, org_name, original_match_name)]
        

    return candidate_match_dict
        
def get_candidate_frequency_dict(org_name_df):
    print('Preparing candidate frequency dictionary.')
    candidate_frequency_dict = {}
    for org_name in tqdm(org_name_df['org_name']):
        for token in org_name.split(" "):
            if token in candidate_frequency_dict:
                candidate_frequency_dict[token] += 1
            else:
                candidate_frequency_dict[token] = 1
    return candidate_frequency_dict

def get_match_candidates(candidate_match_dict, candidate_frequency_dict, org_name):

    # Tokenize the submitter name and org name
    org_tokens = org_name.split(" ")
    
    # Get the frequencies (in the scraped comment db) of the tokens in the submitter name and org name
    # submitter_token_frequencies = sorted([(submitter_token, submitter_frequency_dict[submitter_token]) for submitter_token in submitter_tokens], key=lambda x: x[1])
    org_token_frequencies = [(org_token, candidate_frequency_dict.get(org_token)) for org_token in org_tokens]
    org_token_frequencies = list(filter(lambda item: item[1] is not None, org_token_frequencies))
    org_token_frequencies = sorted(org_token_frequencies, key=lambda x: x[1])
    

    candidate_matches = []
    # Iterate through the candidate matches to the most informative token
    for most_unique_org_token, _ in org_token_frequencies[:]: # uses top 2 most unique tokens
        if most_unique_org_token in candidate_match_dict:
            for row in candidate_match_dict[most_unique_org_token]:
                unique_id = row[0]
                match_name = row[1]
                original_match_name = row[2]
                match_score = get_match_score(candidate_frequency_dict, org_name, match_name)
                candidate_matches.append((match_score, match_name, original_match_name, unique_id))

    # Sort the candidate matches, first by the match score and then by the absolute value of the difference in the number of tokens between the submitter (or org) name and the candidate match org name
    candidate_matches.sort(key=lambda x:(-x[0], abs(len(x[1].split(" ")) - len(org_tokens))))
    candidate_matches = pd.DataFrame(candidate_matches, columns=['match_score','match_name', 'original_match_name', 'unique_id'])


    return candidate_matches


def filter_top_matches(match_dict, elem):
    matches = match_dict[elem]
    if len(matches)== 0:
        result = pd.DataFrame()
    else:
        matches['source'] = matches['unique_id'].str.split('-').str[0]
        result = matches.groupby('source').first()

    return result

def clean_key_names_df(key_names_df):
    key_names_df = key_names_df[['original_organization_name']].copy()
    key_names_df['original_organization_name'] = key_names_df['original_organization_name'].str.replace('\r\n', '\n').str.strip()
    key_names_df = key_names_df.dropna()
    key_names_df['organization'] = key_names_df['original_organization_name']
    lst = list()
    for idx in range(len(key_names_df)):
        row = key_names_df.iloc[idx].to_dict()
        organization = row['organization']
        organization = organization.split('(')[0]
        for org in re.split('; |, |\n',organization):
            row['organization'] = org.strip()
            lst.append(row.copy())
    key_names_df = pd.DataFrame(lst)
    key_names_df['organization'] = key_names_df['organization'].apply(clean_fin_org_names)
    return key_names_df

def get_matches(org_name_df, key_names_df):
    org_name_df = org_name_df.copy()
    org_name_df = org_name_df.dropna()
    org_name_df['original_match_name'] = org_name_df['org_name']
    org_name_df['org_name'] = org_name_df['org_name'].apply(clean_fin_org_names)

    # 1.4: Create a dict mapping from tokens in the gathered org datasets to IDs and org_names that contain that token
    candidate_frequency_dict = get_candidate_frequency_dict(org_name_df)
    candidate_match_dict = get_candidate_match_dict(org_name_df)
        

    # Apply linking dataset
    print('Calculating match scores.')
    # 1.5.1: For each org and submitter name in the scraped comment dataset, get all of the names ('candidate matches') from among the gathered org datasets that have the most important word of the scraped db names in the org's name. Calculate a tf-idf weighted jaccard index match score to choose the best matches among the candidates.
    match_dict = {}#get_match_dict(candidate_match_dict, candidate_frequency_dict, key_names_df)
    unique_cleaned_names = key_names_df['organization'].unique()
    for org_name in tqdm(unique_cleaned_names):

        if not org_name:
            match_dict[org_name] = pd.DataFrame()
            continue

        if org_name in match_dict:
            continue

        match_dict[org_name] = get_match_candidates(candidate_match_dict,candidate_frequency_dict,org_name)


    # 1.5.2: Save the candidate matches and get record counts

    # print("Num scraped records: " + str(len(key_names_df)))


    # 1.6: Extract the scraped records with at least one candidate match and take the top top_matches_num (or all if there are < top_matches_num) matches from the scored candidate matches
    # TODO: speed up by using dataframe top
        

    print('Filtering top matches')
    good_matches = {}
    for elem in tqdm(list(match_dict.keys())):
        good_matches[elem] = filter_top_matches(match_dict, elem)



    ## PART 3: Create a data from to explore commenter covariates
    # covariate_dfs = get_covariate_dfs()
    # 3.2: Make a dataframe organizing the covariates of the gathered datasets
    print('Creating Final Matching Dict')
    lst = []
    for idx in tqdm(range(len(key_names_df))):
        key_name = key_names_df.iloc[idx]
        original_match_name = key_name['original_organization_name']
        org_name = key_name['organization']
        row = {}

        row["comment_org_name"] = org_name
        row["original_organization_name"] = original_match_name
        matches = good_matches[org_name]
        
        for match_idx in range(len(matches)):
            match = matches.iloc[match_idx]
            dataset = match['unique_id'].split('-')[0]

            row.update(dict(zip(dataset+":"+match.index.astype(str), match.values)))

        lst.append(row)

    print('Creating Final Matching DataFrame')
    df = pd.DataFrame(lst).set_index('original_organization_name')
    print('Done')

    return df


def get_best(df):
    score_cols = list(filter(lambda x: "score" in x,df.columns))
    df['best_match_score'] = df[score_cols].max(axis=1)
    df['dataset'] = df[score_cols].idxmax(axis=1).str.split(':').str[0]
    df['cleaned_best_match_name'] = df.apply(lambda row: row.loc[row['dataset'] + ':match_name'] if row['dataset']==row['dataset'] else None, axis=1)
    df['original_best_match_name'] = df.apply(lambda row: row.loc[row['dataset'] + ':original_match_name'] if row['dataset']==row['dataset'] else None, axis=1)
    df = df.groupby(['comment_org_name','original_organization_name']).first()
    return df[['best_match_score', 'cleaned_best_match_name', 'original_best_match_name', 'dataset']].reset_index()

def get_validation(matches_df,key_names_df):
    df = get_best(matches_df)
    key_names_df['x']=1
    df = df.set_index('comment_org_name')
    df['frequency'] = key_names_df.groupby(['organization']).count()['x']
    df = df[['original_organization_name','frequency', 'best_match_score', 'cleaned_best_match_name', 'original_best_match_name', 'dataset']]
    df['hand_match']=''
    df['notes']=''
    df['original_organization_name'] = df.groupby('comment_org_name')['original_organization_name'].apply(list)
    df = df.groupby('comment_org_name').first()
    df = df.sort_values(by=['frequency','best_match_score'], ascending=False)
    return df

def get_comp(df1, df2):
    df1 = df1.groupby('comment_org_name').first()[['cleaned_best_match_name']]
    df2 = df2.groupby('comment_org_name').first()[['cleaned_best_match_name']]
    merged = df1.merge(df2, on ='comment_org_name', suffixes=('_old', '_new'))
    merged = merged[merged['cleaned_best_match_name_old'] != merged['cleaned_best_match_name_new']]
    merged = merged[merged['cleaned_best_match_name_old'] == merged['cleaned_best_match_name_old']]
    return merged

       


# Comparison
# DATA_DIR = '/Users/jameschen/Team Name Dropbox/James Chen/FINREGRULEMAKE2/finreg/data/match_data/'
# old_file = "validation_20230627.csv"
# new_file = "validation_df_20230705170014.csv"
# df1 = pd.read_csv(DATA_DIR + old_file)
# df2 = pd.read_csv(DATA_DIR + new_file)
# comp = get_comp(df1,df2)
# comp.to_csv(DATA_DIR + "comparison_" + old_file + "_" + new_file + ".csv")

# Scoring


def get_validation_scores(validation, new_validation):
    validation = validation.set_index('original_org_name')
    validated_names = validation[validation['hand_match']==validation['hand_match']].index

    new_validation = new_validation.set_index('original_best_match_name')
    overlap_names = list(set(validated_names).intersection(set(new_validation.index)))

    prev_validated = validation.loc[overlap_names]
    new_validated = df.loc[overlap_names]

    prev_score = (prev_validated['cleaned_best_match_name'] == prev_validated['original_best_match_name'].apply(clean_fin_org_names)).mean()
    new_score = (new_validated['cleaned_best_match_name'] == new_validated['original_best_match_name'].apply(clean_fin_org_names)).mean()
    return prev_score, new_score

# def update_validation(validation, new_validation):
#     # first include additional rows
#     new_rows = 
#     pass

# clean validation as well by stripping and whitelines
def update_validation(validation, df):
    validation=validation.set_index('comment_org_name')
    validation.index = validation.index.str.replace('\r\n', '\n').str.strip()
    temp = validation.groupby('comment_org_name').first()

    for idx in range(len(df)):
        try:
            df.iloc[idx, -2] = temp['hand_match'].loc[df.index[idx]]
            df.iloc[idx, -1] = temp['notes'].loc[df.index[idx]]
        except:
            print(df.index[idx])
            
    return df

# org_name_df = get_organization_dataset()
# key_names_df = get_comment_dataset().iloc[:,:]
# # key_names_df = pd.read_csv('/Users/jameschen/Documents/Code/finreg/data/comment_metadata_orgs.csv').rename({'organization': 'original_organization_name'}, axis=1)
# key_names_df = clean_key_names_df(key_names_df)
# matches_df = get_matches(org_name_df, key_names_df)
# df = get_validation(matches_df,key_names_df)


# validation = pd.read_csv('/Users/jameschen/Downloads/Validation 20230706 - test.csv',keep_default_na=False)

# df = update_validation(validation, df)

# df.to_csv(BASE_DIR + "data/match_data/validation_df_" + curr_date + ".csv")
# # new_validation.index = new_validation.index.str.replace('\r\n', '\n').str.strip()


# apply to new dataset
BASE = '/Users/jameschen/Dropbox (HPP)/Petitions Project/BHC/'
rssd_linked_data_path = BASE + 'RSSD Linked Data/'
raw_data_path = BASE + 'Raw Data/'

org_name_df = pd.read_csv(raw_data_path + 'FFIEC banks.csv')
org_name_df = org_name_df.rename({"NM_LGL": "org_name", "#ID_RSSD": "unique_id"},axis=1)
key_names_df = pd.read_csv('/Users/jameschen/Documents/Code/BHC/Raw Data/Linkedin_law_experiences_filtered_keywords.csv',index_col=0,nrows=1000)
key_names_df['organization'] = key_names_df['organization'].astype(str).apply(clean_fin_org_names)


matches_df = get_matches(org_name_df, key_names_df)