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

from datetime import date
today_for_filenames = date.today()
curr_date = str(today_for_filenames.strftime("%Y%m%d"))

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

# function to clean org names
def clean_fin_org_names(name):
    if name is None or not isinstance(name, str) or name == "NA":
        return ""
    else:
        # James strip metadata from name
        name = name.split(',')[0]
        name = re.sub(" [0-9]* [k|m]b pdf","",name)

        name = name.translate(corp_simplify_utils.STR_TABLE)
        name = re.sub(stopword_re, '', name.lower())
        
        
        return corpHash(name)
    

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

def get_match_candidate_score(frequency_dict, org_name, candidate_match_name):
    org_tokens = org_name.split(' ')
    
    # tokenize the candidate match
    candidate_match_tokens = set(candidate_match_name.split(" "))

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
    weight = 1/len(org_name)
    longest_common_substring = get_longest_common_substring(org_name, candidate_match_name)
    match_score -= weight * len(candidate_match_name)/len(longest_common_substring) - weight

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

    # 1.1: Read and clean org names from gathered org datasets
    # sources = ["FFIECInstitutions", "CIK"]

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
        # org_name_df_lst = []
        # for source in sources:
        #     org_name_df_lst.append(org_name_df[org_name_df['financial_dataset']==source]['org_name'].apply(clean_fin_org_names))
        org_name_df['original_org_name'] = org_name_df['org_name']
        org_name_df['org_name'] = org_name_df['org_name'].apply(clean_fin_org_names)

        return org_name_df



def get_comment_dataset():
    # 1.2: Read and clean submitter and org names from scraped comments
    connection=apsw.Connection(DB_PATH)
    c=connection.cursor()

    c.execute("SELECT * FROM comments")
    key_names_list = c.fetchall()

    c.execute("PRAGMA table_info(comments);")
    column_names = [row[1] for row in c.fetchall()]

    print("Starting cleaning")
    df = pd.DataFrame(key_names_list, columns = column_names)
    cols = ['comment_url', 'submitter_name', 'organization', 'agency_acronym', 'docket_id', 'comment_title']

    df = df[cols]
    df['original_organization_name'] = df['organization']

    # FRS, take what is before first comma
    df.loc[df['agency_acronym']=='FRS', "organization"] = df.loc[df['agency_acronym']=='FRS', "organization"].str.split(',').map(lambda x: x[0]).map(lambda x: '' if '(' in x else x)

    # FDIC take before first comma, before with, and before -
    df.loc[df['agency_acronym']=='FDIC', "organization"] = df.loc[df['agency_acronym']=='FDIC', "organization"].str.split(',').map(lambda x: x[0] if x else '')
    df.loc[df['agency_acronym']=='FDIC', "organization"] = df.loc[df['agency_acronym']=='FDIC', "organization"].str.split(' with ').map(lambda x: x[1] if len(x)>1 else x[0])
    df.loc[df['agency_acronym']=='FDIC', "organization"] = df.loc[df['agency_acronym']=='FDIC', "organization"].str.split(' - ').map(lambda x: x[0] if x else '')

    # SEC is difficult
    df['submitter_name'] = df['submitter_name'].map(clean_fin_org_names)
    df['organization'] = df['organization'].map(clean_fin_org_names)

    #replace none
    df.loc[df['submitter_name'].isna(), "submitter_name"] = ''

    # key_names_list = df.iloc[:,:] # include how many to match

    return df


    # 1.3: Create 2 dicts with frequency counts of every token in the org and submitter name fields of the scraped comments db
    # submitter_frequency_dict = {}
    # org_frequency_dict = {}
    # for _, key_name in key_names_list.iterrows():
    #     org_name = key_name[2]
    #     for token in org_name.split(" "):
    #         if token in org_frequency_dict:
    #             org_frequency_dict[token] += 1
    #         else:
    #             org_frequency_dict[token] = 1

def get_matches(org_name_df, key_names_list):
    print('Preparing candidate frequency dictionary.')
    candidate_frequency_dict = {}
    for org_name in tqdm(org_name_df['org_name']):
        for token in org_name.split(" "):
            if token in candidate_frequency_dict:
                candidate_frequency_dict[token] += 1
            else:
                candidate_frequency_dict[token] = 1

    # Create linking dataset
    # 1.4: Create a dict mapping from tokens in the gathered org datasets to IDs and org_names that contain that token
    candidate_match_dict = {}
    print('Preparing candidate match dictionary.')
    for row_idx in tqdm(range(len(org_name_df))):
        row = org_name_df.iloc[row_idx]
        unique_id = row['unique_id']
        org_name = row['org_name']
        original_org_name = row['original_org_name']
        for token in org_name.split(" "):
            if token in candidate_match_dict:
                candidate_match_dict[token].append((unique_id, org_name, original_org_name))
            else:
                candidate_match_dict[token] = [(unique_id, org_name, original_org_name)]
                

    # Apply linking dataset
    # 1.5.1: For each org and submitter name in the scraped comment dataset, get all of the names ('candidate matches') from among the gathered org datasets that have the most important word of the scraped db names in the org's name. Calculate a tf-idf weighted jaccard index match score to choose the best matches among the candidates.
    print('Generating match dictionary.')
    match_dict = {}
    print("Num scraped records: " + str(len(key_names_list)))
    for key_name_idx in tqdm(range(len(key_names_list))):
        key_name = key_names_list.iloc[key_name_idx]
        org_name = key_name['organization']

        if not org_name:
            match_dict[org_name] = pd.DataFrame()
            continue

        if org_name in match_dict:
            continue

        # Tokenize the submitter name and org name
        org_tokens = org_name.split(" ")
        
        # Get the frequencies (in the scraped comment db) of the tokens in the submitter name and org name
        # submitter_token_frequencies = sorted([(submitter_token, submitter_frequency_dict[submitter_token]) for submitter_token in submitter_tokens], key=lambda x: x[1])
        org_token_frequencies = [(org_token, candidate_frequency_dict.get(org_token)) for org_token in org_tokens]
        org_token_frequencies = list(filter(lambda item: item[1] is not None, org_token_frequencies))
        org_token_frequencies = sorted(org_token_frequencies, key=lambda x: x[1])
        

        candidate_matches = []
        # Iterate through the candidate matches to the most informative token
        for most_unique_org_token, _ in org_token_frequencies[:1]: # uses top 2 most unique tokens
            if most_unique_org_token in candidate_match_dict:
                for row in candidate_match_dict[most_unique_org_token]:
                    unique_id = row[0]
                    candidate_match_name = row[1]
                    original_candidate_match_name = row[2]
                    match_score = get_match_candidate_score(candidate_frequency_dict, org_name, candidate_match_name)
                    candidate_matches.append((match_score, candidate_match_name, original_candidate_match_name, unique_id))

        # Sort the candidate matches, first by the match score and then by the absolute value of the difference in the number of tokens between the submitter (or org) name and the candidate match org name
        candidate_matches.sort(key=lambda x:(-x[0], abs(len(x[1].split(" ")) - len(org_tokens))))
        candidate_matches = pd.DataFrame(candidate_matches, columns=['match_score','candidate_match_name', 'original_org_name', 'unique_id'])
        #TODO: remove submitters
        # candidate_matches_list.append([])
        # candidate_matches_list.append(candidate_matches)


        # Record the candidate matches corresponding to the current scraped comment record
        match_dict[org_name] = candidate_matches


    # 1.5.2: Save the candidate matches and get record counts
    # with open(BASE_DIR + "data/finreg_jaccard_match_" + curr_date + ".pkl", 'wb') as pkl_out:
    #     pickle.dump(match_dict, pkl_out)

    print("Num scraped records: " + str(len(key_names_list)))


    # 1.6: Extract the scraped records with at least one candidate match and take the top top_matches_num (or all if there are < top_matches_num) matches from the scored candidate matches
    # DONE: loop until we get top match from each dataset
    counter = 0
    good_matches = {}
    for elem in tqdm(list(match_dict.keys())):

        # Org name
        good_org_matches = []
        collected_sources = set()
        matches = match_dict[elem]

        if len(matches) == 0:
            counter += 1   


        if len(collected_sources) == len(sources):
            break

        for match_candidate_idx in range(len(matches)):
            match_candidate = matches.iloc[match_candidate_idx]
            if len(collected_sources) == len(sources):
                break
            match_candidate_source = match_candidate['unique_id'].split('-')[0]
            if not match_candidate_source in collected_sources:
                good_org_matches.append(match_candidate)
                collected_sources.add(match_candidate_source)

        good_matches[elem] = good_org_matches
        
    ## PART 3: Create a data from to explore commenter covariates
    covariate_dfs = get_covariate_dfs()
    # 3.2: Make a dataframe organizing the covariates of the gathered datasets
    covariate_dict = {}
    frs_counter = 0
    for _, elem in tqdm(list(key_names_list.iterrows())):
        elem = tuple(elem)
        url = elem[0]
        submitter_name = elem[1]
        org_name = elem[2]
        agency_acronym = elem[3]
        if agency_acronym == "FRS":
            frs_counter += 1
        docket_id = elem[4]
        comment_title = elem[5]
        original_org_name = elem[6]

        matches = good_matches[org_name]

        
        org_match_covariate_dict = {}
        org_match_type = ""
        org_best_match_score = np.nan

        org_matches_collected = []
        org_types_collected = []
        if len(matches) > 0:
            for match in matches:
                org_best_match = match
                org_best_match_id = org_best_match['unique_id']
                org_match_type = org_best_match_id.split("-")[0]
                if not org_match_type in org_types_collected:
                    org_matches_collected.append(match)
                    org_types_collected.append(org_match_type)

            for org_match in org_matches_collected:
                org_best_match = org_match
                org_best_match_id = org_best_match['unique_id']
                org_best_match_name = org_best_match['candidate_match_name']
                original_best_match_name = org_best_match['original_org_name']
                org_best_match_score = clean_match_score(org_best_match['match_score'])
                if pd.isnull(org_best_match_score):
                    print("this shouldn't be null")
                org_match_type = org_best_match_id.split("-")[0]
                org_match_row_num = org_best_match_id.split("-")[1]
                org_match_covariate_dict.update(get_data_row(covariate_dfs, org_match_type, int(org_match_row_num), "orgMatch"))
                org_match_covariate_dict[org_match_type + '-orgMatch' + ":best_match_score"] = org_best_match_score
                org_match_covariate_dict[org_match_type + '-orgMatch' + ":best_match_name"] = org_best_match_name
                org_match_covariate_dict[org_match_type + '-orgMatch' + ":original_match_name"] = original_best_match_name

        
        covariate_dict[elem] = {**org_match_covariate_dict}
        covariate_dict[elem]['original_org_name'] = original_org_name
        covariate_dict[elem]['comment_url'] = url
        covariate_dict[elem]['comment_submitter_name'] = submitter_name
        covariate_dict[elem]['comment_org_name'] = org_name
        covariate_dict[elem]['comment_agency'] = agency_acronym
        covariate_dict[elem]['docket_id'] = docket_id
        covariate_dict[elem]['org_match_type'] = org_match_type
        covariate_dict[elem]['org_best_match_score'] = org_best_match_score  

        covariate_dict[elem]['num_org_matches'] = len(org_matches_collected)


    variables = set()
    for elem_idx, elem in tqdm(enumerate(covariate_dict)):
        variables = variables.union(set(covariate_dict[elem].keys()))
    variables = list(variables)
    variables.sort()

    data = []
    for elem in tqdm(covariate_dict.keys()):
        elem_data_dict = covariate_dict[elem]
        elem_data_row = [None]*len(variables)
        for var_idx, variable in enumerate(variables):
            if variable in elem_data_dict:
                elem_data_row[var_idx] = elem_data_dict[variable]
        data.append(elem_data_row)
    print("Finished creating items for df")

    covariate_df = pd.DataFrame(data, columns=variables)

    return covariate_df

def clean_covariate_df(covariate_df):

    # 3.3: Save the dataframe of scraped records with attached covariates

    # filter columns
    common_tails = ['best_match_name',
                    'original_match_name', 
                    'best_match_score', 
                    'CIK', 
                    'CU_NUMBER', 
                    'RSSD', 
                    'CERT', 
                    'FED_RSSD',
                    'FDIC Certificate Number',
                    'IDRSSD',
                    'OCC Charter Number',
                    'SIC',
                    'Ticker',
                    'cik',
                    'cusip',
                    'gvkey',
                    'naics',
                    'sic',
                    'tic',
                    'ein',
                    'name',
                    'parentID'
                    ]
    important_cols = ['original_org_name',
                      'num_org_matches', 
                      'num_submitter_matches', 
                      'comment_agency',
                      'comment_org_name',
                      'comment_submitter_name',
                      'docket_id',
                      'comment_url',
                      'unique_id',]
    important_cols = [x for x in covariate_df.columns if (x.split(':')[-1] in common_tails) or (x in important_cols)]
    covariate_df= covariate_df[important_cols]

    # reorder columns
    cols = covariate_df.columns
    cols = [x for x in cols if not ':' in x] + [x for x in cols if ':' in x]
    covariate_df= covariate_df[cols]

    # covariate_df.to_csv(BASE_DIR + "data/finreg_commenter_covariates_df_" + curr_date + ".csv")

    df = covariate_df
    df = df[list(filter(lambda x: not "submitter" in x,df.columns))]
    # df = df[df['comment_org_name']!='']
    # df.to_csv(BASE_DIR + "data/match_data/match_all_covariates_df_" + curr_date + ".csv")

    # df = pd.read_csv(BASE_DIR + "data/match_data/match_all_covariates_df_" + curr_date + ".csv")
    # df = df.drop("Unnamed: 0", axis=1)

    # threshold = 0.95

    # score_cols = list(filter(lambda x: "score" in x,df.columns))
    # for score_col in score_cols:
    #     threshold_fail = df[score_col]<threshold
    #     all_cols = list(filter(lambda x: score_col.split(':')[0] in x,df.columns))
    #     df.loc[threshold_fail, all_cols] = np.NaN


    # name_cols = list(filter(lambda x: "best_match_name" in x,df.columns))
    # exact_matches = pd.DataFrame()
    # for name_col in name_cols:
    #     exact_matches[name_col] = df[name_col]==df['comment_org_name']

    # new_col = (exact_matches.sum(axis=1)>0).astype(int)
    # df.insert(loc = 5,
    #       column = 'exact_match_present',
    #       value = new_col)
    
    # cols = list(df.columns)
    # front = [
    #     'comment_agency',
    #     'original_org_name',
    #     'comment_url',
    #     'docket_id',
    #     'comment_org_name',
    #     'num_org_matches',
    #     'exact_match_present',
    #     ]
    # cols[:len(front)]= front
    
    # df = df[cols]

    # df.to_csv(BASE_DIR + "data/match_data/match_df_" + curr_date + ".csv")
    return df


       
org_name_df = get_organization_dataset()
key_names_df = get_comment_dataset().iloc[:1000,:]
covariate_df = get_matches(org_name_df,key_names_df)
df = clean_covariate_df(covariate_df)