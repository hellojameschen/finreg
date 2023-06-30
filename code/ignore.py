        # covariate_dict[elem] = {**org_match_covariate_dict}
        # covariate_dict[elem]['comment_url'] = url
        # covariate_dict[elem]['comment_submitter_name'] = submitter_name
        covariate_dict.loc[original_org_name,'comment_org_name'] = org_name
        # covariate_dict[elem]['comment_agency'] = agency_acronym
        # covariate_dict[elem]['docket_id'] = docket_id
        covariate_dict.loc[original_org_name,'org_match_type'] = org_match_type
        covariate_dict.loc[original_org_name,'org_best_match_score'] = org_best_match_score  

        covariate_dict.loc[original_org_name,'num_org_matches'] = len(org_matches_collected)


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
    df = df[df['comment_org_name']!='']
    df.to_csv(BASE_DIR + "data/match_data/match_all_covariates_df_" + curr_date + ".csv")

    df = pd.read_csv(BASE_DIR + "data/match_data/match_all_covariates_df_" + curr_date + ".csv")
    df = df.drop("Unnamed: 0", axis=1)

    threshold = 0.95

    score_cols = list(filter(lambda x: "score" in x,df.columns))
    for score_col in score_cols:
        threshold_fail = df[score_col]<threshold
        all_cols = list(filter(lambda x: score_col.split(':')[0] in x,df.columns))
        df.loc[threshold_fail, all_cols] = np.NaN


    name_cols = list(filter(lambda x: "best_match_name" in x,df.columns))
    exact_matches = pd.DataFrame()
    for name_col in name_cols:
        exact_matches[name_col] = df[name_col]==df['comment_org_name']

    new_col = (exact_matches.sum(axis=1)>0).astype(int)
    df.insert(loc = 5,
          column = 'exact_match_present',
          value = new_col)
    
    cols = list(df.columns)
    front = [
        'comment_agency',
        'original_org_name',
        'comment_url',
        'docket_id',
        'comment_org_name',
        'num_org_matches',
        'exact_match_present',
        ]
    cols[:len(front)]= front
    
    df = df[cols]

    return df