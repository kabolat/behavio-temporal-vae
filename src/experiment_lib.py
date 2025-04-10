import os, json, pickle
import pandas as pd
import numpy as np

from . import utils, preprocess_lib

def collect_results(config_dir, config_file='config.json', sub_folder=""):
    df = pd.DataFrame()
    for i, folder in enumerate(os.listdir(os.path.join(config_dir))):
        # Load config file
        with open(os.path.join(config_dir, folder, config_file), 'r') as f: config = json.load(f)
        config_flt = utils.flatten_dict(config)
        config_flt["model_folder"] = folder

        df_config_val = pd.DataFrame(config_flt, index=[i])

        # Load test results
        if not os.path.exists(os.path.join(config_dir, folder, sub_folder, 'test_results_aggregate.pkl')): df_test = pd.DataFrame()
        else:
            with open(os.path.join(config_dir, folder, sub_folder, 'test_results_aggregate.pkl'), 'rb') as f: test_results = pickle.load(f)
            test_results_flt = utils.flatten_dict(test_results)
            ##add to dataframe
            df_test = pd.DataFrame([test_results_flt], index=[i])

        df = pd.concat([df, pd.concat([df_config_val, df_test], axis=1)], axis=0)
    
    return df

def rename_columns(df_):
    df = df_.rename(columns={ 'data_ampute_params_b': 'Availability Rate (b)',
                        'data_condition_tag_list': 'Conditions',
                        'model_distribution_dict_likelihood_dist_type': 'Likelihood Distribution',
                        'data_user_embedding_kwargs_model_kwargs_num_topics': 'Number of LDA Topics',
                        'data_user_embedding_kwargs_model_kwargs_num_clusters': 'Number of LDA Clusters',
                        'model_distribution_dict_likelihood_vocab_size': 'Pattern Dictionary Size',
                        'model_distribution_dict_likelihood_dropout': 'Dropout',
                        'model_distribution_dict_likelihood_num_hidden_layers': 'Number of Layers',
                        'model_distribution_dict_likelihood_num_neurons': 'Number of Neurons',
                        'train_beta': 'Beta',
                        'test_loglikelihood': 'Log-Likelihood (Test)',
                        'missing_loglikelihood': 'Log-Likelihood (Missing)',
                        'model_distribution_dict_likelihood_sigma_lim': 'epsilon'
                    }, inplace=False)
    return df

def add_columns(df_):
    df = df_.copy()
    df["NLL (Test)"] = -df["Log-Likelihood (Test)"]
    # df["NLL (Missing)"] = -df["Log-Likelihood (Missing)"]

    df.loc[df["Conditions"].apply(lambda x: "users" not in x), "Number of LDA Topics"] = 0
    df.loc[df["Conditions"].apply(lambda x: "users" not in x), "Number of LDA Clusters"] = 0
    df.loc[df["Likelihood Distribution"] != "dict-gauss", "Pattern Dictionary Size"] = 0

    df["Covariance Structure"] = df["Likelihood Distribution"].apply(lambda x: "Diagonal" if x == "normal" else "PDCC")
    df["User-Informed"] = df["Number of LDA Topics"].apply(lambda x: "Yes" if x>0 else "No")

    try:
        df["Expected Missing Days"] = np.round(df['data_ampute_params_a'] / (df['data_ampute_params_a'] + df['Availability Rate (b)']) * 365, 1)
        utils.blockPrint()
        df["Missing Set Size"] = df.apply(lambda x: np.sum(preprocess_lib.generate_random_enrolments(n=365, a=0.85, b=x["Availability Rate (b)"], size=6830, random_seed=x["data_random_seed"])), axis=1)
        utils.enablePrint()
    except:
        df["Expected Missing Days"] = np.nan
        df["Missing Set Size"] = np.nan

    df["Test Set Size"] = (6830*365 - df["Missing Set Size"])*df["data_test_ratio"]
    df["Validation Set Size"] = (6830*365 - df["Missing Set Size"])*df["data_val_ratio"]
    df["Training Set Size"] = (6830*365 - df["Missing Set Size"]) - df["Test Set Size"] - df["Validation Set Size"]

    df["Test Set Ratio"] = df["Test Set Size"] / (6830*365)
    df["Validation Set Ratio"] = df["Validation Set Size"] / (6830*365)
    df["Missing Set Ratio"] = df["Missing Set Size"] / (6830*365)
    df["Training Set Ratio"] = 1 - df["Test Set Ratio"] - df["Validation Set Ratio"] - df["Missing Set Ratio"]
    return df