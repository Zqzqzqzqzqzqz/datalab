import pandas as pd
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
from log_utils import get_logger

logger = get_logger('data_process')

# Constraint on memory: define types
DTYPES = {
    'user_id': 'int32',
    'click_article_id': 'int32',
    'click_timestamp': 'int64',
    'click_environment': 'int8',
    'click_deviceGroup': 'int8',
    'click_os': 'int8',
    'click_country': 'int8',
    'click_region': 'int8',
    'click_referrer_type': 'int8'
}

def get_data_dir(mode):
    if mode == 'valid':
        return '../user_data/data/offline'
    else:
        return '../user_data/data/online'

def make_valid_data(train_path):
    logger.info("Processing Offline Validator Data...")
    
    # Read Train
    df = pd.read_csv(train_path, dtype=DTYPES)
    df = df.sort_values(['user_id', 'click_timestamp'])
    
    # Sample 50k users for validation to speed up
    user_ids = df['user_id'].unique()
    sample_users = np.random.choice(user_ids, size=50000, replace=False)
    
    # Label as Sampled
    df_sample = df[df['user_id'].isin(sample_users)]
    df_other = df[~df['user_id'].isin(sample_users)]
    
    # For Sampled Users: Last Click is Query (Validation Label), Rest is History
    # For Other Users: All is History
    
    logger.info("Splitting History and Query for Validation...")
    
    # Split last click
    val_query = df_sample.groupby('user_id').tail(1)
    
    # History = (Others) + (Sampled excluding last click)
    # 1. Drop val_query from df_sample
    val_history = df_sample.drop(val_query.index)
    
    # 2. Concat
    # history_df includes full history of 'other' users + partial history of 'val' users
    # This forms the "Knowledge Base" for recall
    history_df = pd.concat([df_other, val_history])
    
    # Save
    save_dir = get_data_dir('valid')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Saving to {save_dir}...")
    
    # Save as Pickle
    # Format: 
    # click.pkl -> DataFrame of history
    # query.pkl -> DataFrame of targets (validation labels)
    
    history_df.to_pickle(os.path.join(save_dir, 'click.pkl'))
    val_query.to_pickle(os.path.join(save_dir, 'query.pkl'))
    
    logger.info(f"Saved click.pkl ({len(history_df)}) and query.pkl ({len(val_query)})")

def make_test_data(train_path, test_path):
    logger.info("Processing Online Test Data...")
    
    train = pd.read_csv(train_path, dtype=DTYPES)
    test = pd.read_csv(test_path, dtype=DTYPES)
    
    # For Online:
    # History = All Train + All Test (except the 'future' click we want to predict? 
    # Actually TestA has history. We want to predict the NEXT click after the last one in TestA?
    # Usually TestA contains observed clicks. We predict the NEXT one.
    # So 'History' is everything we have.
    # 'Query' is the user IDs in Test set.
    
    # But for compatibility with recall scripts (which use 'click.pkl' to build connections),
    # we should put all available data in 'click.pkl'.
    
    data = pd.concat([train, test])
    data = data.sort_values(['user_id', 'click_timestamp'])
    
    # But wait, specific recall scripts usually split "History" (to build model/graph) and "Query" (to predict).
    # For Test Users, their entire sequence in TestA IS the history. We predict the NEXT unknown click.
    # So `click.pkl` should contain everything. 
    # `query.pkl` should contain the Test Users to predict for.
    # To match 'valid' format (where query has a 'click_article_id' as label), 
    # we can put a dummy article_id for test query or just the last row?
    # Actually, let's look at how recall scripts use 'query'. 
    # If they use it just for user_id, we are fine.
    
    # Let's save specific rows as 'query' to indicate WHO to predict.
    # We take the last observed click of test users as the 'anchor' (or just specific user list).
    # Let's use `groupby('user_id').tail(1)` of Test Data as the 'Query' placeholder.
    
    save_dir = get_data_dir('test')
    os.makedirs(save_dir, exist_ok=True)
    
    # History = All data
    data.to_pickle(os.path.join(save_dir, 'click.pkl'))
    
    # Query = Unique users in Test set (represented by their last row)
    test_query = test.groupby('user_id').tail(1)
    test_query.to_pickle(os.path.join(save_dir, 'query.pkl'))
    
    logger.info(f"Saved click.pkl ({len(data)}) and query.pkl ({len(test_query)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    TRAIN_PATH = '../tcdata/train_click_log.csv'
    TEST_PATH = '../tcdata/testA_click_log.csv'
    
    if args.mode == 'valid':
        make_valid_data(TRAIN_PATH)
    else:
        make_test_data(TRAIN_PATH, TEST_PATH)
