import pandas as pd
import numpy as np
import pickle
import os
import collections
import math
from tqdm import tqdm
from log_utils import get_logger
import argparse

logger = get_logger('recall_binetwork')

def get_user_item_time_dict(df):
    logger.info("Building User-Item Dict...")
    r = df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: list(zip(x.click_article_id, x.click_timestamp))).to_dict()
    return r

def get_item_user_time_dict(df):
    logger.info("Building Item-User Dict...")
    # This is inverse of user-item
    r = df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(lambda x: list(zip(x.user_id, x.click_timestamp))).to_dict()
    return r

def bi_graph_recall(user_item_time, item_user_time, query_users, topk=100):
    recall_res = {}
    logger.info("Generating Candidates...")
    
    for user in tqdm(query_users):
        rank = collections.defaultdict(float)
        
        history = user_item_time.get(user, [])
        if not history:
            continue
            
        # Only use recent click as seed?
        # User defined strategy: "Recall based on recent 1 click"
        recent_item = history[-1][0]
        
        # Propagation: item -> users -> items
        if recent_item not in item_user_time:
            continue
            
        # Get users who clicked this item
        users_j_list = item_user_time[recent_item]
        
        # Limit connected users to avoid explosion (e.g., top 50 recent users)
        # Sort users by time?
        # users_j_list.sort(key=lambda x: x[1], reverse=True)
        # users_j_list = users_j_list[:50]
        
        for user_j, _ in users_j_list:
            if user_j == user:
                continue
            
            # 2nd Step: Get items clicked by user_j
            if user_j not in user_item_time:
                continue
                
            items_k_list = user_item_time[user_j]
            
            # W = 1 / log(1 + len(users_j)) * 1 / log(1 + len(items_k))
            # Simplified degree normalization
            
            w_uj = 1.0 / math.log(1 + len(users_j_list))
            w_uk = 1.0 / math.log(1 + len(items_k_list))
            
            weight = w_uj * w_uk
            
            for item_k, _ in items_k_list:
                if item_k == recent_item:
                    continue
                rank[item_k] += weight
                
        # Filter history
        clicked_set = set([x[0] for x in history])
        preds = []
        for item, score in sorted(rank.items(), key=lambda x:x[1], reverse=True):
            if item in clicked_set:
                continue
            preds.append((item, score))
            if len(preds) >= topk:
                break
                
        recall_res[user] = preds
        
    return recall_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    BASE_DIR = f'../user_data/data/offline' if args.mode == 'valid' else f'../user_data/data/online'
    CLICK_PATH = os.path.join(BASE_DIR, 'click.pkl')
    QUERY_PATH = os.path.join(BASE_DIR, 'query.pkl')
    SAVE_PATH = os.path.join(BASE_DIR, 'binetwork_recall.pkl')
    
    logger.info(f"Mode: {args.mode}")
    
    if not os.path.exists(CLICK_PATH):
        logger.error(f"Data not found! Run data.py first.")
        exit(1)
        
    click_df = pd.read_pickle(CLICK_PATH)
    query_df = pd.read_pickle(QUERY_PATH)
    
    user_item = get_user_item_time_dict(click_df)
    item_user = get_item_user_time_dict(click_df)
    
    target_users = query_df['user_id'].unique()
    
    res = bi_graph_recall(user_item, item_user, target_users)
    
    logger.info(f"Saving to {SAVE_PATH}...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(res, f)
    logger.info("Done.")
