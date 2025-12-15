
import pandas as pd
import numpy as np
import collections
import pickle
import os
import argparse
from tqdm import tqdm
from log_utils import get_logger

logger = get_logger('recall_hot')

def hot_recall_improved(user_item_time_dict, topk=50):
    """
    Time-Decayed Popularity
    """
    item_time_score = collections.defaultdict(float)
    
    max_time = 0
    for user, items in user_item_time_dict.items():
        for item, time in items:
            if time > max_time:
                max_time = time
                
    # 1 day = 86400000 ms
    logger.info("Calculating Time-Decayed Popularity...")
    for user, items in tqdm(user_item_time_dict.items()):
        for item, time in items:
            diff = max_time - time
            days_diff = diff / 86400000.0
            score = 1.0 / (1.0 + days_diff)
            item_time_score[item] += score
            
    popular_items = sorted(item_time_score.items(), key=lambda x: x[1], reverse=True)[:topk+20]
    
    recall_res = {}
    logger.info("Generating Hot Recall Candidates...")
    for user, history in tqdm(user_item_time_dict.items()):
        clicked_set = set([x[0] for x in history])
        preds = []
        for item, score in popular_items:
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
    SAVE_PATH = os.path.join(BASE_DIR, 'hot_recall.pkl')
    
    if not os.path.exists(CLICK_PATH):
        logger.error("Click data not found.")
        exit(1)
        
    click_df = pd.read_pickle(CLICK_PATH)
    
    # Build history
    logger.info("Building history...")
    user_item_time = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: list(zip(x.click_article_id, x.click_timestamp))).to_dict()
    
    # Run
    res = hot_recall_improved(user_item_time, topk=50)
    
    logger.info(f"Saving to {SAVE_PATH}...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(res, f)
    logger.info("Done.")
