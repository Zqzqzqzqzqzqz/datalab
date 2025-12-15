import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from log_utils import get_logger
import argparse

logger = get_logger('recall_merge')

def normalize_scores(recall_dict):
    """
    Min-Max Normalize scores for each user
    """
    new_dict = {}
    for user, items in recall_dict.items():
        if not items:
            new_dict[user] = []
            continue
            
        scores = [x[1] for x in items]
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s
        
        norm_items = []
        for item, score in items:
            if range_s > 0:
                norm_s = (score - min_s) / range_s
            else:
                norm_s = 1.0
            norm_items.append((item, norm_s))
        new_dict[user] = norm_items
    return new_dict

def merge_recalls(recall_dicts, weights):
    """
    Merge multiple recall results.
    recall_dicts: {'name': dict}
    weights: {'name': float}
    """
    merged_res = {}
    
    # Collect all users
    all_users = set()
    for d in recall_dicts.values():
        all_users.update(d.keys())
        
    logger.info(f"Merging for {len(all_users)} users...")
    
    for user in tqdm(all_users):
        item_scores = {}
        
        for name, r_dict in recall_dicts.items():
            w = weights.get(name, 1.0)
            
            # Normalize first? Yes, assumed normalized before passing or here
            # We implemented normalization outside
            
            items = r_dict.get(user, [])
            for item, score in items:
                item_scores[item] = item_scores.get(item, 0) + score * w
                
        # Sort desc
        merged = sorted(item_scores.items(), key=lambda x:x[1], reverse=True)
        merged_res[user] = merged
        
    return merged_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    BASE_DIR = f'../user_data/data/offline' if args.mode == 'valid' else f'../user_data/data/online'
    
    # Load Recalls
    RECALLS = {
        'itemcf': os.path.join(BASE_DIR, 'itemcf_recall.pkl'),
        'w2v': os.path.join(BASE_DIR, 'w2v_recall.pkl'),
        'hot': os.path.join(BASE_DIR, 'hot_recall.pkl')
    }
    
    # Weights optimization
    WEIGHTS = {
        'itemcf': 0.5,
        'w2v': 0.3,
        'hot': 0.2
    }
    
    loaded_dicts = {}
    for name, path in RECALLS.items():
        if os.path.exists(path):
            logger.info(f"Loading {name} from {path}...")
            with open(path, 'rb') as f:
                d = pickle.load(f)
                # Normalize immediately
                logger.info(f"Normalizing {name}...")
                loaded_dicts[name] = normalize_scores(d)
        else:
            logger.warning(f"{path} not found, skipping {name}")
            
    # Merge
    merged = merge_recalls(loaded_dicts, WEIGHTS)
    
    # Filter Training Data (Valid Mode only)
    # If a user's true label is NOT in the recalled candidates, this sample is "useless" for ranking (all negatives).
    # Some strategies keep them as "easy negatives", but usually we drop them to balance.
    
    if args.mode == 'valid':
        QUERY_PATH = os.path.join(BASE_DIR, 'query.pkl')
        query_df = pd.read_pickle(QUERY_PATH)
        # map user -> label
        labels = dict(zip(query_df['user_id'], query_df['click_article_id']))
        
        valid_merged = {}
        hit_num = 0
        total_num = 0
        
        logger.info("Filtering invalid training samples (no hit)...")
        for user, items in merged.items():
            if user not in labels:
                continue
            
            label = labels[user]
            cand_ids = [x[0] for x in items]
            
            if label in cand_ids:
                valid_merged[user] = items
                hit_num += 1
            total_num += 1
            
        logger.info(f"Recall Hit Rate: {hit_num}/{total_num} = {hit_num/total_num if total_num>0 else 0:.4f}")
        logger.info(f"Filtered Users: {len(merged)} -> {len(valid_merged)}")
        merged = valid_merged
        
    SAVE_PATH = os.path.join(BASE_DIR, 'merged_recall.pkl')
    logger.info(f"Saving merged results to {SAVE_PATH}...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(merged, f)
    logger.info("Done.")
