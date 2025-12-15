import pandas as pd
import numpy as np
import pickle
import os
import collections
import math
from tqdm import tqdm
from log_utils import get_logger
import argparse

logger = get_logger('recall_itemcf')

class ItemCF:
    def __init__(self, use_time_decay=True, alpha=0.9):
        self.use_time_decay = use_time_decay
        self.alpha = alpha
        self.item_sim_matrix = {}
        self.item_popular = collections.defaultdict(int)
        
    def fit(self, df):
        """
        Refined fit with position weighting and IUF
        df: history dataframe [user_id, click_article_id, click_timestamp]
        """
        logger.info("Building User-Item-Time Dict...")
        # Get user -> [(item, time)] list, sorted by time
        user_item_time_dict = self._get_user_item_time(df)
        
        # Count item popularity
        logger.info("Counting item popularity...")
        for user, items in user_item_time_dict.items():
            for item, _ in items:
                self.item_popular[item] += 1
                
        # Calculate Co-occurrence
        sim_item = collections.defaultdict(collections.defaultdict)
        logger.info("Calculating Co-occurrence matrix...")
        
        for user, items in tqdm(user_item_time_dict.items()):
            n_items = len(items)
            for i, (item_i, time_i) in enumerate(items):
                # Optimization: Only consider recent items if history is super long?
                # User's items are sorted by time.
                
                for j, (item_j, time_j) in enumerate(items):
                    if item_i == item_j:
                        continue
                        
                    # 1. Location Weight (Position Decay)
                    # items are sorted by time from old to new. 
                    # loc_weight: items closer to the end (recent) might be more important? 
                    # OR: items closer to EACH OTHER in sequence are related.
                    # The prompt says: "Position weights: closer is larger; positive/reverse order different"
                    # Simple implementation: 
                    # loc_weight = alpha ^ |i - j| ?
                    
                    # Implementation references specific logic:
                    # loc_alpha = 1.0 if time_decay else ...
                    # Let's use simple distance decay logic:
                    
                    loc_weight = 1.0
                    if self.use_time_decay:
                        loc_weight = 1.0 / (math.log(len(items) - i + 2) * math.log(len(items) - j + 2))
                    
                    # 2. IUF (Inverse User Frequency)
                    # Penalize active users: / log(1 + len(items))
                    weight = loc_weight / math.log(1 + len(items))
                    
                    sim_item[item_i][item_j] = sim_item[item_i].get(item_j, 0) + weight
                    
        # Normalize
        logger.info("Normalizing similarity matrix...")
        self.item_sim_matrix = sim_item
        for i, related_items in tqdm(sim_item.items()):
            for j, wij in related_items.items():
                self.item_sim_matrix[i][j] = wij / math.sqrt(self.item_popular[i] * self.item_popular[j])
                
    def recommend(self, user_history, top_k=50):
        rank = collections.defaultdict(float)
        interacted_items = set([x[0] for x in user_history])
        
        # Taking last K items from history to recommend? 
        # Usually last 5 items.
        target_hist = user_history[::-1][:15]
        
        for idx, (item_i, _) in enumerate(target_hist):
            if item_i not in self.item_sim_matrix:
                continue
            
            time_weight = np.exp(-0.1 * idx)
            
            for item_j, w_ij in sorted(self.item_sim_matrix[item_i].items(), key=lambda x:x[1], reverse=True)[:200]:
                if item_j in interacted_items:
                    continue
                rank[item_j] += w_ij * time_weight
                
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _get_user_item_time(self, df):
        # We assume df is already sorted by time if passed from data.py
        # But to be safe sort again or rely on group order
        # Optim: use iterator
        
        # Group by user and collect list
        # Using pandas magic
        logger.info("Grouping data...")
        # To make it fast, we can convert to list of tuples manually or use parallel apply
        # Here we trust pandas groupby
        r = df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: list(zip(x.click_article_id, x.click_timestamp))).to_dict()
        return r
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    BASE_DIR = f'../user_data/data/offline' if args.mode == 'valid' else f'../user_data/data/online'
    CLICK_PATH = os.path.join(BASE_DIR, 'click.pkl')
    QUERY_PATH = os.path.join(BASE_DIR, 'query.pkl')
    SAVE_PATH = os.path.join(BASE_DIR, 'itemcf_recall.pkl')
    
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Loading data from {CLICK_PATH}...")
    
    if not os.path.exists(CLICK_PATH):
        logger.error(f"Data not found! Run data.py --mode {args.mode} first.")
        exit(1)
        
    click_df = pd.read_pickle(CLICK_PATH)
    query_df = pd.read_pickle(QUERY_PATH)
    
    # Fit
    itemcf = ItemCF()
    itemcf.fit(click_df)
    
    # Recall
    # Target users are those in query_df
    # We need their history from click_df to generate recommendations
    logger.info("Generating Candidates...")
    
    user_item_time = itemcf._get_user_item_time(click_df)
    
    recall_res = {}
    target_users = query_df['user_id'].unique()
    
    for user in tqdm(target_users):
        hist = user_item_time.get(user, [])
        preds = itemcf.recommend(hist, top_k=100) # Recall 100
        recall_res[user] = preds
        
    # Save
    logger.info(f"Saving to {SAVE_PATH}...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(recall_res, f)
        
    logger.info("Done.")
