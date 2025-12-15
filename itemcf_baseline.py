import pandas as pd
import numpy as np
import os
import collections
from tqdm import tqdm
import math

class ItemCF:
    def __init__(self, use_time_decay=True, alpha=0.5):
        self.use_time_decay = use_time_decay
        self.alpha = alpha
        self.user_item_time = collections.defaultdict(list)
        self.item_sim_matrix = {}
        self.item_popular = collections.defaultdict(int)
        self.item_count = 0
        
    def get_sim_item(self, df, user_col, item_col, timestamp_col):
        """
        Calculate item similarity matrix
        """
        # Group by user and sort by time
        user_item_vals = df.groupby(user_col)[[item_col, timestamp_col]].apply(lambda x: x.values.tolist()).to_dict()
        
        sim_item = collections.defaultdict(collections.defaultdict)
        # Count item occurrences
        for _, items in tqdm(user_item_vals.items(), desc="Counting items"):
            for item, _ in items:
                self.item_popular[item] += 1
                
        self.item_count = len(self.item_popular)
        print(f"Total items: {self.item_count}")

        # Calculate Co-occurrence
        for _, items in tqdm(user_item_vals.items(), desc="Calculating Co-occurrence"):
            # Sort valid items by timestamp just in case (though we did simple list extract)
            # Actually, the dataframe might not be sorted per group, let's trust the input or sort it
            # items.sort(key=lambda x: x[1]) # Assuming input is sorted or we sort here. 
            # Doing simple n^2 loop for each user history is fine for small length
            
            for i, (item_i, time_i) in enumerate(items):
                # Optimization: Only look at recent window or full history? 
                # Full history for 200k users with short history is fine.
                for j, (item_j, time_j) in enumerate(items):
                    if item_i == item_j:
                        continue
                        
                    # Time decay factor: 1 / (1 + |t_i - t_j|) or similar. 
                    # Usually: 1 / (1 + delta_time_hours) or just simple 1.
                    # Here we use order distance weight or time weight if needed.
                    # Simple first:
                    weight = 1.0
                    
                    if self.item_popular[item_i] > 0 and self.item_popular[item_j] > 0:
                        sim_item[item_i][item_j] = sim_item[item_i].get(item_j, 0) + weight / math.log(1 + len(items))
                        
        # Normalize
        print("Normalizing similarity matrix...")
        self.item_sim_matrix = sim_item
        for i, related_items in tqdm(sim_item.items(), desc="Normalizing"):
            for j, wij in related_items.items():
                self.item_sim_matrix[i][j] = wij / math.sqrt(self.item_popular[i] * self.item_popular[j])

    def recommend(self, user_history, top_k=5):
        """
        Generate recommendations for a single user
        user_history: list of item_ids the user has clicked
        """
        rank = collections.defaultdict(float)
        interacted_items = set(user_history)
        
        for item_i in user_history:
            if item_i not in self.item_sim_matrix:
                continue
                
            for item_j, w_ij in sorted(self.item_sim_matrix[item_i].items(), key=lambda x:x[1], reverse=True)[:20]:
                if item_j in interacted_items:
                    continue
                rank[item_j] += w_ij
                
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

from log_utils import get_logger

# 初始化logger
logger = get_logger('itemcf_baseline')

def main():
    # Paths
    TRAIN_FILE = '../tcdata/train_click_log.csv'
    TEST_FILE = '../tcdata/testA_click_log.csv'
    OUTPUT_FILE = '../prediction_result/result.csv'
    
    # 1. Load Data
    logger.info("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    # Combine for training (using all history to build item connections)
    # Note: In real competition, be careful not to leak future info if strictly sequential.
    # But for ItemCF co-occurrence, using all available history is standard.
    # We only care about predicting 'next' click for Test users.
    
    all_click = pd.concat([train_df, test_df])
    logger.info(f"Total clicks loaded: {len(all_click)}")
    
    # 2. Train ItemCF
    item_cf = ItemCF()
    # Note: Inside ItemCF we could also add logging, but for now we keep it simple or pass logger?
    # Ideally ItemCF class should use logger too, but let's stick to main flow first.
    logger.info("Starting ItemCF training...")
    item_cf.get_sim_item(all_click, 'user_id', 'click_article_id', 'click_timestamp')
    logger.info("ItemCF training finished.")
    
    # 3. Predict for Test Users
    # We need to predict for users in test_df
    # Get their history from all_click (since test users might have appeared in train? 
    # Actually split is by user, so test users are disjoint from train users usually.
    # But testA file contains their click history.)
    
    test_users = test_df['user_id'].unique()
    test_user_histories = test_df.groupby('user_id')['click_article_id'].apply(list).to_dict()
    
    preds = []
    
    logger.info(f"Generating predictions for {len(test_users)} users...")
    for user in tqdm(test_users):
        hist = test_user_histories.get(user, [])
        rec_items = item_cf.recommend(hist, top_k=5)
        
        # If not enough items, pad with popular ones?
        # For now, just take what we have. 
        # (Improvement: Global TopN fallback)
        
        rec_ids = [str(x[0]) for x in rec_items]
        
        # Fallback to popular items if < 5
        if len(rec_ids) < 5:
            # Simple fallback: Top 5 global popular items
            # Precompute these for speed
            popular_fallback = sorted(item_cf.item_popular.items(), key=lambda x:x[1], reverse=True)[:5]
            for pop_item, _ in popular_fallback:
                if str(pop_item) not in rec_ids and str(pop_item) not in [str(x) for x in hist]:
                    rec_ids.append(str(pop_item))
                if len(rec_ids) >= 5:
                    break
        
        preds.append([user] + rec_ids[:5])
        
    # 4. Save
    logger.info("Saving submission...")
    sub_df = pd.DataFrame(preds, columns=['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5'])
    sub_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
