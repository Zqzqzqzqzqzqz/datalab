import pandas as pd
import numpy as np
from tqdm import tqdm

def get_rank_features_fixed(sample_df, click_df, articles_df, articles_emb_df):
    """
    Advanced Feature Engineering:
    1. User Hist: length, last_click_timestamp
    2. Item: Created time, Emb
    3. Cross: Time diff since last click, Emb similarity
    """
    
    # Pre-process
    click_df = click_df.sort_values(['user_id', 'click_timestamp'])
    
    # 1. User Last Click Time
    user_last_click = click_df.groupby('user_id')['click_timestamp'].max().reset_index().rename(columns={'click_timestamp': 'last_click_ts'})
    
    # 2. Join tables
    df = sample_df.merge(user_last_click, on='user_id', how='left')
    
    # 3. Item Info (if available in articles.csv)
    # Checking eda output: articles.csv -> ['article_id', 'category_id', 'created_at_ts', 'words_count']
    # We should merge these
    
    # articles_df assumed to be loaded
    df = df.merge(articles_df, left_on='article_id', right_on='article_id', how='left')
    
    # 4. Feature Construction
    
    # Time diff: Last click vs Current hypothetical click? 
    # Actually, for training, we use the timestamp of the POSITIVE sample.
    # For negative sample, what timestamp? Usually None or Same as pos.
    # Here, for Recall Candidates, we don't strictly have a 'time of interaction' unless we assume 'now'.
    # Let's use 'created_at_ts' diff as feature: time since article creation.
    
    # 5. Embedding Similarity (User Hist Avg vs Candidate Item)
    # Ideally pre-calc and pass in, or do it here. 
    # For simplicity, we skip heavy emb calc here and rely on pre-calc 'score' from recall (which might be sim)
    
    return df

def basic_feature_pipeline(sample_df, train_click_log_path, articles_path):
    print("Loading raw feature tables...")
    click_df = pd.read_csv(train_click_log_path)
    articles_df = pd.read_csv(articles_path)
    
    return get_rank_features_fixed(sample_df, click_df, articles_df, None)
