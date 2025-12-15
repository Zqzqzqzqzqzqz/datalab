import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from log_utils import get_logger
import argparse
import gc

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from log_utils import get_logger
import argparse
import gc
from recall_w2v import load_embeddings

logger = get_logger('rank_feature')

def get_article_features(articles_path):
    logger.info("Loading article features...")
    df = pd.read_csv(articles_path)
    # Basic processing
    # created_at_ts -> datetime? or just keep as int
    return df

def generate_features(recall_pkl, click_pkl, query_pkl, articles_df, articles_emb_path, mode='valid'):
    logger.info(f"Generating features for {mode}...")
    
    with open(recall_pkl, 'rb') as f:
        recall_res = pickle.load(f)
        
    click_df = pd.read_pickle(click_pkl)
    query_df = pd.read_pickle(query_pkl)
    
    # 1. Base Samples Table
    samples = []
    
    # Map label for valid
    labels = {}
    if mode == 'valid':
        labels = dict(zip(query_df['user_id'], query_df['click_article_id']))
        
    logger.info("Constructing base samples...")
    for user, items in tqdm(recall_res.items()):
        
        # If valid mode and label exists
        label_id = labels.get(user, -1)
        
        for i, (item, score) in enumerate(items):
            label = 1 if (mode == 'valid' and item == label_id) else 0
            
            samples.append([user, item, score, i, label])
            
    sample_df = pd.DataFrame(samples, columns=['user_id', 'article_id', 'recall_score', 'rank_idx', 'label'])
    
    logger.info(f"Sample size: {len(sample_df)}")
    if len(sample_df) == 0:
        return sample_df
        
    # 2. Join User History Features
    # Simple features: User click count
    logger.info("Adding User Stats...")
    user_stats = click_df.groupby('user_id').size().reset_index(name='user_click_cnt')
    sample_df = sample_df.merge(user_stats, on='user_id', how='left')
    
    # Last click timestamp of user
    last_click = click_df.groupby('user_id')['click_timestamp'].max().reset_index(name='last_click_ts')
    sample_df = sample_df.merge(last_click, on='user_id', how='left')
    

    
    # 3. Join Article Features
    logger.info("Adding Article Features...")
    # article_id, category_id, created_at_ts, words_count
    sample_df = sample_df.merge(articles_df, left_on='article_id', right_on='article_id', how='left')
    
    # Article Global Popularity (from click_df history)
    logger.info("Adding Article Popularity...")
    article_cnt = click_df.groupby('click_article_id').size().reset_index(name='article_click_cnt')
    sample_df = sample_df.merge(article_cnt, left_on='article_id', right_on='click_article_id', how='left')
    sample_df['article_click_cnt'] = sample_df['article_click_cnt'].fillna(0)
    sample_df.drop(columns=['click_article_id'], inplace=True)

    # 4. Cross Features
    # Time diff: Last click vs Article Create Time (Proxy for freshness/relevance?)
    # Ideally: Prediction Timestamp - Article Create Time
    # Validation Timestamp = Last click info
    # We use 'last_click_ts' as the query time proxy
    
    sample_df['ts_diff'] = sample_df['last_click_ts'] - sample_df['created_at_ts']
    sample_df['words_count'] = sample_df['words_count'].fillna(0) # Simple fill
    
    # 5. Embedding Similarity Features
    logger.info(f"Loading embeddings from {articles_emb_path} for Similarity Features...")
    emb_dict, aids, emb_matrix = load_embeddings(articles_emb_path)
    
    # Calculate User Embeddings (Mean of History)
    logger.info("Calculating User Embeddings...")
    user_embs = {}
    # Optimization: vectorized groupby mean
    # Instead of loop, map article_id to emb vector in click_df? Memory heavy.
    # Iterative is safer for memory.
    
    # Filter click_df to only efficient columns
    hist_df = click_df[['user_id', 'click_article_id']]
    # We only need users appearing in sample_df
    target_users = set(sample_df['user_id'].unique())
    hist_grp = hist_df[hist_df['user_id'].isin(target_users)].groupby('user_id')
    
    for user, group in tqdm(hist_grp):
        # items
        items = group['click_article_id'].values
        # vectors
        vectors = []
        for item in items:
            if item in emb_dict:
                vectors.append(emb_dict[item])
        if vectors:
            user_embs[user] = np.mean(vectors, axis=0)
        else:
            user_embs[user] = np.zeros(emb_matrix.shape[1])
            
    # Compute dot product for each sample
    logger.info("Computing Embedding Similarity...")
    # This vectorizes per-row?
    # Iterate or apply? Apply is slow.
    # Vectorized:
    # Extract User Vectors aligned with sample_df['user_id']
    # Extract Item Vectors aligned with sample_df['article_id']
    
    # Create array A (N, D) and B (N, D)
    # But first we need to map
    
    # Let's do a simple apply for clarity and safety structure first, optimization if needed.
    # Actually, constructing the big matrix `user_vecs` is fine.
    
    # Map users to vectors
    # Default 0 vector
    dim = emb_matrix.shape[1]
    
    # Pre-compute dict for fast lookup? Already in user_embs.
    
    # We can use pd map.
    # But pd map returns objects (arrays).
    # Then np.stack.
    
    # sample_df['user_vec'] = sample_df['user_id'].map(user_embs) 
    # sample_df['item_vec'] = sample_df['article_id'].map(emb_dict)
    
    # Doing this might consume RAM if N is large.
    # If N=200k, D=256 -> 50MB * 2. It's fine.
    
    sim_scores = []
    # Using python loop with lookups might be faster than pandas overhead for object columns
    
    u_ids = sample_df['user_id'].values
    i_ids = sample_df['article_id'].values
    
    for u, i in tqdm(zip(u_ids, i_ids), total=len(u_ids)):
        u_vec = user_embs.get(u)
        i_vec = emb_dict.get(i)
        
        if u_vec is None or i_vec is None:
            sim_scores.append(0.0)
        else:
            sim_scores.append(float(np.dot(u_vec, i_vec)))
            
    sample_df['emb_sim_score'] = sim_scores

    # Cleanup
    del click_df, query_df, emb_dict, emb_matrix
    gc.collect()
    
    return sample_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    BASE_DIR = f'../user_data/data/offline' if args.mode == 'valid' else f'../user_data/data/online'
    
    RECALL_PKL = os.path.join(BASE_DIR, 'merged_recall.pkl')
    CLICK_PKL = os.path.join(BASE_DIR, 'click.pkl')
    QUERY_PKL = os.path.join(BASE_DIR, 'query.pkl')
    OUT_PKL = os.path.join(BASE_DIR, 'feature.pkl')
    
    ARTICLES = '../tcdata/articles.csv'
    ARTICLES_EMB = '../tcdata/articles_emb.csv'
    
    if not os.path.exists(RECALL_PKL):
        logger.error(f"{RECALL_PKL} not found.")
        exit(1)
        
    art_df = get_article_features(ARTICLES)
    
    feat_df = generate_features(RECALL_PKL, CLICK_PKL, QUERY_PKL, art_df, ARTICLES_EMB, mode=args.mode)
    
    logger.info(f"Saving features to {OUT_PKL}...")
    feat_df.to_pickle(OUT_PKL)
    logger.info("Done.")
