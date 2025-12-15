import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from log_utils import get_logger
import argparse
import gc

logger = get_logger('recall_w2v')

def load_embeddings(emb_path):
    logger.info(f"Loading embeddings from {emb_path}...")
    emb_df = pd.read_csv(emb_path)
    # create dict: article_id -> numpy array
    emb_cols = [c for c in emb_df.columns if 'emb' in c]
    emb_matrix = emb_df[emb_cols].values
    article_ids = emb_df['article_id'].values
    
    # Normalize
    norm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norm + 1e-9)
    # Cast to float32 to save memory / speed up
    emb_matrix = emb_matrix.astype(np.float32)
    
    emb_dict = dict(zip(article_ids, emb_matrix))
    return emb_dict, article_ids, emb_matrix

def w2v_recall(user_item_time, query_users, emb_dict, article_ids, emb_matrix, topk=50, batch_size=1000):
    recall_res = {}
    logger.info("Generating Candidates (Batch Mode)...")
    
    item_matrix_T = emb_matrix.T
    
    # Process users in batches
    query_users = list(query_users)
    num_users = len(query_users)
    
    for i in tqdm(range(0, num_users, batch_size)):
        batch_users = query_users[i : i + batch_size]
        
        batch_user_embs = []
        valid_batch_users = []
        batch_histories = []
        
        for user in batch_users:
            history = user_item_time.get(user, [])
            if not history:
                continue
                
            # Mean pooling
            user_emb = np.zeros(emb_matrix.shape[1], dtype=np.float32)
            cnt = 0
            for item, _ in history:
                if item in emb_dict:
                    user_emb += emb_dict[item]
                    cnt += 1
            
            if cnt > 0:
                user_emb /= cnt
                batch_user_embs.append(user_emb)
                valid_batch_users.append(user)
                batch_histories.append(set([x[0] for x in history]))
            
        if not valid_batch_users:
            continue
            
        # Stack: (B, Dim)
        batch_user_embs = np.vstack(batch_user_embs)
        
        # Dot Product: (B, Dim) x (Dim, N) -> (B, N)
        scores = np.dot(batch_user_embs, item_matrix_T)
        
        # Top K
        # Efficient TopK
        # argpartition along axis 1
        top_indices = np.argpartition(scores, -topk, axis=1)[:, -topk:]
        
        # We need to sort these top k per user
        # This is slightly tricky vectorized, but loop is fine for B small
        
        # But we can grab the scores for top k
        # advanced indexing
        row_idx = np.arange(len(valid_batch_users))[:, None]
        top_scores = scores[row_idx, top_indices]
        
        # Now we sort top_scores
        sorted_idx_in_top = np.argsort(top_scores, axis=1)[:, ::-1]
        
        final_indices = top_indices[row_idx, sorted_idx_in_top]
        final_scores = top_scores[row_idx, sorted_idx_in_top]
        
        for j, user in enumerate(valid_batch_users):
            history_set = batch_histories[j]
            preds = []
            
            for k in range(topk):
                idx = final_indices[j, k]
                score = final_scores[j, k]
                item_id = article_ids[idx]
                
                if item_id in history_set:
                    continue
                preds.append((item_id, float(score)))
                
            recall_res[user] = preds
            
    return recall_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    BASE_DIR = f'../user_data/data/offline' if args.mode == 'valid' else f'../user_data/data/online'
    CLICK_PATH = os.path.join(BASE_DIR, 'click.pkl')
    QUERY_PATH = os.path.join(BASE_DIR, 'query.pkl')
    SAVE_PATH = os.path.join(BASE_DIR, 'w2v_recall.pkl')
    EMB_PATH = '../tcdata/articles_emb.csv'
    
    logger.info(f"Mode: {args.mode}")
    
    if not os.path.exists(CLICK_PATH):
        logger.error("Data not found.")
        exit(1)
        
    click_df = pd.read_pickle(CLICK_PATH)
    query_df = pd.read_pickle(QUERY_PATH)
    
    # Build history dict
    logger.info("Building User History...")
    user_item_time = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: list(zip(x.click_article_id, x.click_timestamp))).to_dict()
    
    target_users = query_df['user_id'].unique()
    
    # Load Emb
    emb_dict, aids, emb_mat = load_embeddings(EMB_PATH)
    
    # Recall
    res = w2v_recall(user_item_time, target_users, emb_dict, aids, emb_mat, batch_size=500)
    
    logger.info(f"Saving to {SAVE_PATH}...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(res, f)
    logger.info("Done.")
