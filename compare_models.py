import pandas as pd
import numpy as np
from recall import metrics_recall, get_user_item_time
from recall_itemcf import ItemCF
from recall_hot import hot_recall_improved
from recall_w2v import w2v_recall, load_embeddings
# recall_binetwork is slow for full check, maybe skip or tiny sample

def compare_recalls():
    print("Loading Data...")
    # Use smaller sample for quick comparison
    train_df = pd.read_csv('../tcdata/train_click_log.csv', nrows=20000)
    
    # Split Train/Val
    # Val = Last click of each user
    train_df = train_df.sort_values(['user_id', 'click_timestamp'])
    
    val_df = train_df.groupby('user_id').tail(1)
    
    # Training set is everything excluding the exact validation rows
    # Note: For ItemCF co-occurrence, usually we want ALL history.
    # But for strict evaluation, we should hide the last click.
    # Proper: masked_train = train_df.drop(val_df.index) 
    # But for simplicity let's use full train to build model and check if we can retrieve last item (leakage? yes if used to build, but ItemCF usually predicts NEXT from history).
    # If we use Co-occurrence of (A, B), and user history is A-B-C.
    # If we mask C. We compute Sim(A, B). Input A, B. Models suggests X. We hope X==C.
    # Actually, if we use full history A-B-C to build matrix, we definitely know C is related to B.
    # So strictly, we MUST mask C.
    
    print("Splitting Train/Val (Masking last click)...")
    masked_train = train_df.drop(val_df.index)
    
    # 1. ItemCF
    print("\n--- Running ItemCF ---")
    itemcf = ItemCF()
    itemcf.fit(masked_train)
    
    # Predict for val users
    user_item_time = itemcf._get_user_item_time(masked_train)
    
    itemcf_preds = {}
    for user in val_df['user_id'].unique():
        if user in user_item_time:
            itemcf_preds[user] = itemcf.recommend(user_item_time[user], top_k=50)
            
    score_itemcf = metrics_recall(itemcf_preds, val_df, topk=50)
    print(f"ItemCF Recall@50: {score_itemcf:.4f}")
    
    # 2. Hot Improved
    print("\n--- Running Hot Recall ---")
    hot_preds = hot_recall_improved(user_item_time, topk=50)
    score_hot = metrics_recall(hot_preds, val_df, topk=50)
    print(f"Hot Recall@50: {score_hot:.4f}")
    
    # 3. W2V
    print("\n--- Running Word2Vec Recall ---")
    try:
        emb_dict, aids, emb_matrix = load_embeddings('../tcdata/articles_emb.csv')
        w2v_preds = w2v_recall(user_item_time, emb_dict, aids, emb_matrix, topk=50)
        score_w2v = metrics_recall(w2v_preds, val_df, topk=50)
        print(f"W2V Recall@50: {score_w2v:.4f}")
    except Exception as e:
        print(f"Skipping W2V (Error: {e})")

if __name__ == "__main__":
    compare_recalls()
