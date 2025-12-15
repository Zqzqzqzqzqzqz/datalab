import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from log_utils import get_logger, log_evaluation_callback
from recall_itemcf import ItemCF
from rank_feature import make_rank_samples, get_user_features, get_item_features, merge_features
from rank_lgb import train_lgb_ranker

logger = get_logger('train_pipeline')

# Config
TRAIN_FILE = '../tcdata/train_click_log.csv'
TEST_FILE = '../tcdata/testA_click_log.csv'
SUBMISSION_FILE = '../prediction_result/result.csv'

def run_pipeline():
    logger.info("Starting End-to-End Pipeline...")
    
    # --------------------------------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------------------------------
    logger.info("Loading Data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # --------------------------------------------------------------------------------
    # 2. Preprocess & Split
    # --------------------------------------------------------------------------------
    # Validation Strategy: Last click of Train Users
    # Note: Test set users are disjoint from Train set users in this competition, 
    # but we use their history (in TestA file) to predict their NEXT click.
    
    train_df = train_df.sort_values(['user_id', 'click_timestamp'])
    
    # Split Train into:
    # - Train_History: All clicks except last one (used to model user interest)
    # - Train_Label: Last click (Ground Truth for Ranker)
    val_label_df = train_df.groupby('user_id').tail(1)
    train_history_df = train_df.drop(val_label_df.index)
    
    # For Feature Engineering: calculate statistics on "Full Train History"
    # Ideally should correspond to point-in-time, but here we approximate with train_history_df
    
    # Concatenate TrainHistory + TestHistory for ItemCF Co-occurrence Matrix
    # (We can use Test history to understand item similarities better)
    all_history_df = pd.concat([train_history_df, test_df]) 
    
    # --------------------------------------------------------------------------------
    # 3. Recall Phase
    # --------------------------------------------------------------------------------
    logger.info("Running ItemCF Fit...")
    itemcf = ItemCF()
    itemcf.fit(all_history_df)
    
    # 3a. Recall for Validation (Train Ranker)
    logger.info("Generating Candidates for Training (Validation Set)...")
    train_user_item_time = itemcf._get_user_item_time(train_history_df)
    train_candidates = {}
    
    # We only generate for users present in val_label_df
    target_train_users = val_label_df['user_id'].unique()
    
    for user in tqdm(target_train_users):
        hist = train_user_item_time.get(user, [])
        if not hist: continue
        # Recall Top 50
        candidates = itemcf.recommend(hist, top_k=50)
        train_candidates[user] = candidates
        
    logger.info(f"Generated train candidates for {len(train_candidates)} users.")

    # 3b. Recall for Test (Submission)
    logger.info("Generating Candidates for Test Set...")
    test_user_item_time = itemcf._get_user_item_time(test_df)
    test_candidates = {}
    test_users = test_df['user_id'].unique()
    
    for user in tqdm(test_users):
        hist = test_user_item_time.get(user, [])
        # Recall Top 50
        candidates = itemcf.recommend(hist, top_k=50)
        test_candidates[user] = candidates

    logger.info(f"Generated test candidates for {len(test_candidates)} users.")
    
    # --------------------------------------------------------------------------------
    # 4. Feature Engineering
    # --------------------------------------------------------------------------------
    logger.info("Building Features...")
    
    # 4a. Build Train Samples
    train_samples_df = make_rank_samples(train_candidates, val_label_df, mode='train')
    
    # 4b. Build Test Samples
    test_samples_df = make_rank_samples(test_candidates, mode='test')
    
    # Features calc (Use train_history_df + test_df for statistics)
    # Simple Features: User Click Count, Item Popularity
    # Note: Be careful not to leak label info.
    
    # Stats from history
    u_feat = get_user_features(all_history_df)
    i_feat = get_item_features(all_history_df)
    
    logger.info("Merging Features (Train)...")
    train_data = merge_features(train_samples_df, u_feat, i_feat)
    
    logger.info("Merging Features (Test)...")
    test_data = merge_features(test_samples_df, u_feat, i_feat)
    
    feature_cols = [c for c in train_data.columns if c not in ['user_id', 'article_id', 'label', 'click_timestamp', 'created_at_ts']]
    logger.info(f"Features used: {feature_cols}")
    
    # --------------------------------------------------------------------------------
    # 5. Ranking Model Training
    # --------------------------------------------------------------------------------
    logger.info("Training Ranker...")
    
    # Split train_data into Train/Val for LGB Early Stopping
    # We simply split by user or random
    mask = np.random.rand(len(train_data)) < 0.8
    train_split = train_data[mask]
    val_split = train_data[~mask]
    
    callbacks_list = [
        lgb.early_stopping(stopping_rounds=50),
        log_evaluation_callback(period=50, logger=logger)
    ]
    
    # Need to manually call lgb.train here to inject arguments correctly if using rank_lgb wrapper or just inline it
    # rank_lgb.py: train_lgb_ranker has fixed params. Let's reuse it but we need to ensure callbacks are passed if we modified it?
    # Actually I modified rank_lgb to include callbacks. 
    
    gbm = train_lgb_ranker(train_split, val_split, feature_cols)
    
    # --------------------------------------------------------------------------------
    # 6. Inference & Submission
    # --------------------------------------------------------------------------------
    logger.info("Predicting for Test Set...")
    preds = gbm.predict(test_data[feature_cols], num_iteration=gbm.best_iteration)
    test_data['pred_score'] = preds
    
    # Select Top 5 per User
    logger.info("Generating Submission File...")
    
    # Group by user, sort by score, take top 5
    submit_data = []
    
    # Use fallback popular items if < 5
    # Pre-compute popular fallback
    pop_fallback = sorted(itemcf.item_popular.items(), key=lambda x:x[1], reverse=True)[:10]
    pop_fallback_ids = [str(x[0]) for x in pop_fallback]
    
    # Optimize: Process using pandas groupby is faster but loop is clearer for fallback logic
    # Let's do a quick loop
    
    # Sort test_data by user, score
    test_data = test_data.sort_values(['user_id', 'pred_score'], ascending=[True, False])
    
    # Group
    grouped = test_data.groupby('user_id')
    
    # We need to ensure ALL test users are in submission
    
    processed_users = set()
    
    for user, group in tqdm(grouped):
        processed_users.add(user)
        top_items = group['article_id'].head(5).astype(str).tolist()
        
        # Fallback
        if len(top_items) < 5:
            hist_items = set([str(x[0]) for x in test_user_item_time.get(user, [])])
            for pop in pop_fallback_ids:
                if pop not in top_items and pop not in hist_items:
                    top_items.append(pop)
                if len(top_items) >= 5:
                    break
                    
        submit_data.append([user] + top_items[:5])
        
    # Check for missing users (cold start users who got 0 recall candidates?)
    missing_users = set(test_users) - processed_users
    if missing_users:
        logger.warning(f"Found {len(missing_users)} users with no recall candidates. Filling with popularity.")
        for user in missing_users:
            submit_data.append([user] + pop_fallback_ids[:5])
            
    sub_df = pd.DataFrame(submit_data, columns=['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5'])
    sub_df.to_csv(SUBMISSION_FILE, index=False)
    
    logger.info(f"Submission saved to {SUBMISSION_FILE}. Shape: {sub_df.shape}")
    logger.info("Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_pipeline()
