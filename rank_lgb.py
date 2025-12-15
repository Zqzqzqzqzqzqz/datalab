import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from tqdm import tqdm
from log_utils import get_logger, log_evaluation_callback
from sklearn.model_selection import GroupKFold
import argparse

logger = get_logger('rank_lgb')

def evaluate_metrics(df, topk=5):
    """
    Calculate HitRate@K for validation
    df: must contain 'user_id', 'article_id', 'label', 'pred_score'
    Must perform ranking per user
    """
    # Sort
    df = df.sort_values(['user_id', 'pred_score'], ascending=[True, False])
    
    # Take top K
    # Group by User
    hits = 0
    cnt = 0
    
    # Optimized check
    # Get ground truth map
    # Val sets have label=1 for target. All others 0.
    
    # We want to check if the item with label=1 is in Top K
    
    grouped = df.groupby('user_id')
    
    for user, group in grouped:
        # Check if there is any positive label for this user in the group
        # (Filtered users have at least 1, but we might have others)
        if group['label'].sum() == 0:
            continue
            
        # Get item with label=1
        target_item = group[group['label'] == 1]['article_id'].values[0]
        
        # Get Top K recommendations
        recs = group['article_id'].head(topk).values
        
        if target_item in recs:
            hits += 1
        cnt += 1
        
    return hits / cnt if cnt > 0 else 0

def train_lgb(feature_pkl, mode='valid', n_splits=5):
    logger.info(f"Loading features from {feature_pkl}...")
    df = pd.read_pickle(feature_pkl)
    
    # Features
    dummy_cols = ['user_id', 'article_id', 'label', 'pred_score', 'dt', 'last_click_ts', 'created_at_ts']
    feature_cols = [c for c in df.columns if c not in dummy_cols]
    
    logger.info(f"Features: {feature_cols}")
    
    if mode == 'valid':
        # Single Fold or just Train/Val split? 
        # The prompt says KFold. Let's do KFold CV and average score.
        pass
        
    # Model Params
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # KFold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Store predictions
    # If mode=test, we predict on the test set for each fold and average.
    # If mode=valid, we predict on the 'val' part of each fold (OOF) to compute offline metric.
    
    # Wait, 'valid' mode is offline validation. We strictly use offline data.
    # 'test' mode uses online data.
    # In offline mode, we usually do CV to verify stability.
    
    if mode == 'valid':
        oof_preds = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], groups=df['user_id'])):
            logger.info(f"Fold {fold+1}/{n_splits}")
            
            x_train = df.iloc[train_idx][feature_cols]
            y_train = df.iloc[train_idx]['label']
            x_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx]['label']
            
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
            
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=1000,
                            valid_sets=[lgb_train, lgb_val],
                            callbacks=[lgb.early_stopping(stopping_rounds=50), log_evaluation_callback(period=100, logger=logger)])
                            
            oof_preds[val_idx] = gbm.predict(x_val, num_iteration=gbm.best_iteration)
            
        # Metric
        df['pred_score'] = oof_preds
        hr = evaluate_metrics(df, topk=5)
        logger.info(f"Offline HitRate@5: {hr:.4f}")
        
    else:
        # Online Mode: Train 5 folds on ALL data? 
        # Usually we train on full data 5 times with different seeds, or just 1 time.
        # But 'Stacking' pattern suggests treating the whole 'online/feature.pkl' as Test Set,
        # and we need a Training Set.
        # Wait, `make_test_data` puts TRAIN+TEST in click.pkl but the 'Feature Table' for Online is for Test Users.
        # Where is the Training Data for Online prediction?
        # WE NEED TO LOAD OFFLINE DATA TO TRAIN THE MODEL.
        
        # ONLINE FLOW:
        # 1. Load OFFLINE features (Labeled) -> Train
        # 2. Load ONLINE features (Unlabeled) -> Test
        
        logger.info("Loading Training Data (Offline)...")
        train_pkl = '../user_data/data/offline/feature.pkl'
        if not os.path.exists(train_pkl):
             logger.error("Offline features needed for training! Run offline mode first.")
             exit(1)
        
        train_df = pd.read_pickle(train_pkl)
        test_df = df # The passed arg is online features
        
        test_preds = np.zeros(len(test_df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, train_df['label'], groups=train_df['user_id'])):
            logger.info(f"Fold {fold+1}/{n_splits} (Training on offline data)...")
            
            x_train = train_df.iloc[train_idx][feature_cols]
            y_train = train_df.iloc[train_idx]['label']
            x_val = train_df.iloc[val_idx][feature_cols] # Still use validation to stop
            y_val = train_df.iloc[val_idx]['label']
            
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
            
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=1000,
                            valid_sets=[lgb_train, lgb_val],
                            callbacks=[lgb.early_stopping(stopping_rounds=50), log_evaluation_callback(period=100, logger=logger)])
            
            # Predict on TEST
            test_preds += gbm.predict(test_df[feature_cols], num_iteration=gbm.best_iteration) / n_splits
            
        test_df['pred_score'] = test_preds
        
        # Save Predictions for Ensemble
        res_df = test_df[['user_id', 'article_id', 'pred_score']]
        save_path = '../user_data/data/online/lgb_preds.pkl'
        res_df.to_pickle(save_path)
        logger.info(f"Saved LightGBM predictions to {save_path}")
        
        # Generate Submission
        gen_submission(test_df)
        
def gen_submission(df):
    logger.info("Generating Submission...")
    sub_path = '../prediction_result/result.csv'
    
    df = df.sort_values(['user_id', 'pred_score'], ascending=[True, False])
    grouped = df.groupby('user_id')
    
    res = []
    processed = set()
    
    # Global fallback
    # Should perform popularity fallback if < 5
    # For now assume recall is enough (100)
    
    for user, group in tqdm(grouped):
        processed.add(user)
        items = group['article_id'].head(5).astype(str).tolist()
        
        # Fallback logic here if needed (omitted for brevity)
        if len(items) < 5:
            items = items + ['0'] * (5 - len(items)) # Dummy fallback
            
        res.append([user] + items)
        
    sub = pd.DataFrame(res, columns=['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5'])
    sub.to_csv(sub_path, index=False)
    logger.info(f"Saved submission to {sub_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    BASE_DIR = f'../user_data/data/offline' if args.mode == 'valid' else f'../user_data/data/online'
    FEAT_PKL = os.path.join(BASE_DIR, 'feature.pkl')
    
    train_lgb(FEAT_PKL, mode=args.mode)
