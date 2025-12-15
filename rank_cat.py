
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from log_utils import get_logger
from sklearn.model_selection import GroupKFold
import argparse
from catboost import CatBoostClassifier, Pool

logger = get_logger('rank_cat')

def evaluate_metrics(df, topk=5):
    """
    Calculate HitRate@K for validation
    """
    df = df.sort_values(['user_id', 'pred_score'], ascending=[True, False])
    hits = 0
    cnt = 0
    grouped = df.groupby('user_id')
    
    for user, group in grouped:
        if group['label'].sum() == 0:
            continue
        target_item = group[group['label'] == 1]['article_id'].values[0]
        recs = group['article_id'].head(topk).values
        if target_item in recs:
            hits += 1
        cnt += 1
    return hits / cnt if cnt > 0 else 0

def train_cat(feature_pkl, mode='valid', n_splits=5):
    logger.info(f"Loading features from {feature_pkl}...")
    df = pd.read_pickle(feature_pkl)
    
    dummy_cols = ['user_id', 'article_id', 'label', 'pred_score', 'dt', 'last_click_ts', 'created_at_ts']
    feature_cols = [c for c in df.columns if c not in dummy_cols]
    
    logger.info(f"Features: {feature_cols}")
    
    # CatBoost Params
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'CPU',
        'learning_rate': 0.1,
        'iterations': 1000,
        'depth': 6,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'thread_count': -1
    }
    
    gkf = GroupKFold(n_splits=n_splits)
    
    if mode == 'valid':
        oof_preds = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], groups=df['user_id'])):
            logger.info(f"Fold {fold+1}/{n_splits}")
            
            x_train = df.iloc[train_idx][feature_cols]
            y_train = df.iloc[train_idx]['label']
            x_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx]['label']
            
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
            
            oof_preds[val_idx] = model.predict_proba(x_val)[:, 1]
            
        df['pred_score'] = oof_preds
        hr = evaluate_metrics(df, topk=5)
        logger.info(f"Offline HitRate@5: {hr:.4f}")
        
    else:
        # Online Mode: Train on Offline features, Predict on Online features
        logger.info("Loading Training Data (Offline)...")
        train_pkl = '../user_data/data/offline/feature.pkl'
        if not os.path.exists(train_pkl):
             logger.error("Offline features needed for training! Run offline mode first.")
             exit(1)
        
        train_df = pd.read_pickle(train_pkl)
        test_df = df
        
        test_preds = np.zeros(len(test_df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, train_df['label'], groups=train_df['user_id'])):
            logger.info(f"Fold {fold+1}/{n_splits} (Training on offline data)...")
            
            x_train = train_df.iloc[train_idx][feature_cols]
            y_train = train_df.iloc[train_idx]['label']
            x_val = train_df.iloc[val_idx][feature_cols]
            y_val = train_df.iloc[val_idx]['label']
            
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
            
            test_preds += model.predict_proba(test_df[feature_cols])[:, 1] / n_splits
            
        test_df['pred_score'] = test_preds
        
        # Save Predictions for Ensemble
        res_df = test_df[['user_id', 'article_id', 'pred_score']]
        save_path = '../user_data/data/online/cat_preds.pkl'
        res_df.to_pickle(save_path)
        logger.info(f"Saved CatBoost predictions to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    
    BASE_DIR = f'../user_data/data/offline' if args.mode == 'valid' else f'../user_data/data/online'
    FEAT_PKL = os.path.join(BASE_DIR, 'feature.pkl')
    
    train_cat(FEAT_PKL, mode=args.mode)
