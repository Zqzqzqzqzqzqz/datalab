
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from log_utils import get_logger

logger = get_logger('ensemble')

def gen_submission(df, sub_path):
    logger.info("Generating Submission...")
    
    df = df.sort_values(['user_id', 'pred_score'], ascending=[True, False])
    grouped = df.groupby('user_id')
    
    res = []
    
    for user, group in tqdm(grouped):
        items = group['article_id'].head(5).astype(str).tolist()
        
        if len(items) < 5:
            items = items + ['0'] * (5 - len(items))
            
        res.append([user] + items)
        
    sub = pd.DataFrame(res, columns=['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5'])
    sub.to_csv(sub_path, index=False)
    logger.info(f"Saved submission to {sub_path}")

def ensemble():
    BASE_DIR = '../user_data/data/online'
    LGB_PATH = os.path.join(BASE_DIR, 'lgb_preds.pkl')
    CAT_PATH = os.path.join(BASE_DIR, 'cat_preds.pkl')
    XGB_PATH = os.path.join(BASE_DIR, 'xgb_preds.pkl')
    OUT_PATH = '../prediction_result/result.csv'
    
    preds = []
    
    # 1. LightGBM (Primary)
    if os.path.exists(LGB_PATH):
        logger.info(f"Loading LightGBM from {LGB_PATH}")
        lgb_df = pd.read_pickle(LGB_PATH)
        # Rename score
        lgb_df = lgb_df.rename(columns={'pred_score': 'lgb_score'})
        preds.append(lgb_df)
    else:
        logger.warning("LightGBM predictions not found!")
        
    # 2. CatBoost
    if os.path.exists(CAT_PATH):
        logger.info(f"Loading CatBoost from {CAT_PATH}")
        cat_df = pd.read_pickle(CAT_PATH)
        cat_df = cat_df.rename(columns={'pred_score': 'cat_score'})
        preds.append(cat_df)
    else:
        logger.warning("CatBoost predictions not found!")
        
    # 3. XGBoost
    if os.path.exists(XGB_PATH):
        logger.info(f"Loading XGBoost from {XGB_PATH}")
        xgb_df = pd.read_pickle(XGB_PATH)
        xgb_df = xgb_df.rename(columns={'pred_score': 'xgb_score'})
        preds.append(xgb_df)
    else:
        logger.warning("XGBoost predictions not found!")
        
    if not preds:
        logger.error("No predictions found!")
        return
        
    # Merge
    logger.info("Merging predictions...")
    final_df = preds[0][['user_id', 'article_id']]
    
    for df in preds:
        final_df = final_df.merge(df, on=['user_id', 'article_id'], how='left')
        
    # Weighted Avg
    # Check what columns we have
    cols = final_df.columns
    
    final_df['pred_score'] = 0
    weights = {'lgb_score': 0.4, 'cat_score': 0.3, 'xgb_score': 0.3}
    
    # Normalize weights if missing models
    active_weights = {k:v for k,v in weights.items() if k in cols}
    total_w = sum(active_weights.values())
    active_weights = {k:v/total_w for k,v in active_weights.items()}
    
    logger.info(f"Weights: {active_weights}")
    
    for col, w in active_weights.items():
        final_df[col] = final_df[col].fillna(0)
        final_df['pred_score'] += final_df[col] * w
        
    gen_submission(final_df, OUT_PATH)

if __name__ == "__main__":
    ensemble()
