import pandas as pd
import numpy as np
import collections
from tqdm import tqdm

def hot_recall(user_item_time_dict, topk=50):
    """
    Basic Hot Recall: Recommend the most popular items globally.
    """
    item_popular = collections.defaultdict(int)
    for user, items in user_item_time_dict.items():
        for item, _ in items:
            item_popular[item] += 1
            
    # Sort popular items
    popular_items = sorted(item_popular.items(), key=lambda x: x[1], reverse=True)[:topk+20] # Fetch a bit more to filter history
    
    recall_res = {}
    print("Generating Hot Recall Candidates...")
    for user, history in tqdm(user_item_time_dict.items()):
        clicked_set = set([x[0] for x in history])
        preds = []
        for item, score in popular_items:
            if item in clicked_set:
                continue
            preds.append((item, score))
            if len(preds) >= topk:
                break
        recall_res[user] = preds
    return recall_res

def hot_recall_improved(user_item_time_dict, topk=50):
    """
    Improved Hot Recall:
    - Recent popularity (e.g., last 3 days)?
    - Or just simple weighted popularity?
    Let's implement a 'Recent Hot' logic if timestamps allow, 
    otherwise default to weighted counts.
    Here we'll implement a simple Time-Decay Popularity.
    """
    item_time_score = collections.defaultdict(float)
    
    # Calculate global max time to normalize
    max_time = 0
    for user, items in user_item_time_dict.items():
        for item, time in items:
            if time > max_time:
                max_time = time
                
    # 1 day in ms approx (assuming timestamp is something standard, usually ms or s)
    # Check eda? 'Time range: 1547466649550 to 1548000459880' -> looks like ms (13 digits)
    # 1 day = 86400 * 1000 = 86400000
    
    print("Calculating Time-Decayed Popularity...")
    for user, items in tqdm(user_item_time_dict.items()):
        for item, time in items:
            # Decay score
            diff = max_time - time
            # Decay factor: 1 / (1 + days_diff)
            days_diff = diff / 86400000.0
            score = 1.0 / (1.0 + days_diff)
            item_time_score[item] += score
            
    popular_items = sorted(item_time_score.items(), key=lambda x: x[1], reverse=True)[:topk+20]
    
    recall_res = {}
    for user, history in tqdm(user_item_time_dict.items()):
        clicked_set = set([x[0] for x in history])
        preds = []
        for item, score in popular_items:
            if item in clicked_set:
                continue
            preds.append((item, score))
            if len(preds) >= topk:
                break
        recall_res[user] = preds
        
    return recall_res
