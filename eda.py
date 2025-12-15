import pandas as pd
import os

DATA_DIR = '../tcdata'

def inspect_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"--- Analyzing {filename} ---")
    try:
        df = pd.read_csv(path, nrows=100000) # Read sample to be fast
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(df.head())
        print(df.info())
        print(df.describe())
        
        if 'click_timestamp' in df.columns:
            print(f"Time range: {df['click_timestamp'].min()} to {df['click_timestamp'].max()}")
            
        print("\n")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

def main():
    files = ['train_click_log.csv', 'testA_click_log.csv', 'articles.csv', 'articles_emb.csv', 'sample_submit.csv']
    for f in files:
        inspect_csv(f)

if __name__ == "__main__":
    main()
