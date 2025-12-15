#!/bin/bash

MODE=$1

if [ -z "$MODE" ]; then
    MODE="valid"
fi

echo "Running Pipeline in $MODE mode..."

# 1. Data Processing
python data.py --mode $MODE

# 2. Recall
python recall_itemcf.py --mode $MODE
python recall_binetwork.py --mode $MODE
python recall_w2v.py --mode $MODE

# 3. Merge
python recall.py --mode $MODE

# 4. Features
python rank_feature.py --mode $MODE

# 5. Ranker
python rank_lgb.py --mode $MODE
