import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, dump
from surprise.model_selection import train_test_split
from snapshot_user_histories import snapshot_user_histories

BASE_DIR = Path(__file__).resolve().parent


def train_model(refresh_database: bool = False):
    if refresh_database:
        print("Snapshotting user_histories from PostgreSQL...")
        snapshot_user_histories()

    dataset_path = BASE_DIR / "dataset" / "interactions.csv"
    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    required_columns = {"User-ID", "ISBN", "rating"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("Dataset contains no interactions.")

    reader = Reader(rating_scale=(0, 2))
    data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'rating']], reader)


    print("Splitting data (80/20) and applying random seed 0...")
    trainset, testset = train_test_split(data, test_size=0.20, random_state=0)

    print("Training SVD Model...")
    algo = SVD(random_state=7)
    algo.fit(trainset)

    print("Saving model and test data artifacts...")
    # Save the trained AI model
    dump.dump(str(BASE_DIR / 'library_svd_model.pkl'), algo=algo)
    
    # Save the 20% test dataset for the evaluator script to use later
    with open(BASE_DIR / 'testset.pkl', 'wb') as f:
        pickle.dump(testset, f)
        
    # Save a master list of all unique books 
    with open(BASE_DIR / 'all_books.json', 'w') as f:
        json.dump(df['ISBN'].unique().tolist(), f)

    print("Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-database",
        action="store_true",
        help="Replace interactions.csv with user_histories before training.",
    )
    train_model(refresh_database=parser.parse_args().refresh_database)