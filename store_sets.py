"""
Fetches, processes, and stores multiple classification datasets in a normalized format.

This script downloads datasets from various sources (sklearn, OpenML, UCI),
encodes categorical features using one-hot encoding, normalizes the target
labels to be zero-indexed integers, and saves the final dataframes in the
efficient Parquet format.

This preprocessing is done on the entire dataset without splitting, making it
suitable for creating a repository of clean data ready for modeling pipelines.

Usage:
    python fetch_datasets.py ./path/to/save/data

Dependencies:
    - pandas
    - scikit-learn
    - pyarrow
    - openml
    - ucimlrepo
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Any

# Import dataset loaders
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_openml,
)
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

def get_dataset_loaders() -> dict[str, dict[str, Any]]:
    """
    Defines a registry of datasets to be fetched and processed.

    Each entry contains a loader function and its required arguments.

    Returns:
        A dictionary mapping a dataset name to its loader configuration.
    """
    return {
        # scikit-learn datasets (small, classic, good for testing)
        "iris": {"loader": fetch_sklearn_dataset, "args": {"name": "iris"}},
        "wine": {"loader": fetch_sklearn_dataset, "args": {"name": "wine"}},
        "breast_cancer": {
            "loader": fetch_sklearn_dataset,
            "args": {"name": "breast_cancer"},
        },
        # OpenML datasets (IDs chosen for popular classification tasks)
        "credit_g": {"loader": fetch_openml_dataset, "args": {"data_id": 31}},
        "diabetes_pima": {"loader": fetch_openml_dataset, "args": {"data_id": 37}},
        "phoneme": {"loader": fetch_openml_dataset, "args": {"data_id": 1489}},
        "spambase": {"loader": fetch_openml_dataset, "args": {"data_id": 44}},
        "bank_marketing": {"loader": fetch_openml_dataset, "args": {"data_id": 1461}},
        "titanic": {"loader": fetch_openml_dataset, "args": {"data_id": 40945}},
        "adult_openml": {"loader": fetch_openml_dataset, "args": {"data_id": 1590}},
        "mnist_784": {"loader": fetch_openml_dataset, "args": {"data_id": 554}},
        "fashion_mnist": {"loader": fetch_openml_dataset, "args": {"data_id": 40996}},
        # UCI ML Repo datasets (IDs from ucimlrepo)
        "adult": {"loader": fetch_uci_dataset, "args": {"data_id": 2}},
        "mushroom": {"loader": fetch_uci_dataset, "args": {"data_id": 73}},
        "car_evaluation": {"loader": fetch_uci_dataset, "args": {"data_id": 19}},
        "heart_disease": {"loader": fetch_uci_dataset, "args": {"data_id": 45}},
        "ionosphere": {"loader": fetch_uci_dataset, "args": {"data_id": 52}},
        "banknote_authentication": {"loader": fetch_uci_dataset, "args": {"data_id": 21}},
        "seeds": {"loader": fetch_uci_dataset, "args": {"data_id": 70}},
        "statlog_german_credit": {"loader": fetch_uci_dataset, "args": {"data_id": 31}},
        "yeast": {"loader": fetch_uci_dataset, "args": {"data_id": 73}},
        "abalone": {"loader": fetch_uci_dataset, "args": {"data_id": 1}},
    }

def fetch_sklearn_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads a classification dataset from scikit-learn."""
    print(f"  -> Fetching sklearn dataset: {name}")
    if name == "iris":
        data = load_iris(as_frame=True)
    elif name == "wine":
        data = load_wine(as_frame=True)
    elif name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
    else:
        raise ValueError(f"Unknown sklearn dataset name: {name}")
    return data.data, data.target

def fetch_openml_dataset(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Fetches a classification dataset from OpenML by its ID."""
    print(f"  -> Fetching OpenML dataset ID: {data_id}")
    dataset = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    return dataset.data, dataset.target

def fetch_uci_dataset(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Fetches a classification dataset from the UCI ML Repository by its ID."""
    print(f"  -> Fetching UCI dataset ID: {data_id}")
    repo_fetch = fetch_ucirepo(id=data_id)
    X = repo_fetch.data.features
    y = repo_fetch.data.targets
    # UCI repo often returns target as a DataFrame, so we squeeze it to a Series
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    return X, y


def process_and_save(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path
):
    """
    Processes a raw dataset and saves it to a Parquet file.

    Parameters:
        name: The common name for the dataset.
        X: DataFrame of features.
        y: Series of target labels.
        output_dir: The directory where the Parquet file will be saved.
    """
    print(f"  -> Processing '{name}'...")

    # 1. Handle potential missing values by filling with a placeholder or median
    # For simplicity, we fill categoricals with 'missing' and numerics with median.
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].fillna('missing')
    for col in X.select_dtypes(include=['number']).columns:
        X[col] = X[col].fillna(X[col].median())

    # 2. Encode target labels to be 0-indexed integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_df = pd.DataFrame(y_encoded, columns=["target"], index=X.index)

    # 3. Identify and one-hot encode categorical features
    # Select columns with 'object' or 'category' dtype
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    
    if not categorical_cols.empty:
        print(f"    - Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        X_processed = pd.get_dummies(X, columns=categorical_cols, dummy_na=False, dtype=float)
    else:
        print("    - No categorical columns found.")
        X_processed = X.copy()
        
    # Ensure all feature columns are of a type Parquet supports well
    for col in X_processed.columns:
        if pd.api.types.is_bool_dtype(X_processed[col]):
            X_processed[col] = X_processed[col].astype(int)

    # 4. Combine features and target into a single DataFrame
    final_df = pd.concat([X_processed, y_df], axis=1)

    # 5. Save to Parquet file
    output_path = output_dir / f"{name}.parquet"
    final_df.to_parquet(output_path, index=False)
    print(f"  -> Saved processed data to '{output_path}'")
    print("-" * 40)


def main():
    """Main function to orchestrate the fetching and processing of datasets."""
    parser = argparse.ArgumentParser(
        description="Fetch and process classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the processed Parquet files.",
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir.resolve()}")
    print("=" * 40)

    dataset_loaders = get_dataset_loaders()

    for name, config in dataset_loaders.items():
        try:
            print(f"Processing dataset: '{name}'")
            loader_func = config["loader"]
            loader_args = config["args"]

            features, target = loader_func(**loader_args)
            
            if features.empty or target.empty:
                print(f"  -> Skipping '{name}': No data returned.")
                continue

            process_and_save(name, features, target, args.output_dir)

        except Exception as e:
            print(f"\n[ERROR] Failed to process dataset '{name}'.")
            print(f"  Reason: {e}\n")
            print("-" * 40)

    print("All datasets processed.")


if __name__ == "__main__":
    main()
