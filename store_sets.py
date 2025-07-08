import argparse
import logging
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------

def get_dataset_loaders() -> dict[str, dict[str, Any]]:
    """
    Defines a registry of datasets to fetch and process.

    Returns:
        A dict mapping dataset names to loader functions and arguments.
    """
    return {
        "iris": {"loader": fetch_sklearn_dataset, "args": {"name": "iris"}},
        "wine": {"loader": fetch_sklearn_dataset, "args": {"name": "wine"}},
        "breast_cancer": {"loader": fetch_sklearn_dataset, "args": {"name": "breast_cancer"}},

        "credit_g": {"loader": fetch_openml_dataset, "args": {"data_id": 31}},
        "diabetes_pima": {"loader": fetch_openml_dataset, "args": {"data_id": 37}},
        "phoneme": {"loader": fetch_openml_dataset, "args": {"data_id": 1489}},
        "spambase": {"loader": fetch_openml_dataset, "args": {"data_id": 44}},
        "bank_marketing": {"loader": fetch_openml_dataset, "args": {"data_id": 1461}},
        "titanic": {"loader": fetch_openml_dataset, "args": {"data_id": 40945}},

        "adult": {"loader": fetch_uci_dataset, "args": {"data_id": 2}},
        "mushroom": {"loader": fetch_uci_dataset, "args": {"data_id": 73}},
        "car_evaluation": {"loader": fetch_uci_dataset, "args": {"data_id": 19}},
        "heart_disease": {"loader": fetch_uci_dataset, "args": {"data_id": 45}},
        "ionosphere": {"loader": fetch_uci_dataset, "args": {"data_id": 52}},
        "banknote_authentication": {"loader": fetch_uci_dataset, "args": {"data_id": 267}},
        "monk": {"loader": fetch_uci_dataset, "args": {"data_id": 70}},
        "yeast": {"loader": fetch_uci_dataset, "args": {"data_id": 110}}
    }


# ----------------------------------------------------------------------------
# Fetch functions
# ----------------------------------------------------------------------------

def fetch_sklearn_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    logging.info("Fetching sklearn dataset: %s", name)
    if name == "iris":
        data = load_iris(as_frame=True)
    elif name == "wine":
        data = load_wine(as_frame=True)
    elif name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
    else:
        raise ValueError(f"Unknown sklearn dataset: {name}")
    return data.data, data.target # pyright: ignore[reportAttributeAccessIssue]


def fetch_openml_dataset(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    logging.info("Fetching OpenML ID: %s", data_id)
    ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    return ds.data, ds.target


def fetch_uci_dataset(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    logging.info("Fetching UCI repository ID: %s", data_id)
    repo = fetch_ucirepo(id=data_id)
    X = repo.data.features
    y = repo.data.targets
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y


# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------

def process_dataset(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    min_class_size: int = 10,
    max_cardinality: int = 10,
) -> dict[str, Any] | None:
    """
    Preprocess and save dataset to Parquet.
    Applies functional checks to drop small classes and invalid samples.

    Returns:
        A dictionary with processing stats (name, samples, features) or None if skipped.
    """
    logging.info("Processing dataset: %s", name)

    # Drop samples with missing target
    mask_y = y.notna()
    X, y = X.loc[mask_y], y.loc[mask_y]

    # Drop classes with fewer than min_class_size samples
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= min_class_size].index
    mask_cls = y.isin(valid_classes)
    dropped_cls = len(y) - mask_cls.sum()
    if dropped_cls > 0:
        logging.warning("Dropped %d samples from small classes in %s", dropped_cls, name)
    X, y = X.loc[mask_cls], y.loc[mask_cls]

    # Fill missing values
    for col in X.select_dtypes(include=["object", "category"]):
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.add_categories(["<missing>"])
        X[col] = X[col].fillna("<missing>")
    for col in X.select_dtypes(include=["number"]):
        X[col] = X[col].fillna(X[col].median())

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    high_cardinality_cols = [
        col for col in cat_cols if X[col].nunique() > max_cardinality
    ]
    if high_cardinality_cols:
        logging.warning(
            "In %s, dropping high-cardinality columns: %s", name, high_cardinality_cols
        )
        X = X.drop(columns=high_cardinality_cols)

    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    # Ensure all features numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Final Validation and Alignment
    if len(X) != len(y):
        logging.warning(
            "Mismatched lengths for X (%d) and y (%d) in %s. Aligning by index.",
            len(X), len(y), name
        )
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    mask_x_valid = X.notna().all(axis=1)
    mask_y_valid = y.notna()
    final_mask = mask_x_valid & mask_y_valid

    dropped_count = len(X) - final_mask.sum()
    if dropped_count > 0:
        logging.warning(
            "Dropped %d samples with invalid final values in %s.", dropped_count, name
        )
        X = X.loc[final_mask]
        y = y.loc[final_mask]

    # Encode target
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), name="target", index=X.index) # type: ignore[reportArgumentType]

    # Combine features + target
    df = pd.concat([X, y_enc], axis=1)
    
    if df.empty:
        logging.error("Dataset %s is empty after processing. Skipping.", name)
        return None

    # Save to Parquet
    path = output_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logging.info("Saved dataset %s: %s", name, path)

    # Return statistics for the summary file
    return {
        "dataset": name,
        "samples": df.shape[0],
        "features": X.shape[1], # Number of feature columns
    }

def fetch_and_process_wrapper(name, cfg, output_dir, min_class_size):
    """Fetches and processes a single dataset, handling exceptions."""
    try:
        logging.info("Starting task for dataset: %s", name)
        # 1. Fetch
        loader, loader_args = cfg["loader"], cfg["args"]
        X, y = loader(**loader_args)
        
        # 2. Process
        summary = process_dataset(
            name, X, y, output_dir, min_class_size=min_class_size
        )
        return summary
    except Exception:
        logging.exception("Failed to fetch and process dataset: %s", name)
        return None

# ----------------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch, process, and store classification datasets with quality checks."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory for Parquet files."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers."
    )
    parser.add_argument(
        "--min-class-size", type=int, default=10,
        help="Minimum samples per class (drop smaller classes)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=level
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", output_dir)

    loaders = get_dataset_loaders()
    dataset_summaries = []

    # Parallel execution: fetch data
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_name = {
            executor.submit(
                fetch_and_process_wrapper,
                name,
                cfg,
                output_dir,
                args.min_class_size,
            ): name
            for name, cfg in loaders.items()
        }

        for future in as_completed(future_to_name):
            summary = future.result()
            if summary:
                dataset_summaries.append(summary)

    # Save the summary file
    if dataset_summaries:
        logging.info("Saving dataset summaries...")
        summary_df = pd.DataFrame(dataset_summaries)
        summary_df.sort_values("dataset", inplace=True)
        summary_path = output_dir / "datasets_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info("Dataset summary saved to: %s", summary_path)
    else:
        logging.warning("No datasets were successfully processed; summary file not created.")

    logging.info("All datasets processed.")


if __name__ == "__main__":
    main()