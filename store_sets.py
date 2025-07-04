import argparse
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
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
        "seeds": {"loader": fetch_uci_dataset, "args": {"data_id": 70}},
        "yeast": {"loader": fetch_uci_dataset, "args": {"data_id": 73}}
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
    split: bool = False,
    min_class_size: int = 10,
) -> None:
    """
    Preprocess and save dataset to Parquet (optional train/test splits).
    Applies functional checks to drop small classes and invalid samples.
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
            # For categorical columns, add "<missing>" to categories first
            X[col] = X[col].cat.add_categories(["<missing>"])
        X[col] = X[col].fillna("<missing>")
    for col in X.select_dtypes(include=["number"]):
        X[col] = X[col].fillna(X[col].median())

    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    # Ensure all features numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop samples with any invalid (NaN or infinite) features
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    mask_valid = X.notna().all(axis=1)
    dropped_feat = len(X) - mask_valid.sum()
    if dropped_feat > 0:
        logging.warning("Dropped %d samples with invalid features in %s", dropped_feat, name)
    X, y = X.loc[mask_valid], y.loc[mask_valid]

    # Encode target
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), name="target", index=X.index) # pyright: ignore[reportArgumentType, reportCallIssue]

    # Combine features + target
    df = pd.concat([X, y_enc], axis=1)

    # Save to Parquet
    if split:
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            df, stratify=y_enc, test_size=0.2, random_state=42
        )
        for part, df_part in [("train", train), ("test", test)]:
            path = output_dir / f"{name}_{part}.parquet"
            df_part.to_parquet(path, index=False)
            logging.info("Saved %s split for %s: %s", part, name, path)
    else:
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logging.info("Saved dataset %s: %s", name, path)


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
        "--workers", type=int, default=1, help="Number of parallel workers."
    )
    parser.add_argument(
        "--split", action="store_true", help="Also create train/test splits."
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
    tasks = []

    # Parallel execution: fetch data
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for name, cfg in loaders.items():
            loader, loader_args = cfg["loader"], cfg["args"]
            tasks.append(
                executor.submit(lambda n, fn, kw: (n, fn(**kw)), name, loader, {**loader_args})
            )

        results = []
        for future in as_completed(tasks):
            try:
                name, (X, y) = future.result()
                results.append((name, X, y))
            except Exception:
                logging.exception("Failed to fetch dataset.")

    # Process and save
    for name, X, y in results:
        try:
            process_dataset(
                name, X, y, output_dir,
                split=args.split,
                min_class_size=args.min_class_size,
            )
        except Exception:
            logging.exception("Failed to process dataset %s.", name)

    logging.info("All datasets processed.")


if __name__ == "__main__":
    main()
