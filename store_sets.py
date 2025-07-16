import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import cast

import pandas as pd
from openml import datasets
from sklearn.datasets import (
    fetch_covtype,
    fetch_kddcup99,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo


def preprocess_dataset(df: pd.DataFrame, target_name: str):
    """
    Applies a series of preprocessing steps to a given dataset.

    Args:
        df (pd.DataFrame): The raw dataset.
        target_name (str): The name of the target column.

    Returns:
        tuple: A tuple containing the processed DataFrame and a status message.
               Returns (None, message) if preprocessing fails.
    """
    df_processed = df.copy()

    # 1. Drop samples with missing target values
    df_processed.dropna(subset=[target_name], inplace=True)
    if df_processed.empty:
        return None, "Dropped all samples due to missing targets."

    # 2. Drop samples with any missing feature values
    # We do this after dropping target NaNs to keep as much data as possible
    df_processed.dropna(inplace=True)
    if df_processed.empty:
        return None, "Dropped all samples due to missing features."

    # 3. Encode target to numerical values (before filtering)
    try:
        le_target = LabelEncoder()
        df_processed[target_name] = le_target.fit_transform(df_processed[target_name])
    except Exception as e:
        return None, f"Could not encode target column: {e}"

    # 4. Drop classes with fewer than three samples
    class_counts = df_processed[target_name].value_counts()
    valid_classes = class_counts[class_counts >= 3].index
    df_processed = df_processed[df_processed[target_name].isin(valid_classes)]

    if df_processed.empty:
        return None, "Dropped all samples after filtering small classes."

    # Re-encode target to be sequential (0, 1, 2...) after dropping classes
    df_processed[target_name] = LabelEncoder().fit_transform(df_processed[target_name])

    num_classes = df_processed[target_name].nunique()
    if num_classes < 2:
        return None, f"Dataset has less than 2 classes ({num_classes}) after filtering."

    features_df = df_processed.drop(columns=[target_name])
    target_series = df_processed[target_name]

    # 5. Identify categorical features (of type 'object' or 'category')
    categorical_features = features_df.select_dtypes(
        include=["object", "category"]
    ).columns

    # 6. Drop categorical features with cardinality > number of classes
    for col in categorical_features:
        if features_df[col].nunique() > num_classes:
            features_df.drop(columns=[col], inplace=True)

    # 7. Encode remaining categorical features
    remaining_cat_features = features_df.select_dtypes(
        include=["object", "category"]
    ).columns
    for col in remaining_cat_features:
        le = LabelEncoder()
        features_df[col] = le.fit_transform(features_df[col])

    # 8. Reconstruct the final DataFrame
    final_df = features_df.copy()
    final_df["target"] = target_series

    return final_df, "Success"


def fetch_process_and_save(task: dict):
    """
    Worker function that handles the full pipeline for a single dataset.
    This function is designed to be run in a separate process.

    Args:
        task (dict): A dictionary containing task details.
                     {'id': dataset_id, 'source': 'sklearn'|'openml'|'uci', 'output_dir': str}

    Returns:
        dict: A summary dictionary for this dataset.
    """
    ds_id = task["id"]
    source = task["source"]
    output_dir = task["output_dir"]

    df, target_name, dataset_name, sanitized_name = None, None, None, None

    # --- 1. Fetching ---
    try:
        if source == "sklearn":
            sanitized_name = f"sklearn_{ds_id}"
            loader = {
                "iris": load_iris,
                "wine": load_wine,
                "breast_cancer": load_breast_cancer,
                "digits": load_digits,
                "covtype": fetch_covtype,
                "kddcup99": fetch_kddcup99,
            }[ds_id]
            bunch = loader()
            df = pd.DataFrame(
                data=bunch.data,
                columns=[f"feature_{i}" for i in range(bunch.data.shape[1])],
            )
            df["target"] = bunch.target
            target_name = "target"

        elif source == "openml":
            dataset = datasets.get_dataset(
                ds_id,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )
            dataset_name = dataset.name
            sanitized_name = f"openml_{dataset_name.lower().replace(' ', '-').replace('_', '-').replace('(', '').replace(')', '')}"
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            target_name = dataset.default_target_attribute

        elif source == "uci":
            dataset = fetch_ucirepo(id=ds_id)
            dataset_name = dataset.metadata.get(
                "slug", dataset.metadata.get("name", str(ds_id))
            )
            sanitized_name = f"uci_{dataset_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}"
            df = dataset.data.features.copy()
            df["target"] = dataset.data.targets.iloc[:, 0]
            target_name = "target"

    except Exception as e:
        failed_name = f"{source}_{ds_id}"
        return {
            "dataset_name": failed_name,
            "source": source,
            "n_samples": 0,
            "n_features": 0,
            "n_classes": 0,
            "status": f"Failed: Could not fetch - {e}",
            "saved_file": "N/A",
        }

    # --- 2. Preprocessing ---
    if df is None:
        return {
            "dataset_name": sanitized_name,
            "source": source,
            "n_samples": 0,
            "n_features": 0,
            "n_classes": 0,
            "status": "Failed: Dataframe was empty after fetch.",
            "saved_file": "N/A",
        }

    df = cast(pd.DataFrame, df)  # Ensure df is a DataFrame
    if target_name is None:
        target_name = "target"  # Default target name if not set

    processed_df, status = preprocess_dataset(df, target_name)

    # --- 3. Saving and Reporting ---
    if processed_df is not None:
        output_path = os.path.join(output_dir, f"{sanitized_name}.csv")
        processed_df.to_csv(output_path, index=False)

        n_samples, n_features = processed_df.shape
        n_features -= 1  # Exclude target column
        n_classes = processed_df["target"].nunique()

        return {
            "dataset_name": sanitized_name,
            "source": source,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "status": "Success",
            "saved_file": f"{sanitized_name}.csv",
        }
    else:
        return {
            "dataset_name": sanitized_name,
            "source": source,
            "n_samples": 0,
            "n_features": 0,
            "n_classes": 0,
            "status": f"Failed: {status}",
            "saved_file": "N/A",
        }


def main():
    """
    Main function to orchestrate dataset fetching, processing, and saving in parallel.
    """
    parser = argparse.ArgumentParser(
        description="Download and preprocess classification datasets in parallel.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="The directory where datasets and summary will be saved.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Define all datasets to be processed ---
    tasks = []

    sklearn_datasets = [
        "iris",
        "wine",
        "breast_cancer",
        "digits",
        "covtype",
        "kddcup99",
    ]
    openml_ids = [
        44,
        188,
        1468,
        1461,
        451,
        151,
        40668,
        1489,
        458,
        1464,
        40966,
        41146,
        40996,
        40978,
        41138,
        1462,
    ]
    uci_ids = [22, 2, 109, 850, 45, 159, 145, 17, 28, 30, 12, 1, 42, 563, 174, 50, 144]

    for name in sklearn_datasets:
        tasks.append({"id": name, "source": "sklearn", "output_dir": args.output_dir})
    for ds_id in openml_ids:
        tasks.append({"id": ds_id, "source": "openml", "output_dir": args.output_dir})
    for ds_id in uci_ids:
        tasks.append({"id": ds_id, "source": "uci", "output_dir": args.output_dir})

    summaries = []

    # --- Execute tasks in parallel ---
    print(f"Starting parallel processing for {len(tasks)} datasets...")

    with ProcessPoolExecutor() as executor:
        # Create a dictionary to map futures to their tasks for better error reporting
        future_to_task = {
            executor.submit(fetch_process_and_save, task): task for task in tasks
        }

        # Use tqdm for a progress bar as futures complete
        for future in tqdm(
            as_completed(future_to_task), total=len(tasks), desc="Processing datasets"
        ):
            try:
                summary = future.result()
                summaries.append(summary)
            except Exception as e:
                task = future_to_task[future]
                failed_name = f"{task['source']}_{task['id']}"
                print(f"A task generated an unexpected error: {failed_name} -> {e}")
                summaries.append(
                    {
                        "dataset_name": failed_name,
                        "source": task["source"],
                        "n_samples": 0,
                        "n_features": 0,
                        "n_classes": 0,
                        "status": f"Failed: Critical error - {e}",
                        "saved_file": "N/A",
                    }
                )

    # --- Save final summary report ---
    summary_df = (
        pd.DataFrame(summaries)
        .sort_values(by=["source", "dataset_name"])
        .reset_index(drop=True)
    )
    summary_path = os.path.join(args.output_dir, "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n--- All tasks complete. Summary report saved to {summary_path} ---")


if __name__ == "__main__":
    main()
