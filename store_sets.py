import argparse
import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_iris, load_wine, load_digits, load_breast_cancer, load_diabetes
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

def fetch_sklearn_datasets():
    """
    Fetches standard datasets from Scikit-learn.
    
    Returns:
        dict: A dictionary where keys are dataset names and values are tuples 
              containing a DataFrame and the target column name.
    """
    print("Fetching datasets from Scikit-learn...")
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'digits': load_digits,
        'breast_cancer': load_breast_cancer,
        'diabetes': load_diabetes
    }
    
    processed_datasets = {}
    for name, loader in datasets.items():

        print(f"  - Loading sklearn dataset '{name}'...")

        try:
            bunch = loader()
            df = pd.DataFrame(data=bunch.data, columns=[f'feature_{i}' for i in range(bunch.data.shape[1])])
            df['target'] = bunch.target
            processed_datasets[f"sklearn_{name}"] = (df, 'target')
        except Exception as e:
            print(f"  - Failed to load sklearn dataset '{name}': {e}")
            
    return processed_datasets

def fetch_openml_datasets():
    """
    Fetches specified classification datasets from OpenML.
    
    Returns:
        dict: A dictionary of DataFrames and their target column names.
    """
    print("Fetching datasets from OpenML...")
    # A list of interesting classification dataset IDs from OpenML
    dataset_ids = [
        44,    # spambase
        188,   # eucalyptus
        1468,  # cnae-9
        1461,  # bank-marketing
        151,   # irish
        40668, # phoneme
        1489,  # analcatdata_authorship
        42729, # blood-transfusion-service-center
        40966, # sylvine
        40996, # Fashion-MNIST
        40978, # APSFailure
        1464,  # banknote-authentication
    ]
    
    processed_datasets = {}
    for ds_id in dataset_ids:

        print(f"  - Fetching OpenML dataset ID {ds_id}...")

        try:
            dataset = fetch_openml(data_id=ds_id, as_frame=True, parser='auto')
            df = dataset.frame
            target_name = dataset.target_names[0]
            # Ensure target is of a basic type for consistent processing
            df[target_name] = df[target_name].astype(str)
            processed_datasets[f"openml_{ds_id}"] = (df, target_name)
        except Exception as e:
            print(f"  - Failed to fetch OpenML dataset ID {ds_id}: {e}")
            
    return processed_datasets

def fetch_uci_datasets():
    """
    Fetches specified classification datasets from the UCI ML Repository.
    
    Returns:
        dict: A dictionary of DataFrames and their target column names.
    """
    print("Fetching datasets from the UCI Repository...")
    # A list of interesting classification dataset IDs from UCI
    dataset_ids = [
        22,  # Chess (King-Rook vs. King-Pawn)
        2,   # Adult
        109, # Wine
        850, # Heart Disease
        159, # Statlog (Heart)
        17,  # Breast Cancer Wisconsin (Diagnostic)
        28,  # Contraceptive Method Choice
        110, # Yeast
        12,  # Balance Scale
        1,   # Abalone
        42,  # Glass Identification
        563, # Bike-sharing
        174, # Parkinson's Disease
        50,  # Statlog (German Credit)
    ]

    processed_datasets = {}
    for ds_id in dataset_ids:

        print(f"  - Fetching UCI dataset ID {ds_id}...")

        try:
            dataset = fetch_ucirepo(id=ds_id)
            
            # Use the slug if it exists, otherwise, sanitize the dataset name.
            if hasattr(dataset.metadata, 'slug') and dataset.metadata.slug:
                dataset_name = dataset.metadata.slug
            else:
                # Sanitize the name for use in a filename
                dataset_name = dataset.metadata.name.lower().replace(' ', '-').replace('(', '').replace(')', '')
            
            df = dataset.data.features.copy()
            # The target is often in a separate DataFrame
            df['target'] = dataset.data.targets.iloc[:, 0]
            processed_datasets[f"uci_{dataset_name}"] = (df, 'target')
        except Exception as e:
            print(f"  - Failed to fetch UCI dataset ID {ds_id}: {e}")
            
    return processed_datasets

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
    df_processed.dropna(inplace=True)
    if df_processed.empty:
        return None, "Dropped all samples due to missing features."
    
    # 3. Encode target to numerical values (before filtering)
    le_target = LabelEncoder()
    df_processed[target_name] = le_target.fit_transform(df_processed[target_name])

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
    categorical_features = features_df.select_dtypes(include=['object', 'category']).columns

    # 6. Drop categorical features with cardinality > number of classes
    for col in categorical_features:
        if features_df[col].nunique() > num_classes:
            features_df.drop(columns=[col], inplace=True)
            print(f"      - Dropped feature '{col}': cardinality > num_classes")

    # 7. Encode remaining categorical features
    remaining_cat_features = features_df.select_dtypes(include=['object', 'category']).columns
    for col in remaining_cat_features:
        le = LabelEncoder()
        features_df[col] = le.fit_transform(features_df[col])

    # 8. Reconstruct the final DataFrame
    final_df = features_df.copy()
    final_df['target'] = target_series
    
    return final_df, "Success"

def main():
    """
    Main function to orchestrate dataset fetching, processing, and saving.
    """
    parser = argparse.ArgumentParser(
        description="Download and preprocess classification datasets from various sources.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'output_dir', 
        type=str, 
        help="The directory where datasets and summary will be saved."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Consolidate all dataset fetching functions
    all_datasets = {}
    all_datasets.update(fetch_sklearn_datasets())
    all_datasets.update(fetch_openml_datasets())
    all_datasets.update(fetch_uci_datasets())
    
    summaries = []

    print("\n--- Starting Preprocessing ---\n")
    for name, (df, target_name) in all_datasets.items():
        print(f"Processing dataset: '{name}'...")
        
        if target_name not in df.columns:
            print(f"  - SKIPPING: Target column '{target_name}' not found.")
            continue
            
        processed_df, status = preprocess_dataset(df, target_name)
        
        source = "Scikit-learn"
        if name.startswith('openml_'):
            source = "OpenML"
        elif name.startswith('uci_'):
            source = "UCI"

        if processed_df is not None:
            output_path = os.path.join(args.output_dir, f"{name}.csv")
            processed_df.to_csv(output_path, index=False)
            
            n_samples, n_features = processed_df.shape
            n_features -= 1  # Exclude target column
            n_classes = processed_df['target'].nunique()
            
            summaries.append({
                'dataset_name': name,
                'source': source,
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'status': 'Success',
                'saved_file': f"{name}.csv"
            })
            print(f"  - ✅ Success! Shape: ({n_samples}, {n_features+1}). Saved to {output_path}\n")
        else:
            summaries.append({
                'dataset_name': name,
                'source': source,
                'n_samples': 0, 'n_features': 0, 'n_classes': 0,
                'status': f'Failed: {status}',
                'saved_file': "N/A"
            })
            print(f"  - ❌ Failed: {status}\n")

    # Save summary report
    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.output_dir, "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"--- All tasks complete. Summary report saved to {summary_path} ---")

if __name__ == '__main__':
    main()