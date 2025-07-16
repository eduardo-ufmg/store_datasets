import argparse
import glob
import os
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

def validate_dataset(file_path: str):
    """
    Worker function to load a dataset, build a preprocessing and classification
    pipeline, and evaluate its performance.

    Args:
        file_path (str): The path to the dataset CSV file.

    Returns:
        dict: A dictionary containing the validation results.
    """
    dataset_name = os.path.basename(file_path)
    
    try:
        # 1. Load and prepare data
        df = pd.read_csv(file_path)
        
        # Ensure 'target' column exists
        if 'target' not in df.columns:
            raise ValueError("Column 'target' not found in the dataset.")
            
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data before any fitting to prevent data leakage
        # Using stratify is good practice for classification tasks
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y
        )

        # 2. Define the ML pipeline
        # This is the key to preventing data leakage. Each step is fitted
        # only on the training data.
        pipeline = Pipeline([
            # Step 1: Remove features with low variance
            ('variance_filter', VarianceThreshold(threshold=0.1)),
            
            # Step 2: Scale features (essential for PCA and KNN)
            ('scaler', StandardScaler()),
            
            # Step 3: PCA - keeps components explaining 90% of variance
            ('pca', PCA(n_components=0.9)),
            
            # Step 4: The classifier
            ('knn', KNeighborsClassifier(n_neighbors=3))
        ])

        # 3. Train and evaluate
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)
        duration = time.time() - start_time
        
        # Get the number of features selected by PCA
        n_components_pca = pipeline.named_steps['pca'].n_components_

        return {
            "dataset_name": dataset_name,
            "accuracy": f"{accuracy:.4f}",
            "original_features": X.shape[1],
            "pca_components": n_components_pca,
            "duration_seconds": f"{duration:.2f}",
            "status": "Success"
        }

    except Exception as e:
        # Return a summary in case of any failure during processing
        return {
            "dataset_name": dataset_name,
            "accuracy": 0.0,
            "original_features": "N/A",
            "pca_components": "N/A",
            "duration_seconds": "N/A",
            "status": f"Failed: {str(e)}"
        }

def main():
    """
    Main function to find datasets and run validation in parallel.
    """
    parser = argparse.ArgumentParser(
        description="Validate preprocessed datasets using a parallelized KNN pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help="The directory containing the preprocessed dataset CSV files."
    )
    args = parser.parse_args()

    # Find all dataset files, ignoring the summary file
    dataset_files = [
        f for f in glob.glob(os.path.join(args.input_dir, '*.csv'))
        if not os.path.basename(f).startswith('_')
    ]
    
    if not dataset_files:
        print(f"Error: No dataset CSV files found in '{args.input_dir}'.")
        return

    results = []

    # --- Execute validation tasks in parallel ---
    print(f"🚀 Starting validation for {len(dataset_files)} datasets...")
    
    with ProcessPoolExecutor() as executor:
        # Submit all validation tasks to the pool
        future_to_file = {executor.submit(validate_dataset, f): f for f in dataset_files}
        
        # Process results as they complete, with a progress bar
        for future in tqdm(as_completed(future_to_file), total=len(dataset_files), desc="Validating datasets"):
            result = future.result()
            results.append(result)

    # --- Display and save results ---
    results_df = pd.DataFrame(results).reset_index(drop=True)
    
    print("\n--- ✅ Validation Complete ---")
    print(results_df.to_string())

    # Save the results to a new summary file
    summary_path = os.path.join(args.input_dir, "_validation_results.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\nValidation summary saved to: {summary_path}")


if __name__ == '__main__':
    main()