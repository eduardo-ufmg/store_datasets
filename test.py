import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SETS_DIR = "sets"
SUMMARY_FILE = os.path.join(SETS_DIR, "_summary.csv")


def main():
    summary = pd.read_csv(SUMMARY_FILE)
    results = []

    for _, row in summary.iterrows():
        dataset_file = os.path.join(SETS_DIR, row["saved_file"])
        if not os.path.isfile(dataset_file) or row["status"] != "Success":
            print(f"Skipping {row['dataset_name']} (not found or failed)")
            continue

        print(f"Testing {row['dataset_name']} ...")
        df = pd.read_csv(dataset_file)
        if "target" not in df.columns:
            print(f"  - No target column in {row['dataset_name']}")
            continue

        X = df.drop(columns=["target"])
        y = df["target"]

        # Use stratified split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y
            )
        except Exception as e:
            print(f"  - Split failed: {e}")
            continue

        clf = KNeighborsClassifier()
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"  - Accuracy: {acc:.4f}")
            results.append(
                {
                    "dataset": row["dataset_name"],
                    "n_samples": row["n_samples"],
                    "n_features": row["n_features"],
                    "n_classes": row["n_classes"],
                    "accuracy": acc,
                }
            )
        except Exception as e:
            print(f"  - KNN failed: {e}")
            results.append(
                {
                    "dataset": row["dataset_name"],
                    "n_samples": row["n_samples"],
                    "n_features": row["n_features"],
                    "n_classes": row["n_classes"],
                    "accuracy": None,
                }
            )

    # Print summary
    print("\n=== KNN Results ===")
    for res in results:
        print(f"{res['dataset']}: accuracy={res['accuracy']}")


if __name__ == "__main__":
    main()
