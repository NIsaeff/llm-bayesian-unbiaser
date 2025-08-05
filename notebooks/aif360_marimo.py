import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    # Basic imports
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

    return BinaryLabelDatasetMetric, ClassificationMetric, pd


@app.cell
def _():
    import os
    import sys
    import urllib.request

    # 1. Detect AIF360 package path in the active venv
    import aif360
    aif360_path = os.path.dirname(aif360.__file__)
    adult_data_dir = os.path.join(aif360_path, "data", "raw", "adult")

    # 2. Create directory if missing
    os.makedirs(adult_data_dir, exist_ok=True)
    print(f"Target directory: {adult_data_dir}")

    # 3. URLs for the dataset
    urls = {
        "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    }

    # 4. Download files
    for filename, url in urls.items():
        file_path = os.path.join(adult_data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, file_path)
        else:
            print(f"{filename} already exists, skipping.")

    print("\n✅ Adult dataset is ready for AIF360!")

    return


@app.cell
def _():
    from aif360.datasets import AdultDataset

    # Load dataset globally so other cells can use it
    dataset = AdultDataset()
    print("Features:", dataset.feature_names)
    print("Protected Attributes:", dataset.protected_attribute_names)
    print("Label names:", dataset.label_names)

    return (dataset,)


@app.cell
def _(BinaryLabelDatasetMetric, dataset):
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{'sex': 1}],  # Male
        unprivileged_groups=[{'sex': 0}] # Female
    )

    print("Mean difference:", metric.mean_difference())
    print("Disparate impact:", metric.disparate_impact())

    return


@app.cell
def _(ClassificationMetric, dataset, pd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Convert to pandas
    X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
    y = dataset.labels.ravel()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Create AIF360-compatible test dataset with predictions
    dataset_test = dataset.split([0.7], shuffle=True)[1]
    dataset_pred = dataset_test.copy()
    dataset_pred.labels = y_pred.reshape(-1,1)

    metric_pred = ClassificationMetric(
        dataset_test, dataset_pred,
        unprivileged_groups=[{'sex':0}],
        privileged_groups=[{'sex':1}]
    )

    print("Disparate impact (model):", metric_pred.disparate_impact())
    print("Equal opportunity diff:", metric_pred.equal_opportunity_difference())
    print("Average odds diff:", metric_pred.average_odds_difference())

    return X_test, y_pred


@app.cell
def _(X_test, y_pred):
    import matplotlib.pyplot as plt

    y_priv = y_pred[X_test['sex']==1]
    y_unpriv = y_pred[X_test['sex']==0]

    plt.hist(y_priv, alpha=0.5, label='Male')
    plt.hist(y_unpriv, alpha=0.5, label='Female')
    plt.legend()
    plt.title("Distribution of Predicted Outcomes by Sex")
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
