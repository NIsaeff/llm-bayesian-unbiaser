#!/usr/bin/env python3
"""
aif360_bias_scorer.py

1. Compute AIF360 s-values → prior Normal(μ_prior, σ²_prior)
2. Write out template_with_priors.csv
"""

import os
import numpy as np
import pandas as pd

# AIF360 imports
from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def compute_aif360_prior():
    """Compute μ_prior and σ_prior from AIF360 s-values."""
    # 1) Load your dataset
    dataset = AdultDataset()

    # 2) Define privileged / unprivileged groups
    privileged   = [{'sex': 1}]
    unprivileged = [{'sex': 0}]

    # 3) Train a simple classifier and collect one s-value
    s_values = []
    train, test = dataset.split([0.7], shuffle=True)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='liblinear')
    )
    clf.fit(train.features, train.labels.ravel())

    # 4) Get predictions and wrap them in a new dataset
    preds = clf.predict(test.features).reshape(-1, 1)

    # Single deep‐copy, then overwrite its labels
    test_pred = test.copy(deepcopy=True)
    test_pred.labels = preds

    # 5) Compute the fairness metric (statistical parity difference)
    metric = ClassificationMetric(
        test,             # true labels
        test_pred,        # predicted labels
        privileged_groups=privileged,
        unprivileged_groups=unprivileged
    )
    s = metric.statistical_parity_difference()
    s_values.append(s)

    # 6) Compute prior moments
    mu_prior = float(np.mean(s_values))
    if len(s_values) > 1:
        sigma_prior = float(np.std(s_values, ddof=1))
    else:
        sigma_prior = 0.0
    return mu_prior, sigma_prior

def main():
    # Compute the prior
    mu_prior, sigma_prior = compute_aif360_prior()

    # Load your template CSV
    template = pd.read_csv('template.csv')

    # Add only the prior columns
    template['mu_prior']    = mu_prior
    template['sigma_prior'] = sigma_prior

    # Write out just the priors
    out = "template_with_priors.csv"
    template.to_csv(out, index=False)
    print(f"Wrote results to {out}")

if __name__ == "__main__":
    main()

