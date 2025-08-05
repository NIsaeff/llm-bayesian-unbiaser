#!/usr/bin/env python3
"""
aif360_bias_scorer.py

1. Stream AIF360 s-values → sequential Normal–Inverse–Gamma updates
2. Write out template_with_priors.csv with evolving mu_prior & sigma_prior
"""

import math
import numpy as np
import pandas as pd

from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def compute_statistical_parity_difference():
    """
    Run one train/test split, fit a classifier, and return the
    statistical parity difference (s).
    """
    dataset = AdultDataset()  # load data :contentReference[oaicite:2]{index=2}
    privileged   = [{'sex': 1}]
    unprivileged = [{'sex': 0}]

    train, test = dataset.split([0.7], shuffle=True)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='liblinear')
    )
    clf.fit(train.features, train.labels.ravel())
    preds = clf.predict(test.features).reshape(-1, 1)

    test_pred = test.copy(deepcopy=True)
    test_pred.labels = preds

    metric = ClassificationMetric(
        test, test_pred,
        privileged_groups=privileged,
        unprivileged_groups=unprivileged
    )
    return metric.statistical_parity_difference()

def main():
    # 1) Load your template of entries
    template = pd.read_csv('template.csv')

    # 2) Initialize very vague Normal–Inverse–Gamma prior
    mu0, kappa0    = 0.0, 1e-6
    alpha0, beta0  = 1e-6, 1e-6

    # 3) Prepare output columns
    template['mu_prior']    = 0.0
    template['sigma_prior'] = 0.0

    # 4) Iterate row-by-row, record prior, then update with new s
    for idx, _ in template.iterrows():
        # Prior predictive mean & variance
        mu_pred   = mu0
        var_pred  = beta0 * (kappa0 + 1) / (alpha0 * kappa0)
        sigma_pred = math.sqrt(var_pred)

        # Store these as the prior for this row
        template.at[idx, 'mu_prior']    = mu_pred
        template.at[idx, 'sigma_prior'] = sigma_pred

        # Observe new s-value
        s = compute_statistical_parity_difference()

        # Conjugate updates for Normal–Inverse–Gamma
        kappa1 = kappa0 + 1
        mu1    = (kappa0 * mu0 + s) / kappa1
        alpha1 = alpha0 + 0.5
        beta1  = beta0 + (kappa0 * (s - mu0) ** 2) / (2 * kappa1)

        # Roll forward to next prior
        mu0, kappa0, alpha0, beta0 = mu1, kappa1, alpha1, beta1

    # 5) Write out evolving priors
    out = "template_with_priors.csv"
    template.to_csv(out, index=False)
    print(f"Wrote results to {out}")


    # 1) Compute posterior predictive variance & std. dev.
    var_post = beta0 * (kappa0 + 1) / (alpha0 * kappa0)
    sigma_post = math.sqrt(var_post)

    # 2) The posterior mean is mu0
    B = mu0

    # 3) (Optionally) sample one draw from the predictive
    # B_sample = np.random.normal(B, sigma_post)

    print(f"Final bias estimate B = {B:.4f} ± {sigma_post:.4f}")


if __name__ == "__main__":
    main()

