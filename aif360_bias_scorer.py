#!/usr/bin/env python3
"""
bayesian_bias_scoring.py

1. Compute AIF360 s-values → prior Normal(μ_prior, σ²_prior)
2. Instantiate prior_dist = norm(μ_prior, σ_prior)
3. Load LLM likelihood params (mu_llm, sigma_llm)
4. Conjugate update → posterior Normal(μ_post, σ²_post)
5. Write out template_with_priors_and_posteriors.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# AIF360 & modeling imports
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def compute_aif360_prior():
    """Compute μ_prior and σ_prior from AIF360 s-values."""
    ds = AdultDataset()  # loads & cleans Adult :contentReference[oaicite:7]{index=7}
    priv = [{'sex': 1}]; unpriv = [{'sex': 0}]
    # Pre-model metric
    pre = BinaryLabelDatasetMetric(ds, privileged_groups=priv, unprivileged_groups=unpriv)
    # Fit simple model for EOD and AOD
    train, test = ds.split([0.7], shuffle=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf.fit(train.features, train.labels.ravel())
    pred = test.copy(); pred.labels = clf.predict(test.features).reshape(-1,1)
    post = ClassificationMetric(test, pred, privileged_groups=priv, unprivileged_groups=unpriv)

    # s-values
    s_vals = np.array([
        pre.mean_difference(),
        pre.disparate_impact(),
        post.equal_opportunity_difference(),
        post.average_odds_difference()
    ], dtype=float)

    # Prior parameters μ_prior, σ²_prior :contentReference[oaicite:8]{index=8}
    mu = s_vals.mean()
    var = s_vals.var(ddof=1)
    sigma = np.sqrt(var)
    return mu, sigma

def conjugate_update(mu_prior, sigma_prior, mu_llm, sigma_llm):
    """Perform Normal–Normal conjugate update → (μ_post, σ_post)."""
    prec_prior = 1.0 / (sigma_prior**2)
    prec_llm   = 1.0 / (sigma_llm**2)
    prec_post  = prec_prior + prec_llm
    sigma_post = np.sqrt(1.0 / prec_post)
    mu_post    = (mu_prior*prec_prior + mu_llm*prec_llm) / prec_post  # :contentReference[oaicite:9]{index=9}
    return mu_post, sigma_post

def main():
    # 1) Compute and instantiate the prior distribution
    mu_prior, sigma_prior = compute_aif360_prior()
    prior_dist = norm(loc=mu_prior, scale=sigma_prior)
    print(f"Prior:  μ_prior={mu_prior:.4f}, σ_prior={sigma_prior:.4f}")
    # Example: evaluate prior PDF at 0
    print(f"P(B=0)={prior_dist.pdf(0):.4f}, P(B≤0)={prior_dist.cdf(0):.4f}\n")

    # 2) Load your template with LLM scores
    template = pd.read_csv("template.csv")
    if not {'mu_llm','sigma_llm'}.issubset(template.columns):
        raise KeyError("template.csv must contain 'mu_llm' and 'sigma_llm'")

    # 3) Update each row → posterior
    mus_post, sigmas_post = [], []
    for _, row in template.iterrows():
        μ_post, σ_post = conjugate_update(
            mu_prior, sigma_prior,
            float(row['mu_llm']), float(row['sigma_llm'])
        )
        mus_post.append(μ_post)
        sigmas_post.append(σ_post)

    # 4) Augment & save
    template['mu_prior']    = mu_prior
    template['sigma_prior'] = sigma_prior
    template['mu_post']     = mus_post
    template['sigma_post']  = sigmas_post

    out = "template_with_priors_and_posteriors.csv"
    template.to_csv(out, index=False)
    print(f"Wrote results to {out}")

if __name__ == "__main__":
    main()

