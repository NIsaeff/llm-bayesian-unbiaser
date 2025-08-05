## Bayesian Bias Scorer

### Overview

The Bayesian Bias Scorer provides a sequential Bayesian framework for quantifying and updating bias estimates in machine learning classifiers. Leveraging the AIF360 toolkit, the script computes the statistical parity difference (SPD) for each data instance and updates a Normal–Inverse–Gamma conjugate prior to produce evolving estimates of the mean ($\mu$) and uncertainty ($\sigma$). The final output includes per-entry prior values and an aggregated bias score with associated confidence.

### Implementation

1. **Data Loading**: Reads a template CSV (`template.csv`) containing metadata for each evaluation entry.
2. **Metric Computation**: For each row, performs a train/test split on the Adult dataset, fits a logistic regression pipeline, and computes the SPD as the bias metric.
3. **Bayesian Updating**: Initializes a vague Normal–Inverse–Gamma prior. Iterates through entries, recording the prior predictive mean and standard deviation before observing each SPD, then updates the hyperparameters `(μ₀, κ₀, α₀, β₀)` via conjugate formulas.
4. **Output Generation**:

   * Writes `template_with_priors.csv`, which appends `mu_prior` and `sigma_prior` columns reflecting the state before each observation.
   * Computes the final posterior predictive mean ($B$) and standard deviation ($\sigma_{post}$).

### Motivation

Continuous monitoring of bias is critical for ensuring fairness in deployed models. By framing SPD measurements within a Bayesian conjugate structure, this approach:

* Produces **real-time** bias estimates that evolve with incoming data.
* Quantifies **uncertainty**, aiding researchers in assessing confidence in bias measurements.
* Offers a **transparent and rigorous** mechanism for peer reviewers to evaluate bias trajectories.

### Setup Instructions

1. **Prerequisites**:

   * Python 3.7 or later
   * Git (for cloning the repository)
2. **Environment Setup**:

   ```bash
   git clone https://github.com/your-org/bayesian-bias-scorer.git
   cd bayesian-bias-scorer
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Dependencies** (`requirements.txt` should include):

   * `aif360`
   * `scikit-learn`
   * `pandas`
   * `numpy`
   
4. **Running the Scorer**:

   ```bash
   python aif360_bias_scorer.py --input template.csv --output template_with_priors.csv
   ```
5. **Results**:

   * Per-entry priors are saved in `template_with_priors.csv`.
   * Final aggregated bias estimate and uncertainty are printed to the console.

### Notes for Reviewers

* Ensure the Adult dataset is accessible via AIF360.
* Confirm that `template.csv` matches the expected format (contains the necessary columns for identification).
* Review the conjugate update formulas in `aif360_bias_scorer.py` for mathematical correctness.

