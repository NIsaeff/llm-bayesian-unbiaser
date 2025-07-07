#!/usr/bin/env python3
"""
Bayesian Bias Scoring Tool with CSV Processing
Processes input CSV files and generates standardized output CSV with bias scores,
confidence intervals, and recommendations.
"""

import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
from datetime import datetime
from scipy import stats
import os

warnings.filterwarnings("ignore")


class BayesianBiasScorer:
    """
    Enhanced bias scoring system that processes CSV input and generates
    standardized CSV output with Bayesian analysis.
    """

    def __init__(self, model_version: str = "v1.0.0"):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.model_version = model_version

        # Enhanced evidence database with confidence intervals
        self.evidence_db = {
            "positive_bias": {"prior": 0.3, "weight": 1.0, "variance": 0.05},
            "negative_bias": {"prior": 0.3, "weight": 1.0, "variance": 0.05},
            "neutral_stance": {"prior": 0.4, "weight": 0.8, "variance": 0.03},
            "toxic_language": {"prior": 0.1, "weight": 1.5, "variance": 0.08},
            "subjective_bias": {"prior": 0.25, "weight": 1.2, "variance": 0.06},
        }

        # Bias category mappings for suspected categories
        self.bias_category_mapping = {
            "age": ["negative_bias", "subjective_bias"],
            "gender": ["subjective_bias", "toxic_language"],
            "socioeconomic": ["negative_bias", "subjective_bias"],
            "geographic": ["negative_bias", "subjective_bias"],
            "education": ["subjective_bias", "negative_bias"],
            "class": ["negative_bias", "subjective_bias"],
            "technical_ability": ["negative_bias", "subjective_bias"],
            "ability": ["negative_bias", "toxic_language"],
            "nationality": ["negative_bias", "toxic_language"],
            "language": ["negative_bias", "subjective_bias"],
            "xenophobia": ["toxic_language", "negative_bias"],
            "tokenism": ["subjective_bias", "positive_bias"],
            "diversity": ["subjective_bias", "positive_bias"],
            "parental_status": ["negative_bias", "subjective_bias"],
            "work_culture": ["subjective_bias", "negative_bias"],
            "ableism": ["negative_bias", "toxic_language"],
            "condescension": ["negative_bias", "toxic_language"],
            "learning_ability": ["negative_bias", "subjective_bias"],
            "appearance": ["subjective_bias", "negative_bias"],
        }

    def get_textblob_scores(self, text: str) -> Dict:
        """Extract sentiment scores using TextBlob"""
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    def get_vader_scores(self, text: str) -> Dict:
        """Extract sentiment scores using VADER"""
        return self.vader_analyzer.polarity_scores(text)

    def calculate_bayesian_bias_probability(
        self, evidence_scores: Dict, suspected_categories: List[str] = None
    ) -> Dict:
        """
        Calculate Bayesian posterior probabilities with confidence intervals
        """
        posteriors = {}

        # Weight evidence based on suspected categories
        category_weights = {}
        if suspected_categories:
            for category in suspected_categories:
                if category in self.bias_category_mapping:
                    for bias_type in self.bias_category_mapping[category]:
                        category_weights[bias_type] = (
                            category_weights.get(bias_type, 1.0) + 0.3
                        )

        for bias_type, params in self.evidence_db.items():
            prior = params["prior"]
            weight = params["weight"]
            variance = params["variance"]

            # Apply category weighting
            if bias_type in category_weights:
                weight *= category_weights[bias_type]

            # Calculate likelihood based on sentiment evidence
            if bias_type == "positive_bias":
                likelihood = max(
                    0.1,
                    (evidence_scores.get("polarity", 0) + 1)
                    / 2
                    * evidence_scores.get("pos", 0)
                    * weight,
                )
            elif bias_type == "negative_bias":
                likelihood = max(
                    0.1,
                    abs(min(0, evidence_scores.get("polarity", 0)))
                    * evidence_scores.get("neg", 0)
                    * weight,
                )
            elif bias_type == "neutral_stance":
                neutrality = 1 - abs(evidence_scores.get("polarity", 0))
                likelihood = max(
                    0.1,
                    neutrality * (1 - evidence_scores.get("compound", 0) ** 2) * weight,
                )
            elif bias_type == "toxic_language":
                toxicity_indicator = evidence_scores.get(
                    "neg", 0
                ) * evidence_scores.get("subjectivity", 0)
                likelihood = max(0.1, toxicity_indicator * weight)
            elif bias_type == "subjective_bias":
                likelihood = max(0.1, evidence_scores.get("subjectivity", 0) * weight)

            # Bayesian update
            posterior = (likelihood * prior) / (
                likelihood * prior + (1 - likelihood) * (1 - prior)
            )
            posterior = min(0.99, max(0.01, posterior))

            posteriors[bias_type] = {
                "mean": posterior,
                "variance": variance,
                "evidence": likelihood,
            }

        return posteriors

    def calculate_confidence_interval(
        self, posterior_mean: float, variance: float, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for bias score"""
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * np.sqrt(variance)

        lower = max(0.0, posterior_mean - margin_of_error)
        upper = min(1.0, posterior_mean + margin_of_error)

        return lower, upper

    def get_bias_explanation(
        self, bias_type: str, probability: float, suspected_categories: List[str] = None
    ) -> str:
        """Generate explanation for detected bias"""
        explanations = {
            "positive_bias": "Text shows overly positive sentiment that may indicate favoritism or unrealistic expectations",
            "negative_bias": "Text contains negative sentiment that may unfairly exclude or discriminate against certain groups",
            "toxic_language": "Text contains potentially harmful or discriminatory language",
            "subjective_bias": "Text presents subjective opinions as facts, potentially introducing unfair bias",
            "neutral_stance": "Text appears relatively neutral with minimal bias indicators",
        }

        base_explanation = explanations.get(bias_type, "Bias detected in text")

        if suspected_categories:
            category_context = ", ".join(suspected_categories)
            return f"{base_explanation}. Particularly relevant to: {category_context}"

        return base_explanation

    def get_recommendations(
        self, dominant_bias: str, score: float, suspected_categories: List[str] = None
    ) -> str:
        """Generate recommendations based on bias analysis"""
        recommendations = {
            "positive_bias": "Consider more balanced language; avoid excessive positive claims",
            "negative_bias": "Rephrase to focus on objective criteria rather than exclusionary language",
            "toxic_language": "Review and revise language to ensure respectful and inclusive communication",
            "subjective_bias": "Present information more objectively; distinguish between facts and opinions",
            "neutral_stance": "Text appears balanced; consider if more specific criteria would be helpful",
        }

        base_rec = recommendations.get(dominant_bias, "Review text for potential bias")

        if score > 0.7:
            urgency = "URGENT: "
        elif score > 0.4:
            urgency = "Important: "
        else:
            urgency = "Consider: "

        return urgency + base_rec

    def analyze_text(
        self,
        text: str,
        text_id: str,
        suspected_categories: List[str] = None,
        confidence_prior: float = 0.5,
    ) -> Dict:
        """Complete bias analysis with CSV output formatting"""
        # Get raw sentiment scores
        textblob_scores = self.get_textblob_scores(text)
        vader_scores = self.get_vader_scores(text)
        evidence = {**textblob_scores, **vader_scores}

        # Calculate Bayesian bias probabilities
        bias_probabilities = self.calculate_bayesian_bias_probability(
            evidence, suspected_categories
        )

        # Calculate overall bias score (weighted average)
        bias_weights = {
            "positive_bias": 0.5,
            "negative_bias": 0.5,
            "toxic_language": 1.5,
            "subjective_bias": 0.8,
            "neutral_stance": 0.2,
        }

        weighted_scores = []
        total_weight = 0
        for bt, prob_data in bias_probabilities.items():
            if bt in bias_weights:
                weighted_scores.append(prob_data["mean"] * bias_weights[bt])
                total_weight += bias_weights[bt]

        overall_bias = sum(weighted_scores) / total_weight if total_weight > 0 else 0

        # Find dominant bias type
        dominant_bias = max(
            bias_probabilities.items(),
            key=lambda x: x[1]["mean"] if x[0] != "neutral_stance" else 0,
        )

        # Calculate confidence interval for overall score
        avg_variance = np.mean([p["variance"] for p in bias_probabilities.values()])
        ci_lower, ci_upper = self.calculate_confidence_interval(
            overall_bias, avg_variance
        )

        # Determine severity
        if overall_bias > 0.7:
            severity = "high"
        elif overall_bias > 0.4:
            severity = "medium"
        else:
            severity = "low"

        return {
            "input_id": text_id,
            "original_text": text,
            "bias_category": dominant_bias[0],
            "bias_score": overall_bias,
            "confidence_interval_lower": ci_lower,
            "confidence_interval_upper": ci_upper,
            "confidence_level": 0.95,
            "bayesian_evidence": dominant_bias[1]["evidence"],
            "prior_strength": confidence_prior,
            "posterior_mean": dominant_bias[1]["mean"],
            "posterior_variance": dominant_bias[1]["variance"],
            "bias_explanation": self.get_bias_explanation(
                dominant_bias[0], dominant_bias[1]["mean"], suspected_categories
            ),
            "severity_level": severity,
            "recommendations": self.get_recommendations(
                dominant_bias[0], overall_bias, suspected_categories
            ),
            "processing_timestamp": datetime.now().isoformat(),
            "model_version": self.model_version,
        }

    def process_csv(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Process input CSV and generate output CSV"""
        print(f"Processing input file: {input_file}")

        # Read input CSV
        try:
            df_input = pd.read_csv(input_file)
        except Exception as e:
            print(f"Error reading input file: {e}")
            return None

        # Validate required columns
        required_columns = ["id", "text"]
        missing_columns = [
            col for col in required_columns if col not in df_input.columns
        ]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return None

        print(f"Found {len(df_input)} records to process")

        # Process each row
        results = []
        for idx, row in df_input.iterrows():
            text_id = str(row["id"])
            text = row["text"]

            # Parse suspected categories
            suspected_categories = []
            if "suspected_bias_categories" in row and pd.notna(
                row["suspected_bias_categories"]
            ):
                suspected_categories = [
                    cat.strip()
                    for cat in str(row["suspected_bias_categories"]).split(",")
                ]

            # Get confidence prior
            confidence_prior = float(row.get("confidence_prior", 0.5))

            print(f"Processing record {idx + 1}/{len(df_input)}: ID {text_id}")

            # Analyze text
            result = self.analyze_text(
                text, text_id, suspected_categories, confidence_prior
            )
            results.append(result)

        # Create output DataFrame
        df_output = pd.DataFrame(results)

        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_bias_analysis_{timestamp}.csv"

        # Save output CSV
        df_output.to_csv(output_file, index=False)
        print(f"Output saved to: {output_file}")

        # Print summary
        self.print_summary(df_output)

        return df_output

    def print_summary(self, df_output: pd.DataFrame):
        """Print analysis summary"""
        print(f"\n{'=' * 60}")
        print("BIAS ANALYSIS SUMMARY")
        print(f"{'=' * 60}")

        print(f"Total records processed: {len(df_output)}")
        print(f"Average bias score: {df_output['bias_score'].mean():.3f}")

        # Severity distribution
        severity_counts = df_output["severity_level"].value_counts()
        print(f"\nSeverity distribution:")
        for severity, count in severity_counts.items():
            print(
                f"  {severity.capitalize()}: {count} ({count / len(df_output) * 100:.1f}%)"
            )

        # Top bias categories
        print(f"\nTop bias categories:")
        bias_counts = df_output["bias_category"].value_counts().head(5)
        for bias_type, count in bias_counts.items():
            print(f"  {bias_type.replace('_', ' ').title()}: {count}")

        # High-risk items
        high_risk = df_output[df_output["severity_level"] == "high"]
        if len(high_risk) > 0:
            print(f"\nHigh-risk items requiring attention:")
            for _, row in high_risk.head(3).iterrows():
                print(f"  ID {row['input_id']}: {row['original_text'][:50]}...")


def main():
    """Main execution function with command line interface"""
    parser = argparse.ArgumentParser(description="Bayesian Bias Scoring Tool")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path (optional)")
    parser.add_argument("-v", "--version", default="v1.0.0", help="Model version")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    print("BAYESIAN BIAS SCORING TOOL - CSV PROCESSOR")
    print("=" * 60)

    # Initialize scorer
    scorer = BayesianBiasScorer(model_version=args.version)

    # Process CSV
    results = scorer.process_csv(args.input_file, args.output)

    if results is not None:
        print(f"\nProcessing completed successfully!")
    else:
        print(f"\nProcessing failed. Please check the input file format.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        print("Please install required packages:")
        print("pip install pandas textblob vaderSentiment numpy scipy")
    except Exception as e:
        print(f"EXECUTION ERROR: {e}")
        import traceback

        traceback.print_exc()
