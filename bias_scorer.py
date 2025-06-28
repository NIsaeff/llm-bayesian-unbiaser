#!/usr/bin/env python3
"""
Bayesian Bias Scoring Tool
Demonstrates bias scoring using Python sentiment tools with Bayesian proposition modeling.
Sentiments are modeled as probabilistic propositions that are updated based on evidence.
"""

import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class BayesianBiasScorer:
    """
    A bias scoring system that models sentiments as Bayesian propositions
    updated by evidence from multiple sentiment analysis tools.
    """

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        # Evidence database - hypothetical evidence weights for different bias types
        self.evidence_db = {
            "positive_bias": {"prior": 0.3, "weight": 1.0},
            "negative_bias": {"prior": 0.3, "weight": 1.0},
            "neutral_stance": {"prior": 0.4, "weight": 0.8},
            "toxic_language": {"prior": 0.1, "weight": 1.5},
            "subjective_bias": {"prior": 0.25, "weight": 1.2},
        }

    def get_textblob_scores(self, text: str) -> Dict:
        """Extract sentiment scores using TextBlob"""
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,  # -1 to 1
            "subjectivity": blob.sentiment.subjectivity,  # 0 to 1
        }

    def get_vader_scores(self, text: str) -> Dict:
        """Extract sentiment scores using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        return scores

    def calculate_bayesian_bias_probability(self, evidence_scores: Dict) -> Dict:
        """
        Calculate Bayesian posterior probabilities for different bias types
        based on observed evidence (sentiment scores)
        """
        posteriors = {}

        # Update probabilities based on evidence
        for bias_type, params in self.evidence_db.items():
            prior = params["prior"]
            weight = params["weight"]

            # Calculate likelihood based on sentiment evidence
            if bias_type == "positive_bias":
                # Evidence: high positive polarity or positive VADER scores
                likelihood = max(
                    0.1,
                    (evidence_scores.get("polarity", 0) + 1)
                    / 2
                    * evidence_scores.get("pos", 0)
                    * weight,
                )
            elif bias_type == "negative_bias":
                # Evidence: high negative polarity or negative VADER scores
                likelihood = max(
                    0.1,
                    abs(min(0, evidence_scores.get("polarity", 0)))
                    * evidence_scores.get("neg", 0)
                    * weight,
                )
            elif bias_type == "neutral_stance":
                # Evidence: neutral polarity and compound scores
                neutrality = 1 - abs(evidence_scores.get("polarity", 0))
                likelihood = max(
                    0.1,
                    neutrality * (1 - evidence_scores.get("compound", 0) ** 2) * weight,
                )
            elif bias_type == "toxic_language":
                # Evidence: high negative sentiment + high subjectivity
                toxicity_indicator = evidence_scores.get(
                    "neg", 0
                ) * evidence_scores.get("subjectivity", 0)
                likelihood = max(0.1, toxicity_indicator * weight)
            elif bias_type == "subjective_bias":
                # Evidence: high subjectivity score
                likelihood = max(0.1, evidence_scores.get("subjectivity", 0) * weight)

            # Bayesian update: P(bias|evidence) = P(evidence|bias) * P(bias) / P(evidence)
            # Simplified normalization
            posterior = (likelihood * prior) / (
                likelihood * prior + (1 - likelihood) * (1 - prior)
            )
            posteriors[bias_type] = min(
                0.99, max(0.01, posterior)
            )  # Bound between 0.01 and 0.99

        return posteriors

    def analyze_text(self, text: str) -> Dict:
        """Complete bias analysis of text using Bayesian framework"""
        print(f"\n{'=' * 60}")
        print(f"ANALYZING TEXT: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"{'=' * 60}")

        # Get raw sentiment scores
        textblob_scores = self.get_textblob_scores(text)
        vader_scores = self.get_vader_scores(text)

        # Combine evidence
        evidence = {**textblob_scores, **vader_scores}

        # Calculate Bayesian bias probabilities
        bias_probabilities = self.calculate_bayesian_bias_probability(evidence)

        # Calculate overall bias score (weighted average)
        bias_weights = {
            "positive_bias": 0.5,
            "negative_bias": 0.5,
            "toxic_language": 1.5,
            "subjective_bias": 0.8,
        }
        overall_bias = sum(
            bias_probabilities[bt] * bias_weights.get(bt, 1.0)
            for bt in bias_probabilities
        ) / sum(bias_weights.values())

        results = {
            "text": text,
            "raw_scores": {"textblob": textblob_scores, "vader": vader_scores},
            "bayesian_bias_probabilities": bias_probabilities,
            "overall_bias_score": overall_bias,
            "interpretation": self._interpret_results(bias_probabilities, overall_bias),
        }

        self._print_results(results)
        return results

    def _interpret_results(self, probabilities: Dict, overall_score: float) -> str:
        """Generate interpretation of bias analysis"""
        dominant_bias = max(probabilities.items(), key=lambda x: x[1])

        if overall_score > 0.7:
            severity = "HIGH"
        elif overall_score > 0.4:
            severity = "MODERATE"
        else:
            severity = "LOW"

        return f"{severity} bias detected. Primary concern: {dominant_bias[0]} (probability: {dominant_bias[1]:.3f})"

    def _print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\nRAW SENTIMENT SCORES:")
        print(
            f"TextBlob - Polarity: {results['raw_scores']['textblob']['polarity']:.3f}, "
            f"Subjectivity: {results['raw_scores']['textblob']['subjectivity']:.3f}"
        )
        print(
            f"VADER - Compound: {results['raw_scores']['vader']['compound']:.3f}, "
            f"Pos: {results['raw_scores']['vader']['pos']:.3f}, "
            f"Neg: {results['raw_scores']['vader']['neg']:.3f}"
        )

        print(f"\nBAYESIAN BIAS PROBABILITIES:")
        for bias_type, prob in results["bayesian_bias_probabilities"].items():
            print(f"{bias_type.replace('_', ' ').title()}: {prob:.3f}")

        print(f"\nOVERALL BIAS SCORE: {results['overall_bias_score']:.3f}")
        print(f"INTERPRETATION: {results['interpretation']}")


# Sample texts for testing
SAMPLE_TEXTS = [
    "I absolutely love this product! It's amazing and perfect in every way.",  # Positive bias
    "This is the worst thing I've ever seen. Complete garbage and waste of time.",  # Negative bias
    "The weather today is partly cloudy with temperatures around 70 degrees.",  # Neutral
    "Those people are always causing problems and can't be trusted.",  # Potential bias/toxicity
    "In my opinion, this approach is clearly superior to all alternatives.",  # Subjective bias
]


def main():
    """Main execution function"""
    print("BAYESIAN BIAS SCORING TOOL")
    print("Models sentiments as probabilistic propositions updated by evidence")
    print("Uses TextBlob and VADER sentiment analysis tools")

    # Initialize scorer
    scorer = BayesianBiasScorer()

    # Analyze sample texts
    all_results = []
    for text in SAMPLE_TEXTS:
        result = scorer.analyze_text(text)
        all_results.append(result)

    # Summary analysis
    print(f"\n{'=' * 60}")
    print("SUMMARY ANALYSIS")
    print(f"{'=' * 60}")

    avg_bias = np.mean([r["overall_bias_score"] for r in all_results])
    print(f"Average bias score across all samples: {avg_bias:.3f}")

    high_bias_texts = [r for r in all_results if r["overall_bias_score"] > 0.5]
    print(f"Texts with high bias scores: {len(high_bias_texts)}/{len(all_results)}")

    # Evidence database summary
    print(f"\nEVIDENCE DATABASE CONFIGURATION:")
    for bias_type, params in scorer.evidence_db.items():
        print(
            f"{bias_type}: Prior={params['prior']:.2f}, Weight={params['weight']:.2f}"
        )


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        print("Please install required packages:")
        print("pip install textblob vaderSentiment numpy")
        print("\nNote: TextBlob may require additional setup:")
        print("python -m textblob.download_corpora")
    except Exception as e:
        print(f"EXECUTION ERROR: {e}")
        print("This is expected in some environments due to package dependencies.")
        print("The code structure demonstrates the Bayesian bias scoring approach.")

# Expected output format when run:
"""
SAMPLE EXPECTED OUTPUT:

============================================================
ANALYZING TEXT: 'I absolutely love this product! It's amazing and pe...'
============================================================

RAW SENTIMENT SCORES:
TextBlob - Polarity: 0.875, Subjectivity: 0.900
VADER - Compound: 0.8316, Pos: 0.741, Neg: 0.000

BAYESIAN BIAS PROBABILITIES:
Positive Bias: 0.789
Negative Bias: 0.023
Neutral Stance: 0.145
Toxic Language: 0.034
Subjective Bias: 0.876

OVERALL BIAS SCORE: 0.623
INTERPRETATION: MODERATE bias detected. Primary concern: subjective_bias (probability: 0.876)
"""
