import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    """
    Bayesian Bias Scoring Tool - Marimo Version
    Demonstrates bias scoring using Python sentiment tools with Bayesian proposition modeling.
    """
    import numpy as np
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import json
    from typing import Dict, List, Tuple
    import warnings

    warnings.filterwarnings("ignore")

    mo.md("# Bayesian Bias Scoring Tool")
    return Dict, List, SentimentIntensityAnalyzer, TextBlob, Tuple, json, np, warnings


@app.cell
def __(Dict, List, SentimentIntensityAnalyzer, TextBlob, Tuple, np):
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
                        neutrality
                        * (1 - evidence_scores.get("compound", 0) ** 2)
                        * weight,
                    )
                elif bias_type == "toxic_language":
                    # Evidence: high negative sentiment + high subjectivity
                    toxicity_indicator = evidence_scores.get(
                        "neg", 0
                    ) * evidence_scores.get("subjectivity", 0)
                    likelihood = max(0.1, toxicity_indicator * weight)
                elif bias_type == "subjective_bias":
                    # Evidence: high subjectivity score
                    likelihood = max(
                        0.1, evidence_scores.get("subjectivity", 0) * weight
                    )

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
                "interpretation": self._interpret_results(
                    bias_probabilities, overall_bias
                ),
            }

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

    return (BayesianBiasScorer,)


@app.cell
def __(BayesianBiasScorer):
    # Initialize the scorer
    scorer = BayesianBiasScorer()

    # Sample texts for testing
    SAMPLE_TEXTS = [
        "I absolutely love this product! It's amazing and perfect in every way.",  # Positive bias
        "This is the worst thing I've ever seen. Complete garbage and waste of time.",  # Negative bias
        "The weather today is partly cloudy with temperatures around 70 degrees.",  # Neutral
        "Those people are always causing problems and can't be trusted.",  # Potential bias/toxicity
        "In my opinion, this approach is clearly superior to all alternatives.",  # Subjective bias
    ]

    return SAMPLE_TEXTS, scorer


@app.cell
def __(mo):
    # Interactive text input for analysis
    text_input = mo.ui.text_area(
        label="Enter text to analyze for bias:",
        placeholder="Type or paste text here...",
        value="I absolutely love this product! It's amazing and perfect in every way.",
    )
    text_input
    return (text_input,)


@app.cell
def __(mo, scorer, text_input):
    def analyze_single_text():
        """Analyze the input text and return formatted results"""
        if text_input.value:
            single_result = scorer.analyze_text(text_input.value)

            # Display results
            return mo.vstack([
                mo.md(
                    f"**Analyzing:** {text_input.value[:100]}{'...' if len(text_input.value) > 100 else ''}"
                ),
                mo.md("### Raw Sentiment Scores"),
                mo.ui.table({
                    "Metric": [
                        "TextBlob Polarity",
                        "TextBlob Subjectivity",
                        "VADER Compound",
                        "VADER Positive",
                        "VADER Negative",
                        "VADER Neutral",
                    ],
                    "Score": [
                        f"{single_result['raw_scores']['textblob']['polarity']:.3f}",
                        f"{single_result['raw_scores']['textblob']['subjectivity']:.3f}",
                        f"{single_result['raw_scores']['vader']['compound']:.3f}",
                        f"{single_result['raw_scores']['vader']['pos']:.3f}",
                        f"{single_result['raw_scores']['vader']['neg']:.3f}",
                        f"{single_result['raw_scores']['vader']['neu']:.3f}",
                    ],
                }),
                mo.md("### Bayesian Bias Probabilities"),
                mo.ui.table({
                    "Bias Type": [
                        bias_type.replace("_", " ").title()
                        for bias_type in single_result[
                            "bayesian_bias_probabilities"
                        ].keys()
                    ],
                    "Probability": [
                        f"{prob:.3f}"
                        for prob in single_result[
                            "bayesian_bias_probabilities"
                        ].values()
                    ],
                }),
                mo.md(
                    f"### Overall Bias Score: **{single_result['overall_bias_score']:.3f}**"
                ),
                mo.md(f"### Interpretation: **{single_result['interpretation']}**"),
            ])
        else:
            return mo.md("Enter some text above to analyze for bias.")

    # Call the function to display results
    analyze_single_text()
    return (analyze_single_text,)


@app.cell
def __(SAMPLE_TEXTS, mo, np, scorer):
    def analyze_sample_texts():
        """Analyze all sample texts and return summary"""
        # Analyze all sample texts
        sample_results = []
        for text in SAMPLE_TEXTS:
            sample_result = scorer.analyze_text(text)
            sample_results.append(sample_result)

        # Create summary table
        summary_data = {
            "Text": [
                r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
                for r in sample_results
            ],
            "Overall Bias Score": [
                f"{r['overall_bias_score']:.3f}" for r in sample_results
            ],
            "Primary Concern": [
                r["interpretation"].split(": ")[1].split(" (")[0]
                for r in sample_results
            ],
            "Severity": [r["interpretation"].split(" ")[0] for r in sample_results],
        }

        return mo.vstack([
            mo.md("## Sample Text Analysis"),
            mo.ui.table(summary_data),
            mo.md(
                f"**Average bias score across all samples:** {np.mean([r['overall_bias_score'] for r in sample_results]):.3f}"
            ),
        ])

    # Call the function to display results
    analyze_sample_texts()
    return (analyze_sample_texts,)


if __name__ == "__main__":
    app.run()
