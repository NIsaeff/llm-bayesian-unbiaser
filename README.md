# LLM-Bayesian-Unbiaser
# Bayesian Bias Scoring Tool 🧠📊

An interactive sentiment analysis tool that uses Bayesian probability modeling to detect bias in text. Built with Python and featuring a beautiful interactive interface powered by [marimo](https://marimo.io).

## ✨ Features

- **Bayesian Bias Detection**: Uses probabilistic modeling to identify different types of bias
- **Multi-Tool Analysis**: Combines TextBlob and VADER sentiment analysis for comprehensive scoring
- **Interactive Interface**: Real-time analysis with marimo's reactive notebook interface
- **Multiple Bias Types**: Detects positive bias, negative bias, toxic language, subjective bias, and neutral stance
- **Sample Analysis**: Includes pre-loaded examples to demonstrate different bias patterns

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/NIsaeff/llm-bayesian-unbiaser.git
cd llm-bayesian-unbiaser
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download TextBlob Corpora
```bash
python -m textblob.download_corpora
```

### 4. Run the Interactive Notebook
```bash
marimo edit bias_scorer_marimo.py
```

💻 Command-Line Interface

Batch-process text data from a CSV template:

python bias_scorer.py <input.csv> [-o <output.csv>] [-v <model_version>]

<input.csv>: Path to your input CSV template (e.g., template.csv).

-o, --output: (Optional) Specify an output file path. If omitted, the tool will generate a file named <basename>_bias_analysis_<YYYYMMDD_HHMMSS>.csv.

-v, --version: (Optional) Set the model version tag (default: v1.0.0).

Example

python bias_scorer.py template.csv
# Reads `template.csv` and writes something like `template_bias_analysis_20250709_080642.csv`

After running, the console will display progress messages and the location of the output CSV containing bias scores, confidence intervals, and recommendations.

This will open the interactive interface in your browser at `http://localhost:2718`

## 🔧 Alternative: Run the Original Script
```bash
python paste.txt
```

## 📋 Requirements

- Python 3.7+
- numpy
- textblob
- vaderSentiment
- marimo

See `requirements.txt` for exact versions.

## 🎯 How It Works

### The Science Behind the Tool

1. **Sentiment Analysis**: Uses both TextBlob and VADER to extract multiple sentiment dimensions
2. **Bayesian Updates**: Treats bias detection as a probabilistic inference problem
3. **Evidence Integration**: Combines polarity, subjectivity, and sentiment scores as evidence
4. **Posterior Calculation**: Updates prior beliefs about bias types based on observed evidence

### Bias Types Detected

- **Positive Bias**: Overly favorable language
- **Negative Bias**: Excessively critical language  
- **Toxic Language**: Harmful or offensive content
- **Subjective Bias**: Opinion-heavy rather than factual
- **Neutral Stance**: Objective, balanced language

### How VADER Scores Work

VADER returns four key metrics:
- **Compound (-1 to +1)**: Overall sentiment score (most important)
- **Positive (0 to 1)**: Proportion of positive sentiment
- **Negative (0 to 1)**: Proportion of negative sentiment  
- **Neutral (0 to 1)**: Proportion of neutral sentiment

**Compound Score Interpretation:**
- `> 0.05`: Positive sentiment
- `-0.05 to 0.05`: Neutral sentiment
- `< -0.05`: Negative sentiment

## 🖥️ Using the Interactive Interface

### Text Analysis
1. Enter or paste text in the input area
2. View real-time sentiment scores from TextBlob and VADER
3. See Bayesian bias probabilities for each bias type
4. Get an overall bias score and interpretation

### Sample Analysis
- Explore pre-loaded examples showing different bias patterns
- Compare bias scores across multiple text samples
- View summary statistics

## 📊 Example Output

```
Analyzing Text: "I absolutely love this product! It's amazing and perfect in every way."

Raw Sentiment Scores:
- TextBlob Polarity: 0.875, Subjectivity: 0.900  
- VADER Compound: 0.832, Pos: 0.741, Neg: 0.000

Bayesian Bias Probabilities:
- Positive Bias: 0.789
- Subjective Bias: 0.876
- Negative Bias: 0.023
- Toxic Language: 0.034
- Neutral Stance: 0.145

Overall Bias Score: 0.623
Interpretation: MODERATE bias detected. Primary concern: subjective_bias (probability: 0.876)
```

## 🛠️ Development Setup

### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv bias_scorer_env

# Activate (Windows)
bias_scorer_env\Scripts\activate

# Activate (macOS/Linux)  
source bias_scorer_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download corpora
python -m textblob.download_corpora
```

### Verify Installation
```python
import numpy
import textblob
import vaderSentiment
import marimo
print("All packages installed successfully!")
```

## 📚 Understanding the Code

### Key Components

- `BayesianBiasScorer`: Main class handling bias detection logic
- `get_textblob_scores()`: Extracts polarity and subjectivity 
- `get_vader_scores()`: Extracts comprehensive sentiment metrics
- `calculate_bayesian_bias_probability()`: Core Bayesian inference engine
- `analyze_text()`: Complete analysis pipeline

### Bayesian Framework

The tool models each bias type as a probabilistic proposition:

```
P(bias_type|evidence) = P(evidence|bias_type) × P(bias_type) / P(evidence)
```

Where evidence includes sentiment scores from both TextBlob and VADER.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for social media-optimized sentiment analysis
- [marimo](https://marimo.io) for the interactive notebook interface
- The open source community for making tools like this possible

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/NIsaeff/llm-bayesian-unbiaser/issues) page
2. Create a new issue with detailed information
3. Include your Python version and operating system

---

**Happy analyzing! 🎉** Use this tool responsibly to understand and improve the objectivity of text content.
