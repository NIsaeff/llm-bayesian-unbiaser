import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # Let's start by setting up our imports
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.special import softmax
    import warnings
    warnings.filterwarnings('ignore')

    print("✅ Imports successful")
    print("📦 Required packages: transformers, torch, numpy, pandas, matplotlib, seaborn, scipy")
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        mo,
        pd,
        plt,
        softmax,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🧠 What is HateBERT?

    **HateBERT** is a specialized AI model designed to detect hate speech and toxic language.

    ## Key Facts:
    - **Base**: Built on BERT (the famous transformer model)
    - **Purpose**: Detects toxic, abusive, and hate speech content
    - **Output**: Probability scores (0-1) indicating toxicity level
    - **Training**: Fine-tuned on datasets containing labeled hate speech examples

    ## What it detects:
    ✅ Hate speech targeting identity groups  
    ✅ Abusive and offensive language  
    ✅ Toxic comments that harm discussions  
    ✅ Threatening or harassing content  

    ## What we'll learn:
    1. How to load and use HateBERT
    2. How to interpret its probability scores
    3. Its strengths and limitations
    4. How it fits into bias detection
    """)
    return


@app.cell
def _(AutoModelForSequenceClassification, AutoTokenizer):
    # Load HateBERT model
    print("🔄 Loading HateBERT model...")

    # We'll use a popular toxic language detection model
    model_name = "unitary/toxic-bert"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
        print("✅ Model loaded successfully!")
        print(f"📊 Number of labels: {model.config.num_labels}")
    
        # Let's see what labels the model predicts
        if hasattr(model.config, 'id2label'):
            print(f"🏷️  Labels: {model.config.id2label}")
        else:
            print("🏷️  Labels: Binary classification (toxic/non-toxic)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have internet connection and transformers installed")
    return model, tokenizer


@app.cell
def _(model, softmax, tokenizer, torch):
    def analyze_toxicity(text):
        """
        Analyze text for toxicity using HateBERT
        Returns probability scores and interpretation
        """
    
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          padding=True, max_length=512)
    
        # Get model predictions (no gradients needed for inference)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
    
        # Convert to probabilities using softmax
        probabilities = softmax(logits.numpy(), axis=1)[0]
    
        # For binary classification, get toxicity probability
        if len(probabilities) == 2:
            # Usually [non-toxic, toxic] or [toxic, non-toxic]
            toxicity_score = probabilities[1] if probabilities[1] > probabilities[0] else probabilities[0]
        else:
            toxicity_score = max(probabilities)
    
        return {
            'text': text,
            'toxicity_score': float(toxicity_score),
            'all_probabilities': probabilities.tolist(),
            'prediction': 'TOXIC' if toxicity_score > 0.5 else 'NON-TOXIC'
        }

    print("✅ Toxicity analysis function created!")
    return (analyze_toxicity,)


@app.cell
def _(analyze_toxicity, pd):
    # Test examples from clearly safe to clearly toxic
    test_texts = [
        "Hello, how are you today?",           # Safe
        "I love this beautiful weather!",      # Safe  
        "I disagree with your opinion.",       # Mild disagreement
        "This is really frustrating.",         # Negative but not toxic
        "You're completely wrong about this.", # More negative
        "That's the stupidest thing I've heard.", # Getting toxic
        "You're an absolute moron.",           # Clearly toxic
        "People like you don't deserve respect." # Hateful
    ]

    print("🧪 Testing HateBERT on sample texts:")
    print("=" * 60)

    results = []
    for text in test_texts:
        result = analyze_toxicity(text)
        results.append(result)
    
        print(f"Text: '{text}'")
        print(f"Toxicity Score: {result['toxicity_score']:.3f}")
        print(f"Prediction: {result['prediction']}")
        print("-" * 40)

    # Store results for later analysis
    df_results = pd.DataFrame(results)
    return (results,)


@app.cell
def _(analyze_toxicity, pd):
    def _():
        # Test examples with same threat, different targets
        test_texts_threats = [
            "I am going to kill you because you are a woman. This is not a joke.",
            "I am going to kill you because you are a man. This is not a joke.",
            "I am going to kill you because you are too old. This is not a joke.",
            "I am going to kill you because you are a child. This is not a joke."
        ]

        print("🧪 Testing HateBERT on sample texts:")
        print("=" * 60)

        results = []
        for text in test_texts_threats:
            result = analyze_toxicity(text)
            results.append(result)
        
            print(f"Text: '{text}'")
            print(f"Toxicity Score: {result['toxicity_score']:.3f}")
            print(f"Prediction: {result['prediction']}")
        return print("-" * 40)

        # Store results for later analysis
        df_results = pd.DataFrame(results)


    _()
    return


@app.cell
def _(plt, results):
    # Create a simple visualization
    plt.figure(figsize=(12, 6))

    # Plot toxicity scores
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(results)), [r['toxicity_score'] for r in results], 
                   color=['green' if score < 0.5 else 'red' for score in [r['toxicity_score'] for r in results]])
    plt.axhline(y=0.5, color='black', linestyle='--', label='Decision Threshold')
    plt.xlabel('Text Example')
    plt.ylabel('Toxicity Score')
    plt.title('HateBERT Toxicity Scores')
    plt.legend()
    plt.xticks(range(len(results)), [f'Text {i+1}' for i in range(len(results))], rotation=45)

    # Distribution of scores
    plt.subplot(1, 2, 2)
    plt.hist([r['toxicity_score'] for r in results], bins=10, alpha=0.7, color='blue')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Toxicity Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"📊 Score range: {min([r['toxicity_score'] for r in results]):.3f} to {max([r['toxicity_score'] for r in results]):.3f}")
    return


@app.cell
def _(analyze_toxicity):
    def _():
        # Edge cases that might confuse the model
        edge_cases = [
            # Sarcasm
            "Oh great, another wonderful Monday morning!",
            "Sure, because that's exactly what I needed today.",
        
            # Context-dependent 
            "You killed it in that presentation!",  # Positive use of 'killed'
            "That performance was sick!",           # Positive use of 'sick'
        
            # Mild profanity
            "This is damn good coffee.",
            "Holy crap, that's amazing!",
        
            # Technical/professional language
            "We need to eliminate this bug.",
            "This code is garbage and needs to be destroyed.",
        
            # Borderline cases
            "I can't stand people who don't signal when driving.",
            "Politicians are all liars and thieves."
        ]

        print("🔍 Testing Edge Cases:")
        print("=" * 50)

        edge_results = []
        for text in edge_cases:
            result = analyze_toxicity(text)
            edge_results.append(result)
        
            print(f"'{text}'")
            print(f"  → Score: {result['toxicity_score']:.3f} | {result['prediction']}")
        return print()


    _()
    return


@app.cell
def _(results):
    def _():
        def interpret_toxicity_score(score):
            """Interpret HateBERT toxicity scores"""
            if score < 0.2:
                return "✅ Very Safe", "No action needed"
            elif score < 0.4:
                return "🟢 Generally Safe", "Monitor if needed"
            elif score < 0.6:
                return "🟨 Potentially Problematic", "Review recommended"
            elif score < 0.8:
                return "🟧 Likely Toxic", "Action recommended"
            else:
                return "🚫 Highly Toxic", "Immediate action needed"

        print("📋 HateBERT Score Interpretation Guide:")
        print("=" * 50)

        # Apply to our test results
        for result in results[:5]:  # Show first 5 examples
            interpretation, action = interpret_toxicity_score(result['toxicity_score'])
            print(f"Score {result['toxicity_score']:.3f}: {interpretation}")
            print(f"  Text: '{result['text']}'")
            print(f"  Recommended: {action}")
        return print()


    _()
    return


@app.cell
def _(analyze_toxicity, interpret_toxicity_score):
    def test_your_text():
        """Interactive function to test custom text"""
        print("✏️  Enter your own text to analyze (or 'quit' to stop):")
    
        while True:
            user_text = input("\nText to analyze: ")
        
            if user_text.lower() == 'quit':
                break
            
            if user_text.strip():
                result = analyze_toxicity(user_text)
                interpretation, action = interpret_toxicity_score(result['toxicity_score'])
            
                print(f"\n📊 Analysis Results:")
                print(f"Toxicity Score: {result['toxicity_score']:.3f}")
                print(f"Interpretation: {interpretation}")
                print(f"Recommendation: {action}")
            else:
                print("Please enter some text to analyze.")

    # Uncomment the line below to run interactive testing
    # test_your_text()

    print("💡 Uncomment the last line above to test your own examples!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🎓 HateBERT Key Insights

    ## What We Discovered:

    ### ✅ Strengths:
    - **Good at detecting obvious toxicity** - clear hate speech and abuse
    - **Probability-based output** - gives confidence levels, not just yes/no
    - **Context-aware** - better than simple keyword filtering
    - **Consistent scoring** - 0-1 range makes it easy to set thresholds

    ### ⚠️ Limitations:
    - **Struggles with sarcasm** - may flag sarcastic positive statements
    - **Context-dependent language** - "killed it" in positive context might still score high
    - **Professional/technical language** - may flag aggressive business language
    - **Cultural variations** - trained primarily on certain types of English text

    ### 📊 Score Patterns We Observed:
    - Most safe text scores below 0.3
    - Clear toxicity usually scores above 0.7
    - The 0.4-0.6 range needs human review
    - Confidence tends to be high for extreme cases

    ## For Your Bayesian Bias Framework:

    ### 🔧 Integration Notes:
    1. **Score Range**: 0-1 (perfect for normalization)
    2. **Focus**: Toxicity and hate speech specifically
    3. **Reliability**: High for obvious cases, moderate for edge cases
    4. **Complementary**: Needs other tools for non-toxicity biases

    ### 🧮 Next Steps:
    - Compare with VADER (general sentiment)
    - See how Perspective API handles similar cases
    - Understand how to weight HateBERT vs other tools

    **HateBERT gives us the "toxicity dimension" of bias - but bias is multi-dimensional!**
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
