# IMDBSentiCompare 🎭

> A Comprehensive Sentiment Analysis Comparison Framework

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/weixiubo/IMDBSentiCompare?style=social)](https://github.com/weixiubo/IMDBSentiCompare/stargazers)

## 🌟 Overview

IMDBSentiCompare is a comprehensive sentiment analysis research platform that systematically compares three major approaches to sentiment classification on IMDB movie reviews. This project provides a fair and thorough evaluation of different methodologies, from traditional rule-based systems to cutting-edge large language models.

### 🎯 **Why IMDBSentiCompare?**

- **📊 Comprehensive Comparison**: Framework to systematically compare rule-based, traditional ML, and LLM approaches
- **🔬 Research-Grade**: Rigorous experimental design with proper statistical analysis
- **🛠️ Production-Ready**: Modular architecture suitable for both research and industry applications
- **📈 Reproducible Results**: Complete pipeline with fixed random seeds and detailed documentation

## 🚀 Key Features

### 🔧 **Rule-based Systems**
- **Multi-lexicon Fusion**: VADER + SentiWordNet + Custom dictionaries
- **Advanced Negation Handling**: Context-aware negation processing
- **Weighted Voting Mechanism**: Optimized combination of different sentiment sources

### 🤖 **Traditional Machine Learning**
- **Feature Engineering**: BOW, TF-IDF with 10,000-dimensional sparse matrices
- **Feature Selection**: Chi-squared and Mutual Information with 200/2000 feature subsets
- **Model Variety**: Naive Bayes, Logistic Regression with hyperparameter optimization

### 🧠 **Large Language Models**
- **Multi-Strategy Prompting**: Simple, Few-shot, Chain-of-Thought approaches
- **Structured Output**: JSON-formatted analysis with topic, emotion, and intensity
- **Robust Parsing**: Error-tolerant output processing with fallback mechanisms
- **Cost Optimization**: Smart sampling strategies for API efficiency

### 📊 **Advanced Analytics**
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Statistical Analysis**: SentiWordNet correlation studies
- **Rich Visualizations**: Confusion matrices, performance comparisons, distribution plots
- **Cross-Model Validation**: Comprehensive benchmarking across all approaches

## 📈 Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
|-------|----------|-----------|---------|----------|-------|
| **🥇 LLM (Simple)** | **94.5%** | **97.0%** | **92.4%** | **94.6%** | Best overall performance |
| **🥈 LLM (Few-shot)** | **93.5%** | **95.1%** | **92.4%** | **93.7%** | Consistent and reliable |
| **🥉 Chi² (2000 features)** | **88.3%** | **87.6%** | **89.3%** | **88.4%** | Best traditional ML |
| Mutual Info (2000) | 88.0% | 87.1% | 89.2% | 88.2% | Close second in ML |
| Logistic Regression (BOW) | 87.5% | 87.4% | 87.8% | 87.6% | Solid baseline |
| Logistic Regression (TF-IDF) | 86.9% | 84.7% | 90.3% | 87.4% | High recall |
| Naive Bayes | 85.5% | 86.5% | 84.3% | 85.4% | Fast and interpretable |
| Rule-based System | 69.2% | 64.9% | 83.9% | 73.2% | High recall, lower precision |
| LLM (CoT) | 53.5% | 53.0% | 100.0% | 69.3% | An expected normal situation |

> **Note**: LLM models tested on 200 samples due to API constraints; traditional models on full test set (~5000 samples)

## 🏗️ Project Architecture

```
IMDBSentiCompare/
├── 📁 Core Modules
│   ├── config.py              # Central configuration
│   ├── main.py               # Pipeline orchestrator
│   └── utils.py              # Shared utilities
├── 📊 Data Processing
│   ├── data_analysis.py      # EDA and statistics
│   └── data_preprocessing.py # Text cleaning & vectorization
├── 🔧 Classification Methods
│   ├── rule_based_classifier.py    # Multi-lexicon system
│   ├── ml_models.py                # Traditional ML
│   ├── feature_selection.py        # Advanced feature engineering
│   └── llm_classifier.py           # LLM integration
├── 📈 Evaluation & Analysis
│   └── evaluation.py         # Comprehensive evaluation suite
├── 📂 Data & Results
│   ├── Train.csv / Valid.csv / Test.csv
│   ├── models/               # Saved model artifacts
│   ├── results/              # Evaluation outputs
│   └── figures/              # Generated visualizations
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n senticompare python=3.9
conda activate senticompare

# Install dependencies
conda install pandas numpy scikit-learn nltk matplotlib seaborn tqdm openai
```

### Environment Setup

```bash
# For LLM experiments (optional)
export DEEPSEEK_API_KEY="your_api_key_here"

# Download NLTK resources (automatic on first run)
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger'])"
```

### Basic Usage

```bash
# 🎯 Run complete pipeline
python main.py --run-all

# 🔧 Run specific components
python main.py --run-preprocessing --run-ml --run-evaluation

# 📊 Individual model training
python ml_models.py --model nb                    # Naive Bayes
python rule_based_classifier.py                   # Rule-based system
python llm_classifier.py --sample-size 200        # LLM experiments
```

### Advanced Usage

```bash
# 🎛️ Custom feature selection
python feature_selection.py --features 500 1000 2000

# 🧠 LLM with different strategies
python llm_classifier.py --sample-size 100 --structured-sample-size 20

# 📈 Evaluation only
python evaluation.py --models-dir ./models --output-dir ./custom_results
```

## 📊 Detailed Results & Analysis

### 🎯 **Key Findings**

1. **LLM Superiority**: Simple prompting strategies outperform complex Chain-of-Thought
2. **Feature Selection Impact**: 2000 features provide optimal balance of performance and efficiency
3. **Rule-based Limitations**: High recall but lower precision due to lexicon coverage gaps
4. **Cost-Performance Trade-off**: Traditional ML offers 88%+ accuracy at fraction of LLM cost

### 📈 **Performance Insights**

- **Best Overall**: LLM Simple strategy (94.5% accuracy)
- **Best Traditional**: Chi-squared feature selection (88.3% accuracy)
- **Most Efficient**: Naive Bayes (85.5% accuracy, fastest training)
- **Highest Recall**: Rule-based system (83.9% recall)

## 🔬 Technical Highlights

### 🛠️ **Engineering Excellence**
- **Sparse Matrix Optimization**: Efficient handling of 10,000-dimensional features
- **Robust Error Handling**: Graceful degradation with comprehensive logging
- **Modular Design**: Easy to extend with new models or datasets
- **Reproducible Research**: Fixed random seeds and detailed configuration management

### 🧪 **Research Contributions**
- **Multi-Lexicon Fusion Algorithm**: Novel weighted voting for rule-based classification
- **LLM Prompt Engineering**: Systematic evaluation of different prompting strategies
- **Fair Comparison Framework**: Handling different sample sizes and evaluation constraints
- **Comprehensive Benchmarking**: Statistical analysis with correlation studies

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🎯 **Priority Areas**
- [ ] Additional LLM providers (OpenAI GPT, Anthropic Claude, Google Gemini)
- [ ] New feature engineering techniques (word embeddings, transformers)
- [ ] Alternative datasets (Amazon reviews, Twitter sentiment)
- [ ] Performance optimizations (GPU acceleration, distributed computing)
- [ ] Advanced evaluation metrics (BLEU, ROUGE for structured outputs)

### 📝 **How to Contribute**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Known Issues & Limitations

- **LLM Sample Size**: Due to the limitations of API costs and rates, it is difficult to achieve a test set of 5,000 samples in actual scenarios
- **CoT Parsing**: The chain-of-thought strategy is not applicable to non-reasoning models
- **Memory Usage**: Large feature matrices require a large amount of memory
- **API Dependencies**: LLM experiments require stable internet connection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IMDB** for providing the movie reviews dataset
- **NLTK Team** for comprehensive natural language processing tools
- **DeepSeek** for providing accessible LLM API
- **Scikit-learn** for robust machine learning implementations
- **Open Source Community** for inspiration and foundational tools

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/weixiubo/IMDBSentiCompare/issues)
- **Discussions**: [GitHub Discussions](https://github.com/weixiubo/IMDBSentiCompare/discussions)
- **Email**: 1494849734@qq.com

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

[🐛 Report Bug](https://github.com/weixiubo/IMDBSentiCompare/issues) • [✨ Request Feature](https://github.com/weixiubo/IMDBSentiCompare/issues) • [📖 Documentation](https://github.com/weixiubo/IMDBSentiCompare/wiki)

</div>
