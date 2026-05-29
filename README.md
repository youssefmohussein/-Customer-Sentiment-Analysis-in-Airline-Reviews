# ✈️ Customer Sentiment Analysis in Airline Reviews

> A comparative NLP & Machine Learning framework for classifying passenger sentiment across three real-world airline review datasets.

**Team 01 — Misr International University, Faculty of Computer Science**  
· Youssef Mohamed · Youssef Ahmed · Abdelrahman Montaser · Mariam Tamer · Jana Mohamed

---

## 📌 Project Overview

This project builds an end-to-end **sentiment classification pipeline** for airline passenger reviews. Raw, unstructured text from three Kaggle datasets is cleaned, vectorized using TF-IDF, and fed into five classical machine learning classifiers. The goal is to determine whether a review is **Positive**, **Neutral**, or **Negative** — at scale, without the computational cost of deep learning.

The paper was written and typeset in **Overleaf (LaTeX)**, experiments were run in **Google Colab**, and results are hosted on a **project website** showcasing the best-performing model.

---




## 📊 Datasets

All three datasets were sourced from **Kaggle**:

| # | Dataset | Airlines Covered | Size (approx.) |
|---|---------|-----------------|----------------|
| 1 | [Skytrax User Reviews](https://www.kaggle.com) | 362 airlines | 41,396 reviews |
| 2 | British Airways Reviews | British Airways | ~3,000 reviews |
| 3 | Global Airline Reviews | Worldwide | ~6,000 reviews |

> **Note:** Due to Kaggle terms, raw data files are not included. Download them via KaggleHub or the links above and place CSV files in the `data/` folder.

Key features used: `review_text`, `seat_comfort`, `cabin_staff_service`, `food_beverages`, `value_for_money`, `recommended`.

---

## 🔧 NLP Pipeline

Each dataset passes through the same preprocessing pipeline before model training:

```
Raw Text
   │
   ▼
Lowercase Conversion
   │
   ▼
Noise Removal  (URLs, emails, numbers, special characters — via regex)
   │
   ▼
Tokenization  (NLTK word_tokenize)
   │
   ▼
Stopword Removal  (with Negation Retention — "not good", "never again" preserved)
   │
   ▼
Lemmatization  (WordNet Lemmatizer)
   │
   ▼
Comprehensive Text Cleaning  (rejoin tokens → cleaned review column)
   │
   ▼
TF-IDF Vectorization  (unigrams + bigrams)
   │
   ▼
ANOVA F-test Feature Selection
   │
   ▼
Sentiment Label Encoding  (LabelEncoder → 0=Negative, 1=Neutral, 2=Positive)
```

**Sentiment labeling** was derived from average sub-ratings:
- Dataset 1 (Skytrax): average of 5 service rating categories
- Dataset 2 (British Airways): average of 4 categories (Seat Comfort, Cabin Staff, Food, Value)
- Dataset 3 (Global): average of 3 categories (Seat Comfort, Cabin Staff, Value)

---

## 🤖 Models Evaluated

| Model | Type | Notes |
|-------|------|-------|
| **Logistic Regression** ⭐ | Linear | Best overall F1; fast, interpretable |
| **SGD Classifier** | Linear | Fastest; best raw accuracy on Dataset 3 |
| **Random Forest** | Ensemble | Robust to noise; higher compute cost |
| **Naïve Bayes** | Probabilistic | Strong baseline for sparse TF-IDF features |
| **Decision Tree** | Tree-based | Lowest performance; prone to overfitting |

Train/Test split: **80% / 20%**

---

## 📈 Results Summary

### Dataset 1 — Skytrax (41,396 reviews)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.6942** | **0.6762** | **0.6942** | **0.6804** |
| SGD Classifier | 0.6700 | 0.6790 | 0.6700 | 0.6583 |
| Naive Bayes | 0.6487 | 0.6206 | 0.6487 | 0.6154 |
| Random Forest | 0.6465 | 0.6239 | 0.6465 | 0.5607 |
| Decision Tree | 0.6000 | 0.5694 | 0.6000 | 0.5740 |

### Dataset 2 — British Airways

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.6509** | **0.6429** | **0.6509** | **0.6437** |
| SGD Classifier | 0.6225 | 0.6172 | 0.6225 | 0.6192 |
| Naive Bayes | 0.6279 | 0.6022 | 0.6279 | 0.5963 |
| Random Forest | 0.6238 | 0.6054 | 0.6238 | 0.5852 |
| Decision Tree | 0.5467 | 0.5322 | 0.5467 | 0.5247 |

### Dataset 3 — Global Airlines

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SGD Classifier | **0.7534** | 0.7027 | 0.7534 | 0.7043 |
| **Logistic Regression** | 0.7527 | **0.7132** | 0.7527 | **0.7189** |
| Naive Bayes | 0.7397 | 0.6956 | 0.7397 | 0.6828 |
| Random Forest | 0.7308 | 0.5946 | 0.7308 | 0.6545 |
| Decision Tree | 0.7007 | 0.6567 | 0.7007 | 0.6678 |

### 🏆 Best Model: Logistic Regression
- Most **consistent** F1-Score across all three datasets
- Best balance between precision and recall
- Low computational overhead — runs in seconds
- Highly interpretable (coefficients map directly to TF-IDF word weights)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud kagglehub
```

Download NLTK resources (run once):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running the Notebooks

1. Open any notebook in `notebooks/` via **Google Colab**
2. Mount your Google Drive or upload datasets to `/content/data/`
3. Run all cells top-to-bottom — preprocessing, training, and evaluation are fully sequential

### Quick Inference with the Best Model

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model and vectorizer
with open('models/logistic_regression_best.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

def predict_sentiment(review_text: str) -> str:
    cleaned = preprocess(review_text)          # apply your NLP pipeline
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[prediction]

# Example
print(predict_sentiment("The cabin crew was incredibly friendly and the seat was comfortable."))
# → Positive

print(predict_sentiment("Flight delayed by 3 hours. Staff were rude and unhelpful."))
# → Negative
```

---

## 🌐 Website

The project website provides an interactive demo of the **Logistic Regression** model. Users can paste any airline review and receive an instant sentiment prediction.

To run locally:

```bash
cd website
# No build step needed — open index.html directly in your browser
open index.html
```

The website includes:
- **Live sentiment prediction** (calls the deployed model API)
- **Results dashboard** with all model comparison charts
- **Dataset explorer** with word clouds and bigram visualizations
- **About the team** section

---

## 📝 Paper

The full research paper was typeset in **Overleaf (LaTeX)**. It is available in `paper/` as a compiled PDF and full source.

To recompile:
1. Upload the `paper/` folder to [Overleaf](https://www.overleaf.com)
2. Set compiler to **pdfLaTeX**
3. Click **Recompile**

Citation (BibTeX):
```bibtex
@article{andelmoneim2025airline,
  title     = {Customer Sentiment Analysis in Airline Reviews: A Comparative Analysis
               of Optimized Machine Learning and NLP Techniques},
  author    = {Andelmoneim, Diaa and Mohamed, Youssef and Ahmed, Youssef and
               Montaser, Abdelrahman and Tamer, Mariam and Mohamed, Jana},
  journal   = {Misr International University -- Faculty of Computer Science},
  year      = {2025}
}
```

---

## 🔑 Key Findings

- **Logistic Regression** is the most reliable classifier for airline review sentiment across diverse datasets and review sources.
- The **Neutral class** is the hardest to predict in all experiments — a known challenge in 3-class sentiment tasks.
- **Negation-aware stopword removal** (preserving "not good", "never again") meaningfully improves classification.
- **Bigrams** (e.g., "great service", "flight delayed") capture sentiment signals that unigrams alone miss.
- Traditional ML + TF-IDF delivers competitive results compared to BERT/LSTM at a fraction of the compute cost.

---

## 🔭 Future Work

- [ ] Fine-tune **BERT** / **RoBERTa** on domain-specific airline review data
- [ ] Apply **SMOTE** to address class imbalance, especially for the Neutral class
- [ ] Extend to **aspect-based sentiment analysis** (e.g., separate scores for seat comfort vs. staff)
- [ ] Build a real-time airline feedback monitoring dashboard
- [ ] Explore multilingual sentiment analysis for non-English reviews

---

## 👥 Team

| Name | Student ID | Email |
|------|-----------|-------|
| Youssef Mohamed Hussein | 23038201 | youssef23038201@miuegypt.edu.eg |
| Youssef Ahmed | 23077362 | youssef23077362@miuegypt.edu.eg |
| Abdelrahman Montaser | 23049973 | abdelrhman23049973@miuegypt.edu.eg |
| Mariam Tamer | 23012524 | mariam23012524@miuegypt.edu.eg |
| Jana Mohamed | 23080055 | jana23080055@miuegypt.edu.eg |

**Supervisor:** Dr. Diaa Andelmoneim

---

## 📄 License

This project is submitted as academic work for Misr International University. All code is available for educational and research use. Datasets are subject to their original Kaggle licenses.

---

<div align="center">
  <sub>Built with ❤️ at Misr International University · Cairo, Egypt · 2025</sub>
</div>