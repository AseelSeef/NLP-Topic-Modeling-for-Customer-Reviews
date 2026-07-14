# NLP Topic Modeling for Customer Reviews

## Project Overview

This project demonstrates an end-to-end Natural Language Processing (NLP) pipeline for analysing customer reviews and extracting actionable insights from unstructured text.

The project focuses on identifying common discussion topics, understanding customer emotions, and generating business recommendations through a combination of traditional NLP techniques, topic modelling, transformer-based models, and Large Language Models (LLMs).

**Note:** The original dataset used in this project is proprietary and subject to a confidentiality agreement. Therefore, the dataset, outputs, and any company-specific information have been removed from this public repository.

---

## Objectives

- Preprocess and clean customer review text
- Explore frequently occurring terms using word frequency analysis and word clouds
- Discover hidden topics using BERTopic
- Compare BERTopic with Gensim LDA topic modelling
- Perform emotion analysis using a BERT-based transformer model
- Analyse highly negative reviews to identify key customer pain points
- Extract review topics using a Large Language Model (LLM)
- Generate actionable business recommendations from customer feedback

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- BERTopic
- Gensim (LDA)
- Hugging Face Transformers
- Ollama (Phi-4-mini)
- Matplotlib
- WordCloud

---

## Project Workflow

1. Data preprocessing
   - Remove missing values
   - Convert text to lowercase
   - Remove stopwords and numbers
   - Tokenise and lemmatise text

2. Exploratory text analysis
   - Word frequency analysis
   - Word cloud visualisation

3. Topic modelling
   - BERTopic
   - Topic visualisation
   - Topic similarity analysis
   - Comparison with Gensim LDA

4. Emotion analysis
   - BERT emotion classification
   - Distribution of customer emotions
   - Analysis of highly negative reviews

5. Large Language Models
   - Topic extraction using Phi-4-mini
   - Generation of actionable recommendations

---

## Repository Contents

```
.
├── nlp_topic_modeling_customer_reviews.py
├── README.md
└── requirements.txt
```

---

## Results

The project demonstrates how NLP and modern machine learning techniques can be combined to:

- Identify recurring themes within customer reviews
- Discover common customer concerns
- Understand emotional patterns in customer feedback
- Generate interpretable topics from large collections of text
- Produce data-driven recommendations for improving customer experience

The repository focuses on the implementation and methodology. All proprietary data, outputs, and identifying information have been removed to comply with confidentiality requirements.

---

## Confidentiality

The original dataset used for this project cannot be shared publicly because it is subject to a confidentiality agreement.

Only the source code and methodology are included in this repository. No proprietary data, review content, company identifiers, or generated outputs are distributed.

---

## Author

**Aseel Seef Azmy**

- LinkedIn: [https://www.linkedin.com/in/aseel-seef-ab1778178/](https://www.linkedin.com/in/aseel-seef-azmy/)
