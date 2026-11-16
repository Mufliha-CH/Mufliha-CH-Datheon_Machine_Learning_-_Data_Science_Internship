# Datheon Machine Learning & Data Science Internship: Task 4

## NLP Sentiment Classification on Customer Reviews (TF-IDF)

**Goal:** Classify movie reviews as positive or negative using TF-IDF features, Multinomial Naive Bayes and Logistic Regression.

## Structure
```
.
├── README.md
├── requirements.txt
└── Task4_Sentiment_Classification/
    ├── Task_04.ipynb
    └── movie_reviews.csv
```

## Approach
1. Loaded 2,000 labelled movie reviews (1,000 positive, 1,000 negative) from the NLTK `movie_reviews` corpus.
2. Cleaned the text: removed HTML tags and punctuation, lowercased, tokenized, removed stopwords (keeping negations such as "not"), and lemmatized with NLTK.
3. Split 80/20 (stratified) and converted text to numerical vectors with `TfidfVectorizer(max_features=5000)`, fitted on training data only.
4. Trained `MultinomialNB` and `LogisticRegression`.
5. Evaluated accuracy, classification reports and confusion matrices; plotted the most influential words.
6. Built a `predict_sentiment()` function (plus an interactive `input()` helper) to test custom phrases.

## Results

| Model | Test accuracy |
|---|---|
| Multinomial Naive Bayes | 0.8025 |
| Logistic Regression | 0.8500 |

Limitation: TF-IDF ignores word order, so negation ("not good") can be misclassified. Bigrams or a transformer model would improve this.

**Tools:** Python, NLTK, Scikit-Learn (TfidfVectorizer, Naive Bayes), Pandas, Seaborn, Matplotlib

## How to run
```bash
pip install -r requirements.txt
cd Task4_Sentiment_Classification
jupyter notebook Task_04.ipynb
```
