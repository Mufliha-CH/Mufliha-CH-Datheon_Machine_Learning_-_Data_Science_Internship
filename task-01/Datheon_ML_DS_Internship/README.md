# Datheon Machine Learning & Data Science Internship: Task 1

## Multi-Class Benchmark Classification (Iris)

**Goal:** Build a supervised classifier on the Iris dataset and benchmark Logistic Regression against a Decision Tree.

## Structure
```
.
├── README.md
├── requirements.txt
└── Task1_Iris_Classification/
    ├── Task_01.ipynb
    └── iris.csv
```

## Approach
1. Loaded the Iris dataset with Pandas and explored it with a Seaborn pairplot.
2. Split the data 80/20 (stratified) into training and test sets.
3. Scaled features with `StandardScaler` inside a Scikit-Learn `Pipeline`, so the scaler is fitted only on training data.
4. Trained `LogisticRegression` and `DecisionTreeClassifier`.
5. Evaluated accuracy, precision, recall and F1-score; plotted confusion matrix heatmaps and decision boundaries (petal length vs petal width).

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

Setosa is perfectly separated; all errors are between versicolor and virginica, whose measurements overlap. With only 30 test samples, the tie is not strong evidence that the models are equal.

**Tools:** Python, Scikit-Learn, Pandas, Seaborn, Matplotlib

## How to run
```bash
pip install -r requirements.txt
cd Task1_Iris_Classification
jupyter notebook Task_01.ipynb
```
