# Breast Cancer Diagnosis Using Machine Learning

Machine Learning – Assignment 2
Name: SITARTH R | BITS ID: 2026aj05590

## a. Problem Statement

Early and accurate diagnosis of breast cancer matters because a tumour classified as benign when it is actually malignant can delay treatment. In this project, I used measurements taken from digitized images of breast mass cells to predict whether a tumour is malignant or benign.

I trained five classification models on the same dataset and compared them using six evaluation metrics. I also created a Streamlit application that loads the trained models and evaluates them on uploaded test data. The application does not retrain the models.

The target variable is `target`:

- `0`: The tumour is malignant.
- `1`: The tumour is benign.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository (also bundled with scikit-learn as `load_breast_cancer`)
- **Dataset link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains 569 records, with 30 numeric input features and one target column (`target`). The 30 features are the mean, the standard error and the worst (largest) value of ten measurements of the cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

There are 357 benign records (62.7%) and 212 malignant records (37.3%). Because the two classes are not perfectly balanced, I considered other metrics alongside accuracy.

### Data Preparation

The dataset has no missing values and no duplicate rows. I used an 80:20 stratified train-test split (`random_state=42`), giving 455 training records and 114 test records (72 benign and 42 malignant in the test set). The test set is saved as `test_data.csv`. Features were standardized with `StandardScaler` inside the pipelines for Logistic Regression, KNN and SVM; the Decision Tree and Gaussian Naive Bayes use the raw values. No hyperparameter tuning was done.

## c. GitHub Repository Link

- **GitHub repository:** https://github.com/rsitarth-coder/breast-cancer
- **Live Streamlit app:** https://breast-cancer-2dbdpr5ufkvbserd3vnohv.streamlit.app/

The project repository is intended to contain the following files:

```
breast-cancer/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── ML_Assignment_2.ipynb
└── model/
    └── saved trained models for all five classifiers (.joblib)
```

The saved models are complete pipelines, so the scaling step is included with the models that need it. Model filenames must match the paths used in `app.py`.

## d. Models Used

I implemented the five classification models required for this assignment:

1. **Logistic Regression:** Uses a linear relationship to estimate the probability that a tumour is benign.
2. **Decision Tree:** Makes predictions by learning a series of decision rules from the data.
3. **K-Nearest Neighbors (KNN):** Predicts based on the five most similar records in the training data.
4. **Gaussian Naive Bayes:** Uses probabilities, assuming each feature is normally distributed, to classify each record.
5. **Support Vector Machine (SVM):** Learns an RBF-kernel boundary to separate benign and malignant tumours.

### Model Comparison Table

These values are from the saved results in my `ML_Assignment_2.ipynb` notebook (test set of 114 records).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes (Gaussian) | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| SVM (RBF) | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Evaluation Metrics

- **Accuracy:** How often the model makes the correct prediction overall.
- **AUC:** How well the model separates benign and malignant tumours across different thresholds.
- **Precision:** Out of the tumours predicted as benign, how many were actually benign.
- **Recall:** Out of the tumours that were actually benign, how many the model identified.
- **F1 Score:** A combined measure of precision and recall.
- **MCC:** A measure of prediction quality that considers all four outcomes in the confusion matrix; useful when the classes are not perfectly balanced.

### Observations About Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It had the highest AUC (0.9954) and, together with SVM, the highest accuracy (0.9825), precision, recall, F1 and MCC (0.9623). It made only two mistakes on the 114 test records: one malignant tumour predicted as benign and one benign tumour predicted as malignant. |
| Decision Tree | It performed worst, with accuracy 0.9123, AUC 0.9157 and MCC 0.8174. It made ten mistakes, including three malignant tumours predicted as benign, which is the costlier kind of error in diagnosis. |
| KNN | It reached 0.9561 accuracy and 0.9722 recall with five mistakes. Scaling the features before computing distances was important for this model. |
| Naive Bayes (Gaussian) | It achieved 0.9386 accuracy and a high AUC of 0.9878, but made seven mistakes. Its assumption that features are independent is weak here because many of the measurements are strongly correlated. |
| SVM (RBF) | It matched Logistic Regression on accuracy (0.9825), F1 (0.9861) and MCC (0.9623), with a marginally lower AUC (0.9950). |
| **Overall Winner** | Logistic Regression was the best choice for this dataset: it had the highest MCC and AUC, with SVM tied on the other metrics. I considered F1 and MCC alongside accuracy because the classes are not perfectly balanced. |

### Results and Best-Model Observation

- Highest accuracy: Logistic Regression and SVM (0.9825)
- Highest AUC: Logistic Regression (0.9954)
- Highest precision: Logistic Regression and SVM (0.9861)
- Highest recall: Logistic Regression and SVM (0.9861)
- Highest F1: Logistic Regression and SVM (0.9861)
- Highest MCC: Logistic Regression and SVM (0.9623)

Logistic Regression was the strongest overall model in my test results, with SVM almost identical. The Decision Tree was clearly the weakest. The test set has only 114 records, so a difference of one or two predictions moves the scores noticeably and small gaps between models should be read with care.

## Streamlit Application

The application, titled Breast Cancer Diagnostic Evaluation, lets users choose one of the five saved models (the active classifier) in the sidebar and upload a labeled test CSV that contains the target column. The main page also shows a pre-computed baseline comparison table of all five models on the six evaluation metrics. The application uses the saved trained models without retraining.

To run the application locally:

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## What I Learned

Assignment helped me understand how different classification algorithms perform on the same data. I found that scaling matters for distance-based and margin-based models such as KNN and SVM, and that a simple linear model like Logistic Regression can match a more complex one on a well-behaved dataset. On a small test set, every single prediction moves the metrics, so I looked at the confusion matrices and not only at accuracy. I also learned how to build a Streamlit application and deploy it.

## Links

- **GitHub repository:** https://github.com/rsitarth-coder/breast-cancer
- **Live Streamlit app:** https://breast-cancer-2dbdpr5ufkvbserd3vnohv.streamlit.app/
- **Dataset:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
