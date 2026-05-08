# Cross-Selling Prediction

Predicts potential vehicle insurance buyers from a health insurance customer database. Compares multiple classifiers to identify the best model for targeting cross-sell campaigns.

## Dataset

- 102,351 records with customer attributes (age, vehicle age, annual premium, region, etc.)
- Binary target: interested in vehicle insurance (1) or not (0)
- 70/30 train/test split, stratified

## Pipeline

1. **EDA** — class distribution, null value handling, feature analysis
2. **Preprocessing** — one-hot encoding for categorical features, log transform for skewed `Annual_Premium`, StandardScaler for numerical features
3. **Dimensionality** — PCA evaluated but not applied (no sharp variance drop)
4. **Model comparison** — hyperparameter search via F1-score

## Results

| Model | AUC | Notes |
|-------|-----|-------|
| KNN (k=110) | 0.84 | Baseline |
| Naive Bayes | — | Lowest performance |
| Neural Network | ~0.85 | Similar to RF |
| **Random Forest** | **Best** | Highest F1-score — selected |

## Files

- `Insurance_Predict.ipynb` — full EDA, preprocessing, and model comparison notebook

## Stack

Python · scikit-learn · pandas · NumPy · seaborn · matplotlib
