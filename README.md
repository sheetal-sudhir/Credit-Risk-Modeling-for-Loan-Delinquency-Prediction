# Credit-Risk-Modeling-for-Loan-Delinquency-Prediction
This project builds a binary classification model to predict loan delinquency using real-world credit data with extreme class imbalance (~0.34% delinquent).

## Objective
Develop a credit risk model to identify potentially delinquent loans, enabling better risk management for financial institutions.

## Contents
- `credit_risk_modeling.ipynb`: Jupyter Notebook with full implementation
- Data preprocessing and EDA
- Handling class imbalance using **SMOTE combined with Tomek Links**
- Feature engineering and selection
- Model training using **Logistic Regression** and **Decision Tree**
- Evaluation using metrics suited for imbalanced classification

## Techniques Used
- Python (pandas, scikit-learn, imbalanced-learn)
- Binary classification
- **SMOTE + Tomek Links**: Oversampling the minority class and removing borderline examples to improve class separation
- Precision, Recall, F1-score, ROC-AUC, and PR AUC
- Confusion matrix analysis

## Challenges
- Severe class imbalance (0.34% positive class)
- Need for metrics beyond accuracy (e.g., recall and PR AUC)
- Managing noise and overlapping classes when oversampling

## Results
Using SMOTE with Tomek Links significantly improved recall on the minority class while reducing noise introduced by synthetic examples, resulting in a more robust model.

## Future Improvements
- Evaluate advanced ensemble methods (e.g., XGBoost, LightGBM)
- Perform feature importance analysis to improve interpretability
- Explore cost-sensitive learning approaches

---
