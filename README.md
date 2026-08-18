# MLOps_Training

## In order to setup this project on your local machine :
* ### Requirements :
    - Python installed
    - DVC installed
    - Git installed
EDA notebook: Proper datetime feature engineering, percentile analysis, and multiple thoughtful distribution comparisons (amount vs. fraud, gender vs. fraud, category vs. fraud) — with real written interpretation of results, not just charts.
ML modeling notebook: Uses SMOTE to handle class imbalance — the correct, non-beginner technique for fraud detection specifically. Compares Logistic Regression vs. Random Forest. Uses precision/recall/F1, not naive accuracy — and correctly reasons about why recall matters more than precision for fraud detection. Uses MLflow autologging for experiment tracking.
Architecture: Separate CI, CT (Continuous Training), and CD Jenkins pipelines is a genuinely sophisticated MLOps pattern — most junior candidates only ever do CI/CD for code, not automated retraining pipelines. Real DVC integration for data versioning, pushing to GitHub as remote storage.
