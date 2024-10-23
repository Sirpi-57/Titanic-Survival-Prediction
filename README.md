# Titanic-Survival-Prediction
The project aims to predict the survival of passengers on the Titanic using machine learning algorithms, specifically Logistic Regression and K-Nearest Neighbors (KNN), to analyze key factors affecting survival and evaluate model performance.

**Project Workflow:**

**Data Understanding:**
Used the Titanic dataset to inspect survival distribution and feature correlations.

**Exploratory Data Analysis (EDA):**
Visualized survival rates by gender, passenger class, and other features.

**Data Preprocessing:**
Imputed missing values (Age, Embarked, and Fare) and dropped Cabin due to excessive missing data.
Created a Family column by combining SibSp and Parch.
Grouped Age into categories and applied one-hot encoding to categorical variables (Sex, Embarked, age_categories).

**Modeling & Results:**

**Logistic Regression:**

**Accuracy: 80.2%**
Precision/Recall (Survival): 0.79/0.69
ROC AUC: 0.83

**K-Nearest Neighbors (KNN):**

Optimal k: 21 (selected via cross-validation)
**Accuracy: 78.3%**
Precision/Recall (Survival): 0.77/0.65
ROC AUC: 0.81

**Optimal Cutoff (Logistic Regression):**
Chose 0.4 threshold for better recall, achieving Accuracy: 78.1%, Precision/Recall: 0.74/0.72.

**Model Evaluation:**

Confusion matrices and classification reports were generated for both models.
ROC curves plotted for model comparison.

**Key Concepts**

**Logistic Regression:** A classification algorithm used to predict survival probabilities.

**K-Nearest Neighbors (KNN):** A distance-based classifier used for predicting survival based on the majority class of nearest neighbors.

**ROC Curve & AUC:** Used to evaluate the trade-off between sensitivity and specificity.

**Cross-Validation:** Used to fine-tune model hyperparameters and prevent overfitting.

**Deliverables**

**Predictive Models:** Logistic Regression and KNN with respective optimal parameters.

**Model Evaluation Metrics:** Accuracy, precision, recall, ROC AUC.

**Optimal K for KNN:** Selected via cross-validation (k = 21).

**Future Scope**

**Advanced Models:** Implement Random Forest, Gradient Boosting, or ensemble methods to improve performance.

**Feature Engineering:** Explore additional features such as family survival rates or advanced feature combinations.

**Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV for further model optimization.

**Results**

**Best Model:** Logistic Regression with an accuracy of 80.2% and ROC AUC of 0.83.

**Optimal K for KNN:** Found to be 21 using cross-validation, yielding an accuracy of 78.3%.
