# **Titanic Survival Prediction Project**

### **Objective:**

This project aims to predict the survival of passengers aboard the Titanic using machine learning models. Various data preprocessing steps are applied to clean the dataset, and classification algorithms like Logistic Regression and K-Nearest Neighbors (KNN) are used to make predictions.

### **Dataset:**

* **Train dataset**: Used for training the model.  
* **Test dataset**: Used for testing the model. Both datasets come from the famous Titanic dataset, which contains passenger information, such as age, sex, fare, and survival status.

---

### **Approach:**

The project is divided into the following steps:

1. **Exploratory Data Analysis (EDA)**  
2. **Data Cleaning and Feature Engineering**  
3. **Model Building**  
4. **Evaluation**  
5. **Fine-Tuning and Cross-Validation**

---

## **1\. Exploratory Data Analysis (EDA):**

The project starts by exploring the data:

* **Survival Count**: The number of survivors and non-survivors is counted using `train.Survived.value_counts()`.  
* **Visualization**:  
  * The survival count is visualized using seaborn's `sns.countplot`.  
  * Gender-wise survival rate is also explored with `sns.countplot(x = train['Survived'], hue = train['Sex'])`.  
* **Survival Rate by Gender**:  
  * Calculations are done separately for males and females to determine survival rates based on gender.

---

## **2\. Data Cleaning and Feature Engineering:**

The Titanic dataset contains missing values and unnecessary columns. The following cleaning steps are applied:

* **Missing Values**:  
  * The `train.isnull().sum()` function identifies missing values.  
  * For the `Age` column, the missing values are filled based on the average age for each passenger class using the `add_age()` function.  
  * Missing values in the `Embarked` column are filled with the mode, and missing values in `Fare` are filled with the mean.  
* **Dropping Irrelevant Columns**:  
  * The `Cabin` column is dropped due to a high number of missing values, as well as the `PassengerId`, `Name`, `Ticket`, and other non-influential columns.  
* **Combining Family Features**:  
  * The `combine()` function creates a new column called `Family` by combining the `SibSp` and `Parch` columns.  
* **Binning Age into Categories**:  
  * The `process_age()` function is used to bin the age data into categories such as Infant, Child, Young Adult, Adult, etc.  
* **One-Hot Encoding**:  
  * The `one_hot_encoding()` function is applied to categorical variables like `Sex`, `Embarked`, and `age_categories` to convert them into numerical form.

---

## **3\. Model Building:**

### **Logistic Regression:**

* The primary model used in the project is **Logistic Regression**.  
* **Model Training**:  
  * The dataset is split into training and testing sets using `train_test_split`.  
  * The `LogisticRegression()` model is fitted on the training data.  
* **Model Evaluation**:  
  * The `confusion_matrix()` and `classification_report()` functions evaluate the model's performance.  
  * The **ROC curve** is plotted using `RocCurveDisplay` to measure the model's discrimination ability.  
* **ROC-AUC Score**:  
  * A custom `draw_roc()` function is implemented to visualize the ROC curve.  
* **Cutoff Threshold Analysis**:  
  * Various probability cutoff points are tested to evaluate accuracy, sensitivity, and specificity using `cutoff_df.plot.line()`.

### **K-Nearest Neighbors (KNN):**

* **Model Training**:  
  * The KNN model is fitted using `KNeighborsClassifier(n_neighbors=3)`.  
* **Optimal K Value**:  
  * The error rate for different K values is plotted, and the optimal K value is selected using cross-validation.

---

## **4\. Cross-Validation:**

* The project uses **cross-validation** to assess model performance more robustly.  
* The `cross_val_score()` function is used to perform 5-fold cross-validation on Logistic Regression and KNN models.

---

## **5\. Fine-Tuning:**

* After identifying the optimal K value for KNN, the model is re-trained using this value.  
* Both Logistic Regression and KNN models are compared using various performance metrics like accuracy, precision, recall, and F1-score.

---

## **Functions Used:**

1. **`add_age(cols)`**: Fills missing values in the `Age` column based on the average age of passengers in the same class.  
2. **`combine(df, col1, col2)`**: Combines `SibSp` and `Parch` columns to create a new `Family` column.  
3. **`process_age(df, cut_points, label_names)`**: Bins the `Age` column into specified categories.  
4. **`one_hot_encoding(df, column_name)`**: Performs one-hot encoding on categorical variables.  
5. **`draw_roc(actual, probs)`**: Plots the ROC curve and calculates the AUC score.

---

## **Results:**

1. Developed and **hyper-tuned** a **Logistic Regression model**, achieving an **accuracy** of **79%** and an **AUC score of 0.80** for predicting passenger survival.  
2. •Implemented **K-Nearest Neighbors (KNN)** for comparative analysis, yielding an **accuracy** of **80%** and an **AUC score of 0.83**.

---

## **Future Scope:**

* **Feature Engineering**: Additional features like family titles or cabin location could be engineered to improve model accuracy.  
* **Ensemble Learning**: Incorporating other machine learning algorithms (e.g., Random Forest, Gradient Boosting) to improve prediction accuracy.  
* **Hyperparameter Tuning**: Tuning hyperparameters for Logistic Regression and KNN to further optimize performance.

