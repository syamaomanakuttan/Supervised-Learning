# Supervised-Learning
Supervised Learning

## Supervised Learning
### 1. Loading and Preprocessing
#### Loading the Dataset

The breast cancer dataset can be loaded directly from the sklearn library:
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
#### Preprocessing

The preprocessing steps include handling missing values and feature scaling. In this dataset, there are no missing values,
but we'll still perform a check and apply feature scaling.
##### Handling Missing Values

# Check for missing values
print(X.isnull().sum().sum())
###### Since there are no missing values, we can skip imputation.
#### Feature Scaling
Feature scaling is necessary for many machine learning algorithms, especially those that rely on distance calculations, such as k-NN and SVM.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
### 2. Classification Algorithm Implementation
##### We will implement and describe each of the five classification algorithms.


##### 1. Logistic Regression
Description: Logistic Regression is a linear model used for binary classification. It uses a logistic function to model a binary dependent variable.
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_scaled, y)
##### 2. Decision Tree Classifier
Description: Decision Tree Classifier splits the dataset into subsets based on the value of the best features, 
chosen using metrics like Gini impurity or information gain.
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_scaled, y)
##### 3. Random Forest Classifier
Description: Random Forest is an ensemble method that combines multiple decision trees to improve accuracy and control overfitting.
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
forest.fit(X_scaled, y)
##### 4. Support Vector Machine (SVM)
Description: SVM finds the hyperplane that best separates the data into different classes. It is effective in high-dimensional spaces.
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_scaled, y)
##### 5. k-Nearest Neighbors (k-NN)
Description: k-NN is a non-parametric method that classifies a data point based on the majority class among its k-nearest neighbors.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_scaled, y)
### 3. Model Comparison
##### We will compare the performance of these models using cross-validation and accuracy as the evaluation metric.
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": tree,
    "Random Forest": forest,
    "Support Vector Machine": svm,
    "k-Nearest Neighbors": knn
}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"{name}: {scores.mean():.2f} ± {scores.std():.2f}")

#### Summary of Results
After running the above code, we'll get the average accuracy and standard deviation for each model. 
The model with the highest mean accuracy performs the best, while the one with the lowest mean accuracy performs the worst.
#### Conclusion
Based on the results, the Support Vector Machine (SVM) might perform the best with the highest mean accuracy, while the Decision Tree might perform the worst with the lowest mean accuracy.

This analysis gives an idea of which algorithm is most suitable for this dataset, considering both the performance and the characteristics of the data.

