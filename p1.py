# employee_attrition_analysis.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import warnings

# Set display options and warnings filter
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Load dataset
data = pd.read_csv('C:\\Users\\Pratiksha\\Desktop\\mscit project\\Employee-Attrition.csv')


# Check for duplicates and remove them
print(data.duplicated().value_counts())
data.drop_duplicates(inplace=True)

# Check for null values
print(data.isnull().sum())

# Plot attrition count
plt.figure(figsize=(15,5))
sns.countplot(y='Attrition', data=data)
plt.show()

# Plot attrition with respect to departments
plt.figure(figsize=(12,5))
sns.countplot(x='Department', hue='Attrition', data=data, palette='hot')
plt.title("Attrition w.r.t Department")
plt.show()

# Plot attrition with respect to education fields
plt.figure(figsize=(12,5))
sns.countplot(x='EducationField', hue='Attrition', data=data, palette='hot')
plt.title("Attrition w.r.t EducationField")
plt.xticks(rotation=45)
plt.show()

# Plot attrition with respect to job roles
plt.figure(figsize=(12,5))
sns.countplot(x='JobRole', hue='Attrition', data=data, palette='hot')
plt.title("JobRole w.r.t Attrition")
plt.xticks(rotation=45)
plt.show()

# Plot attrition with respect to gender
plt.figure(figsize=(12,5))
sns.countplot(x='Gender', hue='Attrition', data=data, palette='hot')
plt.title("Gender w.r.t Attrition")
plt.show()

# Plot age distribution
plt.figure(figsize=(12,5))
sns.distplot(data['Age'], hist=False)
plt.show()

# Map Education levels
edu_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
plt.figure(figsize=(12,5))
sns.countplot(x=data['Education'].map(edu_map), hue='Attrition', data=data, palette='hot')
plt.title("Education W.R.T Attrition")
plt.show()

# Encode target variable and other binary features
data['Attrition'] = data['Attrition'].replace({'No': 0, 'Yes': 1})
data['OverTime'] = data['OverTime'].map({'No': 0, 'Yes': 1})
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Label encode categorical features
encoding_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
label_encoders = {}
for column in encoding_cols:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Define features and target
X = data.drop(['Attrition', 'Over18'], axis=1)
y = data['Attrition'].values

# Handle class imbalance using RandomOverSampler
rus = RandomOverSampler(random_state=42)
X_over, y_over = rus.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions and evaluate the model
prediction = logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test, prediction)
print("Accuracy Score -", accuracy_score(y_test, prediction))

# Plot confusion matrix
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(1,2,1)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Expected')

# Plot ROC curve
ax2 = fig.add_subplot(1,2,2)
y_pred_proba = logreg.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, prediction)
plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.legend(loc=4)
plt.show()
