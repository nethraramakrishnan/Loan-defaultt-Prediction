import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score

df = pd.read_csv('loan_default_prediction_project.csv')

df.describe()
df.shape
df.isna().sum()
df.dropna(inplace  = True)
df[df.duplicated()]
df.Gender.value_counts()
# Plot to check the count of male and female customers
sns.countplot(data=df,x='Gender')
plt.show()

# Label encodingSuburban
df.replace({"Loan_Status":{'Default':0,'Non-Default':1}},inplace = True)
df.replace({"Gender":{'Male':0,'Female':1}},inplace = True)
df.replace({"Employment_Status":{'Unemployed':0,'Employed':1}},inplace = True)
df.replace({"Location":{'Rural':0,'Suburban':1,'Urban':2}},inplace = True)

df.head()
# Data Visualization Employment status Vs Loan Status
sns.countplot(x = 'Employment_Status',hue = 'Loan_Status',data = df)

# Data Visualization Gender Vs Loan Status
sns.countplot(x = 'Gender',hue = 'Loan_Status',data = df)

X = df.drop(columns = 'Loan_Status',axis = 1 )
Y = df['Loan_Status']
# print(X)
# print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Support Vector Machine Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Predictions
svm_test_prediction = classifier.predict(X_test)

# Metrics
svm_accuracy = accuracy_score(Y_test, svm_test_prediction)
svm_precision = precision_score(Y_test, svm_test_prediction)
svm_recall = recall_score(Y_test, svm_test_prediction)
svm_f1 = f1_score(Y_test, svm_test_prediction)
svm_classification_report = classification_report(Y_test, svm_test_prediction)

print("Support Vector Machine:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)
print("Classification Report:\n", svm_classification_report)

# Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, Y_train)
rf_test_prediction = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(Y_test, rf_test_prediction)
rf_precision = precision_score(Y_test, rf_test_prediction)
rf_recall = recall_score(Y_test, rf_test_prediction)
rf_f1 = f1_score(Y_test, rf_test_prediction)
rf_classification_report = classification_report(Y_test, rf_test_prediction)

print("\nRandom Forest:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print("Classification Report:\n", rf_classification_report)

# Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)
nb_test_prediction = nb_clf.predict(X_test)
nb_accuracy = accuracy_score(Y_test, nb_test_prediction)
nb_precision = precision_score(Y_test, nb_test_prediction)
nb_recall = recall_score(Y_test, nb_test_prediction)
nb_f1 = f1_score(Y_test, nb_test_prediction)
nb_classification_report = classification_report(Y_test, nb_test_prediction)

print("\nNaive Bayes:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print("Classification Report:\n", nb_classification_report)

# Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)
dt_test_prediction = dt_clf.predict(X_test)
dt_accuracy = accuracy_score(Y_test, dt_test_prediction)
dt_precision = precision_score(Y_test, dt_test_prediction)
dt_recall = recall_score(Y_test, dt_test_prediction)
dt_f1 = f1_score(Y_test, dt_test_prediction)
dt_classification_report = classification_report(Y_test, dt_test_prediction)

print("\nDecision Tree:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print("Classification Report:\n", dt_classification_report)

# K-Nearest Neighbors
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, Y_train)
kn_test_prediction = kn_clf.predict(X_test)
kn_accuracy = accuracy_score(Y_test, kn_test_prediction)
kn_precision = precision_score(Y_test, kn_test_prediction)
kn_recall = recall_score(Y_test, kn_test_prediction)
kn_f1 = f1_score(Y_test, kn_test_prediction)
kn_classification_report = classification_report(Y_test, kn_test_prediction)

print("\nK-Nearest Neighbors:")
print("Accuracy:", kn_accuracy)
print("Precision:", kn_precision)
print("Recall:", kn_recall)
print("F1 Score:", kn_f1)
print("Classification Report:\n", kn_classification_report)


# Define algorithms and their corresponding metrics
algorithms = ['SVM', 'Random Forest', 'Naive Bayes', 'Decision Tree', 'K-Nearest Neighbors']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Define metric values for each algorithm
accuracy_scores = [svm_accuracy, rf_accuracy, nb_accuracy, dt_accuracy, kn_accuracy]
precision_scores = [svm_precision, rf_precision, nb_precision, dt_precision, kn_precision]
recall_scores = [svm_recall, rf_recall, nb_recall, dt_recall, kn_recall]
f1_scores = [svm_f1, rf_f1, nb_f1, dt_f1, kn_f1]

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plotting accuracy
axes[0, 0].bar(algorithms, accuracy_scores, color='skyblue')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_ylim([0, 1])

# Plotting precision
axes[0, 1].bar(algorithms, precision_scores, color='lightgreen')
axes[0, 1].set_title('Precision')
axes[0, 1].set_ylim([0, 1])

# Plotting recall
axes[1, 0].bar(algorithms, recall_scores, color='salmon')
axes[1, 0].set_title('Recall')
axes[1, 0].set_ylim([0, 1])

# Plotting F1 score
axes[1, 1].bar(algorithms, f1_scores, color='gold')
axes[1, 1].set_title('F1 Score')
axes[1, 1].set_ylim([0, 1])

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


