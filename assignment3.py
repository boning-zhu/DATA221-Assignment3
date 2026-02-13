# QUESTIONS 1
import pandas as pd

# Load crime dataset
crime = pd.read_csv("crime1.csv")

# Select the column
col = crime["ViolentCrimesPerPop"]

# Compute statistics
print("Mean:", col.mean())
print("Median:", col.median())
print("Standard Deviation:", col.std())
print("Minimum:", col.min())
print("Maximum:", col.max())

# The mean is larger than the median, suggesting the distribution is slightly right-skewed.
# This indicates that there may be some higher extreme values.
# Extreme values affect the mean more than the median.
# The mean uses all values in its calculation.
# The median is more robust to outliers.


# QUESTION 2
import matplotlib.pyplot as plt

plt.hist(col, bins=20)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.close()

plt.boxplot(col)
plt.title("Box Plot of Violent Crimes Per Population")
plt.ylabel("ViolentCrimesPerPop")
plt.savefig("boxplot.png")
plt.close()

# The histogram shows how the crime values are distributed across different ranges.
# It helps us see whether the data is skewed or symmetric.
# The box plot shows the median as the line inside the box.
# The box represents the interquartile range (IQR).
# If there are points outside the whiskers, they indicate potential outliers.
# The box plot provides a clear summary of spread and extreme values.


#QUESTION 3
from sklearn.model_selection import train_test_split

kidney = pd.read_csv("kidney_disease.csv")

print(kidney.columns)

X = kidney.drop("CKD", axis=1)
y = kidney["CKD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training size:", len(X_train))
print("Testing size:", len(X_test))

# We should not train and test on the same data because the model would memorize the training data.
# This would give an overly optimistic performance estimate.
# The testing set is used to evaluate how well the model generalizes to new, unseen data.
# It simulates real-world prediction performance.


#QUESTION 4

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# True Positive means correctly predicting a patient has kidney disease.
# True Negative means correctly predicting a patient is healthy.
# False Positive means predicting disease when the patient is actually healthy.
# False Negative means failing to detect kidney disease in a patient.
# Accuracy alone may not be enough because the dataset could be imbalanced.
# If missing kidney disease is very serious, recall is the most important metric.
# Recall measures how many actual disease cases were correctly identified.

#QUESTION 5

k_values = [1, 3, 5, 7, 9]
results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append((k, acc))

results_df = pd.DataFrame(results, columns=["k", "Accuracy"])
print(results_df)

best_k = results_df.loc[results_df["Accuracy"].idxmax()]
print("Best k:", best_k["k"])

# Changing k affects the complexity of the KNN model.
# Small values of k make the model sensitive to noise and may cause overfitting.
# Large values of k make the model smoother and may cause underfitting.
# Choosing an appropriate k balances bias and variance.
