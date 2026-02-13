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
