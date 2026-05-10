import pandas as pd
import numpy as np

# load data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# look
print("SHAPE")
print(df.shape) # rows and cols

# first 5 rows
print("First 5 rows")
print(df.head())

# info vs descrive (infor to get information about the data, kya kya variable hai and all, desrive to get mean std and all)
print("INFO")
print(df.info())

# sttas
print("DESCRIBE")
print(df.describe())

# missing values
print("Missing Values")
print(df.isnull().sum())

## data cleaning
# 77% missing values, tpp much impute so dropped
df = df.drop(columns=["Cabin"])

# will fill the age with median better than mean whne data is skewed!
# 20% was missing data, so just added median values
df["Age"]=df["Age"].fillna(df["Age"].median())

# only 2 missing so safer to drop rows
df = df.dropna(subset=["Embarked"])
# no missing values remaining
print("Missing after cleaning")
print(df.isnull().sum())
print(f"\nRows remaining: {len(df)}")