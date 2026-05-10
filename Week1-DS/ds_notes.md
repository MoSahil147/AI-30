df.info() → tells you the structure — column names, data types, how many non-null values. Answers: "what does my data look like and is anything missing?"

df.describe() → tells you statistics — mean, min, max, standard deviation for numeric columns. Answers: "what are the numbers doing?"

df.info()
## Output:
## age      1000 non-null int64
## city      987 non-null object   ← 13 missing values!
## salary   1000 non-null float64

df.describe()
## Output:
##         age    salary
## mean    34.2   65000
## min     18     22000
## max     67     180000
## std     8.4    21000

df.info()     = structure, dtypes, missing values
df.describe() = statistics, mean/min/max/std
First thing on ANY new dataset: run both

Pandas: Provides high-performance, easy-to-use data structures like DataFrames for data manipulation and analysis.

NumPy: The foundation for numerical computing, offering powerful N-dimensional array objects and complex mathematical functions.

Matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python.

Seaborn: A statistical data visualization library based on Matplotlib that provides a high-level interface for drawing attractive graphics.

Scikit-learn: The go-to library for machine learning, featuring simple and efficient tools for predictive data analysis and modeling.

Jupyter: An open-source web application that allows you to create and share documents containing live code, equations, and narrative text.

Q1: 177/891 = roughly 20% missing. That's significant — you can't just ignore it. But you also can't drop the column because Age is probably important for survival prediction. So you fill it with mean or median. That's called imputation.
Q2: Correct — drop it. 687/891 = 77% missing. A column that's mostly empty tells you nothing useful. Dropping is the right call.
Q3: Close but backwards. Mean of 0.38 means 38% survived, 62% died. Since values are only 0 or 1, the mean literally = the survival rate. Smart way to read binary columns.

EDA First Steps on any dataset:
1. df.shape      - how big is it?
2. df.info()     - structure, dtypes, missing values
3. df.describe() - statistics
4. df.isnull().sum() - missing value count

Missing value rules:
- <20% missing → impute (fill with mean/median)
- >50% missing → drop the column
- Binary columns: mean = rate (0.38 = 38% are 1s)