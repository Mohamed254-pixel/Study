from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Path("charts").mkdir(exist_ok=True)


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]


df = pd.read_csv(
    Path(__file__).with_name("adult-all-1.csv"),
    names=columns,
    header=None,
    skipinitialspace=True
)


for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip().str.rstrip(".")


df = df.replace("?", np.nan)
df = df.dropna()
df = df.drop_duplicates()


print(df.head())
print(df.describe().round(2))
print(df["income"].value_counts())