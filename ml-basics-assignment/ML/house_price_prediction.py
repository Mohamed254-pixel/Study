import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Data source: Ames Housing Dataset from OpenML
# OpenML ID: 42165
# https://www.openml.org/d/42165
# Contains 1,460 house records

housing = fetch_openml(name="house_prices", as_frame=True)


# Select the columns needed for this assignment
df = housing.frame[
    ["GrLivArea", "Neighborhood", "SalePrice"]
].copy()


# Rename columns to make them easier to understand
df = df.rename(columns={
    "GrLivArea": "square_footage",
    "Neighborhood": "location",
    "SalePrice": "price"
})


# Remove any missing values
df = df.dropna()


# Show how much data we are using
print("Number of houses:", len(df))
print(df.head())


# Features and target
X = df[["square_footage", "location"]]
y = df["price"]


# Convert the location column into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            ),
            ["location"]
        )
    ],
    remainder="passthrough"
)


# Create pipeline with preprocessing and Linear Regression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


# Predict the price of a new 2000 sq ft house
new_house = pd.DataFrame({
    "square_footage": [2000],
    "location": ["NAmes"]
})


predicted_price = model.predict(new_house)


print(
    f"\nPredicted price for a 2000 sq ft house in NAmes: "
    f"${predicted_price[0]:,.2f}"
)


# Get feature names
feature_names = (
    model.named_steps["preprocessor"]
    .named_transformers_["location"]
    .get_feature_names_out(["location"])
).tolist() + ["square_footage"]


# Get model coefficients
coefficients = model.named_steps["regressor"].coef_


# Display model coefficients
print("\nModel Coefficients:")

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.2f}")