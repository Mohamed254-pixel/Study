import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Data source: IBM Telco Customer Churn Dataset
# Source: IBM GitHub
# Contains 7,043 customer records

df = pd.read_csv(
    "ml-basics-assignment/ML/data/Telco-Customer-Churn.csv"
)


# Convert Churn from Yes/No to 1/0
df["churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# Select features
X = df[
    [
        "tenure",
        "MonthlyCharges",
        "Contract",
        "InternetService"
    ]
]

y = df["churn"]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            ["tenure", "MonthlyCharges"]
        ),
        (
            "cat",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            ),
            ["Contract", "InternetService"]
        )
    ]
)


# Create pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Create a new customer
new_customer = pd.DataFrame({
    "tenure": [12],
    "MonthlyCharges": [75],
    "Contract": ["Month-to-month"],
    "InternetService": ["Fiber optic"]
})


# Predict churn probability
churn_probability = model.predict_proba(new_customer)[0][1]


# Classify using 0.5 threshold
threshold = 0.5

churn_prediction = (
    1 if churn_probability > threshold else 0
)


print(
    f"Churn Probability: "
    f"{churn_probability:.2f}"
)

print(
    f"Churn Prediction "
    f"(1 = churn, 0 = no churn): "
    f"{churn_prediction}"
)

# Display model coefficients
feature_names = (
    ["tenure", "MonthlyCharges"] +
    model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(
        ["Contract", "InternetService"]
    ).tolist()
)

coefficients = model.named_steps["classifier"].coef_[0]

print("\nModel Coefficients:")

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.2f}")