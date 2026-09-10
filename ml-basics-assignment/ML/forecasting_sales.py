import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Data source: FRED
# Series: New One Family Houses Sold in the United States (HSN1F)
# Source: U.S. Census Bureau and HUD

df = pd.read_csv(
    "ml-basics-assignment/ML/data/housing_sales.csv"
)


# Rename columns
df = df.rename(columns={
    "observation_date": "date",
    "HSN1F": "sales"
})


# Convert date column
df["date"] = pd.to_datetime(df["date"])


# Remove missing sales values
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
df = df.dropna()


# Create month numbers for the model
df["month"] = range(1, len(df) + 1)


# Features and target
X = df[["month"]]
y = df["sales"]


# Train model
model = LinearRegression()
model.fit(X, y)


# Predict next 6 months
future_months = pd.DataFrame({
    "month": range(
        df["month"].max() + 1,
        df["month"].max() + 7
    )
})

predictions = model.predict(future_months)


# Create future dates
future_dates = pd.date_range(
    start=df["date"].max() + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)


# Print forecast
print("Next 6 Months Forecast:")

for date, prediction in zip(future_dates, predictions):
    print(f"{date.strftime('%B %Y')}: {prediction:.2f}")


# Plot results
plt.figure(figsize=(10, 5))

plt.plot(
    df["date"],
    df["sales"],
    label="Historical Sales"
)

plt.plot(
    future_dates,
    predictions,
    label="Predicted Sales",
    linestyle="--"
)

plt.xlabel("Date")
plt.ylabel("Home Sales")
plt.title("Housing Sales Forecast")
plt.legend()

plt.savefig(
    "ml-basics-assignment/ML/housing_sales_forecast.png"
)

plt.show()


# Assignment notes
print("\nAssumption:")
print("The model assumes the overall sales trend continues.")

print("\nChallenge:")
print("Housing sales can change because of the economy and interest rates.")

print("\nPotential Improvement:")
print("The model could include more factors and seasonal patterns.")