import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Data source: IBM Telco Customer Churn Dataset
# Source: IBM GitHub
# Contains 7,043 customer records

df = pd.read_csv(
    "ml-basics-assignment/ML/data/Telco-Customer-Churn.csv"
)


# Convert TotalCharges to numbers
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Remove missing values
df = df.dropna(
    subset=["tenure", "MonthlyCharges", "TotalCharges"]
)


# Select features for clustering
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[features]


# Scale the data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Elbow Method
inertia = []

K = range(1, 6)

for k in K:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


# Create elbow graph
plt.figure(figsize=(8, 5))

plt.plot(K, inertia, "bo-")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")

plt.savefig(
    "ml-basics-assignment/ML/elbow_plot.png"
)

plt.close()


# Use 3 clusters
optimal_k = 2

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42
)


# Assign each customer to a cluster
df["cluster"] = kmeans.fit_predict(X_scaled)


# Show average characteristics of each cluster
cluster_summary = (
    df.groupby("cluster")[features]
    .mean()
    .round(2)
)

print("Cluster Characteristics:")

print(cluster_summary)


# Give a strategy for each cluster
for cluster in range(optimal_k):

    print(f"\nCluster {cluster} Strategy:")

    if cluster_summary.loc[cluster, "MonthlyCharges"] > 70:

        print(
            "High-paying customers: "
            "Offer loyalty rewards."
        )

    elif cluster_summary.loc[cluster, "tenure"] > 40:

        print(
            "Long-term customers: "
            "Offer renewal discounts."
        )

    else:

        print(
            "Newer customers: "
            "Send engagement offers."
        )


# Save customer clusters
df.to_csv(
    "ml-basics-assignment/ML/customer_segments.csv",
    index=False
)

print("\nCustomer segments saved to customer_segments.csv")
print("Elbow plot saved to elbow_plot.png")