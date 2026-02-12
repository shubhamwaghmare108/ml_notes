import numpy as np
import pandas as pd

np.random.seed(42)

# Number of customers
n = 5000

# Features
tenure = np.random.randint(1, 72, n)

monthly_charges = np.random.uniform(20, 120, n)

total_charges = tenure * monthly_charges + np.random.normal(0, 50, n)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    n,
    p=[0.6, 0.25, 0.15]
)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"],
    n,
    p=[0.4, 0.4, 0.2]
)

support_calls = np.random.poisson(2, n)

payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    n
)

# Churn logic (realistic)
churn_prob = (
    0.3
    + 0.002 * monthly_charges
    - 0.003 * tenure
    + 0.05 * support_calls
)

# Contract impact
churn_prob += np.where(contract == "Month-to-month", 0.25, -0.1)

# Fiber users churn slightly more
churn_prob += np.where(internet_service == "Fiber optic", 0.1, 0)

# Clip probabilities
churn_prob = np.clip(churn_prob, 0, 0.95)

churn = np.random.binomial(1, churn_prob)

# Build dataframe
df = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges.round(2),
    "total_charges": total_charges.round(2),
    "contract": contract,
    "internet_service": internet_service,
    "support_calls": support_calls,
    "payment_method": payment_method,
    "churn": churn
})

# Save dataset
df.to_csv("synthetic_churn.csv", index=False)

print("Dataset generated!")
print(df.head())
print("\nChurn Rate:", df['churn'].mean())
