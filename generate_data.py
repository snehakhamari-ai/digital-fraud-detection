# Step 1: create a fake (synthetic) payment dataset

import numpy as np          # for random numbers
import pandas as pd         # for tables (DataFrames)

rng = np.random.default_rng(42)   # random generator; 42 makes results repeatable
N = 20000                          # number of transactions

df = pd.DataFrame({
    "txn_id": np.arange(1, N + 1),                         # 1, 2, 3 ... 20000
    "customer_id": rng.integers(1, 1501, N),               # 1500 different customers
    "amount": np.round(rng.lognormal(6.0, 1.0, N), 2),     # many small, few large payments
    "hour": rng.integers(0, 24, N),                        # hour of the day, 0-23
    "merchant_type": rng.choice(
        ["grocery", "fuel", "food", "shopping", "travel", "gaming", "crypto"], N,
        p=[.25, .15, .2, .2, .08, .07, .05]),              # p = how common each type is
    "channel": rng.choice(["UPI", "card", "wallet"], N, p=[.55, .3, .15]),
    "location_mismatch": rng.choice([0, 1], N, p=[.93, .07]),   # 1 = unusual location
    "new_device": rng.choice([0, 1], N, p=[.92, .08]),          # 1 = new device
    "txns_last_hour": rng.poisson(1.0, N),                      # payments in last hour
})

# Feature 1: how big is this payment compared with the customer's average?
cust_avg = df.groupby("customer_id").amount.transform("mean")
df["amount_ratio"] = (df.amount / cust_avg).round(2)

# Feature 2: is this the first payment to this payee?
df["is_new_payee"] = rng.choice([0, 1], N, p=[.8, .2])

# Decide which payments are fraud: risky behaviour raises the chance
z = (-7.7 + 0.35 * df.amount_ratio + 2.6 * df.location_mismatch
     + 2.4 * df.new_device + 1.6 * df.is_new_payee
     + 0.8 * df.txns_last_hour + 1.5 * df.hour.isin([0, 1, 2, 3])
     + 1.4 * df.merchant_type.isin(["crypto", "gaming"]))
df["is_fraud"] = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)   # 1 = fraud

df.to_csv("transactions.csv", index=False)    # save as a file
print(df.shape, "fraud rate:", round(df.is_fraud.mean() * 100, 2), "%")