# %% [markdown]
# # HelioLead AI - EDA & Preprocessing
# Open this file in VS Code with the Python extension. Each `# %%` block is a
# runnable cell (like a Jupyter notebook) - click "Run Cell" above each block,
# or run the whole file top to bottom.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/heliolead_solar_leads.csv")
df.shape

# %%
df.info()
df.describe(include="all").T

# %%
df.isnull().sum()

# %% [markdown]
# ## Class balance

# %%
fig, ax = plt.subplots(figsize=(5, 4))
df["adopted_solar"].value_counts(normalize=True).plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("Solar Adoption Class Balance")
ax.set_xticklabels(["Not Adopted (0)", "Adopted (1)"], rotation=0)
plt.tight_layout()
plt.savefig("figures/class_balance.png", dpi=150)
plt.show()

# %% [markdown]
# ## Correlation between numeric features and adoption

# %%
numeric_cols = [
    "avg_sunlight_hours_per_day", "roof_area_sqft", "household_income_annual_inr",
    "monthly_electricity_bill_inr", "property_age_years", "household_size",
    "has_ev", "previous_solar_inquiry", "state_subsidy_pct", "adopted_solar",
]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png", dpi=150)
plt.show()

# %% [markdown]
# ## Categorical drivers: orientation, ownership, state

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col in zip(axes, ["roof_orientation", "home_ownership", "state"]):
    rate = df.groupby(col)["adopted_solar"].mean().sort_values(ascending=False)
    rate.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title(f"Adoption Rate by {col}")
    ax.set_ylabel("Adoption rate")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("figures/categorical_adoption_rates.png", dpi=150)
plt.show()

# %% [markdown]
# ## Sunlight hours vs. income, colored by adoption

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=df.sample(1500, random_state=1),
    x="avg_sunlight_hours_per_day", y="household_income_annual_inr",
    hue="adopted_solar", alpha=0.5, palette={0: "#4C72B0", 1: "#DD8452"}, ax=ax,
)
ax.set_yscale("log")
ax.set_title("Sunlight Hours vs Household Income (log scale)")
plt.tight_layout()
plt.savefig("figures/sunlight_vs_income.png", dpi=150)
plt.show()

# %% [markdown]
# ## Feature engineering
# - One-hot encode categoricals (roof_orientation, home_ownership, state)
# - Derive `bill_to_income_ratio` - a common domain feature: households
#   spending a larger share of income on electricity have a stronger financial
#   incentive to switch to solar.

# %%
df["bill_to_income_ratio"] = (df["monthly_electricity_bill_inr"] * 12) / df["household_income_annual_inr"]

categorical_cols = ["roof_orientation", "home_ownership", "state"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

feature_cols = [c for c in df_encoded.columns if c != "adopted_solar"]
X = df_encoded[feature_cols]
y = df_encoded["adopted_solar"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)
print("Train adoption rate:", y_train.mean().round(3), "| Test adoption rate:", y_test.mean().round(3))

# %%
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("Saved processed train/test splits to data/")
