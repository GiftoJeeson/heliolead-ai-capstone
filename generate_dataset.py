"""
generate_dataset.py
--------------------
HelioLead AI - Synthetic Lead Dataset Generator

WHY THIS EXISTS:
There is no public, row-level, labeled dataset of "household features -> did they
adopt solar" that you can just download. Real sources (Google Project Sunroof,
NREL NSRDB, US EIA/RECS survey, DeepSolar) are either aggregated at the census-tract
level or omit an adoption label entirely.

For a capstone lead-scoring project, the standard practice is to generate a
synthetic dataset whose feature RANGES are grounded in real published sources,
and whose LABEL is produced by a probabilistic function (not a simple if/else
rule), so the resulting classification problem is non-trivial and realistic.

Grounding used for the ranges below (cite these in your report):
- Avg sunlight hours/day by Indian region: 4-7 hrs/day (typical solar-irradiance
  literature for India, ~1600-2200 kWh/m2/year)
- Income vs. adoption skew: solar adopters skew higher-income but span all
  income bands (LBNL "Income Trends Among U.S. Residential Solar Adopters", 2020)
- Home ownership matters a lot (renters rarely install rooftop solar)
- Roof orientation: south-facing roofs (northern hemisphere) receive the most
  usable sunlight, followed by east/west, with north-facing/flat being weakest
- State subsidy % varies by government incentive scheme

You can swap these distributions for a different country/market later without
touching any other file in the project - this is the ONLY file that should
change if your team decides to target a different geography.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ROWS = 6000

rng = np.random.default_rng(RNG_SEED)

STATES = [
    "Tamil Nadu", "Karnataka", "Maharashtra", "Gujarat", "Rajasthan",
    "Andhra Pradesh", "Telangana", "Kerala", "Punjab", "Uttar Pradesh",
]
# Rough relative solar-irradiance ranking used only to bias sunlight hours per state
STATE_SUNLIGHT_BIAS = {
    "Rajasthan": 0.9, "Gujarat": 0.7, "Andhra Pradesh": 0.6, "Telangana": 0.55,
    "Karnataka": 0.5, "Maharashtra": 0.45, "Tamil Nadu": 0.4, "Punjab": 0.3,
    "Uttar Pradesh": 0.2, "Kerala": 0.1,
}
# Rough relative state subsidy generosity (0-1 scale, purely illustrative)
STATE_SUBSIDY_BIAS = {
    "Gujarat": 0.9, "Rajasthan": 0.7, "Punjab": 0.6, "Karnataka": 0.6,
    "Maharashtra": 0.5, "Tamil Nadu": 0.5, "Telangana": 0.45,
    "Andhra Pradesh": 0.4, "Uttar Pradesh": 0.3, "Kerala": 0.3,
}

ORIENTATIONS = ["South", "East", "West", "North", "Flat"]
# Higher = more usable sunlight captured for a given orientation
ORIENTATION_SCORE = {"South": 1.0, "East": 0.75, "West": 0.7, "Flat": 0.6, "North": 0.35}


def sample_rows(n: int) -> pd.DataFrame:
    states = rng.choice(STATES, size=n)
    sunlight_bias = np.array([STATE_SUNLIGHT_BIAS[s] for s in states])
    subsidy_bias = np.array([STATE_SUBSIDY_BIAS[s] for s in states])

    avg_sunlight_hours = np.clip(
        4.0 + sunlight_bias * 2.5 + rng.normal(0, 0.4, n), 3.0, 7.5
    ).round(2)

    orientation = rng.choice(ORIENTATIONS, size=n, p=[0.30, 0.22, 0.20, 0.13, 0.15])
    orientation_score = np.array([ORIENTATION_SCORE[o] for o in orientation])

    roof_area_sqft = np.clip(rng.normal(1200, 400, n), 300, 4000).round(0)

    # Lognormal income distribution (INR/year) - right-skewed like real income data
    household_income = np.clip(
        rng.lognormal(mean=13.3, sigma=0.5, size=n), 150_000, 8_000_000
    ).round(-3)

    monthly_bill = np.clip(
        household_income / 12 * rng.uniform(0.015, 0.05, n) + rng.normal(0, 300, n),
        500, 25000,
    ).round(0)

    home_ownership = rng.choice(["Owned", "Rented"], size=n, p=[0.72, 0.28])
    property_age_years = np.clip(rng.normal(15, 8, n), 0, 60).round(0)
    household_size = rng.integers(1, 8, n)
    has_ev = rng.choice([0, 1], size=n, p=[0.88, 0.12])
    previous_inquiry = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    state_subsidy_pct = np.clip(subsidy_bias * 40 + rng.normal(0, 5, n), 5, 45).round(1)

    df = pd.DataFrame({
        "state": states,
        "avg_sunlight_hours_per_day": avg_sunlight_hours,
        "roof_orientation": orientation,
        "roof_area_sqft": roof_area_sqft,
        "household_income_annual_inr": household_income,
        "monthly_electricity_bill_inr": monthly_bill,
        "home_ownership": home_ownership,
        "property_age_years": property_age_years,
        "household_size": household_size,
        "has_ev": has_ev,
        "previous_solar_inquiry": previous_inquiry,
        "state_subsidy_pct": state_subsidy_pct,
    })

    # ---- Probabilistic label: adopted_solar ----
    # Standardize a few continuous drivers so weights are comparable
    def z(x):
        return (x - x.mean()) / x.std()

    logit = (
        1.1 * z(avg_sunlight_hours)
        + 0.9 * z(orientation_score)
        + 0.8 * z(np.log(household_income))
        + 0.5 * z(monthly_bill)
        + 0.7 * (home_ownership == "Owned").astype(float)
        + 0.6 * z(state_subsidy_pct)
        + 0.9 * previous_inquiry
        + 0.3 * has_ev
        - 0.4 * z(property_age_years)
        - 3.4  # intercept, keeps overall adoption rate realistic (~18-22%)
        + rng.normal(0, 0.6, n)  # irreducible noise so the problem isn't trivial
    )
    prob_adopt = 1 / (1 + np.exp(-logit))
    df["adopted_solar"] = (rng.uniform(0, 1, n) < prob_adopt).astype(int)

    return df


if __name__ == "__main__":
    df = sample_rows(N_ROWS)
    out_path = "data/heliolead_solar_leads.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Positive class rate (adopted_solar=1): {df['adopted_solar'].mean():.3f}")
    print(df.head())
