"""
app.py
-------
Streamlit dashboard: sales teams enter a property's details and get an
instant solar-adoption conversion probability + lead tier.

Run with:  streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HelioLead AI", page_icon="\u2600", layout="centered")

STATES = [
    "Tamil Nadu", "Karnataka", "Maharashtra", "Gujarat", "Rajasthan",
    "Andhra Pradesh", "Telangana", "Kerala", "Punjab", "Uttar Pradesh",
]
ORIENTATIONS = ["South", "East", "West", "North", "Flat"]


@st.cache_resource
def load_model():
    model = joblib.load("models/heliolead_model.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")
    return model, feature_cols


model, feature_cols = load_model()

st.title("\u2600 HelioLead AI")
st.caption("Predictive lead scoring for residential solar sales teams")

with st.form("lead_form"):
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", STATES)
        orientation = st.selectbox("Roof orientation", ORIENTATIONS)
        roof_area = st.number_input("Roof area (sq ft)", 200, 5000, 1200)
        income = st.number_input("Household annual income (INR)", 100_000, 10_000_000, 800_000, step=50_000)
        bill = st.number_input("Monthly electricity bill (INR)", 200, 30_000, 3000, step=100)
    with col2:
        ownership = st.selectbox("Home ownership", ["Owned", "Rented"])
        property_age = st.number_input("Property age (years)", 0, 80, 10)
        household_size = st.number_input("Household size", 1, 10, 4)
        has_ev = st.checkbox("Owns an electric vehicle")
        previous_inquiry = st.checkbox("Has previously inquired about solar")
        subsidy_pct = st.slider("State subsidy (%)", 0, 45, 20)

    submitted = st.form_submit_button("Score this lead")

if submitted:
    sunlight_by_state = {
        "Rajasthan": 6.2, "Gujarat": 5.7, "Andhra Pradesh": 5.5, "Telangana": 5.4,
        "Karnataka": 5.2, "Maharashtra": 5.1, "Tamil Nadu": 5.0, "Punjab": 4.7,
        "Uttar Pradesh": 4.4, "Kerala": 4.1,
    }
    raw = {
        "avg_sunlight_hours_per_day": sunlight_by_state[state],
        "roof_area_sqft": roof_area,
        "household_income_annual_inr": income,
        "monthly_electricity_bill_inr": bill,
        "property_age_years": property_age,
        "household_size": household_size,
        "has_ev": int(has_ev),
        "previous_solar_inquiry": int(previous_inquiry),
        "state_subsidy_pct": subsidy_pct,
        "bill_to_income_ratio": (bill * 12) / income,
    }
    row = pd.DataFrame([raw])

    for s in STATES:
        row[f"state_{s}"] = 1 if state == s else 0
    for o in ORIENTATIONS:
        row[f"roof_orientation_{o}"] = 1 if orientation == o else 0
    row["home_ownership_Rented"] = 1 if ownership == "Rented" else 0

    # Align to the exact training column order; any missing dummy = 0
    row = row.reindex(columns=feature_cols, fill_value=0)

    proba = model.predict_proba(row)[0, 1]

    if proba >= 0.6:
        tier, color = "Hot Lead", "green"
    elif proba >= 0.3:
        tier, color = "Warm Lead", "orange"
    else:
        tier, color = "Cold Lead", "red"

    st.metric("Conversion probability", f"{proba:.1%}")
    st.markdown(f"### Lead tier: :{color}[{tier}]")
    st.progress(min(proba, 1.0))
