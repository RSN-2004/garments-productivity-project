import streamlit as st
import pandas as pd
import joblib

# Page config (makes it look like a real app)
st.set_page_config(page_title="Garment Productivity AI", layout="wide")

# Load model
model = joblib.load("model.pkl")

# HEADER
st.title("👕 Garment Worker Productivity Prediction System")
st.markdown("### AI-powered productivity analysis dashboard")

st.divider()

# Layout (3 columns = impressive UI)
col1, col2, col3 = st.columns(3)

with col1:
    team = st.number_input("Team", 1, 12, 1)
    smv = st.number_input("SMV", value=10.0)
    over_time = st.number_input("Over Time", value=1000)

with col2:
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.5)
    wip = st.number_input("WIP", value=1000.0)
    incentive = st.number_input("Incentive", value=0)

with col3:
    idle_time = st.number_input("Idle Time", value=0.0)
    idle_men = st.number_input("Idle Men", value=0)
    no_of_workers = st.number_input("Workers", value=30.0)

no_of_style_change = st.number_input("Style Changes", value=0)

st.divider()

# Predict button centered
if st.button("🚀 Predict Productivity"):

    input_data = pd.DataFrame(
        [[team, targeted_productivity, smv, wip, over_time,
          incentive, idle_time, idle_men,
          no_of_style_change, no_of_workers] + [0]*12],
        columns=model.feature_names_in_
    )

    prediction = model.predict(input_data)[0]

    # Result box
    st.success(f"Predicted Productivity: {round(prediction, 2)}")

    # Interpretation
    if prediction > 0.8:
        st.info("High Productivity 👍")
    elif prediction > 0.5:
        st.warning("Moderate Productivity ⚠️")
    else:
        st.error("Low Productivity ❌")