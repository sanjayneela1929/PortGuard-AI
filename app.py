
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "port_congestion_data.csv"
MODEL_PATH = BASE_DIR / "models" / "congestion_model.pkl"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PortGuard AI",
    page_icon="🚢",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    data = load_data()
    model = load_model()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚢 PortGuard AI")
st.subheader("AI-Powered Port Congestion Prediction")

st.write(
    "Analyze port traffic conditions and predict "
    "the likelihood of congestion using Machine Learning."
)

st.divider()


# --------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------

st.header("📊 Port Congestion Overview")

total_records = len(data)
congested_records = int(data["congestion"].sum())
normal_records = total_records - congested_records

average_delay = data["delay_hours"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", total_records)
col2.metric("Congested Records", congested_records)
col3.metric("Normal Records", normal_records)
col4.metric("Average Delay", f"{average_delay:.2f} hrs")

st.divider()


# --------------------------------------------------
# DATA VISUALIZATION
# --------------------------------------------------

st.header("📈 Data Analysis")

left_col, right_col = st.columns(2)

with left_col:

    st.subheader("Congestion Distribution")

    congestion_counts = (
        data["congestion"]
        .value_counts()
        .rename(index={
            0: "Normal",
            1: "Congested"
        })
    )

    st.bar_chart(congestion_counts)


with right_col:

    st.subheader("Average Delay by Port")

    port_delay = (
        data.groupby("port_name")["delay_hours"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(port_delay)

st.divider()


# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.header("🔍 Predict Port Congestion")

st.write(
    "Enter the current port conditions below "
    "to predict whether congestion is likely."
)

port_capacities = {
    "Chennai": 30,
    "Mumbai": 40,
    "Visakhapatnam": 25,
    "Kolkata": 28,
    "Cochin": 22
}

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        port_name = st.selectbox(
            "Select Port",
            list(port_capacities.keys())
        )

        vessel_arrivals = st.number_input(
            "Vessel Arrivals",
            min_value=5,
            max_value=50,
            value=30
        )

        port_capacity = st.number_input(
            "Port Capacity",
            min_value=1,
            max_value=50,
            value=port_capacities[port_name]
        )

        queue_length = st.number_input(
            "Queue Length",
            min_value=0,
            max_value=30,
            value=15
        )

    with col2:

        weather_severity = st.slider(
            "Weather Severity",
            min_value=0,
            max_value=10,
            value=5
        )

        historical_avg_delay = st.number_input(
            "Historical Average Delay (hours)",
            min_value=0.0,
            max_value=8.0,
            value=3.0,
            step=0.5
        )

        hour_of_day = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=0
        )

    submitted = st.form_submit_button(
        "Predict Congestion",
        use_container_width=True
    )


# --------------------------------------------------
# DISPLAY PREDICTION AND EXPLANATION
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "port_name": port_name,
        "vessel_arrivals": vessel_arrivals,
        "port_capacity": port_capacity,
        "queue_length": queue_length,
        "weather_severity": weather_severity,
        "historical_avg_delay": historical_avg_delay,
        "hour_of_day": hour_of_day,
        "month": month
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    congestion_index = list(model.classes_).index(1)

    congestion_probability = (
        probabilities[congestion_index] * 100
    )

    st.divider()
    st.subheader("📋 Prediction Result")

    if prediction == 1:

        st.error("⚠️ CONGESTION DETECTED")

    else:

        st.success("✅ NORMAL PORT CONDITIONS")

    st.metric(
        "Congestion Probability",
        f"{congestion_probability:.2f}%"
    )

    st.progress(
        float(congestion_probability / 100)
    )

    st.caption(
        "This is a machine-learning prediction based on "
        "synthetic training data, not a confirmed operational forecast."
    )

    # --------------------------------------------------
    # EXPLAINABLE PREDICTION
    # --------------------------------------------------

    st.divider()
    st.subheader("🔎 Prediction Explanation")

    st.write(
        "This section compares the entered port conditions "
        "against typical values from the training dataset."
    )

    # Create a baseline using typical dataset values
    baseline_input = pd.DataFrame([{
        "port_name": data["port_name"].mode()[0],
        "vessel_arrivals": data["vessel_arrivals"].median(),
        "port_capacity": data["port_capacity"].median(),
        "queue_length": data["queue_length"].median(),
        "weather_severity": data["weather_severity"].median(),
        "historical_avg_delay": data["historical_avg_delay"].median(),
        "hour_of_day": data["hour_of_day"].median(),
        "month": data["month"].median()
    }])

    baseline_probability = (
        model.predict_proba(baseline_input)[0][congestion_index] * 100
    )

    # Feature display names
    feature_labels = {
        "port_name": "Port Name",
        "vessel_arrivals": "Vessel Arrivals",
        "port_capacity": "Port Capacity",
        "queue_length": "Queue Length",
        "weather_severity": "Weather Severity",
        "historical_avg_delay": "Historical Average Delay",
        "hour_of_day": "Hour of Day",
        "month": "Month"
    }

    sensitivity_results = []

    # Compare each input feature against the baseline
    for feature in input_data.columns:

        modified_input = baseline_input.copy()

        modified_input[feature] = input_data.iloc[0][feature]

        modified_probability = (
            model.predict_proba(modified_input)[0][congestion_index] * 100
        )

        probability_change = (
            modified_probability - baseline_probability
        )

        sensitivity_results.append({
            "Feature": feature_labels[feature],
            "Actual Value": input_data.iloc[0][feature],
            "Baseline Probability (%)": round(baseline_probability, 2),
            "What-if Probability (%)": round(modified_probability, 2),
            "Probability Change (percentage points)": round(
                probability_change, 2
            )
        })

    sensitivity_df = pd.DataFrame(sensitivity_results)

    sensitivity_df["Absolute Change"] = (
        sensitivity_df[
            "Probability Change (percentage points)"
        ].abs()
    )

    sensitivity_df = sensitivity_df.sort_values(
        by="Absolute Change",
        ascending=False
    )

    # Display baseline probability
    st.metric(
        "Baseline Congestion Probability",
        f"{baseline_probability:.2f}%"
    )

    st.write("**Individual Feature Sensitivity Analysis**")

    st.dataframe(
        sensitivity_df.drop(columns=["Absolute Change"]),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # TOP THREE INPUT SENSITIVITIES
    # --------------------------------------------------

    st.write("**Top 3 Input Sensitivities**")

    strongest_features = sensitivity_df.head(3)

    for _, row in strongest_features.iterrows():

        change = row[
            "Probability Change (percentage points)"
        ]

        if change > 0:

            st.warning(
                f"**{row['Feature']}** increased the estimated "
                f"congestion probability by {change:.2f} "
                "percentage points compared with the baseline."
            )

        elif change < 0:

            st.success(
                f"**{row['Feature']}** reduced the estimated "
                f"congestion probability by {abs(change):.2f} "
                "percentage points compared with the baseline."
            )

        else:

            st.info(
                f"**{row['Feature']}** made no change to the "
                "estimated probability in this comparison."
            )

    st.caption(
        "Note: These are one-feature-at-a-time sensitivity comparisons. "
        "They are not causal effects or exact SHAP contributions. "
        "Feature interactions may affect the results."
    )


# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("View Dataset Preview"):

    st.dataframe(
        data.head(20),
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PortGuard AI | Machine Learning Based Port "
    "Congestion Prediction Project"
)
