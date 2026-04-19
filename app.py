import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide"
)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Feature Ranges (from dataset stats) ──────────────────────
FEATURES = {
    'ph':                  {'min': 0.0,  'max': 14.0, 'default': 7.0,    'unit': 'pH'},
    'Hardness':            {'min': 47.0, 'max': 323.0,'default': 196.0,  'unit': 'mg/L'},
    'Solids':              {'min': 320.0,'max': 61227.0,'default': 22014.0,'unit': 'ppm'},
    'Chloramines':         {'min': 0.35, 'max': 13.13, 'default': 7.12,  'unit': 'ppm'},
    'Sulfate':             {'min': 129.0,'max': 481.0, 'default': 333.0, 'unit': 'mg/L'},
    'Conductivity':        {'min': 181.0,'max': 753.0, 'default': 426.0, 'unit': 'μS/cm'},
    'Organic_carbon':      {'min': 2.2,  'max': 28.3,  'default': 14.0,  'unit': 'ppm'},
    'Trihalomethanes':     {'min': 0.74, 'max': 124.0, 'default': 66.0,  'unit': 'μg/L'},
    'Turbidity':           {'min': 1.45, 'max': 6.49,  'default': 3.97,  'unit': 'NTU'},
}

# ── Sidebar: Model Info ───────────────────────────────────────
with st.sidebar:
    st.title("💧 Water Quality Monitor")
    st.markdown("---")
    st.subheader("📊 Model Info")
    st.info("Using: **Random Forest Classifier**\n\nDataset: Water Potability\n\nFeatures: 9 water parameters")

    st.markdown("---")
    st.subheader("🎯 Safe Thresholds (WHO)")
    st.markdown("""
    - **pH:** 6.5 – 8.5
    - **Turbidity:** < 4 NTU
    - **Chloramines:** < 4 mg/L
    - **Conductivity:** < 500 μS/cm
    """)

# ── Main Page ─────────────────────────────────────────────────
st.title("💧 Real-Time Water Quality Monitoring System")
st.markdown("Using IoT sensor simulation + Machine Learning Classifiers")
st.markdown("---")

tab1, tab2 = st.tabs(["🔬 Manual Test", "📡 Sensor Simulation"])

# ══ TAB 1: Manual Input ══════════════════════════════════════
with tab1:
    st.subheader("Enter Water Parameters")
    st.caption("Adjust the sliders to match your sensor readings, then click Predict.")

    col1, col2, col3 = st.columns(3)
    values = {}
    feature_list = list(FEATURES.keys())

    for i, feat in enumerate(feature_list):
        info = FEATURES[feat]
        col = [col1, col2, col3][i % 3]
        with col:
            values[feat] = st.slider(
                f"{feat} ({info['unit']})",
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=float(info['default']),
                step=round((info['max'] - info['min']) / 100, 3)
            )

    st.markdown("")
    if st.button("🔍 Predict Water Quality", use_container_width=True, type="primary"):
        input_arr = np.array([[values[f] for f in feature_list]])
        input_sc  = scaler.transform(input_arr)
        prediction = model.predict(input_sc)[0]
        proba = model.predict_proba(input_sc)[0]

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])

        with res_col2:
            if prediction == 1:
                st.success("## ✅ SAFE TO DRINK")
                st.markdown(f"**Confidence:** {proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
            else:
                st.error("## ⚠️ CONTAMINATED")
                st.markdown(f"**Confidence:** {proba[0]*100:.1f}%")
                st.progress(float(proba[0]))

        # Show input summary
        st.markdown("### Input Summary")
        summary_df = pd.DataFrame({
            'Parameter': feature_list,
            'Value': [round(values[f], 3) for f in feature_list],
            'Unit': [FEATURES[f]['unit'] for f in feature_list]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ══ TAB 2: Sensor Simulation ══════════════════════════════════
with tab2:
    st.subheader("📡 Simulated IoT Sensor Feed")
    st.caption("Generates 30 random sensor readings and predicts water quality for each.")

    n_readings = st.slider("Number of readings to simulate", 10, 50, 30)

    if st.button("▶ Run Simulation", use_container_width=True, type="primary"):
        readings = []
        predictions_list = []
        probabilities = []

        progress = st.progress(0, text="Running sensor simulation...")

        for i in range(n_readings):
            # Generate random reading with slight noise around dataset means
            reading = {
                feat: np.clip(
                    np.random.normal(FEATURES[feat]['default'],
                                     (FEATURES[feat]['max'] - FEATURES[feat]['min']) * 0.1),
                    FEATURES[feat]['min'],
                    FEATURES[feat]['max']
                )
                for feat in feature_list
            }
            readings.append(reading)

            arr = np.array([[reading[f] for f in feature_list]])
            arr_sc = scaler.transform(arr)
            pred = model.predict(arr_sc)[0]
            prob = model.predict_proba(arr_sc)[0]
            predictions_list.append(pred)
            probabilities.append(prob[1])  # probability of being safe

            progress.progress((i + 1) / n_readings,
                               text=f"Reading {i+1}/{n_readings}...")
            time.sleep(0.05)

        progress.empty()

        # ── Summary Metrics ──
        safe_count = sum(predictions_list)
        unsafe_count = n_readings - safe_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Readings", n_readings)
        m2.metric("✅ Safe",       safe_count,   delta=f"{safe_count/n_readings*100:.0f}%")
        m3.metric("⚠️ Contaminated", unsafe_count, delta=f"-{unsafe_count/n_readings*100:.0f}%",
                  delta_color="inverse")

        # ── Line Chart ──
        st.markdown("### Safety Probability Over Time")
        chart_df = pd.DataFrame({
            'Reading #': range(1, n_readings + 1),
            'Safe Probability': probabilities
        }).set_index('Reading #')
        st.line_chart(chart_df, color='#22c55e')

        # ── Alert Log ──
        alert_indices = [i+1 for i, p in enumerate(predictions_list) if p == 0]
        if alert_indices:
            st.markdown("### 🚨 Alert Log — Contaminated Readings")
            alert_data = []
            for idx in alert_indices:
                row = readings[idx-1].copy()
                row['Reading #'] = idx
                row['Safe Prob'] = round(probabilities[idx-1], 3)
                alert_data.append(row)
            alert_df = pd.DataFrame(alert_data)[['Reading #', 'Safe Prob'] + feature_list[:4]]
            st.dataframe(alert_df.round(3), use_container_width=True, hide_index=True)
        else:
            st.success("✅ No contamination detected in this simulation run!")

        # ── Save alert log to CSV ──
        full_df = pd.DataFrame(readings)
        full_df['Prediction'] = predictions_list
        full_df['Safe_Probability'] = probabilities
        full_df.to_csv('sensor_log.csv', index=False)
        st.caption("📁 Full sensor log saved to sensor_log.csv")
