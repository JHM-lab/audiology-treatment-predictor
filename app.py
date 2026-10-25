import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Audiology Treatment Predictor & Comparator", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('lightgbm_audiology_model.pkl')
    le = joblib.load('label_encoder.pkl')
    features = joblib.load('feature_cols.pkl')
    ref_df = pd.read_csv('historical_reference_data.csv')
    return model, le, features, ref_df

model, label_encoder, feature_cols, ref_df = load_artifacts()

st.title("Audiology Treatment Predictor & Dataset Comparator (AC & BC)")
st.markdown("Enter patient Air Conduction (AC) and Bone Conduction (BC) thresholds below to predict treatment recommendations and evaluate audiometric profiles.")

frequencies = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]

# Initialize Session State for all inputs to 10.0 dB by default
for f in frequencies:
    if f'L{f}C' not in st.session_state:
        st.session_state[f'L{f}C'] = 10.0
    if f'R{f}C' not in st.session_state:
        st.session_state[f'R{f}C'] = 10.0
    if f'BL{f}C' not in st.session_state:
        st.session_state[f'BL{f}C'] = 10.0
    if f'BR{f}C' not in st.session_state:
        st.session_state[f'BR{f}C'] = 10.0

# Sidebar for User Inputs
st.sidebar.header("Patient Audiological Data Entry")

# Reset Button
if st.sidebar.button("Reset All to 10 dB", type="secondary"):
    for f in frequencies:
        st.session_state[f'L{f}C'] = 10.0
        st.session_state[f'R{f}C'] = 10.0
        st.session_state[f'BL{f}C'] = 10.0
        st.session_state[f'BR{f}C'] = 10.0
    st.rerun()

left_ac, right_ac = {}, {}
left_bc, right_bc = {}, {}

st.sidebar.subheader("Left Ear - Air Conduction (AC)")
for f in frequencies:
    left_ac[f'L{f}C'] = st.sidebar.number_input(f"Left AC {f} Hz", min_value=0.0, max_value=120.0, step=5.0, key=f'L{f}C')

st.sidebar.subheader("Right Ear - Air Conduction (AC)")
for f in frequencies:
    right_ac[f'R{f}C'] = st.sidebar.number_input(f"Right AC {f} Hz", min_value=0.0, max_value=120.0, step=5.0, key=f'R{f}C')

st.sidebar.subheader("Left Ear - Bone Conduction (BC)")
for f in frequencies:
    left_bc[f'BL{f}C'] = st.sidebar.number_input(f"Left BC {f} Hz", min_value=0.0, max_value=120.0, step=5.0, key=f'BL{f}C')

st.sidebar.subheader("Right Ear - Bone Conduction (BC)")
for f in frequencies:
    right_bc[f'BR{f}C'] = st.sidebar.number_input(f"Right BC {f} Hz", min_value=0.0, max_value=120.0, step=5.0, key=f'BR{f}C')

# Construct user feature dictionary
user_data = {**left_ac, **right_ac, **left_bc, **right_bc}
user_df = pd.DataFrame([user_data])

# Calculate overall means
user_df['Right_Mean'] = user_df[[f'R{f}C' for f in frequencies]].mean(axis=1)
user_df['Left_Mean'] = user_df[[f'L{f}C' for f in frequencies]].mean(axis=1)
user_df['BRight_Mean'] = user_df[[f'BR{f}C' for f in frequencies]].mean(axis=1)
user_df['BLeft_Mean'] = user_df[[f'BL{f}C' for f in frequencies]].mean(axis=1)

# Frequency bands splits (Low vs High)
H_freqs = [3000, 4000, 6000, 8000]
L_freqs = [250, 500, 750, 1000, 1500, 2000]

# Follow-up High & Low means (AC)
user_df['HRight_Mean'] = user_df[[f'R{f}C' for f in H_freqs]].mean(axis=1)
user_df['HLeft_Mean'] = user_df[[f'L{f}C' for f in H_freqs]].mean(axis=1)
user_df['LRight_Mean'] = user_df[[f'R{f}C' for f in L_freqs]].mean(axis=1)
user_df['LLeft_Mean'] = user_df[[f'L{f}C' for f in L_freqs]].mean(axis=1)

# Baseline High & Low means (BC)
user_df['HBRight_Mean'] = user_df[[f'BR{f}C' for f in H_freqs]].mean(axis=1)
user_df['HBLeft_Mean'] = user_df[[f'BL{f}C' for f in H_freqs]].mean(axis=1)
user_df['LBRight_Mean'] = user_df[[f'BR{f}C' for f in L_freqs]].mean(axis=1)
user_df['LBLeft_Mean'] = user_df[[f'BL{f}C' for f in L_freqs]].mean(axis=1)

user_df['Ear_Asymmetry'] = np.abs(user_df['Right_Mean'] - user_df['Left_Mean'])

# Calculate air-bone gap delta metrics if needed
user_df['RAC_BC'] = np.abs(user_df['Right_Mean'] - user_df['BRight_Mean'])
user_df['LAC_BC'] = np.abs(user_df['Left_Mean'] - user_df['BLeft_Mean'])

# Align columns to match model expectations
X_user = user_df[feature_cols]

# Main Dashboard layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Prediction")
    if st.button("Run Prediction & Compare", type="primary"):
        prediction_encoded = model.predict(X_user)[0]
        prediction_proba = model.predict_proba(X_user)[0]
        predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
        
        st.success(f"**Recommended Treatment Category:** {predicted_label}")
        
        st.write("### Class Probability Breakdown:")
        prob_df = pd.DataFrame({
            'Treatment Category': label_encoder.classes_,
            'Probability': prediction_proba
        }).sort_values(by='Probability', ascending=False)
        st.dataframe(prob_df, hide_index=True)

with col2:
    st.subheader("Patient Audiogram (Air & Bone Conduction)")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    right_ac_vals = [right_ac[f'R{f}C'] for f in frequencies]
    left_ac_vals = [left_ac[f'L{f}C'] for f in frequencies]
    right_bc_vals = [right_bc[f'BR{f}C'] for f in frequencies]
    left_bc_vals = [left_bc[f'BL{f}C'] for f in frequencies]
    
    ax.plot(frequencies, right_ac_vals, marker='o', color='red', label='Right AC (Air)')
    ax.plot(frequencies, left_ac_vals, marker='x', color='blue', label='Left AC (Air)')
    ax.plot(frequencies, right_bc_vals, marker='^', color='orange', linestyle='--', label='Right BC (Bone)')
    ax.plot(frequencies, left_bc_vals, marker='v', color='cyan', linestyle='--', label='Left BC (Bone)')
    
    ax.invert_yaxis()  # Audiograms standardly invert dB HL
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Hearing Threshold (dB HL)")
    ax.set_title("Entered Patient Audiogram")
    ax.legend(fontsize=8)
    ax.grid(True)
    st.pyplot(fig)

# Distribution Comparison for Both Ears
st.subheader("Patient Mean Hearing Thresholds vs Historical Population")
col_dist1, col_dist2 = st.columns(2)

with col_dist1:
    fig_r, ax_r = plt.subplots(figsize=(6, 3))
    sns.histplot(ref_df['Right_Mean'], color='gray', label='Historical Right Mean', kde=True, ax=ax_r, stat="density")
    ax_r.axvline(user_df['Right_Mean'].values[0], color='red', linestyle='-', linewidth=3, label='Patient Right Mean')
    ax_r.set_title("Right Ear Mean Comparison")
    ax_r.legend(fontsize=8)
    st.pyplot(fig_r)

with col_dist2:
    fig_l, ax_l = plt.subplots(figsize=(6, 3))
    sns.histplot(ref_df['Left_Mean'], color='gray', label='Historical Left Mean', kde=True, ax=ax_l, stat="density")
    ax_l.axvline(user_df['Left_Mean'].values[0], color='blue', linestyle='-', linewidth=3, label='Patient Left Mean')
    ax_l.set_title("Left Ear Mean Comparison")
    ax_l.legend(fontsize=8)
    st.pyplot(fig_l)