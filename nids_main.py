import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

st.title("AI-Powered Network Intrusion Detection System")
st.markdown("""
### Project Overview
This system uses **Random Forest Machine Learning** to detect network intrusions  
and **Groq AI** to explain why a packet is classified as benign or malicious.
""")

# --------------------------------------------------
# HELPER: RULE-BASED EXPLANATION
# --------------------------------------------------
def explain_prediction(packet):
    reasons = []

    if packet['Flow Duration'].values[0] < 1000:
        reasons.append("Very short flow duration")

    if packet['Total Fwd Packets'].values[0] > 100:
        reasons.append("High number of forward packets")

    if packet['Packet Length Mean'].values[0] > 1000:
        reasons.append("Abnormally large packet size")

    return reasons

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        nrows=15000
    )
    df.columns = df.columns.str.strip()

    selected_features = [
        'Destination Port',
        'Flow Duration',
        'Total Fwd Packets',
        'Packet Length Mean',
        'Active Mean',
        'Label'
    ]

    df = df[selected_features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (starts with gsk_)",
    type="password"
)

st.sidebar.header("2. Model Parameters")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=(100 - split_size) / 100,
    random_state=42
)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
st.divider()
st.subheader("1. Model Training")

if st.button("Train Model"):
    with st.spinner("Training Random Forest..."):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

    st.success("Model trained successfully!")

# --------------------------------------------------
# METRICS
# --------------------------------------------------
if "model" in st.session_state:
    st.divider()
    st.subheader("2. Performance Metrics")

    model = st.session_state.model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Total Samples", len(df))
    col3.metric("Detected Attacks", np.sum(y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# LIVE SIMULATION
# --------------------------------------------------
if "model" in st.session_state:
    st.divider()
    st.subheader("3. Live Traffic Simulation")

    if st.button("Simulate Random Packet"):
        idx = np.random.randint(0, len(st.session_state.X_test))

        packet = st.session_state.X_test.iloc[[idx]]
        actual_label = st.session_state.y_test.iloc[idx]
        prediction = st.session_state.model.predict(packet)[0]

        st.session_state.packet = packet
        st.session_state.prediction = prediction
        st.session_state.actual_label = actual_label

    if "packet" in st.session_state:
        packet = st.session_state.packet
        prediction = st.session_state.prediction

        st.write("### Packet Details")
        st.dataframe(packet)

        if prediction == 1:
            st.error("🚨 MALICIOUS TRAFFIC DETECTED")
            st.write("### Rule-Based Explanation")
            for r in explain_prediction(packet):
                st.write("•", r)
        else:
            st.success("✅ BENIGN TRAFFIC")

        st.caption(f"Ground Truth Label: {st.session_state.actual_label}")

        # --------------------------------------------------
        # GROQ AI EXPLANATION
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("4. Ask AI Analyst (Groq)")

        if st.button("Generate AI Explanation"):
            if not groq_api_key:
                st.warning("Please enter your Groq API key.")
            else:
                client = Groq(api_key=groq_api_key)

                prompt = f"""
                You are a cybersecurity analyst.

                Prediction: {prediction}
                Packet Details:
                {packet.to_string()}

                Explain in simple student-friendly terms
                why this traffic is classified as malicious or benign.
                """

                with st.spinner("Groq AI analyzing..."):
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.6
                    )

                st.info(response.choices[0].message.content)
