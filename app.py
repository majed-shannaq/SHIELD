import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="SHIELD - Fraud Detection", page_icon="🛡️", layout="wide")

FEATURES = ['input_speed_ms', 'dwell_time_ms', 'time_to_confirm',
            'session_duration', 'transaction_chain_speed', 'amount']

FEATURE_LABELS = {
    'input_speed_ms': 'Input Speed (ms)',
    'dwell_time_ms': 'Key Dwell Time (ms)',
    'time_to_confirm': 'Time to Confirm (s)',
    'session_duration': 'Session Duration (s)',
    'transaction_chain_speed': 'Transaction Chain Speed (s)',
    'amount': 'Amount (SAR)',
}

ANOMALY_LABELS = {
    'input_speed': 'Unusual input speed (+5)',
    'dwell_time': 'Mechanical key rhythm (+12)',
    'time_to_confirm': 'No review before confirm (+5)',
    'session_duration': 'Abnormally short session (+5)',
    'chain_speed': 'Rapid consecutive transactions (+13)',
    'amount': 'Amount exceeds user pattern (+10)',
}


# ---------- Model building (runs once, cached) ----------

@st.cache_resource(show_spinner="Building 50 personal models...")
def build_system():
    np.random.seed(42)

    n_users = 50
    sessions_per_user = 80
    all_data = []

    for user_id in range(1, n_users + 1):
        user_baseline = {
            'input_speed_ms': np.random.uniform(120, 240),
            'dwell_time_ms': np.random.uniform(70, 130),
            'time_to_confirm': np.random.uniform(15, 40),
            'session_duration': np.random.uniform(120, 240),
            'transaction_chain_speed': np.random.uniform(30, 70),
            'usual_amount': np.random.uniform(100, 1000),
        }

        for _ in range(sessions_per_user):
            all_data.append({
                'user_id': user_id,
                'input_speed_ms': np.random.normal(user_baseline['input_speed_ms'], 15),
                'dwell_time_ms': np.random.normal(user_baseline['dwell_time_ms'], 10),
                'time_to_confirm': np.random.normal(user_baseline['time_to_confirm'], 4),
                'session_duration': np.random.normal(user_baseline['session_duration'], 25),
                'transaction_chain_speed': np.random.normal(user_baseline['transaction_chain_speed'], 8),
                'amount': np.random.normal(user_baseline['usual_amount'], 150),
                'is_fraud': 0, 'fraud_type': 'normal'
            })

        for _ in range(2):
            all_data.append({
                'user_id': user_id,
                'input_speed_ms': np.random.normal(55, 8),
                'dwell_time_ms': np.random.normal(40, 5),
                'time_to_confirm': np.random.normal(2.5, 0.5),
                'session_duration': np.random.normal(45, 10),
                'transaction_chain_speed': np.random.normal(6, 2),
                'amount': np.random.uniform(500, 3000),
                'is_fraud': 1, 'fraud_type': 'bot'
            })

        for _ in range(2):
            all_data.append({
                'user_id': user_id,
                'input_speed_ms': np.random.normal(user_baseline['input_speed_ms'] * 0.85, 30),
                'dwell_time_ms': np.random.normal(user_baseline['dwell_time_ms'] * 0.85, 18),
                'time_to_confirm': np.random.normal(12, 4),
                'session_duration': np.random.normal(110, 25),
                'transaction_chain_speed': np.random.normal(22, 7),
                'amount': np.random.uniform(1500, 5000),
                'is_fraud': 1, 'fraud_type': 'ato'
            })

        for _ in range(2):
            all_data.append({
                'user_id': user_id,
                'input_speed_ms': np.random.normal(130, 30),
                'dwell_time_ms': np.random.normal(78, 18),
                'time_to_confirm': np.random.normal(18, 5),
                'session_duration': np.random.normal(100, 22),
                'transaction_chain_speed': np.random.normal(28, 8),
                'amount': np.random.uniform(1800, 4500),
                'is_fraud': 1, 'fraud_type': 'social'
            })

        for _ in range(2):
            mimicry_factor = np.random.uniform(0.90, 1.10)
            all_data.append({
                'user_id': user_id,
                'input_speed_ms': np.random.normal(user_baseline['input_speed_ms'] * mimicry_factor, 25),
                'dwell_time_ms': np.random.normal(user_baseline['dwell_time_ms'] * mimicry_factor, 18),
                'time_to_confirm': np.random.normal(user_baseline['time_to_confirm'] * 0.85, 6),
                'session_duration': np.random.normal(user_baseline['session_duration'] * 0.85, 30),
                'transaction_chain_speed': np.random.normal(user_baseline['transaction_chain_speed'] * 0.80, 12),
                'amount': np.random.normal(user_baseline['usual_amount'] * 2, 250),
                'is_fraud': 1, 'fraud_type': 'targeted'
            })

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    user_models = {}
    user_scalers = {}
    user_baselines = {}
    val_indices = []

    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id]
        user_normal = user_data[user_data['is_fraud'] == 0]
        user_fraud = user_data[user_data['is_fraud'] == 1]

        n_normal = len(user_normal)
        n_train = int(n_normal * 0.6)
        n_val = int(n_normal * 0.2)

        train_data = user_normal.iloc[:n_train]
        val_normal = user_normal.iloc[n_train:n_train + n_val]

        n_fraud_val = int(len(user_fraud) * 0.5)
        val_fraud = user_fraud.iloc[:n_fraud_val]

        val_indices.extend(val_normal.index.tolist() + val_fraud.index.tolist())

        user_baselines[user_id] = {
            'input_speed_ms': train_data['input_speed_ms'].mean(),
            'dwell_time_ms': train_data['dwell_time_ms'].mean(),
            'time_to_confirm': train_data['time_to_confirm'].mean(),
            'session_duration': train_data['session_duration'].mean(),
            'transaction_chain_speed': train_data['transaction_chain_speed'].mean(),
            'usual_amount': train_data['amount'].mean(),
        }

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[FEATURES])

        model = IsolationForest(n_estimators=100, contamination=0.05,
                                random_state=42, n_jobs=-1)
        model.fit(train_scaled)

        user_models[user_id] = model
        user_scalers[user_id] = scaler

    # Thresholds from validation data only
    val_scores, val_labels = [], []
    for idx in val_indices:
        session = df.loc[idx]
        uid = session['user_id']
        arr = [[session[f] for f in FEATURES]]
        scaled = user_scalers[uid].transform(arr)
        val_scores.append(user_models[uid].score_samples(scaled)[0])
        val_labels.append(session['is_fraud'])

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    fraud_scores = val_scores[val_labels == 1]
    normal_scores = val_scores[val_labels == 0]

    thresholds = {
        'high': np.percentile(fraud_scores, 50),
        'medium': np.percentile(fraud_scores, 75),
        'low': np.percentile(normal_scores, 25),
    }

    return user_models, user_scalers, user_baselines, thresholds


# ---------- Scoring logic (identical to notebook) ----------

def detect_anomalies(session, user_baseline):
    anomalies = {}
    if session['input_speed_ms'] < user_baseline['input_speed_ms'] * 0.6:
        anomalies['input_speed'] = 5
    if session['dwell_time_ms'] < 60 or session['dwell_time_ms'] > user_baseline['dwell_time_ms'] * 1.4:
        anomalies['dwell_time'] = 12
    if session['time_to_confirm'] < user_baseline['time_to_confirm'] * 0.3:
        anomalies['time_to_confirm'] = 5
    if session['session_duration'] < user_baseline['session_duration'] * 0.5:
        anomalies['session_duration'] = 5
    if session['transaction_chain_speed'] < user_baseline['transaction_chain_speed'] * 0.4:
        anomalies['chain_speed'] = 13
    if session['amount'] > user_baseline['usual_amount'] * 2.5:
        anomalies['amount'] = 10
    return anomalies


def calculate_behavior_score(session, user_baseline):
    anomalies = detect_anomalies(session, user_baseline)
    score = sum(anomalies.values())
    if len(anomalies) >= 3:
        score += 10
    return min(score, 50), list(anomalies.keys())


def calculate_total_score(session, user_baseline, user_id, models, scalers, thresholds):
    manual_score, anomalies = calculate_behavior_score(session, user_baseline)

    arr = [[session[f] for f in FEATURES]]
    scaled = scalers[user_id].transform(arr)
    model_score = models[user_id].score_samples(scaled)[0]

    if model_score < thresholds['high']:
        model_bonus, model_signal = 25, "high"
    elif model_score < thresholds['medium']:
        model_bonus, model_signal = 18, "medium"
    elif model_score < thresholds['low']:
        model_bonus, model_signal = 10, "low"
    else:
        model_bonus, model_signal = 0, "none"

    total = min(manual_score + model_bonus, 50)
    return {
        'total': total, 'manual_score': manual_score,
        'model_bonus': model_bonus, 'anomalies': anomalies,
        'model_signal': model_signal
    }


def get_decision(total_score):
    if total_score >= 90:
        return "BLOCK", "Freeze account immediately", "red"
    if total_score >= 70:
        return "VERIFY", "Request face biometric", "orange"
    if total_score >= 40:
        return "MONITOR", "Soft check", "yellow"
    return "ALLOW", "Proceed silently", "green"


# ---------- Scenario presets ----------

def get_scenario_values(scenario, baseline):
    if scenario == "Normal":
        return {f: baseline[f if f != 'amount' else 'usual_amount'] for f in FEATURES}
    if scenario == "Bot":
        return {'input_speed_ms': 55, 'dwell_time_ms': 40, 'time_to_confirm': 2.5,
                'session_duration': 45, 'transaction_chain_speed': 6, 'amount': 2000}
    if scenario == "ATO":
        return {'input_speed_ms': baseline['input_speed_ms'] * 0.85,
                'dwell_time_ms': baseline['dwell_time_ms'] * 0.85,
                'time_to_confirm': 12, 'session_duration': 110,
                'transaction_chain_speed': 22, 'amount': 3500}
    if scenario == "Social Engineering":
        return {'input_speed_ms': 130, 'dwell_time_ms': 78, 'time_to_confirm': 18,
                'session_duration': 100, 'transaction_chain_speed': 28, 'amount': 3000}
    if scenario == "Targeted":
        return {'input_speed_ms': baseline['input_speed_ms'] * 1.0,
                'dwell_time_ms': baseline['dwell_time_ms'] * 1.0,
                'time_to_confirm': baseline['time_to_confirm'] * 0.85,
                'session_duration': baseline['session_duration'] * 0.85,
                'transaction_chain_speed': baseline['transaction_chain_speed'] * 0.80,
                'amount': baseline['usual_amount'] * 2}
    return None


# ---------- UI ----------

models, scalers, baselines, thresholds = build_system()

st.title("SHIELD - Behavioral Fraud Detection")
st.caption("Live demo: the model runs in real time. Behavior layer (0-50) + context layer (0-50) = Risk Score / 100")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Select user")
    user_id = st.selectbox("User", list(range(1, 51)), format_func=lambda x: f"User {x}")
    baseline = baselines[user_id]

    st.markdown("**This user's behavioral baseline:**")
    bl_df = pd.DataFrame({
        'Metric': [FEATURE_LABELS[f] for f in FEATURES],
        'Baseline': [round(baseline[f if f != 'amount' else 'usual_amount'], 1) for f in FEATURES]
    })
    st.dataframe(bl_df, hide_index=True, use_container_width=True)

    st.subheader("2. Load a scenario (or set values manually)")
    scenario = st.radio("Scenario", ["Normal", "Bot", "ATO", "Social Engineering", "Targeted", "Manual"],
                        horizontal=True)

with col_right:
    st.subheader("3. Session behavior")

    preset = get_scenario_values(scenario, baseline) if scenario != "Manual" else None

    session = {}
    session['input_speed_ms'] = st.slider(FEATURE_LABELS['input_speed_ms'], 20.0, 400.0,
        float(preset['input_speed_ms']) if preset else float(baseline['input_speed_ms']))
    session['dwell_time_ms'] = st.slider(FEATURE_LABELS['dwell_time_ms'], 20.0, 200.0,
        float(preset['dwell_time_ms']) if preset else float(baseline['dwell_time_ms']))
    session['time_to_confirm'] = st.slider(FEATURE_LABELS['time_to_confirm'], 1.0, 60.0,
        float(preset['time_to_confirm']) if preset else float(baseline['time_to_confirm']))
    session['session_duration'] = st.slider(FEATURE_LABELS['session_duration'], 20.0, 400.0,
        float(preset['session_duration']) if preset else float(baseline['session_duration']))
    session['transaction_chain_speed'] = st.slider(FEATURE_LABELS['transaction_chain_speed'], 2.0, 120.0,
        float(preset['transaction_chain_speed']) if preset else float(baseline['transaction_chain_speed']))
    session['amount'] = st.slider(FEATURE_LABELS['amount'], 50.0, 10000.0,
        float(preset['amount']) if preset else float(baseline['usual_amount']))

    context_score = st.slider("Context score from security layer (device, IP, location)", 0, 50, 0,
                              help="In production this comes from the contextual security engine")

st.divider()

if st.button("Assess Risk", type="primary", use_container_width=True):
    result = calculate_total_score(session, baseline, user_id, models, scalers, thresholds)
    total_risk = min(result['total'] + context_score, 100)
    action, description, color = get_decision(total_risk)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Behavior Score", f"{result['total']}/50")
    c2.metric("Context Score", f"{context_score}/50")
    c3.metric("Total Risk", f"{total_risk}/100")
    c4.metric("Model Signal", result['model_signal'].upper())

    if color == "green":
        st.success(f"**{action}** — {description}")
    elif color == "yellow":
        st.warning(f"**{action}** — {description}")
    else:
        st.error(f"**{action}** — {description}")

    if result['anomalies']:
        st.markdown("**Detected anomalies:**")
        for a in result['anomalies']:
            st.markdown(f"- {ANOMALY_LABELS[a]}")
    else:
        st.markdown("*No rule-based anomalies. "
                    + ("Isolation Forest flagged a hidden pattern." if result['model_bonus'] > 0
                       else "Behavior matches this user's personal pattern.")
                    + "*")

st.divider()
st.caption("SHIELD | Amad Hackathon 2026 | Team: Rawan, Majed, Al-Baraa, Rayhana")
