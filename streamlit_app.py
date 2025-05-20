import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data on-the-fly; no CSV required
def generate_data(n_matches=20, interval=5):
    np.random.seed(42)
    rows = []
    match_duration = 13 * 60  # seconds
    for match_id in range(1, n_matches + 1):
        label = 1 if match_id <= n_matches // 2 else 0
        for i in range(match_duration // interval):
            # simulate distances
            if label == 1:
                dist_to_tank = np.random.normal(4.5, 1.0)
                dist_to_healer = np.random.normal(5.5, 1.5)
                dist_to_duelist = np.random.normal(4.0, 1.0)
                dist_to_team_centroid = np.random.normal(2.5, 0.8)
                dist_to_enemy_centroid = np.random.normal(9.0, 2.0)
            else:
                dist_to_tank = np.random.normal(6.0, 1.2)
                dist_to_healer = np.random.normal(7.0, 1.8)
                dist_to_duelist = np.random.normal(5.0, 1.3)
                dist_to_team_centroid = np.random.normal(4.5, 1.2)
                dist_to_enemy_centroid = np.random.normal(7.5, 2.5)
            # enforce non-negative distances
            d = lambda x: max(x, 0)
            rows.append({
                'dist_to_tank': d(dist_to_tank),
                'dist_to_healer': d(dist_to_healer),
                'dist_to_duelist': d(dist_to_duelist),
                'dist_to_team_centroid': d(dist_to_team_centroid),
                'dist_to_enemy_centroid': d(dist_to_enemy_centroid),
                'near_objective': np.random.choice([0, 1], p=[0.4, 0.6]),
                'label': label
            })
    return pd.DataFrame(rows)

@st.cache_data
# Train the model via Random Forest
fn train_model():
    df = generate_data()
    features = ['dist_to_tank', 'dist_to_healer', 'dist_to_duelist',
                'dist_to_team_centroid', 'dist_to_enemy_centroid', 'near_objective']
    X = df[features]
    y = df['label']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Streamlit UI
st.title("Adam Warlock Positioning Predictor")
st.write("Enter your positioning metrics and get instant feedback.")

# Input sliders
tank = st.slider("Distance to Tank", 0.0, 15.0, 5.0)
healer = st.slider("Distance to Healer", 0.0, 15.0, 6.0)
duelist = st.slider("Distance to Duelist", 0.0, 15.0, 4.0)
team_ctr = st.slider("Distance to Team Centroid", 0.0, 10.0, 3.0)
enemy_ctr = st.slider("Distance to Enemy Centroid", 0.0, 20.0, 8.0)
near_obj = st.selectbox("Near Objective?", [0, 1])

if st.button("Predict Positioning Quality"):
    model = train_model()
    input_df = pd.DataFrame([{ 'dist_to_tank': tank,
                                'dist_to_healer': healer,
                                'dist_to_duelist': duelist,
                                'dist_to_team_centroid': team_ctr,
                                'dist_to_enemy_centroid': enemy_ctr,
                                'near_objective': near_obj }])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    result = "Good (Top Player)" if pred == 1 else "Bad (Needs Improvement)"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {proba*100:.1f}%")
