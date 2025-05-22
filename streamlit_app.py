import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# 1) Load and train on the built-in backbone dataset
@st.cache_data
def load_and_train():
    # Change this filename if yours is different
    df = pd.read_csv("adam_warlock_positions_20_matches.csv")
    features = [
        'dist_to_tank', 'dist_to_healer', 'dist_to_duelist',
        'dist_to_team_centroid', 'dist_to_enemy_centroid', 'near_objective'
    ]
    X = df[features].values
    y = df['label'].values

    # Scale for SVM
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def main():
    st.title("Adam Warlock Match Evaluator")
    st.write("This app trains on our backbone dataset and then rates your entire match.")

    # Train once on the included CSV
    model, scaler = load_and_train()

    st.write("---")
    st.write("## 1. Upload Your Match Data")
    st.write("Upload a CSV of your match, sampled every 5 seconds (same schema).")
    uploaded = st.file_uploader("Your Match CSV", type=["csv"])
    if not uploaded:
        st.info("Please upload your game CSV to proceed.")
        return

    # Load the user's match
    match_df = pd.read_csv(uploaded)
    st.write("### Your Match Preview")
    st.dataframe(match_df.head())

    # 2) Run predictions over every interval
    features = [
        'dist_to_tank', 'dist_to_healer', 'dist_to_duelist',
        'dist_to_team_centroid', 'dist_to_enemy_centroid', 'near_objective'
    ]
    X_user = match_df[features].values
    X_user_scaled = scaler.transform(X_user)
    preds = model.predict(X_user_scaled)
    probs = model.predict_proba(X_user_scaled)

    # 3) Summarize match-level results
    good_pct = (preds == 1).mean() * 100
    st.write(f"### Good-Positioning Intervals: {good_pct:.1f}% of your match")
    overall = "Good Match ✅" if good_pct >= 50 else "Needs Improvement ❌"
    st.subheader(f"Overall Assessment: {overall}")

    # 4) Show distribution
    st.write("#### Interval Breakdown")
    st.bar_chart(pd.Series(preds).value_counts(normalize=True))

if __name__ == "__main__":
    main()
