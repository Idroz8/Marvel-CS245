import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# 1) Load & train on the built-in backbone dataset
@st.cache_data
def load_and_train():
    df = pd.read_csv("adam_warlock_positions_20_matches.csv")
    features = [
        'dist_to_tank', 'dist_to_healer', 'dist_to_duelist',
        'dist_to_team_centroid', 'dist_to_enemy_centroid', 'near_objective'
    ]
    X = df[features].values
    y = df['label'].values

    # Scale features for SVM
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM for predictions
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_scaled, y)

    # Train Random Forest for feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(df[features], y)

    return svm, rf, scaler, df

def main():
    st.title("Adam Warlock Match Evaluator")
    st.write("This app is trained on our backbone data and then rates your entire match.")

    # Load and train once
    svm, rf, scaler, train_df = load_and_train()

    st.write("---")
    st.write("## 1. Upload Your Match Data")
    uploaded = st.file_uploader("Your match CSV (5s intervals)", type=["csv"])
    if not uploaded:
        st.info("Please upload your game CSV to proceed.")
        return

    # Load user data
    match_df = pd.read_csv(uploaded)
    st.write("### Match Preview")
    st.dataframe(match_df.head())

    features = [
        'dist_to_tank', 'dist_to_healer', 'dist_to_duelist',
        'dist_to_team_centroid', 'dist_to_enemy_centroid', 'near_objective'
    ]

    # 2) Run SVM predictions over every interval
    X_user = match_df[features].values
    X_user_scaled = scaler.transform(X_user)
    preds = svm.predict(X_user_scaled)

    # 3) Summarize match-level results
    good_pct = (preds == 1).mean() * 100
    st.write(f"### Good-positioning intervals: {good_pct:.1f}% of your match")
    overall = "Good Match ✅" if good_pct >= 50 else "Needs Improvement ❌"
    st.subheader(f"Overall Assessment: {overall}")

    # 4) Feature feedback using RF importances
    # Compute mean feature values for top (label=1) and for the user's match
    good_means = train_df[train_df['label'] == 1][features].mean()
    user_means = match_df[features].mean()
    diff = good_means - user_means

    importances = pd.Series(rf.feature_importances_, index=features)
    # Impact = absolute difference × importance
    impact = (diff.abs() * importances)
    lacking = impact.idxmax()
    lacking_diff = diff[lacking]
    lacking_imp = importances[lacking]

    st.write("---")
    st.write("## Feature Feedback")
    st.write(f"**Most lacking feature:** {lacking}")
    st.write(f"- Difference from top players: {lacking_diff:.2f}")
    st.write(f"- Feature importance weight: {lacking_imp:.2f}")
    st.write(f"- Combined impact score: {impact[lacking]:.2f}")

if __name__ == "__main__":
    main()
