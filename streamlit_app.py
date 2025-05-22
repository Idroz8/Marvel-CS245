import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Friendly labels for features
FEATURE_LABELS = {
    'dist_to_tank': 'Distance to Tank',
    'dist_to_healer': 'Distance to Healer',
    'dist_to_duelist': 'Distance to Duelist',
    'dist_to_team_centroid': 'Distance to Team Center',
    'dist_to_enemy_centroid': 'Distance to Enemies',
    'near_objective': 'Time Near Objective'
}

@st.cache_data
def load_and_train():
    df = pd.read_csv("adam_warlock_positions_20_matches.csv")
    features = list(FEATURE_LABELS.keys())
    X = df[features].values
    y = df['label'].values

    # Scale for SVM
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_scaled, y)

    # Train RF for feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(df[features], y)

    return svm, rf, scaler, df

def main():
    st.set_page_config(page_title="Adam Warlock Match Evaluator", layout="wide")
    st.title("🎮 Adam Warlock Positioning Evaluator")
    st.write("Upload your match data and get both an overall rating and personalized feedback.")

    svm, rf, scaler, train_df = load_and_train()

    st.markdown("## 1️⃣ Upload Your Match Data")
    uploaded = st.file_uploader("CSV of your 5-second interval data", type=["csv"])
    if not uploaded:
        st.info("Please upload a match CSV to continue.")
        return

    match_df = pd.read_csv(uploaded)
    st.markdown("**Preview of your data:**")
    st.dataframe(match_df.head())

    features = list(FEATURE_LABELS.keys())

    # 2️⃣ Predictions
    st.markdown("## 2️⃣ Match Evaluation")
    X_user = match_df[features].values
    X_user_scaled = scaler.transform(X_user)
    preds = svm.predict(X_user_scaled)

    good_pct = (preds == 1).mean() * 100
    overall = "🏆 Good Match" if good_pct >= 50 else "🔍 Needs Improvement"

    col1, col2 = st.columns([1,2])
    col1.metric("Good-Positioning Rate", f"{good_pct:.1f}%")
    col2.markdown(f"### Overall: {overall}")

    # 3️⃣ Feature-level feedback
    st.markdown("## 3️⃣ Personalized Feedback")

    # Compute mean stats
    good_means = train_df[train_df['label']==1][features].mean()
    user_means = match_df[features].mean()
    diffs = (good_means - user_means).abs()
    importances = pd.Series(rf.feature_importances_, index=features)
    impact = diffs * importances

    # Identify top lacking
    worst_feat = impact.idxmax()
    label = FEATURE_LABELS[worst_feat]
    diff_val = (good_means[worst_feat] - user_means[worst_feat])
    imp_val = importances[worst_feat]

    st.write(f"**🔴 Most Improving Needed:** {label}")
    st.write(f"- You were `{user_means[worst_feat]:.2f}` vs. pro average `{good_means[worst_feat]:.2f}`")
    st.write(f"- Feature importance weight: `{imp_val:.2f}`")
    st.write(f"- Overall impact score: `{impact[worst_feat]:.2f}`")

    # 4️⃣ Impact bar chart
    st.markdown("### Feature Impact Across the Match")
    impact_sorted = impact.sort_values(ascending=True)
    impact_df = pd.DataFrame({
        'Feature': [FEATURE_LABELS[f] for f in impact_sorted.index],
        'Impact': impact_sorted.values
    }).set_index('Feature')
    st.bar_chart(impact_df)

if __name__ == "__main__":
    main()
