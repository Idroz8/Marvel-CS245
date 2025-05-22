import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def train_model(df):
    features = [
        'dist_to_tank', 'dist_to_healer', 'dist_to_duelist',
        'dist_to_team_centroid', 'dist_to_enemy_centroid', 'near_objective'
    ]
    X = df[features].values
    y = df['label'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def main():
    st.title("Adam Warlock Positioning Predictor")
    st.write("Upload your CSV of 5-second interval data to train and use the SVM model.")

    uploaded_file = st.file_uploader("Upload match data CSV", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
        return

    df = load_data(uploaded_file)
    st.write("**Dataset preview:**")
    st.dataframe(df.head())

    model, scaler = train_model(df)

    st.write("---")
    st.write("## Single Interval Prediction")
    st.write("Adjust the sliders to input your 5-second snapshot features:")

    tank = st.slider("Distance to Tank", 0.0, float(df['dist_to_tank'].max()), float(df['dist_to_tank'].mean()))
    healer = st.slider("Distance to Healer", 0.0, float(df['dist_to_healer'].max()), float(df['dist_to_healer'].mean()))
    duelist = st.slider("Distance to Duelist", 0.0, float(df['dist_to_duelist'].max()), float(df['dist_to_duelist'].mean()))
    team_ctr = st.slider("Distance to Team Centroid", 0.0, float(df['dist_to_team_centroid'].max()), float(df['dist_to_team_centroid'].mean()))
    enemy_ctr = st.slider("Distance to Enemy Centroid", 0.0, float(df['dist_to_enemy_centroid'].max()), float(df['dist_to_enemy_centroid'].mean()))
    near_obj = st.selectbox("Near Objective?", [0, 1])

    if st.button("Predict Positioning Quality"):
        input_df = pd.DataFrame([{
            'dist_to_tank': tank,
            'dist_to_healer': healer,
            'dist_to_duelist': duelist,
            'dist_to_team_centroid': team_ctr,
            'dist_to_enemy_centroid': enemy_ctr,
            'near_objective': near_obj
        }])
        X_input = scaler.transform(input_df.values)
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][pred]
        label = "Good (Top Player)" if pred == 1 else "Bad (Needs Improvement)"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {proba*100:.1f}%")

if __name__ == "__main__":
    main()
