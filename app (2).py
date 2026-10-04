import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("Mental_Health_Model.pkl")


model = load_model()

st.title("🧠 Student Social Media & Mental Health Predictor")
st.write(
    "Enter a student's habits and profile to predict their **Mental Health Score** "
    "(higher = better mental well-being)."
)

PLATFORMS = [
    "Instagram", "TikTok", "Facebook", "LinkedIn", "YouTube", "Twitter",
    "Snapchat", "WhatsApp", "LINE", "VKontakte", "KakaoTalk", "WeChat",
]
COUNTRIES = ["Australia", "Canada", "France", "Germany", "India",
             "Mexico", "Turkey", "UK", "USA", "Other"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profile")
    age = st.number_input("Age", min_value=18, max_value=24, value=21, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("India"))
    platform = st.selectbox("Most Used Platform", PLATFORMS)
    purpose = st.selectbox("Purpose of Use", ["Education", "Entertainment", "Networking", "News"])

with col2:
    st.subheader("Daily Habits")
    usage_hours = st.slider("Avg Daily Social Media Usage (hours)", 1.0, 9.0, 5.0, 0.1)
    unlocks = st.slider("Daily Phone Unlocks", 60, 280, 170, 1)
    study_hours = st.slider("Study Hours per Day", 0.3, 8.5, 3.0, 0.1)
    physical = st.slider("Physical Activity (hours per day)", 0.0, 4.5, 1.7, 0.1)
    sleep = st.slider("Sleep Hours per Night", 3.5, 10.0, 6.6, 0.1)
    stress = st.select_slider("Stress Level", options=["Low", "Medium", "High", "Very High"], value="Medium")

if st.button("Predict Mental Health Score", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Age": age,
        "Avg_Daily_Usage_Hours": usage_hours,
        "Daily_Unlocks": unlocks,
        "Physical_Activity_Hours": physical,
        "Sleep_Hours_Per_Night": sleep,
        "Stress_Level": stress,
        "Gender": gender,
        "Academic_Level": academic_level,
        "Most_Used_Platform": platform,
        "Purpose_Of_Use": purpose,
        "Grouped_Country": country,
    }])

    score = float(model.predict(input_df)[0])
    score = max(0.0, min(10.0, score))

    st.divider()
    st.metric("Predicted Mental Health Score", f"{score:.2f} / 10")
    st.progress(score / 10)

    if score >= 7:
        st.success("Good mental well-being predicted.")
    elif score >= 5.5:
        st.warning("Moderate mental well-being predicted. Consider healthier sleep, activity and screen-time habits.")
    else:
        st.error("Lower mental well-being predicted. Reducing screen time and improving sleep may help.")

    with st.expander("Input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.caption("Model: Random Forest Regressor trained on the Student Social Media & Mental Health dataset. "
           "This is an educational tool, not a medical diagnosis.")
