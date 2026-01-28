import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ---------------- BACKGROUND IMAGE ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(0,0,0,0.65),
                rgba(0,0,0,0.65)
            ),
            url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        label, .stSlider label {{
            color: white !important;
            font-weight: 600;
        }}

        h1 {{
            color: white !important;
            font-weight: 800;
            font-size: 3rem !important;
            text-align: center;
        }}

        h2, h3, p {{
            color: white !important;
        }}

        .card {{
            background: rgba(0,0,0,0.55);
            padding: 1.8rem;
            border-radius: 16px;
            margin-bottom: 1rem;
        }}

        .result {{
            padding: 0.9rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
            color: white;
        }}

        .high {{
            background: #2ecc71;
        }}

        .avg {{
            background: #f1c40f;
            color: black;
        }}

        .low {{
            background: #e74c3c;
        }}

        button {{
            width: 100%;
            height: 3rem;
            font-size: 1.1rem;
            border-radius: 10px;
            background: linear-gradient(90deg, #e74c3c, #f39c12);
            color: white !important;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: 0.3s ease;
        }}

        button:hover {{
            background: linear-gradient(90deg, #f39c12, #e74c3c);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.jpg")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1>🍷 Wine Quality Prediction</h1>
    <p style="text-align:center;">Simple & Premium ML Web App</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- TRAIN REAL MODEL ----------------
@st.cache_resource
def load_model():
    # Load dataset
    df = pd.read_csv("winequality-red.csv", sep=";")

    # Map quality into 3 labels: 0=Low, 1=Average, 2=High
    df["label"] = df["quality"].apply(lambda q: 2 if q >= 7 else 1 if q >= 5 else 0)

    X = df.drop(["quality", "label"], axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist()  # Return feature names too

model, feature_names = load_model()

# ---------------- LAYOUT ----------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🍇 Wine Details")

    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.6, 0.45)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.0)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.04)
    free_so2 = st.slider("Free Sulfur Dioxide", 1, 80, 30)
    total_so2 = st.slider("Total Sulfur Dioxide", 6, 300, 100)
    density = st.slider("Density", 0.9900, 1.0050, 0.9940, step=0.0001)
    ph = st.slider("pH", 2.8, 4.0, 3.3)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.75)
    alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 12.0)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Result")

    if st.button("🍷 Predict Wine Quality"):
        # Create DataFrame with feature names to avoid sklearn warning
        features = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid,
                                  residual_sugar, chlorides, free_so2,
                                  total_so2, density, ph, sulphates, alcohol]],
                                columns=feature_names)

        pred = model.predict(features)[0]

        if pred == 2:
            st.markdown("<div class='result high'>🍾 High Quality Wine</div>", unsafe_allow_html=True)
        elif pred == 1:
            st.markdown("<div class='result avg'>🍷 Average Quality Wine</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result low'>❌ Low Quality Wine</div>", unsafe_allow_html=True)

    else:
        st.info("Adjust values and click Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#ddd;">Made with ❤️ using Streamlit</p>
    """,
    unsafe_allow_html=True
)
