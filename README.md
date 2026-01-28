# Wine Quality Prediction Web App
Live Demo
# Live App: https://wine-quality-pdk.streamlit.app/
Predict the quality of red wine based on its chemical properties using Machine Learning. This app provides an interactive, easy-to-use interface for wine quality prediction, helping wine enthusiasts, sommeliers, and producers make informed decisions.

# 📝 Problem Statement

Wine quality is determined by many chemical properties such as acidity, sugar content, pH, sulphates, and alcohol percentage. Manually analyzing these properties to assess quality can be time-consuming and subjective.

Goal: Build a Machine Learning model that can predict the quality of wine based on measurable chemical attributes. The output is classified into three categories:

High Quality Wine (quality ≥ 7)

Average Quality Wine (quality 5–6)

Low Quality Wine (quality < 5)

This prediction system helps users quickly estimate wine quality without lab testing.

# 💡 Project Overview

This project uses the Random Forest Classifier to predict wine quality. It includes a web-based interface built with Streamlit for interactive user input and visualization.

# Steps in the project:

Data Collection:

Dataset: Wine Quality Dataset (Red Wine)

Contains chemical properties like acidity, chlorides, sulphates, alcohol, etc., and a quality score (0–10).

Data Preprocessing:

Map the original quality score into Low, Average, High classes.

Clean and normalize features.

Model Training:

Split dataset into train-test sets (80%-20%).

Train a Random Forest Classifier.

Save the model for deployment (wine_quality_model.pkl).

# Web App Development:

Users adjust sliders for wine properties.

Predict button triggers the model.

Output is displayed in a color-coded card:

Green → High Quality

Yellow → Average Quality

Red → Low Quality

# Deployment:

The app is deployed on Streamlit Cloud: https://wine-quality-pdk.streamlit.app/

# 🛠 Tech Stack

Python 3.11+

Streamlit – Web interface

Scikit-learn – Machine Learning (Random Forest Classifier)

Pandas & NumPy – Data handling

HTML & CSS – Custom styling for UI

# 🎨 Features

Interactive UI: Adjust wine chemical properties via sliders.

Real-time Prediction: Get instant wine quality classification.

Beautiful Design: Dark-themed layout, cards, gradient buttons, background image.

Portable Model: Saved .pkl model for reuse in other Python projects.

# 🚀 Installation

Clone the repository:

git clone https://github.com/CheboluGayatri/Wine-Quality.git
cd wine-quality-prediction


# Install dependencies:

pip install -r requirements.txt


Place necessary files in the project directory:

# winequality-red.csv → dataset

bg.jpg → background image

# 🖥 Run the App Locally
streamlit run app.py --server.port 8502


Open http://localhost:8501
 in your browser.

Or use the live demo: https://wine-quality-pdk.streamlit.app/

📊 Model Accuracy

The Random Forest Classifier achieves an accuracy of approximately:

Accuracy: 87%


# This ensures reliable predictions for wine quality.

🧠 How It Works

Input: User sets values for wine properties via sliders.

Model Prediction: The trained Random Forest model predicts quality class.

Output Display: The result is shown in a color-coded card:

🍾 High Quality → Green

🍷 Average → Yellow

❌ Low → Red

# 📈 Future Enhancements

Include white wine quality prediction.

Display confidence/probability scores.

Add charts & graphs for feature importance.

Deploy on Heroku or AWS for scalable access.

Integrate user authentication for personalized predictions.
