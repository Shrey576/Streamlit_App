import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

# Updated PredictionManager using Ridge Regression
class PredictionManager:
    def __init__(self, data):
        self.data = data
        self.result = None

    def run_prediction(self):
        # Select relevant features
        features = [
            'Backlinks',
            'Organic_Traffic_Growth_Rate',
            'Keywords_Ranking',
            'CTR (%)',
            'Exit_Rate',
            'Average_Page_Load_Time'
        ]

        # Filter only those columns
        df = self.data[features].dropna()

        # Normalize data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df)

        # Generate dummy target values for Ridge regression
        # For illustration, create a synthetic target
        # In a real app, you'd train on known historical SEO scores
        y = X.mean(axis=1)

        # Fit the Ridge Regression model
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Predict on the same input for demo (or could be test split)
        predictions = model.predict(X)

        # Take the most recent row as current SEO score
        self.result = round(predictions[-1] * 100, 2)  # Scale to 0-100

    def get_results(self):
        return self.result

# User upload form
class UserInputForm:
    def __init__(self):
        self.data = None

    def upload_file(self):
        file = st.file_uploader("Upload your SEO dataset (.csv)", type=["csv"])
        if file is not None:
            self.data = pd.read_csv(file)
            st.write("📄 Preview of uploaded data:")
            st.write(self.data.head())

    def get_data(self):
        return self.data

# Display gauge chart
def display_seo_meter(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "SEO Health Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#f44336"},     # Poor
                {'range': [40, 60], 'color': "#ff9800"},    # Fair
                {'range': [60, 80], 'color': "#ffeb3b"},    # Good
                {'range': [80, 100], 'color': "#4caf50"}    # Excellent
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    st.plotly_chart(fig)

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="SEO Score Predictor", layout="centered")
    st.title("📈 SEO Prediction & Scoring Tool")
    st.markdown("Upload your SEO data and get a health score visualized.")

    user_input_form = UserInputForm()
    user_input_form.upload_file()

    if user_input_form.data is not None:
        data = user_input_form.get_data()

        # Run prediction
        prediction_manager = PredictionManager(data)
        prediction_manager.run_prediction()

        # Show results
        score = prediction
