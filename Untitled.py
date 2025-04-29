import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge

# --- Prediction Manager with Bayesian Ridge ---
class PredictionManager:
    def __init__(self, data):
        self.data = data
        self.model = BayesianRidge()
        self.scaler = MinMaxScaler()
        self.X = None
        self.y = None

    def prepare_data(self):
        features = [
            'Backlinks',
            'Organic_Traffic_Growth_Rate',
            'Keywords_Ranking',
            'Exit_Rate',
            'Average_Page_Load_Time'
        ]
        df = self.data.dropna()
        self.X = self.scaler.fit_transform(df[features])
        self.y = df['CTR (%)']

    def train_model(self):
        self.model.fit(self.X, self.y)

    def predict_ctr(self, new_data=None):
        if new_data is not None:
            new_scaled = self.scaler.transform(new_data)
            pred_mean, pred_std = self.model.predict(new_scaled, return_std=True)
        else:
            pred_mean, pred_std = self.model.predict(self.X, return_std=True)
        return pred_mean, pred_std

# --- User Input Form ---
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

# --- SEO Optimization Simulator ---
def simulate_optimization(data):
    data_sim = data.copy()
    st.subheader("🔧 Simulate SEO Optimization")

    backlinks_boost = st.slider("Increase Backlinks (%)", 0, 100, 20)
    exit_rate_reduction = st.slider("Decrease Exit Rate (%)", 0, 50, 10)
    load_time_reduction = st.slider("Decrease Load Time (%)", 0, 50, 15)

    data_sim['Backlinks'] *= (1 + backlinks_boost/100)
    data_sim['Exit_Rate'] *= (1 - exit_rate_reduction/100)
    data_sim['Average_Page_Load_Time'] *= (1 - load_time_reduction/100)

    return data_sim

# --- SEO Meter Display ---
def display_seo_meter(pre_score, post_score):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pre_score,
        title={"text": "Pre-Optimization SEO CTR (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red"}}
    ))

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=post_score,
        title={"text": "Post-Optimization SEO CTR (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green"}}
    ))

    st.plotly_chart(fig)

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="SEO Bayesian CTR Predictor", layout="centered")
    st.title("🔍 SEO Bayesian Prediction Tool")
    st.markdown("Upload your SEO dataset, predict CTR, and simulate SEO improvements!")

    user_input = UserInputForm()
    user_input.upload_file()

    if user_input.data is not None:
        data = user_input.get_data()

        pm = PredictionManager(data)
        pm.prepare_data()
        pm.train_model()

        # Pre-Optimization Prediction
        pre_mean, pre_std = pm.predict_ctr()
        pre_score = round(pre_mean[-1], 2)

        # Simulate Optimization
        optimized_data = simulate_optimization(data)

        # Post-Optimization Prediction
        optimized_features = optimized_data[[
            'Backlinks',
            'Organic_Traffic_Growth_Rate',
            'Keywords_Ranking',
            'Exit_Rate',
            'Average_Page_Load_Time'
        ]]
        post_mean, post_std = pm.predict_ctr(new_data=optimized_features)
        post_score = round(post_mean[-1], 2)

        # Display Results
        st.subheader("📈 Prediction Results")
        st.write(f"**Pre-Optimization CTR Prediction:** {pre_score}%")
        st.write(f"**Post-Optimization CTR Prediction:** {post_score}%")

        display_seo_meter(pre_score, post_score)

if __name__ == "__main__":
    main()
