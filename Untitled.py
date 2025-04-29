import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
import shap

# --- Ridge Linear Regression Skeleton (BayesianRidge) ---
# This supports interoperability and provides a lightweight probabilistic core

class PredictionManager:
    def __init__(self, data):
        self.data = data
        self.model = BayesianRidge()
        self.scaler = MinMaxScaler()
        self.X = None
        self.y = None
        self.explainer = None

    # --- Stage 1: Hidden State Identification ---
    # Simulating latent factors through data inspection and preprocessing
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

    # --- Stage 2: Gaussian Probability Estimation ---
    # Fit Bayesian Ridge under Gaussian prior assumptions
    def train_model(self):
        self.model.fit(self.X, self.y)
        self.explainer = shap.LinearExplainer(self.model, self.X)

    # --- Stages 3 & 4: Likelihood Estimation & Posterior Computation ---
    # Handled internally in BayesianRidge using evidence approximation
    def predict_ctr(self, new_data=None):
        if new_data is not None:
            new_scaled = self.scaler.transform(new_data)
            pred_mean, pred_std = self.model.predict(new_scaled, return_std=True)
        else:
            pred_mean, pred_std = self.model.predict(self.X, return_std=True)
        return pred_mean, pred_std

    # --- Stage 5: Monte Carlo Sampling (Simulated) ---
    def monte_carlo_simulation(self, mean, std, n_samples=100):
        return np.random.normal(loc=mean, scale=std, size=n_samples)

    # --- Stage 6: Model Evaluation & Tuning (Simplified) ---
    def evaluate_model(self):
        return self.model.score(self.X, self.y)

    # --- Stage 7: SHAP Feature Importance ---
    def get_shap_values(self):
        shap_values = self.explainer(self.X)
        return shap_values

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
# Simulates the effects of SEO strategy improvements

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

# --- Stage 8: SEO Score Calculation ---
# Shows predicted CTR % as a final actionable metric

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

        pre_mean, pre_std = pm.predict_ctr()
        pre_score = round(pre_mean[-1], 2)

        optimized_data = simulate_optimization(data)
        optimized_features = optimized_data[[
            'Backlinks',
            'Organic_Traffic_Growth_Rate',
            'Keywords_Ranking',
            'Exit_Rate',
            'Average_Page_Load_Time'
        ]]
        post_mean, post_std = pm.predict_ctr(new_data=optimized_features)
        post_score = round(post_mean[-1], 2)

        st.subheader("📈 Prediction Results")
        st.write(f"**Pre-Optimization CTR Prediction:** {pre_score}%")
        st.write(f"**Post-Optimization CTR Prediction:** {post_score}%")

        st.subheader("🧪 Simulated Monte Carlo Sampling")
        samples = pm.monte_carlo_simulation(post_mean[-1], post_std[-1])
        st.write(f"Prediction 95% Range: {round(np.percentile(samples, 2.5), 2)}% - {round(np.percentile(samples, 97.5), 2)}%")

        display_seo_meter(pre_score, post_score)

        st.subheader("📊 SHAP Feature Importance")
        shap_values = pm.get_shap_values()
        st.pyplot(shap.summary_plot(shap_values, pm.X, feature_names=[
            'Backlinks',
            'Organic_Traffic_Growth_Rate',
            'Keywords_Ranking',
            'Exit_Rate',
            'Average_Page_Load_Time'
        ], show=False))

if __name__ == "__main__":
    main()
