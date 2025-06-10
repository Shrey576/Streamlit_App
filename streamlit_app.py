import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from scipy.optimize import minimize

# --- Core Modular Classes ---

class CTRForecaster:
    def __init__(self, model, X_base):
        self.model = model
        self.X_base = X_base

    def forecast_gaussian(self):
        X_future = np.array([self.X_base * (1 + 0.05 * i) for i in range(5)])
        mean, std = self.model.predict(X_future, return_std=True)
        samples = np.random.normal(loc=mean, scale=std, size=(1000, len(mean)))
        return samples, mean, std


class ClusteringEngine:
    def __init__(self, samples):
        self.samples = samples

    def apply_kmeans(self, n_clusters=3):
        mean_forecasts = self.samples.mean(axis=0)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        km.fit(self.samples)
        centroids = km.cluster_centers_.mean(axis=1)
        optimal_ctr = max(centroids)
        return optimal_ctr, centroids


class ScoreEvaluator:
    @staticmethod
    def compute_deviation(optimal, actual):
        return (optimal - actual) / actual

    @staticmethod
    def compute_seo_score(deviation):
        return 1 / (1 + np.exp(deviation))


class PredictionManager:
    def __init__(self, data):
        self.data = data.dropna()
        self.features = [
            'Backlinks',
            'Organic_Traffic_Growth_Rate (%)',
            'Keywords_Ranking',
            'Exit_Rate (%)',
            'Page_Load_Time (sec)'
        ]
        self.target = 'CTR (%)'
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.data[self.features])
        self.y = self.data[self.target]
        self.model = BayesianRidge()
        self.lr = LinearRegression()

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.lr.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_lr_pred = self.lr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Baseline SEO score from linear regression
        baseline_score = np.mean(y_lr_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Baseline SEO Score": baseline_score
        }

    def run_pipeline(self):
        X_base = self.X.mean(axis=0)
        forecaster = CTRForecaster(self.model, X_base)
        samples, mean_forecast, std_forecast = forecaster.forecast_gaussian()

        clustering = ClusteringEngine(samples)
        optimal_ctr, centroids = clustering.apply_kmeans()

        actual_ctr = self.y.mean()
        deviation = ScoreEvaluator.compute_deviation(optimal_ctr, actual_ctr)
        seo_score = ScoreEvaluator.compute_seo_score(deviation)

        return {
            "forecast_samples": samples,
            "mean_forecast": mean_forecast,
            "std_forecast": std_forecast,
            "optimal_ctr": optimal_ctr,
            "actual_ctr": actual_ctr,
            "deviation": deviation,
            "seo_score": seo_score,
            "centroids": centroids
        }

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="CTR Forecasting Tool", layout="centered")
    st.title("CTR Forecasting and SEO Scoring Tool")

    uploaded_file = st.file_uploader("Upload your SEO dataset (.csv)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data")
        st.dataframe(data.head())

        manager = PredictionManager(data)
        metrics = manager.train_models()
        results = manager.run_pipeline()

        # Display Metrics
        st.subheader("Model Performance")
        for k, v in metrics.items():
            st.write(f"**{k}**: {v:.3f}")

        # CTR Forecast Table
        st.subheader("Forecasted CTR (2025–2029)")
        forecast_df = pd.DataFrame({
            "Year": [2025, 2026, 2027, 2028, 2029],
            "Mean CTR (µ)": results['mean_forecast'],
            "Std Dev (σ)": results['std_forecast']
        })
        st.dataframe(forecast_df)

        st.subheader("Optimisation Summary")
        st.write(f"**Actual CTR (2024):** {results['actual_ctr']:.3f}")
        st.write(f"**Optimised CTR (max cluster):** {results['optimal_ctr']:.3f}")
        st.write(f"**Deviation:** {results['deviation']:.3f}")
        st.write(f"**Final SEO Score:** {results['seo_score']:.3f}")

        st.subheader("CTR Clustering Centroids")
        st.write(results['centroids'])

if __name__ == "__main__":
    main()
