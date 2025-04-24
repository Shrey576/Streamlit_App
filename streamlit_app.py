# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

# === 1. DataConsistencyChecker ===
class DataConsistencyChecker:
    def __init__(self, df):
        self.df = df

    def preprocess(self, features, target):
        df = self.df.dropna(subset=features + [target])
        X = df[features].select_dtypes(include='number')
        y = df[target].astype(float)
        return X, y

# === 2. BayesianProgram ===
class BayesianProgram:
    def __init__(self, X, y):
        self.X_raw = X
        self.y = y
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X_raw)
        self.model = None

    def tune_hyperparameters(self):
        def black_box(lambda_1, lambda_2):
            m = BayesianRidge(lambda_1=lambda_1, lambda_2=lambda_2)
            m.fit(self.X, self.y)
            return -np.sqrt(mean_squared_error(self.y, m.predict(self.X)))

        pbounds = {'lambda_1': (1e-6, 1e-1), 'lambda_2': (1e-6, 1e-1)}
        optimizer = BayesianOptimization(f=black_box, pbounds=pbounds, verbose=0, random_state=1)
        optimizer.maximize(init_points=3, n_iter=5)
        best = optimizer.max['params']
        self.model = BayesianRidge(**best).fit(self.X, self.y)

    def forecast_score(self, n_samples=1000):
        means, stds = self.model.predict(self.X, return_std=True)
        loc   = means[np.newaxis, :]
        scale = stds[np.newaxis, :]
        draws = np.random.normal(loc=loc, scale=scale, size=(n_samples, len(means)))
        row_means = draws.mean(axis=0)
        return row_means.mean()

# === Streamlit UI ===
def main():
    st.title("🔮 Bayesian SEO Score Predictor")

    st.write("Automatically uses predefined features and target to compute a single SEO score.")

    uploaded = st.file_uploader("Upload your SEO CSV", type=['csv'])
    if not uploaded:
        return

    df = pd.read_csv(uploaded)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # >>> DEFINE YOUR FEATURE SET HERE <<<
    FEATURES = [
        'CTR (%)',
        'Keywords_Ranking',
        'Organic_Traffic_Growth_Rate (%)',
        'Average_Position',
        'Domain_Authority'
    ]
    TARGET = 'Conversion_Rate (%)'

    # Run the pipeline
    checker = DataConsistencyChecker(df)
    X, y = checker.preprocess(FEATURES, TARGET)

    prog = BayesianProgram(X, y)
    with st.spinner("Tuning & training Bayesian model..."):
        prog.tune_hyperparameters()

    with st.spinner("Sampling posterior and computing SEO score..."):
        seo_score = prog.forecast_score(n_samples=2000)

    st.success("✅ Computation complete!")
    st.metric("Your Predicted SEO Score", f"{seo_score:.2f}")

if __name__ == "__main__":
    main()
