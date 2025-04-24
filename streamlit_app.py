# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.mixture import BayesianGaussianMixture

# === 1. DataConsistencyChecker ===
class DataConsistencyChecker:
    def __init__(self, df):
        self.df = df

    def preprocess(self, features, target):
        # Drop NA, select only numeric, split X/y
        df = self.df.dropna(subset=features + [target])
        X = df[features].select_dtypes(include='number')
        y = df[target].astype(float)
        return X, y

# === 2. HiddenStateIdentifier (GMM) ===
class HiddenStateIdentifier:
    def __init__(self, n_components=5):
        self.model = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42
        )

    def fit_predict(self, X):
        X_scaled = StandardScaler().fit_transform(X)
        return self.model.fit_predict(X_scaled)

# === 3-6. BayesianProgram ===
class BayesianProgram:
    def __init__(self, X, y):
        self.X_raw = X
        self.y = y
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X_raw)
        self.model = None
        self.best_params = {}

    def tune_hyperparameters(self, init_points=3, n_iter=5):
        # Define black-box for Bayesian Ridge (only valid kwargs)
        def black_box(lambda_1, lambda_2):
            m = BayesianRidge(lambda_1=lambda_1, lambda_2=lambda_2)
            m.fit(self.X, self.y)
            y_pred = m.predict(self.X)
            return -np.sqrt(mean_squared_error(self.y, y_pred))

        pbounds = {
            'lambda_1': (1e-6, 1e-1),
            'lambda_2': (1e-6, 1e-1)
        }

        optimizer = BayesianOptimization(
            f=black_box,
            pbounds=pbounds,
            random_state=1,
            verbose=0
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        self.best_params = optimizer.max['params']
        return self.best_params

    def train_model(self):
        # If tuned, apply best params
        if self.best_params:
            self.model = BayesianRidge(**self.best_params)
        else:
            self.model = BayesianRidge()
        self.model.fit(self.X, self.y)

    def posterior_samples(self, X_new, n_samples=1000):
        Xs = self.scaler.transform(X_new)
        means, stds = self.model.predict(Xs, return_std=True)
        draws = np.random.normal(
            loc=means[:, None],
            scale=stds[:, None],
            size=(n_samples, len(means))
        )
        return draws

# === 7. ShapleyValueAnalyzer ===
class ShapleyValueAnalyzer:
    def __init__(self, model, X):
        self.explainer = shap.Explainer(model, X)

    def shap_values(self, X):
        return self.explainer(X)

# === 8. SEOForecast (inherits BayesianProgram) ===
class SEOForecast(BayesianProgram):
    def __init__(self, X, y, forecast_period=12):
        super().__init__(X, y)
        self.forecast_period = forecast_period

    def forecast(self, n_samples=500):
        draws = self.posterior_samples(self.X_raw, n_samples=n_samples)
        row_means = draws.mean(axis=0)
        overall = row_means.mean()
        return {
            'row_means': row_means,
            'overall_score': overall,
            'posterior_draws': draws
        }

# === 9. PredictionManager ===
class PredictionManager:
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.results = {}

    def run_all(self):
        # 1. Preprocess
        checker = DataConsistencyChecker(self.df)
        X, y = checker.preprocess(self.features, self.target)

        # 2. Hidden states
        hsi = HiddenStateIdentifier(n_components=5)
        clusters = hsi.fit_predict(X)
        self.results['clusters'] = clusters

        # 3. Tune & train
        prog = SEOForecast(X, y, forecast_period=5)
        prog.tune_hyperparameters()
        prog.train_model()

        # 4. Forecast
        forecast = prog.forecast(n_samples=1000)
        self.results.update(forecast)

        # 5. Shapley
        shap_analyzer = ShapleyValueAnalyzer(prog.model, prog.X)
        shap_vals = shap_analyzer.shap_values(prog.X)
        self.results['shap_vals'] = shap_vals

        # 6. Store for UI
        self.results['features'] = self.features
        return self.results

# === Streamlit UI ===
def main():
    st.title("🔮 Full Bayesian SEO Analysis & Forecast")

    uploaded = st.file_uploader("Upload your SEO CSV", type=['csv'])
    if not uploaded:
        return

    df = pd.read_csv(uploaded)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Select features & target
    cols = list(df.columns)
    features = st.multiselect("Feature columns", cols, default=cols[:-1])
    target = st.selectbox("Target column", cols, index=len(cols)-1)
    if not features or not target:
        st.warning("Choose features and target!")
        return

    # Run all Bayesian steps
    pm = PredictionManager(df, features, target)
    with st.spinner("Running Bayesian analysis…"):
        res = pm.run_all()

    # 9.1 Clustering plot
    st.subheader("SEO Behavior Clustering (BGMM)")
    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(
        df[features[1]], df[features[0]], c=res['clusters'], cmap='tab10', alpha=0.7
    )
    ax1.set_xlabel(features[1])
    ax1.set_ylabel(features[0])
    ax1.set_title("Fig 1: Clusters of SEO Behavior")
    st.pyplot(fig1)

    # 9.2 Conversion fit (mean vs actual)
    st.subheader("Conversion Prediction vs Actual")
    fig2, ax2 = plt.subplots()
    y_true = df[target].astype(float)
    y_pred = res['row_means']
    ax2.errorbar(y_true, y_pred, fmt='o', alpha=0.6)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Fig 2: Predicted vs Actual SEO Score")
    st.pyplot(fig2)

    # 9.3 Posterior histogram & overall score
    st.subheader("Posterior Distribution & Aggregated SEO Score")
    fig3, ax3 = plt.subplots()
    ax3.hist(res['row_means'], bins=30, alpha=0.7, label='Per-row means')
    ax3.set_title("Posterior & Per-row SEO Scores")
    ax3.set_xlabel("SEO Score")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)
    st.metric("Overall aggregated SEO Score", f"{res['overall_score']:.2f}")

    # 9.4 Feature importance via Shapley
    st.subheader("Feature Contributions (SHAP values)")
    shap.plots.bar(res['shap_vals'], max_display=len(features))
    st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()
