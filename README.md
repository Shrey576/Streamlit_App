
# CTR Forecasting & SEO Scoring Tool  

## Overview  
This project presents a **predictive SEO analytics tool** built using **Bayesian Ridge Regression (BRR)** for **Click-Through Rate (CTR) forecasting**. Designed for **small and medium-sized enterprises (SMEs)**, it enables data-driven SEO optimization by offering transparent, probabilistic predictions with confidence intervals.  

## Key Features  
- **CTR Forecasting** – Uses Bayesian Ridge Regression to predict future CTR trends.  
- **Monte Carlo Sampling** – Simulates possible CTR distributions for **uncertainty-aware forecasting**.  
- **K-Means Clustering** – Identifies **optimal CTR scenarios** by segmenting predictions.  
- **Linear Regression Baseline** – Provides an **SEO Score** based on weighted feature contributions.  
- **Streamlit Interface** – Allows users to **upload datasets, run forecasts, and visualize results interactively**.  

## Installation  
### 1. Clone the Repository  
```bash  
git clone https://github.com/your-repo-name.git  
cd your-repo-name  
```  
### 2. Install Dependencies  
Ensure you have Python **3.8+** installed, then run:  
```bash  
pip install -r requirements.txt
NB: Please also install the plotly library/use a machine that has plotly installed to avoid dependency issues 
```  
### 3. Run the Application  
Launch the Streamlit app with:  
```bash  
streamlit run streamlit_app.py  
```  
### 4. Upload Dataset
Upload Demo .csv dataset provided. 

## Usage  
1. **Upload a `.csv` dataset** containing SEO metrics.  
2. **Run the prediction pipeline** to generate CTR forecasts.  
3. **View probabilistic CTR outputs** alongside the SEO Score.  
4. **Analyze confidence intervals and cluster insights** for optimization.  

## Tech Stack  
- **Python** (Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib, Plotly)  
- **Machine Learning** (Bayesian Ridge Regression, K-Means Clustering)  
- **Monte Carlo Sampling** (Uncertainty Quantification)  
- **Streamlit** (Interactive UI for real-time SEO predictions)  

## Future Work  
- Implement **Bayesian Neural Networks** for deeper uncertainty modeling.  
- Introduce **real-time SEO data integration** via web scraping or APIs.  
- Enhance **ticket-based architecture** for scalable asynchronous predictions.  

## Contributor 
- **Shreya Sharma** – **University of Reading, Computer Science**

## Supervisor – Dr. Todd Jones  

## License  
This project is released under the **MIT License**.  

 

