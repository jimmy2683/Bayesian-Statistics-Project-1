# Bayesian Estimation of Daily Return and Volatility of XRP/USDT using MCMC

**Project-1**  
**MA4740: Bayesian Statistics**  
**(Under the guidance of Prof. Arunabha Majumdar)**

---

## Project Description

This project applies Bayesian inference to XRP/USDT daily price data. Using a Normal-Inverse-Gamma model and Rejection Sampling (an MCMC method), we estimate posterior distributions for the mean return and volatility, compute credible intervals, and visualize the posterior predictive distribution to analyze uncertainty in cryptocurrency price behavior.

## Features

- Data processing and descriptive analysis
- Computation of log returns and volatility measures
- Bayesian inference using Rejection Sampling
- Posterior distribution estimation for return mean and volatility
- Posterior predictive distribution for future returns
- Visualization of price trends, return distributions, and volatility
- Price forecasts for next day, week, and month horizons
- Comprehensive trading analysis report generation
- LaTeX report with embedded code and visualizations

## How to Run

1. Ensure you have the required libraries installed:
   ```
   pip install numpy pandas matplotlib scipy
   ```

2. Run the analysis:
   ```
   chmod +x run.sh
   ./run.sh
   ```

3. View the generated images in the `images` directory.

4. Compile the LaTeX report (optional):
   ```
   pdflatex report.tex
   ```

## Output Files

### Visualizations

The project generates the following visualization files in the `images` directory:

- `price_trend.png`: Shows the XRP price trend over time
- `returns_histogram.png`: Displays the distribution of daily log returns
- `volatility_analysis.png`: Shows the rolling mean and volatility of price
- `mcmc_posteriors.png`: Displays the posterior distributions of mean return (μ) and volatility (σ)
- `posterior_predictive.png`: Shows the predicted distribution of future returns
- `price_forecast.png`: Visualizes price predictions for different time horizons

### Reports

- `analysis.md`: Comprehensive trading analysis report with:
  - Executive summary with key findings
  - Price forecasts with confidence intervals
  - Trading recommendations and signals
  - Detailed risk assessment
  - Short-term and extended horizon predictions

- `report.tex`: LaTeX document containing:
  - Complete methodology description
  - Embedded source code (C++ and Python)
  - All generated visualizations
  - Mathematical formulations using proper notation (μ, σ)

## Methodology

1. **Data Preprocessing**: Load and clean XRP price data, compute log returns
2. **Descriptive Analysis**: Calculate statistical properties of returns
3. **Bayesian Modeling**: 
   - Use Normal likelihood for returns
   - Apply Normal prior for mean return
   - Apply Inverse-Gamma prior for return variance
4. **MCMC Estimation**: Rejection sampling to generate posterior samples
5. **Posterior Analysis**: Calculate credible intervals and predictive distributions
6. **Trading Analysis**: Generate price forecasts and risk-adjusted recommendations

## Interpretation

The results provide a Bayesian perspective on XRP's price behavior, allowing for a probabilistic assessment of future returns and volatility. The credible intervals quantify uncertainty in these estimates, and the trading analysis report provides actionable insights based on the Bayesian model.

---

**Course:** MA4740 - Bayesian Statistics  
**Instructor:** Prof. Arunabha Majumdar  
**Academic Year:** 2024-2025
