#!/bin/bash

# Exit on any error
set -e

echo "Compiling C++ code..."
g++ main.cpp -o data_processor -std=c++11

echo "Running data processor..."
./data_processor

echo "Running Python plotting script..."
python3 plots.py

echo "Running MCMC estimation script..."
python3 mcmc.py

echo "Done. Images are saved in the 'images' directory:"
echo "- price_trend.png: Shows the XRP price over time"
echo "- returns_histogram.png: Distribution of log returns" 
echo "- volatility_analysis.png: Rolling mean and volatility"
echo "- mcmc_posteriors.png: Bayesian posterior distributions for mean and volatility"
echo "- posterior_predictive.png: Predicted distribution of future returns"
echo "- price_forecast.png: Price forecasts for next day, week, and month"
echo ""
echo "Trading analysis and comprehensive price forecasts have been saved to analysis.md"

# Clean up intermediate files (optional)
rm data_processor prices.csv returns.csv rolling_stats.csv