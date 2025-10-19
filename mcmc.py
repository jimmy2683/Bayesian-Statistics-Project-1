import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm
import os

def run_mcmc_estimation():
    """
    Performs Bayesian estimation of return mean and volatility using Rejection Sampling.
    """
    # Create images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
        print("Created 'images' directory for storing plots")

    # 1. Load the log returns data
    try:
        returns = pd.read_csv('returns.csv')['LogReturn'].dropna().to_numpy()
    except FileNotFoundError:
        print("Error: returns.csv not found. Please run the data_processor first.")
        return

    n = len(returns)
    r_bar = np.mean(returns)
    r_var = np.var(returns)

    # 2. Set Priors (non-informative)
    # For mu: Normal(mu_0, sigma_0^2)
    mu_0 = 0.0
    sigma_0_sq = 1000.0
    # For sigma^2: Inv-Gamma(alpha_0, beta_0)
    alpha_0 = 0.001
    beta_0 = 0.001

    # 3. Rejection Sampling setup
    n_samples = 2000  # Target number of samples
    max_attempts = 1000000  # Maximum attempts to avoid infinite loops
    
    # Initialize storage for accepted samples
    mu_samples = []
    sigma_sq_samples = []

    # 4. Run Rejection Sampling
    print("Running Rejection Sampling...")
    attempts = 0
    accepted = 0
    
    # Define proposal distributions (use simple Normal and Inv-Gamma)
    # Proposal for mu: Normal centered at sample mean with larger variance
    mu_proposal_mean = r_bar
    mu_proposal_std = np.sqrt(r_var / n + sigma_0_sq)
    
    # Proposal for sigma^2: Inverse-Gamma with parameters based on data
    sigma_proposal_alpha = n / 2
    sigma_proposal_beta = 0.5 * np.sum((returns - r_bar)**2)
    
    # Calculate envelope constant M (upper bound on acceptance ratio)
    # For simplicity, use a conservative value
    M = 10.0
    
    while accepted < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Draw from proposal distributions
        mu_prop = np.random.normal(mu_proposal_mean, mu_proposal_std)
        sigma_sq_prop = invgamma.rvs(a=sigma_proposal_alpha, scale=sigma_proposal_beta)
        
        # Calculate likelihood: product of Normal(mu, sigma^2) for each return
        likelihood = np.prod(norm.pdf(returns, loc=mu_prop, scale=np.sqrt(sigma_sq_prop)))
        
        # Calculate prior
        prior_mu = norm.pdf(mu_prop, loc=mu_0, scale=np.sqrt(sigma_0_sq))
        prior_sigma_sq = invgamma.pdf(sigma_sq_prop, a=alpha_0, scale=beta_0)
        prior = prior_mu * prior_sigma_sq
        
        # Calculate proposal density
        proposal_mu = norm.pdf(mu_prop, loc=mu_proposal_mean, scale=mu_proposal_std)
        proposal_sigma_sq = invgamma.pdf(sigma_sq_prop, a=sigma_proposal_alpha, scale=sigma_proposal_beta)
        proposal = proposal_mu * proposal_sigma_sq
        
        # Calculate unnormalized posterior
        posterior = likelihood * prior
        
        # Calculate acceptance ratio
        if proposal > 0:
            acceptance_ratio = posterior / (M * proposal)
        else:
            acceptance_ratio = 0
        
        # Accept or reject
        if np.random.uniform(0, 1) < acceptance_ratio:
            mu_samples.append(mu_prop)
            sigma_sq_samples.append(sigma_sq_prop)
            accepted += 1

        # Print progress every 100000 attempts
        if attempts % 100000 == 0:
            acceptance_rate = accepted / attempts * 100
            print(f"Attempts: {attempts}, Accepted: {accepted}, Acceptance Rate: {acceptance_rate:.2f}%")
    
    if accepted < n_samples:
        print(f"Warning: Only {accepted} samples accepted out of {n_samples} target after {attempts} attempts")
        print(f"Acceptance rate: {accepted/attempts*100:.2f}%")
    else:
        print(f"Successfully generated {n_samples} samples with acceptance rate: {accepted/attempts*100:.2f}%")
    
    # Convert to numpy arrays
    mu_posterior = np.array(mu_samples)
    sigma_posterior = np.sqrt(np.array(sigma_sq_samples))  # Convert variance to std dev

    # 5. Visualize posterior distributions
    plt.figure(figsize=(14, 6))

    # Posterior for mu
    plt.subplot(1, 2, 1)
    plt.hist(mu_posterior, bins=50, density=True, alpha=0.7, label='Posterior of μ')
    plt.title('Posterior Distribution of Mean Return (μ)')
    plt.xlabel('Mean Return')
    plt.ylabel('Density')
    plt.grid(True)

    # Posterior for sigma
    plt.subplot(1, 2, 2)
    plt.hist(sigma_posterior, bins=50, density=True, alpha=0.7, label='Posterior of σ', color='green')
    plt.title('Posterior Distribution of Volatility (σ)')
    plt.xlabel('Volatility (Std. Dev. of Returns)')
    plt.ylabel('Density')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('images/mcmc_posteriors.png')
    plt.close()
    print("Generated mcmc_posteriors.png")

    # 6. Generate and visualize posterior predictive distribution
    # Sample future returns based on the posterior distributions
    posterior_predictive = np.random.normal(
        loc=np.random.choice(mu_posterior, size=5000),
        scale=np.random.choice(sigma_posterior, size=5000)
    )
    
    plt.figure(figsize=(10, 6))
    plt.hist(posterior_predictive, bins=50, density=True, alpha=0.7, color='purple')
    plt.title('Posterior Predictive Distribution of Future Return')
    plt.xlabel('Predicted Return')
    plt.ylabel('Density')
    plt.grid(True)
    
    # Add vertical line at 0 for reference
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add a 95% credible interval on the plot
    pred_ci = np.percentile(posterior_predictive, [2.5, 97.5])
    plt.axvline(x=pred_ci[0], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=pred_ci[1], color='blue', linestyle='--', alpha=0.5)
    plt.annotate(f'95% CI: [{pred_ci[0]:.4f}, {pred_ci[1]:.4f}]', 
                xy=(0.05, 0.92), xycoords='axes fraction', fontsize=10)
    
    plt.savefig('images/posterior_predictive.png')
    plt.close()
    print("Generated posterior_predictive.png")

    # 7. Report credible intervals
    mu_ci = np.percentile(mu_posterior, [2.5, 97.5])
    sigma_ci = np.percentile(sigma_posterior, [2.5, 97.5])

    print("\n--- Rejection Sampling Results ---")
    print(f"95% Credible Interval for Mean Return (mu): [{mu_ci[0]:.6f}, {mu_ci[1]:.6f}]")
    print(f"95% Credible Interval for Volatility (sigma): [{sigma_ci[0]:.6f}, {sigma_ci[1]:.6f}]")
    print(f"95% Credible Interval for Predictive Return: [{pred_ci[0]:.6f}, {pred_ci[1]:.6f}]")
    print("\nInterpretation:")
    print(f"- Expected daily return is approximately {np.mean(mu_posterior):.6f}")
    print(f"- The average volatility is {np.mean(sigma_posterior):.6f}")
    print("- The wide credible intervals reflect high uncertainty in cryptocurrency returns")
    print("- There is a {:.1f}% probability of a positive return on any given day".format(
        100 * np.mean(posterior_predictive > 0)))
        
    # 8. Generate dynamic analysis.md file
    generate_analysis_report(returns, mu_posterior, sigma_posterior, posterior_predictive, 
                           mu_ci, sigma_ci, pred_ci)

def generate_analysis_report(returns, mu_posterior, sigma_posterior, posterior_predictive, 
                           mu_ci, sigma_ci, pred_ci):
    """
    Generates a comprehensive analysis report based on MCMC results
    """
    # Try to determine the asset name from the plots.py file
    asset_name = "Financial Asset"
    try:
        with open('plots.py', 'r') as f:
            for line in f:
                if "plt.title('" in line and "Price Trend')" in line:
                    start = line.find("plt.title('") + len("plt.title('")
                    end = line.find(" Price Trend')")
                    if end > start:
                        asset_name = line[start:end]
                    break
    except:
        pass
    
    # Calculate key statistics
    mean_return = np.mean(mu_posterior)
    mean_volatility = np.mean(sigma_posterior)
    prob_positive = np.mean(posterior_predictive > 0) * 100
    
    # Determine risk level
    if mean_volatility > 0.1:
        risk_level = "HIGH"
    elif mean_volatility > 0.05:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Determine trading signal
    if prob_positive > 60:
        signal = "BUY"
        confidence = "HIGH" if mu_ci[0] > 0 else "MEDIUM"
    elif prob_positive < 40:
        signal = "SELL"
        confidence = "HIGH" if mu_ci[1] < 0 else "MEDIUM"
    else:
        signal = "NEUTRAL"
        confidence = "MEDIUM" if (mu_ci[1] - mu_ci[0]) < 0.01 else "LOW"
    
    # Calculate position sizing recommendation
    base_position = 1.0  # Base position size (100%)
    target_volatility = 0.05  # Target volatility level
    vol_adjustment = min(1.0, target_volatility / mean_volatility)
    directional_confidence = (prob_positive - 50) * 0.02  # -1.0 to 1.0
    position_size = base_position * vol_adjustment * (0.5 + (directional_confidence * 0.5))
    position_size = max(0.1, min(1.0, position_size))  # Clamp between 10% and 100%
    
    # Format the position size as a percentage
    position_size_pct = position_size * 100
    
    # Get recent price data
    recent_price = None
    try:
        prices_df = pd.read_csv('prices.csv')
        if len(prices_df) > 0:
            recent_price = prices_df['Price'].iloc[-1]
    except:
        pass
    
    # Generate price predictions for different time horizons
    price_predictions = {}
    if recent_price is not None:
        # Next day prediction - simulate using one-day returns
        next_day_returns = np.random.normal(
            loc=np.random.choice(mu_posterior, size=10000),
            scale=np.random.choice(sigma_posterior, size=10000)
        )
        next_day_prices = recent_price * np.exp(next_day_returns)
        next_day_ci = np.percentile(next_day_prices, [2.5, 97.5])
        
        # Next week prediction - simulate 5 trading days
        next_week_returns = np.zeros(10000)
        for i in range(5):  # 5 trading days in a week
            daily_returns = np.random.normal(
                loc=np.random.choice(mu_posterior, size=10000),
                scale=np.random.choice(sigma_posterior, size=10000)
            )
            next_week_returns += daily_returns
        next_week_prices = recent_price * np.exp(next_week_returns)
        next_week_ci = np.percentile(next_week_prices, [2.5, 97.5])
        
        # Next month prediction - simulate 22 trading days
        next_month_returns = np.zeros(10000)
        for i in range(22):  # ~22 trading days in a month
            daily_returns = np.random.normal(
                loc=np.random.choice(mu_posterior, size=10000),
                scale=np.random.choice(sigma_posterior, size=10000)
            )
            next_month_returns += daily_returns
        next_month_prices = recent_price * np.exp(next_month_returns)
        next_month_ci = np.percentile(next_month_prices, [2.5, 97.5])
        
        price_predictions = {
            'next_day': {
                'low': next_day_ci[0],
                'high': next_day_ci[1],
                'expected': recent_price * np.exp(mean_return)
            },
            'next_week': {
                'low': next_week_ci[0],
                'high': next_week_ci[1],
                'expected': recent_price * np.exp(5 * mean_return)
            },
            'next_month': {
                'low': next_month_ci[0],
                'high': next_month_ci[1],
                'expected': recent_price * np.exp(22 * mean_return)
            }
        }
    
    # Create the analysis report
    with open('analysis.md', 'w') as f:
        f.write(f"# {asset_name} Trading Analysis Report\n\n")
        f.write("## Executive Summary\n")
        f.write(f"Based on Bayesian MCMC analysis of historical price data, the following insights and recommendations are generated:\n\n")
        
        f.write("### Key Findings\n")
        f.write(f"- **Expected Daily Return:** {mean_return:.6f} ({'+' if mean_return > 0 else ''}{mean_return*100:.4f}%)\n")
        f.write(f"- **Daily Volatility:** {mean_volatility:.6f} ({mean_volatility*100:.4f}%)\n")
        f.write(f"- **Probability of Positive Return:** {prob_positive:.1f}%\n")
        f.write(f"- **Risk Level:** {risk_level}\n")
        f.write(f"- **95% Credible Interval for Mean Return:** [{mu_ci[0]:.6f}, {mu_ci[1]:.6f}]\n")
        f.write(f"- **95% Credible Interval for Volatility:** [{sigma_ci[0]:.6f}, {sigma_ci[1]:.6f}]\n")
        f.write(f"- **95% Predictive Interval for Next-Day Return:** [{pred_ci[0]:.6f}, {pred_ci[1]:.6f}]\n\n")
        
        # Add price predictions if available
        if price_predictions and recent_price is not None:
            f.write("### Price Forecasts (95% Credible Intervals)\n")
            f.write(f"Current Price: {recent_price:.6f}\n\n")
            
            f.write("| Time Horizon | Expected Price | Lower Bound | Upper Bound | Range |\n")
            f.write("|--------------|---------------|-------------|-------------|--------------|\n")
            
            next_day = price_predictions['next_day']
            f.write(f"| Next Day | {next_day['expected']:.6f} | {next_day['low']:.6f} | {next_day['high']:.6f} | {next_day['high'] - next_day['low']:.6f} |\n")
            
            next_week = price_predictions['next_week']
            f.write(f"| Next Week | {next_week['expected']:.6f} | {next_week['low']:.6f} | {next_week['high']:.6f} | {next_week['high'] - next_week['low']:.6f} |\n")
            
            next_month = price_predictions['next_month']
            f.write(f"| Next Month | {next_month['expected']:.6f} | {next_month['low']:.6f} | {next_month['high']:.6f} | {next_month['high'] - next_month['low']:.6f} |\n\n")
            
            # Additional price forecast visualizations
            plt.figure(figsize=(12, 8))
            
            # Plot current price and forecasts
            horizons = ['Current', 'Next Day', 'Next Week', 'Next Month']
            expected_prices = [recent_price, next_day['expected'], next_week['expected'], next_month['expected']]
            lower_bounds = [recent_price, next_day['low'], next_week['low'], next_month['low']]
            upper_bounds = [recent_price, next_day['high'], next_week['high'], next_month['high']]
            
            x = range(len(horizons))
            plt.plot(x, expected_prices, 'o-', color='blue', linewidth=2, label='Expected Price')
            plt.fill_between(x, lower_bounds, upper_bounds, color='blue', alpha=0.2, label='95% Credible Interval')
            
            plt.xlabel('Time Horizon')
            plt.ylabel('Price')
            plt.title(f'{asset_name} Price Forecast')
            plt.xticks(x, horizons)
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('images/price_forecast.png')
            plt.close()
            f.write("![Price Forecast](images/price_forecast.png)\n\n")
        
        f.write("## Trading Recommendation\n\n")
        f.write(f"### Signal: {signal} ({confidence} Confidence)\n\n")
        
        if signal == "BUY":
            f.write(f"**Recommendation:** Enter a long position with {position_size_pct:.1f}% of available capital.\n\n")
            if recent_price:
                stop_loss = recent_price * (1 - 1.5 * mean_volatility)
                take_profit = recent_price * (1 + 2 * mean_volatility)
                f.write(f"- **Entry Price:** {recent_price:.6f}\n")
                f.write(f"- **Stop Loss:** {stop_loss:.6f} (approximately {1.5 * mean_volatility * 100:.1f}% below entry)\n")
                f.write(f"- **Take Profit:** {take_profit:.6f} (approximately {2 * mean_volatility * 100:.1f}% above entry)\n\n")
        elif signal == "SELL":
            f.write(f"**Recommendation:** Enter a short position with {position_size_pct:.1f}% of available capital.\n\n")
            if recent_price:
                stop_loss = recent_price * (1 + 1.5 * mean_volatility)
                take_profit = recent_price * (1 - 2 * mean_volatility)
                f.write(f"- **Entry Price:** {recent_price:.6f}\n")
                f.write(f"- **Stop Loss:** {stop_loss:.6f} (approximately {1.5 * mean_volatility * 100:.1f}% above entry)\n")
                f.write(f"- **Take Profit:** {take_profit:.6f} (approximately {2 * mean_volatility * 100:.1f}% below entry)\n\n")
        else:  # NEUTRAL
            f.write(f"**Recommendation:** Hold current positions or consider a neutral strategy.\n\n")
            f.write(f"- Consider allocating {position_size_pct/2:.1f}% to long positions and {position_size_pct/2:.1f}% to short positions.\n")
            f.write(f"- Alternatively, await stronger directional signals before entering new positions.\n\n")
        
        # Additional trading strategy based on forecast
        if price_predictions and recent_price is not None:
            next_day = price_predictions['next_day']
            expected_move_pct = (next_day['expected'] - recent_price) / recent_price * 100
            
            f.write("### Short-Term Strategy Based on Price Forecast\n\n")
            if expected_move_pct > 1.0:
                f.write(f"The expected price movement for tomorrow is strongly positive ({expected_move_pct:.2f}%). Consider a more aggressive long position, potentially using call options or leveraged products if appropriate for your risk tolerance.\n\n")
            elif expected_move_pct < -1.0:
                f.write(f"The expected price movement for tomorrow is strongly negative ({expected_move_pct:.2f}%). Consider a more aggressive short position, potentially using put options or leveraged products if appropriate for your risk tolerance.\n\n")
            else:
                f.write(f"The expected price movement for tomorrow is relatively small ({expected_move_pct:.2f}%). Consider focusing on range-bound trading strategies or accumulating positions at favorable prices within the predicted range.\n\n")
            
            range_width_pct = (next_day['high'] - next_day['low']) / recent_price * 100
            f.write(f"The predicted price range for tomorrow spans {range_width_pct:.2f}% of the current price, which suggests {'significant' if range_width_pct > 5 else 'moderate' if range_width_pct > 2 else 'limited'} intraday trading opportunities.\n\n")

        # Add the rest of the detailed analysis
        f.write("## Detailed Analysis\n\n")
        f.write("### Return Distribution\n")
        f.write(f"The analysis of historical returns shows an expected daily return of {mean_return:.6f} with a volatility of {mean_volatility:.6f}. ")
        
        if mu_ci[0] < 0 < mu_ci[1]:
            f.write("The 95% credible interval for the mean return contains zero, indicating uncertainty about the true direction of returns.\n\n")
        elif mu_ci[0] > 0:
            f.write("The 95% credible interval for the mean return is entirely positive, suggesting a reliable upward trend.\n\n")
        else:
            f.write("The 95% credible interval for the mean return is entirely negative, suggesting a reliable downward trend.\n\n")
        
        f.write("### Volatility Analysis\n")
        f.write(f"The estimated volatility of {mean_volatility:.6f} indicates ")
        if mean_volatility > 0.1:
            f.write("high levels of price fluctuation, typical of cryptocurrency markets. Proper risk management is essential.\n\n")
        elif mean_volatility > 0.05:
            f.write("moderate levels of price fluctuation. Standard risk management practices are advised.\n\n")
        else:
            f.write("relatively stable price behavior. Tighter stop-losses can be considered.\n\n")
        
        f.write("### Prediction for Next Trading Day\n")
        f.write(f"Based on the posterior predictive distribution, there is a {prob_positive:.1f}% probability of a positive return on the next trading day. ")
        f.write(f"The 95% predictive interval for the next-day return is [{pred_ci[0]:.6f}, {pred_ci[1]:.6f}].\n\n")
        
        if price_predictions and recent_price is not None:
            f.write("### Extended Time Horizon Predictions\n\n")
            
            # Next week analysis
            next_week = price_predictions['next_week']
            next_week_expected_change = (next_week['expected'] - recent_price) / recent_price * 100
            f.write(f"**One-Week Outlook:** The expected price after one week is {next_week['expected']:.6f}, representing a {'+' if next_week_expected_change >= 0 else ''}{next_week_expected_change:.2f}% change from the current price. ")
            f.write(f"The 95% credible interval for the one-week price is [{next_week['low']:.6f}, {next_week['high']:.6f}].\n\n")
            
            # Next month analysis
            next_month = price_predictions['next_month']
            next_month_expected_change = (next_month['expected'] - recent_price) / recent_price * 100
            f.write(f"**One-Month Outlook:** The expected price after one month is {next_month['expected']:.6f}, representing a {'+' if next_month_expected_change >= 0 else ''}{next_month_expected_change:.2f}% change from the current price. ")
            f.write(f"The 95% credible interval for the one-month price is [{next_month['low']:.6f}, {next_month['high']:.6f}].\n\n")
            
            f.write("These extended forecasts become increasingly uncertain with time horizon. The predictions incorporate both parameter uncertainty and random market movements, resulting in wider intervals for longer horizons.\n\n")
        
        f.write("### Risk Assessment\n")
        f.write(f"The current risk level is assessed as {risk_level}, based on the estimated volatility and uncertainty in return predictions. ")
        
        if risk_level == "HIGH":
            f.write("This suggests using smaller position sizes and wider stop-losses.\n\n")
        elif risk_level == "MEDIUM":
            f.write("This suggests standard position sizing and stop-loss practices.\n\n")
        else:
            f.write("This suggests the potential for larger position sizes, but still with appropriate risk controls.\n\n")
        
        f.write("## Methodology\n")
        f.write("This analysis uses Bayesian inference with Rejection Sampling to estimate the posterior distributions of mean return and volatility. ")
        f.write("The rejection sampling algorithm draws candidates from proposal distributions and accepts them based on the ratio of the posterior to the proposal density. ")
        f.write("This method provides exact samples from the posterior distribution but can be less efficient than Gibbs sampling for high-dimensional problems. ")
        f.write("The posterior predictive distribution incorporates both parameter uncertainty and inherent market randomness.\n\n")
        
        f.write("For multi-period forecasts (weekly and monthly), the analysis simulates multiple daily returns using random draws from the posterior predictive distribution ")
        f.write("and compounds them to generate price paths. The reported intervals represent the 95% credible range of these simulated paths.\n\n")
        
        f.write("## Limitations and Disclaimers\n")
        f.write("1. This analysis is based solely on historical price data and does not incorporate fundamental factors, news events, or market sentiment.\n")
        f.write("2. Past performance is not indicative of future results. Financial markets are complex systems subject to numerous influences.\n")
        f.write("3. This report is generated automatically and should be used as one input among many for trading decisions.\n")
        f.write("4. The model assumes returns follow a relatively stable distribution, which may not hold during market regime changes.\n")
        f.write("5. Longer-term forecasts are subject to increasing uncertainty and should be interpreted with appropriate caution.\n\n")
        
        f.write("## Report Generation\n")
        f.write(f"This analysis was generated automatically on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} based on available historical data.")
        
    print("Generated analysis.md with trading recommendations including price forecasts for next day, week, and month")

if __name__ == '__main__':
    run_mcmc_estimation()
