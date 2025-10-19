import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def ensure_images_dir():
    """Creates the images directory if it doesn't exist."""
    if not os.path.exists('images'):
        os.makedirs('images')
        print("Created 'images' directory for storing plots")

def plot_price_trend():
    """Plots price trend over time."""
    ensure_images_dir()
    df = pd.read_csv('prices.csv', parse_dates=['Date'])
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'])
    plt.title('XRP Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    plt.savefig('images/price_trend.png')
    plt.close()
    print("Generated price_trend.png")

def plot_returns_histogram():
    """Plots histogram of log returns."""
    ensure_images_dir()
    df = pd.read_csv('returns.csv')
    plt.figure(figsize=(10, 6))
    df['LogReturn'].hist(bins=50, density=True, alpha=0.7)
    plt.title('Histogram of Daily Log Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig('images/returns_histogram.png')
    plt.close()
    print("Generated returns_histogram.png")

def plot_rolling_stats():
    """Plots rolling mean and standard deviation."""
    ensure_images_dir()
    df = pd.read_csv('rolling_stats.csv', parse_dates=['Date'])
    plt.figure(figsize=(12, 8))
    
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['Price'], label='Price', alpha=0.5)
    plt.plot(df['Date'], df['RollingMean'], label='20-Day Rolling Mean', color='orange')
    plt.title('Price and 20-Day Rolling Mean')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(df['Date'], df['RollingStd'], label='20-Day Rolling Std Dev', color='green')
    plt.title('20-Day Rolling Standard Deviation (Volatility)')
    plt.xlabel('Date')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True)
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('images/volatility_analysis.png')
    plt.close()
    print("Generated volatility_analysis.png")

if __name__ == '__main__':
    plot_price_trend()
    plot_returns_histogram()
    plot_rolling_stats()
